"""The evidence journal: schema, round-trip, artifacts, and a real run.

The journal is the artifact a human reads to decide whether to believe the
agent, so these tests treat its *shape* as the contract:

* what the writer puts on disk equals what the pure serializers produce;
* the credential cannot reach the file;
* a malformed file names its own line instead of failing anonymously;
* an absent advisor is recorded as absent, not as ``null``;
* every line is flushed as it is written, so a killed run leaves a valid prefix;
* the observer drives a real ``ExecutionLoop`` over a real emulator;
* the two Pillow drawings are real PNGs, and an artifact that cannot be written
  degrades the tick instead of failing the run;
* ``frame_index`` and ``emulator_tick`` are recorded separately and differ.

Only the first couple of tests build objects by hand; the run tests use a real
``MarioEmulator(real_time=False)`` and the real ``ExecutionLoop`` with a scripted
transport, so the records are genuine rows rather than fixtures.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import mario_jev.journal
import numpy as np
import pytest
from PIL import Image

from mario_jev.advisor import (
    REASON_TORCH_MISSING,
    AdvisorAbsence,
    AdvisorAdvice,
    AdvisorMove,
    PlayJevAdvisor,
)
from mario_jev.config import Settings
from mario_jev.emulator import MarioEmulator, build_actions
from mario_jev.jev import ChoiceAnswer, HttpResult, JevClient, JevResponse, Usage
from mario_jev.journal import (
    Journal,
    JournalFormatError,
    JournalWriter,
    advisor_record,
    iter_journal,
    read_journal,
    run_record,
    summary_record,
    tick_record,
    write_frame_png,
    write_trajectory_png,
)
from mario_jev.perception import detect
from mario_jev.policy import (
    DecisionSource,
    ExecutionLoop,
    RunSummary,
    TickDecision,
    TickRecord,
)
from mario_jev.state import (
    MarioState,
    NearestEnemy,
    PlayerFeatures,
    Progress,
    TerminalOutcome,
    TerrainAhead,
)
from mario_jev.trajectory import CandidateRollout, CostSource, RolloutTable

#: A synthetic credential. A real key is never used, and never printed by a test.
SENTINEL_KEY = "judge0_SENTINEL_journal_do_not_leak"

#: The injected instant. Every deterministic test pins this and never sleeps.
STARTED_AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

#: The seven ``simple`` actions, keyed by name.
ACTIONS = {action.name: action for action in build_actions("simple")}

#: The PNG signature, checked before Pillow is asked to open anything.
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

SCREEN = (240, 256, 3)


# -- builders -----------------------------------------------------------------------


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "jev_api_key": SENTINEL_KEY,
        "decision_cadence_frames": 1,
        "confidence_threshold": 0.6,
        "rollout_budget_ms": 120.0,
        "real_time": False,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def fixed_clock() -> datetime:
    """A clock that never moves, so a test never sleeps and never flakes."""
    return STARTED_AT


def make_state(tick: int = 0) -> MarioState:
    return MarioState(
        tick=tick,
        player=PlayerFeatures(
            x=40, y=176, screen_x=40, feet_row=191, on_ground=True, facing="right"
        ),
        nearest_enemy=NearestEnemy(distance_px=64, approaching=True),
        terrain_ahead=TerrainAhead(
            gap_columns=0,
            gap_distance_px=None,
            wall_height=0,
            frames_to_jump_window=None,
            vertical_clearance=0,
            lookahead_px=64,
        ),
        progress=Progress(world="1-1", x_max=40, lives=3, time_left=400),
        speed_px_per_tick=2.0,
    )


def make_candidate(
    name: str,
    *,
    dx: int,
    risk: float,
    outcome: TerminalOutcome = TerminalOutcome.LANDS_SAFE,
) -> CandidateRollout:
    return CandidateRollout(
        action=ACTIONS[name],
        horizon=8,
        frames=8,
        start_x=40,
        final_x=40 + dx,
        x_max=40 + dx,
        outcome=outcome,
        risk=risk,
        final_enemy_distance=None,
        terminated=outcome in {TerminalOutcome.FALLS_IN_GAP, TerminalOutcome.HITS_ENEMY},
        truncated=False,
    )


def make_table(
    *candidates: CandidateRollout, tick: int = 0
) -> RolloutTable:
    return RolloutTable(
        tick=tick,
        start_x=40,
        horizon=8,
        budget_ms=120.0,
        elapsed_ms=1.0,
        sweep_ms=0.9,
        base_ms=0.5,
        per_frame_ms=1.0,
        calibration_frames=0,
        passes=1,
        cost_source=CostSource.CALIBRATED,
        steps_taken=16,
        candidates=tuple(candidates),
    )


def default_table(tick: int = 0) -> RolloutTable:
    return make_table(
        make_candidate("right", dx=12, risk=0.2),
        make_candidate("left", dx=-3, risk=0.5, outcome=TerminalOutcome.NO_PROGRESS),
        tick=tick,
    )


def make_record(
    *,
    tick: int = 0,
    frame_index: int = 0,
    emulator_tick: int = 0,
    advisor: AdvisorAdvice | None = None,
    advisor_absence: AdvisorAbsence | None = None,
    frame: np.ndarray | None = None,
    terminated: bool = False,
    truncated: bool = False,
    jev_response: dict[str, object] | None = None,
) -> TickRecord:
    pixels = np.zeros(SCREEN, dtype=np.uint8) if frame is None else frame
    return TickRecord(
        tick=tick,
        frame_index=frame_index,
        emulator_tick=emulator_tick,
        frame=pixels,
        perception=detect(pixels, tick=frame_index),
        state=make_state(tick=tick),
        rollout=default_table(tick=tick),
        jev_questions=None,
        jev_response=jev_response,
        decision=TickDecision(
            action="right",
            source=DecisionSource.FALLBACK_NO_KEY,
            confidence=None,
            risk=None,
            reason="no_credential",
            jev_called=False,
        ),
        advisor=advisor,
        advisor_absence=advisor_absence,
        latencies_ms={"rollout": 1.0, "jev": 0.0, "advisor": 0.0, "total": 2.0},
        degradations=(),
        terminated=terminated,
        truncated=truncated,
    )


def make_summary(**overrides: object) -> RunSummary:
    values: dict[str, object] = {
        "ticks": 2,
        "frames": 2,
        "decisions": 1,
        "sources": {DecisionSource.FALLBACK_NO_KEY: 1, DecisionSource.JEV: 1},
        "degradations": {"timing_overrun": 1},
        "elapsed_s": 0.25,
        "decision_rate_hz": 4.0,
        # No default on ``RunSummary.termination``: every construction has to
        # say why the run stopped, and this builder is a construction.
        "termination": "max_ticks",
    }
    values.update(overrides)
    return RunSummary(**values)  # type: ignore[arg-type]


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def open_png(path: Path) -> Image.Image:
    assert path.read_bytes()[:8] == PNG_MAGIC, f"{path} is not a PNG"
    assert path.stat().st_size > 0
    image = Image.open(path)
    assert image.format == "PNG"
    return image


# -- 1. round-trip ------------------------------------------------------------------


def test_a_small_run_round_trips_through_the_file(tmp_path: Path) -> None:
    settings = make_settings()
    first = make_record(tick=0, frame_index=0, emulator_tick=0)
    second = make_record(tick=1, frame_index=1, emulator_tick=40)
    summary = make_summary()
    path = tmp_path / "run.jsonl"

    writer = JournalWriter(path, settings=settings, clock=fixed_clock)
    writer.observer()(first)
    writer.observer()(second)
    writer.write_summary(summary)
    writer.close()

    journal = read_journal(path)

    assert journal.run == run_record(settings, started_at=STARTED_AT)
    assert journal.ticks == (tick_record(first), tick_record(second))
    assert journal.summary == summary_record(summary)

    raw = path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    lines = raw.splitlines()
    assert len(lines) == 4
    kinds = [json.loads(line)["kind"] for line in lines]
    assert kinds == ["run", "tick", "tick", "summary"]
    for line in lines:
        assert isinstance(json.loads(line), dict)


def test_every_line_is_plain_json_data(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    writer = JournalWriter(path, settings=make_settings(), clock=fixed_clock)
    writer.observer()(make_record())
    writer.write_summary(make_summary())
    writer.close()

    entries = list(iter_journal(path))

    assert [entry["kind"] for entry in entries] == ["run", "tick", "summary"]
    assert entries[1]["degradations"] == []
    assert isinstance(entries[1]["rollout"]["candidates"], list)
    assert entries[1]["state"]["progress"]["world"] == "1-1"


def test_a_run_header_omits_artifacts_dir_until_artifacts_are_enabled(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    with JournalWriter(path, settings=make_settings(), clock=fixed_clock) as writer:
        writer.write_summary(make_summary())

    journal = read_journal(path)

    assert "artifacts_dir" not in journal.run
    assert journal.run["settings"] == make_settings().redacted


def test_the_header_is_written_on_construction(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    writer = JournalWriter(path, settings=make_settings(), clock=fixed_clock)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["kind"] == "run"
    writer.close()


# -- 1b. the end of the run reaches the file ----------------------------------------


#: Every key a tick line carries, and every key a summary line carries. These are
#: the schema, so they are written out here rather than read off the code under
#: test: a key that appears without this list changing is an unreviewed schema
#: change, and a reader of the journal has no way to notice one.
TICK_LINE_KEYS = (
    "kind",
    "tick",
    "frame_index",
    "emulator_tick",
    "state",
    "rollout",
    "questions",
    "response",
    "decision",
    "advisor",
    "latencies_ms",
    "degradations",
    "terminated",
    "truncated",
    "frame_png",
    "trajectory_png",
)
SUMMARY_LINE_KEYS = (
    "kind",
    "ticks",
    "frames",
    "decisions",
    "sources",
    "degradations",
    "elapsed_s",
    "decision_rate_hz",
    "termination",
)


def test_the_written_lines_say_the_run_died_and_which_tick_ended_it(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    writer = JournalWriter(path, settings=make_settings(), clock=fixed_clock)
    writer.observer()(make_record(tick=0))
    writer.observer()(make_record(tick=1, frame_index=1, terminated=True))
    writer.write_summary(make_summary(ticks=2, termination="death"))
    writer.close()

    journal = read_journal(path)

    assert journal.summary["termination"] == "death"
    assert [tick["terminated"] for tick in journal.ticks] == [False, True]
    assert [tick["truncated"] for tick in journal.ticks] == [False, False]
    # Read off the raw bytes too: the flags must survive JSON as booleans, not
    # as the strings a careless serializer would leave behind.
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines[2]["terminated"] is True
    assert lines[3]["termination"] == "death"


def test_a_truncated_run_is_written_as_truncated(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    with JournalWriter(path, settings=make_settings(), clock=fixed_clock) as writer:
        writer.observer()(make_record(truncated=True))
        writer.write_summary(make_summary(ticks=1, termination="truncated"))

    journal = read_journal(path)

    assert journal.ticks[0]["truncated"] is True
    assert journal.ticks[0]["terminated"] is False
    assert journal.summary["termination"] == "truncated"


def test_the_written_key_sets_are_exactly_the_documented_ones(tmp_path: Path) -> None:
    """The schema docstrings enumerate the fields; a stale one is a false claim.

    Two halves, because either alone can pass while the file lies: the key sets
    on disk must be the pinned ones, and every pinned key must appear in the
    enumerations a reader is pointed at.
    """
    path = tmp_path / "run.jsonl"
    with JournalWriter(path, settings=make_settings(), clock=fixed_clock) as writer:
        writer.observer()(make_record())
        writer.write_summary(make_summary())

    journal = read_journal(path)

    assert tuple(journal.ticks[0]) == TICK_LINE_KEYS
    assert tuple(journal.summary) == SUMMARY_LINE_KEYS

    package_doc = mario_jev.journal.__doc__ or ""
    tick_doc = tick_record.__doc__ or ""
    summary_doc = summary_record.__doc__ or ""
    for key in TICK_LINE_KEYS:
        assert f"``{key}``" in package_doc, f"{key} is undocumented in the package"
        assert f"``{key}``" in tick_doc, f"{key} is missing from tick_record's docstring"
    for key in SUMMARY_LINE_KEYS:
        assert f"``{key}``" in package_doc, f"{key} is undocumented in the package"
        assert f"``{key}``" in summary_doc, f"{key} is missing from summary_record's"


def test_a_written_tick_carries_the_attempt_count_and_the_header_the_budget(
    tmp_path: Path,
) -> None:
    """Both halves of the reconstruction have to survive the file.

    A 20 s stall is only reconstructible from the journal because the header says
    the timeout was 20 s and the tick's response says the call took two attempts.
    Read back off disk, not off the in-memory record.
    """
    settings = make_settings(jev_timeout_s=20.0, jev_max_attempts=3)
    retried = JevResponse(
        model="jev-test",
        answers={
            "best_action": ChoiceAnswer(
                type="choice",
                choice="right",
                confidence=0.9,
                probabilities={"right": 1.0},
            )
        },
        usage=Usage(input_tokens=10, output_tokens=5),
        latency_ms=577.0,
        attempts=2,
    )
    path = tmp_path / "run.jsonl"
    with JournalWriter(path, settings=settings, clock=fixed_clock) as writer:
        writer.observer()(make_record(jev_response=retried.as_dict()))
        writer.write_summary(make_summary())

    journal = read_journal(path)

    assert journal.run["settings"]["jev_timeout_s"] == 20.0
    assert journal.run["settings"]["jev_max_attempts"] == 3
    assert journal.ticks[0]["response"]["attempts"] == 2
    assert journal.ticks[0]["response"]["latency_ms"] == 577.0
    # Off the raw bytes too: the count must survive JSON as a number.
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines[1]["response"]["attempts"] == 2


# -- 2. no credential material ------------------------------------------------------


def test_no_credential_material_reaches_the_journal(tmp_path: Path) -> None:
    settings = make_settings()
    path = tmp_path / "run.jsonl"
    writer = JournalWriter(path, settings=settings, clock=fixed_clock)
    writer.observer()(make_record())
    writer.write_summary(make_summary())
    writer.close()

    text = path.read_text(encoding="utf-8")
    journal = read_journal(path)

    assert SENTINEL_KEY not in text
    for start in range(len(SENTINEL_KEY)):
        for end in range(start + 5, len(SENTINEL_KEY) + 1):
            fragment = SENTINEL_KEY[start:end]
            assert fragment not in text, fragment
    assert journal.run["settings"] == settings.redacted
    assert journal.run["settings"]["jev_available"] is True


#: A credential smuggled into the endpoint URL's userinfo instead of the key field.
URL_PASSWORD = "url_pw_this_must_never_reach_the_journal_0987654321"
ENDPOINT_WITH_USERINFO = f"https://alice:{URL_PASSWORD}@example.invalid/v1/systemone"


def test_no_credential_in_the_endpoint_url_reaches_the_header(tmp_path: Path) -> None:
    """A password in the endpoint URL is a credential, and the header is a log.

    The key field was already safe; this is the other place a credential can enter
    a URL, and ``redacted`` used to emit the endpoint verbatim. The test writes a
    real file and reads it back, because the claim is about the bytes on disk and
    not about the value of a property.
    """
    settings = make_settings(jev_endpoint=ENDPOINT_WITH_USERINFO)
    path = tmp_path / "run.jsonl"
    writer = JournalWriter(path, settings=settings, clock=fixed_clock)
    writer.observer()(make_record())
    writer.write_summary(make_summary())
    writer.close()

    text = path.read_text(encoding="utf-8")
    header = json.loads(text.splitlines()[0])

    assert URL_PASSWORD not in text
    assert "alice" not in text
    assert header["settings"]["jev_endpoint"] == "https://example.invalid/v1/systemone"
    # The host survives: a reader must still be able to tell where the run pointed.
    assert "example.invalid" in text


# -- 3. format errors name the line -------------------------------------------------


def header_line() -> str:
    return json.dumps(run_record(make_settings(), started_at=STARTED_AT))


def tick_line() -> str:
    return json.dumps(tick_record(make_record()))


def summary_line() -> str:
    return json.dumps(summary_record(make_summary()))


@pytest.mark.parametrize(
    ("lines", "expected_line", "reason"),
    [
        ([header_line(), '{"kind": "summary", "ticks": 1'], 2, "truncated"),
        ([header_line(), "{not json}", summary_line()], 2, "bad JSON"),
        ([header_line(), '{"kind": "mystery"}', summary_line()], 2, "unknown kind"),
        ([header_line(), "[1, 2, 3]", summary_line()], 2, "not an object"),
        ([header_line(), "", tick_line(), summary_line()], 2, "blank line"),
        ([tick_line(), summary_line()], 1, "missing header"),
        ([header_line(), tick_line()], 2, "missing trailer"),
        ([header_line()], 1, "header only"),
        ([header_line(), summary_line(), tick_line()], 3, "tick after summary"),
        ([header_line(), tick_line(), header_line(), summary_line()], 3, "second header"),
        ([header_line(), summary_line(), summary_line()], 3, "second trailer"),
        ([], 1, "empty file"),
    ],
    ids=[
        "truncated",
        "bad-json",
        "unknown-kind",
        "not-an-object",
        "blank-line",
        "missing-header",
        "missing-trailer",
        "header-only",
        "tick-after-summary",
        "second-header",
        "second-trailer",
        "empty-file",
    ],
)
def test_a_malformed_journal_names_the_offending_line(
    tmp_path: Path, lines: list[str], expected_line: int, reason: str
) -> None:
    path = tmp_path / "broken.jsonl"
    write_lines(path, lines)

    with pytest.raises(JournalFormatError) as excinfo:
        read_journal(path)

    assert excinfo.value.line == expected_line, reason
    assert str(expected_line) in str(excinfo.value)
    assert excinfo.value.reason


def test_a_non_utf8_line_is_a_format_error_naming_it(tmp_path: Path) -> None:
    """An undecodable line is a malformed line, not a decoder exception.

    The journal is UTF-8 JSONL, and the reader's contract is that a line it cannot
    read is reported with its own number. A byte sequence that is not UTF-8 is such
    a line; letting ``UnicodeDecodeError`` escape would break the contract for the
    one input a consumer cannot anticipate from the shape of the file.
    """
    path = tmp_path / "not-utf8.jsonl"
    path.write_bytes((header_line() + "\n").encode("utf-8") + b"\xff\xfe not utf-8\n")

    with pytest.raises(JournalFormatError) as excinfo:
        read_journal(path)

    assert excinfo.value.line == 2
    assert "UTF-8" in excinfo.value.reason


def test_a_well_formed_journal_has_no_error(tmp_path: Path) -> None:
    path = tmp_path / "good.jsonl"
    write_lines(path, [header_line(), tick_line(), summary_line()])

    journal = read_journal(path)

    assert journal.run["kind"] == "run"
    assert [entry["kind"] for entry in journal.ticks] == ["tick"]
    assert journal.summary["kind"] == "summary"


# -- 4. the advisor block -----------------------------------------------------------


def test_advisor_absence_is_recorded_as_absent_not_null() -> None:
    disabled = tick_record(make_record(advisor=None))

    assert disabled["advisor"] == {
        "available": False,
        "reason": "disabled",
        "measured": False,
    }
    assert disabled["advisor"] is not None


def test_advisor_record_has_exactly_three_truthful_shapes() -> None:
    move = AdvisorMove(name="right", probability=0.8)
    answer = AdvisorAdvice(
        model_id="OmniJev/PlayJev-0.8B",
        backend="playjev-torch",
        moves=(move,),
        top=move,
        latency_ms=1.0,
    )
    measured = AdvisorAbsence(model_id="OmniJev/PlayJev-0.8B", reason=REASON_TORCH_MISSING)

    present = advisor_record(answer)
    absent = advisor_record(None, measured)
    disabled = advisor_record(None, None)

    assert present["available"] is True
    assert "measured" not in present
    assert absent == {
        "available": False,
        "reason": REASON_TORCH_MISSING,
        "measured": True,
    }
    assert disabled == {"available": False, "reason": "disabled", "measured": False}
    assert absent != disabled


def test_advisor_block_separates_absence_from_an_empty_answer() -> None:
    move = AdvisorMove(name="right", probability=0.8)
    answer = AdvisorAdvice(
        model_id="OmniJev/PlayJev-0.8B",
        backend="playjev-torch",
        moves=(move, AdvisorMove(name="NOOP", probability=0.2)),
        top=move,
        latency_ms=2.5,
    )
    empty = AdvisorAdvice(
        model_id="OmniJev/PlayJev-0.8B",
        backend="playjev-torch",
        moves=(),
        top=move,
        latency_ms=0.5,
    )

    absent = tick_record(make_record(advisor=None))["advisor"]
    present = tick_record(make_record(advisor=answer))["advisor"]
    blank = tick_record(make_record(advisor=empty))["advisor"]

    assert absent["available"] is False
    assert present["available"] is True
    assert present["top"] == {"name": "right", "probability": 0.8}
    assert blank["available"] is True
    assert blank["moves"] == []
    assert absent != blank
    assert present != blank


# -- 5. flush per line --------------------------------------------------------------


def test_each_line_is_flushed_before_close(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    writer = JournalWriter(path, settings=make_settings(), clock=fixed_clock)

    writer.observer()(make_record(tick=0, frame_index=0, emulator_tick=0))

    # Read the file while the writer is still open: no flush, no line.
    entries = list(iter_journal(path))
    assert [entry["kind"] for entry in entries] == ["run", "tick"]
    assert entries[1]["tick"] == 0
    assert not path.read_text(encoding="utf-8").endswith("\n\n")

    writer.observer()(make_record(tick=1, frame_index=1, emulator_tick=9))
    assert len(list(iter_journal(path))) == 3
    writer.close()


# -- 6. a real loop, a real emulator, a scripted transport --------------------------


class ScriptedTransport:
    """Hands back queued results and remembers the requests; never opens a socket."""

    def __init__(self, *results: HttpResult) -> None:
        self.queued = list(results)
        self.requests: list[object] = []

    def __call__(self, request: object) -> HttpResult:
        self.requests.append(request)
        if not self.queued:
            raise AssertionError("the client sent more requests than the test queued")
        return self.queued.pop(0)


class SilentSleeper:
    def __call__(self, delay: float) -> None:
        raise AssertionError("a retry must not sleep in these tests")


def ok_result(choice: str = "right") -> HttpResult:
    body = json.dumps(
        {
            "model": "jev-test",
            "answers": {
                "best_action": {
                    "type": "choice",
                    "choice": choice,
                    "confidence": 0.9,
                    "probabilities": {choice: 1.0},
                },
                "risk": {
                    "type": "score",
                    "score": 0.2,
                    "confidence": 0.9,
                    "legend": {"0": "safe", "1": "minor risk"},
                    "probabilities": {"0": 0.6, "1": 0.4},
                },
                "lethal_ahead": {"type": "noul", "noul": 0.1},
                "gap_severity": {
                    "type": "score",
                    "score": 0.0,
                    "confidence": 0.9,
                    "legend": {"0": "no gap"},
                    "probabilities": {"0": 1.0},
                },
            },
            "usage": {"input_tokens": 5, "output_tokens": 5},
        }
    )
    return HttpResult(status=200, body=body, elapsed_ms=1.0)


def run_observed(
    tmp_path: Path, emulator: MarioEmulator, *, max_ticks: int = 2
) -> tuple[Journal, RunSummary, Settings]:
    """Drive a real loop into a real journal and hand back what was written."""
    settings = make_settings()
    transport = ScriptedTransport(*[ok_result("right") for _ in range(max_ticks)])
    client = JevClient(
        endpoint=settings.jev_endpoint,
        model=settings.jev_model,
        api_key=settings.jev_api_key,
        timeout_s=settings.jev_timeout_s,
        max_attempts=1,
        transport=transport,
        sleeper=SilentSleeper(),
    )
    loop = ExecutionLoop(settings, emulator=emulator, client=client)
    path = tmp_path / "run.jsonl"
    writer = JournalWriter(path, settings=settings, clock=fixed_clock)
    summary = loop.run(max_ticks=max_ticks, observer=writer.observer())
    writer.write_summary(summary)
    writer.close()
    assert transport.requests, "the loop must have asked Jev"
    return read_journal(path), summary, settings


def run_with_advisor(
    path: Path,
    emulator: MarioEmulator,
    *,
    advisor: object | None,
    advisor_enabled: bool,
) -> Journal:
    """Drive one real tick whose journal records the advisor block and nothing else new."""
    settings = make_settings(advisor_enabled=advisor_enabled)
    transport = ScriptedTransport(ok_result("right"))
    client = JevClient(
        endpoint=settings.jev_endpoint,
        model=settings.jev_model,
        api_key=settings.jev_api_key,
        timeout_s=settings.jev_timeout_s,
        max_attempts=1,
        transport=transport,
        sleeper=SilentSleeper(),
    )
    loop = ExecutionLoop(
        settings, emulator=emulator, client=client, advisor=advisor  # type: ignore[arg-type]
    )
    writer = JournalWriter(path, settings=settings, clock=fixed_clock)
    summary = loop.run(max_ticks=1, observer=writer.observer())
    writer.write_summary(summary)
    writer.close()
    return read_journal(path)


def test_the_observer_drives_a_real_run_end_to_end(
    tmp_path: Path, emulator: MarioEmulator
) -> None:
    journal, summary, settings = run_observed(tmp_path, emulator, max_ticks=2)

    assert journal.run == run_record(settings, started_at=STARTED_AT)
    assert len(journal.ticks) == 2
    assert [tick["frame_index"] for tick in journal.ticks] == [0, 1]
    assert [tick["tick"] for tick in journal.ticks] == [0, 1]
    assert journal.summary == summary_record(summary)
    assert journal.summary["ticks"] == 2
    assert [tick["decision"]["source"] for tick in journal.ticks] == ["jev", "jev"]
    assert all(tick["questions"] is not None for tick in journal.ticks)
    assert all(tick["response"] is not None for tick in journal.ticks)
    assert all(
        tick["advisor"] == {"available": False, "reason": "disabled", "measured": False}
        for tick in journal.ticks
    )
    assert all(tick["frame_png"] is None for tick in journal.ticks)
    assert all(tick["trajectory_png"] is None for tick in journal.ticks)


def test_a_real_run_distinguishes_a_measured_absence_from_a_disabled_advisor(
    tmp_path: Path, emulator_factory: object
) -> None:
    def fresh() -> MarioEmulator:
        instance: MarioEmulator = emulator_factory()  # type: ignore[operator]
        instance.reset(seed=0)
        return instance

    model_id = "OmniJev/PlayJev-0.8B"
    absent = PlayJevAdvisor(
        model_id, absence=AdvisorAbsence(model_id, REASON_TORCH_MISSING)
    )

    measured = run_with_advisor(
        tmp_path / "measured.jsonl",
        fresh(),
        advisor=absent,
        advisor_enabled=True,
    )
    disabled = run_with_advisor(
        tmp_path / "disabled.jsonl",
        fresh(),
        advisor=None,
        advisor_enabled=False,
    )

    assert measured.ticks[0]["advisor"] == {
        "available": False,
        "reason": REASON_TORCH_MISSING,
        "measured": True,
    }
    assert disabled.ticks[0]["advisor"] == {
        "available": False,
        "reason": "disabled",
        "measured": False,
    }
    # The whole point: an enabled-but-absent advisor is never written as disabled.
    assert measured.ticks[0]["advisor"] != disabled.ticks[0]["advisor"]


# -- 9. frame_index and emulator_tick are different quantities -----------------------


def test_emulator_tick_is_recorded_separately_from_the_frame_index(
    tmp_path: Path, emulator: MarioEmulator
) -> None:
    journal, _, _ = run_observed(tmp_path, emulator, max_ticks=2)

    assert [tick["frame_index"] for tick in journal.ticks] == [0, 1]
    assert [tick["emulator_tick"] for tick in journal.ticks][0] == 0
    second = journal.ticks[1]
    assert isinstance(second["emulator_tick"], int)
    assert second["emulator_tick"] > second["frame_index"], (
        "the rollout sweep steps the emulator counter; conflation is a defect"
    )


# -- 7. artifacts -------------------------------------------------------------------


def test_both_artifact_pngs_are_written_and_recorded_relative_to_the_dir(
    tmp_path: Path, emulator: MarioEmulator
) -> None:
    frame = emulator.frame()
    artifacts_dir = tmp_path / "artifacts"
    path = tmp_path / "run.jsonl"
    writer = JournalWriter(
        path,
        settings=make_settings(),
        clock=fixed_clock,
        artifacts_dir=artifacts_dir,
    )
    writer.observer()(make_record(tick=1, frame_index=3, frame=frame))
    writer.write_summary(make_summary(ticks=1, frames=1))
    writer.close()

    journal = read_journal(path)
    tick = journal.ticks[0]

    assert journal.run["artifacts_dir"] == str(artifacts_dir)
    assert tick["frame_png"] == "frame_000003.png"
    assert tick["trajectory_png"] == "trajectory_000001.png"
    assert tick["degradations"] == []
    for key in ("frame_png", "trajectory_png"):
        relative = tick[key]
        assert isinstance(relative, str) and not Path(relative).is_absolute()
        asset = artifacts_dir / relative
        assert asset.is_file()
        image = open_png(asset)
        assert image.size[0] > 0 and image.size[1] > 0
        image.close()


def test_write_frame_png_scales_the_real_frame(
    tmp_path: Path, emulator: MarioEmulator
) -> None:
    frame = emulator.frame()
    target = write_frame_png(
        tmp_path / "nested" / "frame.png", frame, detect(frame, tick=0), scale=3
    )

    image = open_png(target)
    assert image.size == (SCREEN[1] * 3, SCREEN[0] * 3)
    image.close()


def test_trajectory_png_survives_an_empty_candidate_table(tmp_path: Path) -> None:
    target = write_trajectory_png(tmp_path / "empty.png", make_table())

    assert target == tmp_path / "empty.png"
    image = open_png(target)
    assert image.size[0] > 0
    image.close()


def test_trajectory_png_marks_the_chosen_action(tmp_path: Path) -> None:
    chosen = make_record().decision
    target = write_trajectory_png(
        tmp_path / "trajectory.png", default_table(), decision=chosen
    )

    image = open_png(target)
    assert image.size[0] > 0
    image.close()


# -- 8. an artifact failure degrades, never fails the run ---------------------------


def test_an_unwritable_artifact_degrades_instead_of_failing_the_tick(
    tmp_path: Path,
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("this path is a file, not a directory", encoding="utf-8")
    artifacts_dir = blocker / "artifacts"  # cannot exist: its parent is a file
    path = tmp_path / "run.jsonl"
    writer = JournalWriter(
        path,
        settings=make_settings(),
        clock=fixed_clock,
        artifacts_dir=artifacts_dir,
    )

    writer.observer()(make_record(tick=0, frame_index=0))
    writer.write_summary(make_summary(ticks=1, frames=1))
    writer.close()

    journal = read_journal(path)
    tick = journal.ticks[0]

    assert tick["frame_png"] is None
    assert tick["trajectory_png"] is None
    assert any(
        entry.startswith("artifact_") for entry in tick["degradations"]
    ), tick["degradations"]
    assert journal.summary["ticks"] == 1


def test_the_writer_draws_the_frame_the_record_already_carries(tmp_path: Path) -> None:
    frame = np.full(SCREEN, 7, dtype=np.uint8)
    path = tmp_path / "run.jsonl"
    writer = JournalWriter(
        path,
        settings=make_settings(),
        clock=fixed_clock,
        artifacts_dir=tmp_path / "artifacts",
    )

    writer.observer()(make_record(frame=frame))
    writer.write_summary(make_summary(ticks=1, frames=1))
    writer.close()

    tick = read_journal(path).ticks[0]

    assert tick["frame_png"] == "frame_000000.png"
    assert tick["trajectory_png"] == "trajectory_000000.png"
    # No frame source is injected: the pixels and the perception come off the record,
    # so nothing is skipped and nothing is guessed.
    assert tick["degradations"] == []
    assert (tmp_path / "artifacts" / tick["frame_png"]).is_file()


# -- writer lifecycle ---------------------------------------------------------------


def test_close_is_idempotent_and_the_writer_is_a_context_manager(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    with JournalWriter(path, settings=make_settings(), clock=fixed_clock) as writer:
        writer.observer()(make_record())
        writer.write_summary(make_summary(ticks=1, frames=1))

    writer.close()
    writer.close()

    assert len(read_journal(path).ticks) == 1
    with pytest.raises(RuntimeError):
        writer.observer()(make_record())


def test_the_writer_refuses_a_tick_after_its_trailer(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    writer = JournalWriter(path, settings=make_settings(), clock=fixed_clock)
    writer.write_summary(make_summary())

    with pytest.raises(RuntimeError):
        writer.observer()(make_record())

    with pytest.raises(RuntimeError):
        writer.write_summary(make_summary())
    writer.close()


def test_a_naive_timestamp_is_refused() -> None:
    with pytest.raises(ValueError):
        run_record(make_settings(), started_at=datetime(2026, 1, 1, 0, 0, 0))
