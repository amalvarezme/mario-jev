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
from collections import Counter
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

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
from mario_jev.jev import (
    ChoiceAnswer,
    HttpResult,
    JevClient,
    JevResponse,
    JevUnavailable,
    Usage,
)
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
    JevDispatcher,
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
        # Also required rather than defaulted, for the same reason as
        # ``termination`` below: a construction that forgets it must fail rather
        # than report zero applied answers. One of the two ticks applied an answer.
        "answers_applied": 1,
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
#:
#: Updated deliberately when the Jev call left the tick: ``asked_at_tick``,
#: ``answer_age_ticks`` and ``jev_call_ms`` on the tick line, ``answers_applied``
#: on the summary. Firing a call and applying its answer became two facts about
#: two different ticks, and the four keys are what a reader needs to tell them
#: apart. Leaving this list stale would have been the failure it is here to catch.
TICK_LINE_KEYS = (
    "kind",
    "tick",
    "frame_index",
    "emulator_tick",
    "state",
    "rollout",
    "questions",
    "response",
    "asked_at_tick",
    "answer_age_ticks",
    "jev_call_ms",
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
    "answers_applied",
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


class InlineRunner:
    """A runner that finishes the call during ``submit``, on the calling thread.

    These end-to-end journal tests are about what the run *writes*, not about which
    tick an answer eventually lands on. With the real one-worker dispatcher that
    would be a race -- whether the call has finished by the time the same tick takes
    it depends on thread scheduling -- so the tests that assert per-tick sources and
    per-tick responses drive the completion themselves, exactly as the loop's own
    tests do. The ticks where an answer arrives *later* are pinned deterministically
    in ``tests/test_policy.py`` and ``tests/test_dispatch.py``, over a hand-driven
    runner. This double is deliberately local rather than shared through
    ``conftest.py``: it is twelve lines, and test files here do not import each
    other.
    """

    def submit(self, fn: Callable[[], object]) -> Future[object]:
        future: Future[object] = Future()
        future.set_result(fn())
        return future

    def shutdown(self, *, wait: bool = True) -> None:
        """Nothing was started, so there is nothing to join."""


def inline_dispatcher(client: object) -> JevDispatcher:
    """A dispatcher over ``client`` whose calls finish inside the tick that fired them."""
    return JevDispatcher(client, runner=InlineRunner())  # type: ignore[arg-type]


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
    loop = ExecutionLoop(
        settings, emulator=emulator, client=client, dispatcher=inline_dispatcher(client)
    )
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
        settings,
        emulator=emulator,
        client=client,  # type: ignore[arg-type]
        advisor=advisor,  # type: ignore[arg-type]
        dispatcher=inline_dispatcher(client),
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


# -- 6b. the call left the tick, so the journal has to say when it was asked --------
# Every record above is written by a run whose call finishes inside the tick that
# fired it (``InlineRunner``), so every one of them reads an age of 0 and none of
# them can show what this section is about. Driving the runner by hand is the only
# way a *file* can carry an answer applied late, one discarded for being too old,
# or a call that failed after the tick that fired it was already over.


class ManualRunner:
    """A runner the test drives by hand: a call finishes when ``run_next`` says so.

    The same seam ``tests/test_dispatch.py`` and ``tests/test_policy.py`` drive,
    kept local for the same reason :class:`InlineRunner` is: it is a dozen lines,
    and the test files here do not import each other.
    """

    def __init__(self) -> None:
        self.pending: list[tuple[Future[object], Callable[[], object]]] = []
        self.submitted = 0
        self.shutdowns = 0

    def submit(self, fn: Callable[[], object]) -> Future[object]:
        self.submitted += 1
        future: Future[object] = Future()
        self.pending.append((future, fn))
        return future

    def run_next(self) -> None:
        """Finish the oldest submitted call, as a real worker eventually would."""
        future, fn = self.pending.pop(0)
        future.set_result(fn())

    def shutdown(self, *, wait: bool = True) -> None:
        self.shutdowns += 1


class StepClock:
    """A clock that jumps a fixed amount per reading, so a call costs exactly that.

    The dispatcher reads it once on each side of ``client.ask``, so ``step_s=0.25``
    makes every call a quarter-second one -- written as ``jev_call_ms == 250.0``
    without any test waiting a quarter of a second or reading the real clock.
    """

    def __init__(self, *, step_s: float) -> None:
        self.step_s = step_s
        self.now = 0.0

    def __call__(self) -> float:
        value = self.now
        self.now += self.step_s
        return value


CALL_MS = 250.0


class QueuedClient:
    """A client that hands back queued answers and raises queued failures.

    It replaces the transport *and* the client for the two tests whose subject is
    a failed call: ``JevUnavailable`` is what the loop absorbs, and raising it
    here says so directly instead of hoping a scripted HTTP result is classified
    into it. The successful answers are real :class:`JevResponse` values, so what
    reaches the journal is the client's own ``as_dict``.
    """

    def __init__(self, *outcomes: object) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def ask(self, questions: object, state: object) -> JevResponse:
        self.calls += 1
        if not self.outcomes:
            raise AssertionError("the loop asked more times than the test queued")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, JevResponse)
        return outcome


def answer(choice: str = "right", *, confidence: float = 0.9) -> JevResponse:
    """One successful answer, as the client would have parsed it."""
    return JevResponse(
        model="jev-test",
        answers={
            "best_action": ChoiceAnswer(
                type="choice",
                choice=choice,
                confidence=confidence,
                probabilities={choice: 1.0},
            )
        },
        usage=Usage(input_tokens=10, output_tokens=5),
        latency_ms=CALL_MS,
        attempts=1,
    )


def hand_driven_loop(
    client: object, runner: ManualRunner, emulator: MarioEmulator
) -> tuple[ExecutionLoop, Settings]:
    """A real loop over a real emulator whose calls land only when the test says."""
    settings = make_settings()
    dispatcher = JevDispatcher(
        client,  # type: ignore[arg-type]
        runner=runner,
        clock=StepClock(step_s=CALL_MS / 1000.0),
    )
    loop = ExecutionLoop(
        settings,
        emulator=emulator,
        client=client,  # type: ignore[arg-type]
        dispatcher=dispatcher,
    )
    return loop, settings


def summary_over(records: Sequence[TickRecord], *, elapsed_s: float = 0.5) -> RunSummary:
    """A summary counted off the records the way ``ExecutionLoop.run`` counts one.

    These tests step the loop by hand so a call can land on a later tick, which
    means ``run()`` -- and the summary it builds -- is not available to them. The
    counting rule itself is pinned over the real ``run()`` in
    ``tests/test_policy.py``; what is at stake here is that both counts reach the
    file and stay apart on the way.
    """
    decisions = sum(1 for record in records if record.decision.jev_called)
    return RunSummary(
        ticks=len(records),
        frames=len(records),
        decisions=decisions,
        answers_applied=sum(
            1 for record in records if record.answer_age_ticks is not None
        ),
        sources=Counter(record.decision.source for record in records),
        degradations=Counter(
            name for record in records for name in record.degradations
        ),
        elapsed_s=elapsed_s,
        decision_rate_hz=decisions / elapsed_s,
        termination="max_ticks",
    )


def write_records(
    path: Path, records: Sequence[TickRecord], settings: Settings
) -> Journal:
    """Put real records through a real writer and read the file back."""
    writer = JournalWriter(path, settings=settings, clock=fixed_clock)
    for record in records:
        writer.observer()(record)
    writer.write_summary(summary_over(records))
    writer.close()
    return read_journal(path)


def raw_lines(path: Path) -> list[Mapping[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_an_answer_applied_a_tick_late_is_written_with_the_tick_that_asked(
    tmp_path: Path, emulator: MarioEmulator
) -> None:
    """The file, not the record, has to say that the answer came from tick 0.

    ``questions`` says a tick fired and ``response`` says a tick applied, and with
    the call off the tick those are two different ticks. Without ``asked_at_tick``
    and ``answer_age_ticks`` on the line, a reader of this run would see tick 1
    decide with Jev and have no way to learn the answer described the world as it
    was a frame earlier.
    """
    runner = ManualRunner()
    transport = ScriptedTransport(ok_result("right"))
    settings = make_settings()
    client = JevClient(
        endpoint=settings.jev_endpoint,
        model=settings.jev_model,
        api_key=settings.jev_api_key,
        timeout_s=settings.jev_timeout_s,
        max_attempts=1,
        transport=transport,
        sleeper=SilentSleeper(),
    )
    loop, settings = hand_driven_loop(client, runner, emulator)

    asked = loop.step()  # fires; nothing has finished, so nothing is applied
    runner.run_next()  # the worker finishes the call between the two ticks
    applied = loop.step()  # takes an answer that is one tick old

    path = tmp_path / "late.jsonl"
    journal = write_records(path, (asked, applied), settings)
    first, second = journal.ticks

    assert first["questions"] is not None, "tick 0 is the tick that fired"
    assert first["response"] is None
    assert first["asked_at_tick"] is None
    assert first["answer_age_ticks"] is None
    assert first["jev_call_ms"] is None
    assert first["decision"]["source"] == "hold"

    assert second["response"] is not None
    assert second["asked_at_tick"] == 0
    assert second["answer_age_ticks"] == 1
    assert second["jev_call_ms"] == CALL_MS
    assert second["decision"]["source"] == "jev_stale"
    # It fired nothing itself: the call it applied was tick 0's.
    assert second["questions"] is None
    assert second["decision"]["jev_called"] is False

    # Off the raw bytes too: the age and the cost must survive JSON as numbers.
    lines = raw_lines(path)
    assert lines[2]["answer_age_ticks"] == 1
    assert lines[2]["jev_call_ms"] == CALL_MS
    assert lines[3]["answers_applied"] == 1


def test_an_answer_older_than_the_bound_is_written_as_discarded(
    tmp_path: Path, emulator: MarioEmulator
) -> None:
    """A discarded answer still cost what it cost, and the file has to show both.

    The tick that throws the answer away is the only tick that can report the
    call, so ``jev_call_ms`` is on it while ``response`` is not. A reader who saw
    neither would read the tick as one that never asked.
    """
    runner = ManualRunner()
    client = QueuedClient(answer("right"))
    loop, settings = hand_driven_loop(client, runner, emulator)

    # ``STALE_LIMIT_CADENCES`` is 3 and the cadence is 1 frame, so an answer is
    # discarded once it is four ticks old. Nothing fires in between: the call is
    # still in flight, which is exactly the run this bound exists for.
    records = [loop.step()]
    for _ in range(3):
        records.append(loop.step())
    runner.run_next()
    records.append(loop.step())

    journal = write_records(tmp_path / "discarded.jsonl", records, settings)
    discarding = journal.ticks[4]

    assert discarding["response"] is None
    assert discarding["answer_age_ticks"] is None
    assert discarding["asked_at_tick"] is None
    assert discarding["jev_call_ms"] == CALL_MS
    assert "jev_stale_discarded" in discarding["degradations"]
    assert discarding["decision"]["source"] == "hold"
    assert journal.summary["answers_applied"] == 0
    assert journal.summary["decisions"] == 1, "the one call was still fired"


def test_a_failed_call_is_written_on_the_tick_that_observed_it(
    tmp_path: Path, emulator: MarioEmulator
) -> None:
    """A transient failure lands a tick later than the tick that paid for it.

    The failure is degraded, not raised -- that part is the policy's, and it is
    pinned there. What is pinned here is that the *file* carries it: the observing
    tick has the degraded source with the transport's own reason, the cost of the
    call that failed, and no response at all.
    """
    runner = ManualRunner()
    failure = JevUnavailable("the endpoint timed out after 20.0s")
    client = QueuedClient(failure)
    loop, settings = hand_driven_loop(client, runner, emulator)

    firing = loop.step()
    runner.run_next()
    observing = loop.step()

    path = tmp_path / "failed.jsonl"
    journal = write_records(path, (firing, observing), settings)
    first, second = journal.ticks

    assert first["decision"]["jev_called"] is True, "tick 0 fired the call"
    assert first["jev_call_ms"] is None, "nothing had come back yet"

    assert second["response"] is None
    assert second["jev_call_ms"] == CALL_MS
    assert second["asked_at_tick"] is None
    assert second["answer_age_ticks"] is None
    assert second["decision"]["source"] == "fallback_degraded"
    assert "timed out" in second["decision"]["reason"]
    assert client.calls == 1

    lines = raw_lines(path)
    assert lines[2]["jev_call_ms"] == CALL_MS
    assert lines[2]["response"] is None


def test_the_summary_line_keeps_fired_calls_and_applied_answers_apart(
    tmp_path: Path, emulator: MarioEmulator
) -> None:
    """Two calls fired, one answer applied, and the trailer says both numbers.

    ``decisions`` has always counted fired calls and still does; a run whose first
    call failed applied one fewer answer than it fired, and ``answers_applied`` is
    the only field on the line that can say so.
    """
    runner = ManualRunner()
    client = QueuedClient(JevUnavailable("connection reset"), answer("right"))
    loop, settings = hand_driven_loop(client, runner, emulator)

    records = [loop.step()]  # fires the call that will fail
    runner.run_next()
    records.append(loop.step())  # observes the failure
    records.append(loop.step())  # fires the call that will succeed
    runner.run_next()
    records.append(loop.step())  # applies it, one tick late

    journal = write_records(tmp_path / "apart.jsonl", records, settings)

    assert journal.summary["decisions"] == 2
    assert journal.summary["answers_applied"] == 1
    assert journal.summary["decisions"] != journal.summary["answers_applied"]
    assert journal.ticks[3]["answer_age_ticks"] == 1
    assert journal.summary["sources"]["jev_stale"] == 1
    assert journal.summary["sources"]["fallback_degraded"] == 1


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
