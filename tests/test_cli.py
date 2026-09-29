"""The command line and the static dashboard, driven in-process.

Nothing here spawns a subprocess and nothing is skipped: ``main([...])`` is called
directly with an explicit ``argv``, and the one network seam — the Jev transport —
is a scripted callable. The emulator is the real native one for the end-to-end
runs (with ``real_time=False``), so the journal these tests read back is a real
one.

The dashboard tests are what pin the honesty rules of the artifact: a tick that
made no Jev call is text ("no call") and a gap in the series, never a zero; a
missing picture is the words "not recorded"; and a reason taken from the journal
is escaped so it cannot inject markup.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from mario_jev import cli
from mario_jev.dashboard import build_html, render_text
from mario_jev.emulator import StepResult, build_actions
from mario_jev.jev import HttpResult, JevClient
from mario_jev.journal import Journal, read_journal
from mario_jev.state import TerminalOutcome
from mario_jev.trajectory import CandidateRollout, CostSource, RolloutTable

#: A synthetic credential. It is never a real key and is never printed.
SENTINEL_KEY = "apikey_SENTINEL_cli_do_not_leak"

#: A credential smuggled into the endpoint URL's userinfo instead of the key field.
#: Both readers here print the endpoint, so both have to redact it.
URL_PASSWORD = "url_pw_SENTINEL_cli_do_not_leak"
ENDPOINT_WITH_USERINFO = f"https://alice:{URL_PASSWORD}@example.invalid/v1/systemone"


# -- crafted journals --------------------------------------------------------------


def _run_header(*, stage: str = "1-1", artifacts_dir: object | None = None,
                jev_available: bool = True) -> dict[str, object]:
    run: dict[str, object] = {
        "kind": "run",
        "started_at": "2026-01-01T00:00:00+00:00",
        "stage": stage,
        "seed": 0,
        "settings": {
            "jev_endpoint": "https://api.typesafe.ai/v1/systemone",
            "jev_model": "jev-latest",
            "jev_available": jev_available,
            "advisor_enabled": False,
        },
    }
    if artifacts_dir is not None:
        run["artifacts_dir"] = str(artifacts_dir)
    return run


def _tick(
    number: int,
    *,
    source: str,
    action: str = "NOOP",
    confidence: float | None = None,
    risk: float | None = None,
    reason: str = "recorded",
    jev_called: bool = False,
    jev_latency: float = 0.0,
    frame_png: object | None = None,
    trajectory_png: object | None = None,
    probabilities: dict[str, float] | None = None,
    choice: str | None = None,
    terminated: bool = False,
    truncated: bool = False,
    attempts: int = 1,
    asked_at_tick: int | None = None,
    answer_age_ticks: int | None = None,
    jev_call_ms: float | None = None,
    jev_phase_ms: float | None = None,
    degradations: tuple[str, ...] = (),
) -> dict[str, object]:
    """One crafted tick line, in the schema the writer produces today.

    A tick that carries a response is a tick that *applied* an answer, so unless
    the caller says otherwise it applied a fresh one: age ``0``, asked on itself,
    and ``jev_call_ms`` equal to the call's own latency. That is the shape the
    writer produces, and leaving the three new keys off would make every crafted
    journal here look like one written before they existed.

    ``jev_phase_ms`` overrides ``latencies_ms["jev"]``, which since the call left
    the tick measures only the submit and the take. The two fields are given the
    same value by default and pulled apart in ``_decoupled_journal``, which is
    where a reader that takes its Jev latency from the tick instead of the call
    stops agreeing with one that takes it from the call.
    """
    response = None
    if probabilities is not None:
        response = {
            "model": "jev-latest",
            "latency_ms": jev_latency,
            "attempts": attempts,
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "answers": {
                "best_action": {
                    "type": "choice",
                    "choice": choice,
                    "confidence": confidence,
                    "probabilities": probabilities,
                }
            },
        }
    applied = response is not None and asked_at_tick is None and answer_age_ticks is None
    return {
        "kind": "tick",
        "tick": number,
        "frame_index": number,
        "emulator_tick": number * 185,
        "state": {},
        "rollout": {},
        "questions": None,
        "response": response,
        "asked_at_tick": number if applied and asked_at_tick is None else asked_at_tick,
        "answer_age_ticks": 0 if applied and answer_age_ticks is None else answer_age_ticks,
        "jev_call_ms": jev_latency if (jev_call_ms is None and response is not None) else jev_call_ms,
        "decision": {
            "action": action,
            "source": source,
            "confidence": confidence,
            "risk": risk,
            "reason": reason,
            "jev_called": jev_called,
        },
        "advisor": {"available": False, "reason": "disabled", "measured": False},
        "latencies_ms": {
            "rollout": 1.0,
            "jev": jev_latency if jev_phase_ms is None else jev_phase_ms,
            "advisor": 0.0,
            "total": 2.0,
        },
        "degradations": list(degradations),
        "terminated": terminated,
        "truncated": truncated,
        "frame_png": frame_png,
        "trajectory_png": trajectory_png,
    }


def _summary_for(
    ticks: tuple[dict[str, object], ...],
    *,
    elapsed_s: float = 2.0,
    termination: str = "max_ticks",
) -> dict[str, object]:
    sources: dict[str, int] = {}
    decisions = 0
    applied = 0
    for tick in ticks:
        decision = tick["decision"]
        assert isinstance(decision, dict)
        sources[str(decision["source"])] = sources.get(str(decision["source"]), 0) + 1
        if decision["jev_called"]:
            decisions += 1
        if tick.get("answer_age_ticks") is not None:
            applied += 1
    return {
        "kind": "summary",
        "ticks": len(ticks),
        "frames": len(ticks),
        "decisions": decisions,
        # Counted apart from ``decisions`` for the same reason the loop counts them
        # apart: one is the calls fired, the other the answers acted on.
        "answers_applied": applied,
        "sources": sources,
        "degradations": {},
        "elapsed_s": elapsed_s,
        "decision_rate_hz": (decisions / elapsed_s) if elapsed_s > 0 else 0.0,
        "termination": termination,
    }


def _journal(
    ticks: tuple[dict[str, object], ...],
    *,
    run: dict[str, object] | None = None,
    elapsed_s: float = 2.0,
    termination: str = "max_ticks",
) -> Journal:
    return Journal(
        run=_run_header() if run is None else run,
        ticks=ticks,
        summary=_summary_for(ticks, elapsed_s=elapsed_s, termination=termination),
    )


def _write_journal(path: Path, journal: Journal) -> Path:
    lines = [json.dumps(journal.run)]
    lines += [json.dumps(tick) for tick in journal.ticks]
    lines.append(json.dumps(journal.summary))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# -- the real-loop run harness -----------------------------------------------------


class Responder:
    """A scripted Jev transport that answers the real questions with a real body."""

    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[object] = []

    def __call__(self, request: object) -> HttpResult:
        self.calls += 1
        self.requests.append(request)
        questions = request.json_body["questions"]  # type: ignore[attr-defined]
        choice = next(iter(questions["best_action"]["criteria"]))
        body = json.dumps(
            {
                "model": "jev-test",
                "answers": {
                    "best_action": {
                        "type": "choice",
                        "choice": choice,
                        "confidence": 0.9,
                        "probabilities": {choice: 0.7, "NOOP": 0.3},
                    },
                    "risk": {
                        "type": "score",
                        "score": 0.2,
                        "confidence": 0.9,
                        "legend": {"0": "safe"},
                        "probabilities": {"0": 1.0},
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
        return HttpResult(status=200, body=body, elapsed_ms=3.0)


def _client_with(settings, responder: Responder) -> JevClient:
    return JevClient(
        endpoint=settings.jev_endpoint,
        model=settings.jev_model,
        api_key=settings.jev_api_key,
        timeout_s=settings.jev_timeout_s,
        max_attempts=1,
        transport=responder,
        sleeper=lambda delay: None,
    )


def _run_real_journal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    max_ticks: int = 3,
    key: str | None = SENTINEL_KEY,
) -> tuple[int, Path, Responder]:
    monkeypatch.setenv("MARIO_DECISION_CADENCE", "1")
    if key is None:
        monkeypatch.delenv("JEV_API_KEY", raising=False)
    else:
        monkeypatch.setenv("JEV_API_KEY", key)
    responder = Responder()
    monkeypatch.setattr(cli, "_build_client", lambda settings: _client_with(settings, responder))
    journal_path = tmp_path / "run.jsonl"
    code = cli.main(
        [
            "run",
            "--journal",
            str(journal_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--fast",
            "--max-ticks",
            str(max_ticks),
        ]
    )
    return code, journal_path, responder


# 1. doctor ------------------------------------------------------------------------


def test_doctor_is_ready_and_never_prints_the_key(monkeypatch, capsys) -> None:
    monkeypatch.setenv("JEV_API_KEY", SENTINEL_KEY)

    code = cli.main(["doctor"])
    out = capsys.readouterr().out

    assert code == 0
    key_lines = [line for line in out.splitlines() if "JEV_API_KEY" in line]
    assert key_lines, "the report must carry a JEV_API_KEY readiness line"
    assert "configured" in key_lines[0]
    # The readiness is a boolean; the value never appears anywhere in the report.
    assert SENTINEL_KEY not in out

    torch_lines = [line for line in out.splitlines() if line.startswith("torch:")]
    assert torch_lines
    if importlib.util.find_spec("torch") is None:
        assert "advisor unavailable" in torch_lines[0]


def test_doctor_never_prints_a_credential_hidden_in_the_endpoint_url(
    monkeypatch, capsys
) -> None:
    # The doctor report prints the endpoint. A URL may carry its own password, so
    # "the key is a boolean here" is not enough on its own -- the endpoint is the
    # other way a credential reaches this output.
    monkeypatch.setenv("JEV_API_KEY", SENTINEL_KEY)
    monkeypatch.setenv("JEV_ENDPOINT", ENDPOINT_WITH_USERINFO)

    code = cli.main(["doctor"])
    out = capsys.readouterr().out

    assert code == 0
    assert URL_PASSWORD not in out
    assert "alice" not in out
    assert "endpoint: https://example.invalid/v1/systemone" in out


def test_the_doctor_report_value_itself_carries_no_userinfo() -> None:
    # Redacted into the report rather than at print time: the report is a dict a
    # caller may serialise, and a value that is only safe once formatted is not safe.
    report = cli.collect_doctor(
        importable=lambda name: True,
        rom_path=lambda: str(Path(__file__)),
        writable=lambda path: True,
        settings=cli.Settings.load(
            {"JEV_API_KEY": SENTINEL_KEY, "JEV_ENDPOINT": ENDPOINT_WITH_USERINFO}
        ),
    )

    assert report["endpoint"] == "https://example.invalid/v1/systemone"
    assert URL_PASSWORD not in json.dumps(report)


def test_doctor_exits_one_when_a_core_dependency_is_missing(monkeypatch, capsys) -> None:
    real_probe = cli._importable

    def probe(name: str) -> bool:
        return False if name == "numpy" else real_probe(name)

    monkeypatch.setattr(cli, "_importable", probe)

    code = cli.main(["doctor"])
    out = capsys.readouterr().out

    assert code == 1
    assert "numpy: MISSING" in out


def test_doctor_exit_code_ignores_the_optional_stack() -> None:
    report = cli.collect_doctor(
        importable=lambda name: name not in {"torch", "transformers"},
        rom_path=lambda: str(Path(__file__)),
        writable=lambda path: True,
        settings=cli.Settings.load({"JEV_API_KEY": SENTINEL_KEY}),
    )

    assert report["rom_present"] is True
    assert cli.doctor_exit_code(report) == 0


def test_doctor_missing_rom_exits_one() -> None:
    report = cli.collect_doctor(
        importable=lambda name: True,
        rom_path=lambda: None,
        writable=lambda path: True,
        settings=cli.Settings.load({}),
    )

    assert report["rom_present"] is False
    assert cli.doctor_exit_code(report) == 1


# 2. replay ------------------------------------------------------------------------


def test_replay_prints_recorded_decisions_and_is_byte_identical(monkeypatch, capsys, tmp_path) -> None:
    code, journal_path, responder = _run_real_journal(monkeypatch, tmp_path, max_ticks=3)
    assert code == 0
    journal = read_journal(journal_path)
    # The exact number of calls is timing-dependent now, and pinning it would be a
    # race: with at most one call in flight over a real worker, whether a call has
    # finished by the time the next tick *takes* it decides how many are fired. What
    # is deterministic -- and what this test is about -- is that the journal's own
    # count is the truth about the calls that were really made, and that every tick
    # executed a frame. The firing rule itself is pinned deterministically in
    # tests/test_dispatch.py and tests/test_policy.py, over a hand-driven runner.
    assert 1 <= responder.calls <= 3
    assert responder.calls == journal.summary["decisions"]
    assert len(journal.ticks) == 3
    capsys.readouterr()  # drop the run report

    first_code = cli.main(["replay", str(journal_path)])
    first = capsys.readouterr().out
    second_code = cli.main(["replay", str(journal_path)])
    second = capsys.readouterr().out

    assert first_code == 0 and second_code == 0
    assert first == second
    for tick in journal.ticks:
        assert f"tick {tick['tick']} frame {tick['frame_index']}" in first
    assert "source=jev" in first


def test_replay_never_builds_a_client_emulator_or_planner(monkeypatch, capsys, tmp_path) -> None:
    code, journal_path, _ = _run_real_journal(monkeypatch, tmp_path, max_ticks=2)
    assert code == 0
    capsys.readouterr()

    def explode(*args: object, **kwargs: object) -> object:
        raise AssertionError("replay must not construct a collaborator")

    class BoomClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("replay must not construct a JevClient")

        @classmethod
        def from_settings(cls, *args: object, **kwargs: object) -> object:
            raise AssertionError("replay must not construct a JevClient")

    monkeypatch.setattr(cli, "JevClient", BoomClient)
    for name in ("_build_client", "_build_emulator", "_build_planner", "_build_advisor"):
        monkeypatch.setattr(cli, name, explode)

    first = cli.main(["replay", str(journal_path)])
    output = capsys.readouterr().out

    assert first == 0
    assert "mario-jev run report" in output


def test_replay_missing_file_is_an_actionable_error(capsys, tmp_path) -> None:
    code = cli.main(["replay", str(tmp_path / "absent.jsonl")])
    captured = capsys.readouterr()

    assert code != 0
    assert "journal not found" in captured.err
    assert captured.out == ""


def test_replay_malformed_journal_is_an_actionable_error(capsys, tmp_path) -> None:
    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"kind": "run"}\nnot json at all\n', encoding="utf-8")

    code = cli.main(["replay", str(broken)])
    captured = capsys.readouterr()

    assert code != 0
    assert "unreadable" in captured.err
    assert captured.out == ""


def test_replay_non_utf8_journal_is_an_actionable_error(capsys, tmp_path) -> None:
    # The journal reader raises UnicodeDecodeError for non-UTF-8 bytes rather
    # than its own format error; the CLI still reports it instead of crashing.
    binary = tmp_path / "binary.jsonl"
    binary.write_bytes(b"\xff\xfe not utf-8\n")

    code = cli.main(["replay", str(binary)])
    captured = capsys.readouterr()

    assert code != 0
    assert "unreadable" in captured.err
    assert captured.out == ""


# 3. run ---------------------------------------------------------------------------


def test_run_end_to_end_writes_a_readable_journal(monkeypatch, capsys, tmp_path) -> None:
    code, journal_path, responder = _run_real_journal(monkeypatch, tmp_path, max_ticks=3)
    out = capsys.readouterr().out

    assert code == 0
    assert 1 <= responder.calls <= 3
    assert journal_path.is_file()
    journal = read_journal(journal_path)
    assert len(journal.ticks) == 3
    assert responder.calls == journal.summary["decisions"]
    summary = journal.summary
    # The printed report matches the journal's own summary line.
    assert f"ticks: {summary['ticks']}" in out
    assert f"decisions: {summary['decisions']}" in out
    assert f"decision_rate_hz: {summary['decision_rate_hz']}" in out
    assert "source=jev" in out


def test_run_without_a_credential_degrades_to_fallback_no_key(monkeypatch, capsys, tmp_path) -> None:
    code, journal_path, responder = _run_real_journal(
        monkeypatch, tmp_path, max_ticks=2, key=None
    )
    captured = capsys.readouterr()

    assert code == 0
    assert responder.calls == 0  # no credential: no request is ever sent
    journal = read_journal(journal_path)
    assert all(tick["decision"]["source"] == "fallback_no_key" for tick in journal.ticks)
    assert journal.summary["sources"] == {"fallback_no_key": 2}
    assert "fallback_no_key" in captured.out
    # The missing credential is an actionable warning, not a crash.
    assert "JEV_API_KEY" in captured.err


class _EndingEmulator:
    """A real emulator that reports the episode over after N executed frames."""

    def __init__(self, inner: object, *, terminate_at: int) -> None:
        self._inner = inner
        self._terminate_at = terminate_at
        self._executed = 0

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    def step(self, action: object) -> StepResult:
        result = self._inner.step(action)  # type: ignore[attr-defined]
        self._executed += 1
        if self._executed >= self._terminate_at:
            return StepResult(
                frame=result.frame,
                info=result.info,
                reward=result.reward,
                terminated=True,
                truncated=result.truncated,
                tick=result.tick,
            )
        return result

    def close(self) -> None:
        self._inner.close()  # type: ignore[attr-defined]


def _fixed_table() -> RolloutTable:
    actions = {action.name: action for action in build_actions("simple")}
    candidate = CandidateRollout(
        action=actions["right"],
        horizon=8,
        frames=8,
        start_x=40,
        final_x=80,
        x_max=80,
        outcome=TerminalOutcome.LANDS_SAFE,
        risk=0.2,
        final_enemy_distance=None,
        terminated=False,
        truncated=False,
    )
    return RolloutTable(
        tick=0,
        start_x=40,
        horizon=8,
        budget_ms=120.0,
        elapsed_ms=1.0,
        sweep_ms=1.0,
        base_ms=0.5,
        per_frame_ms=1.0,
        calibration_frames=0,
        passes=1,
        cost_source=CostSource.REUSED,
        steps_taken=0,
        candidates=(candidate,),
    )


class _FixedPlanner:
    """A planner that returns one constructed table and never steps the emulator."""

    def __init__(self, table: RolloutTable) -> None:
        self._table = table

    def plan(self, emulator: object) -> RolloutTable:
        return self._table


def test_run_survives_an_episode_that_ends_early(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    real_build = cli._build_emulator

    def build(settings):
        return _EndingEmulator(real_build(settings), terminate_at=2)

    monkeypatch.setattr(cli, "_build_emulator", build)
    monkeypatch.setattr(cli, "_build_planner", lambda settings: _FixedPlanner(_fixed_table()))

    journal_path = tmp_path / "early.jsonl"
    code = cli.main(
        [
            "run",
            "--journal",
            str(journal_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--fast",
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    journal = read_journal(journal_path)
    assert len(journal.ticks) == 2  # the loop stopped at the ended step
    assert journal.summary["ticks"] == 2
    assert f"ticks: {journal.summary['ticks']}" in out


# 4. dashboard ---------------------------------------------------------------------


def _two_tick_journal(*, artifacts_dir: object | None = None, reason: str = "recorded") -> Journal:
    ticks = (
        _tick(0, source="fallback_no_key", action="right+A+B", reason=reason),
        _tick(
            1,
            source="jev",
            action="right",
            confidence=0.9,
            risk=0.2,
            reason="Jev chose 'right'",
            jev_called=True,
            jev_latency=5.0,
            probabilities={"right": 0.7, "left": 0.3},
            choice="right",
        ),
    )
    return _journal(ticks, run=_run_header(artifacts_dir=artifacts_dir))


def test_dashboard_writes_html_with_stage_and_tick_rows(capsys, tmp_path) -> None:
    journal_path = _write_journal(tmp_path / "evidence.jsonl", _two_tick_journal())

    code = cli.main(["dashboard", str(journal_path)])
    out = capsys.readouterr().out

    assert code == 0
    target = Path(out.strip())
    assert target == tmp_path / "evidence.html"
    assert target.is_file()
    html = target.read_text(encoding="utf-8")
    assert "1-1" in html
    assert "Tick 0" in html and "Tick 1" in html
    assert "fallback_no_key" in html and "jev" in html


def test_dashboard_honours_an_explicit_out_path(capsys, tmp_path) -> None:
    journal_path = _write_journal(tmp_path / "trace.jsonl", _two_tick_journal())

    code = cli.main(["dashboard", str(journal_path), "--out", str(tmp_path / "report.html")])
    out = capsys.readouterr().out

    assert code == 0
    assert Path(out.strip()) == tmp_path / "report.html"
    assert (tmp_path / "report.html").is_file()


def test_dashboard_marks_a_no_call_tick_and_never_as_zero() -> None:
    html = build_html(_two_tick_journal())

    section = html.split('<section class="tick" id="tick-0">', 1)[1].split('<section class="tick"', 1)[0]

    assert "no call" in section
    # The raw jev latency for the skipped branch is 0.0; it must not be rendered.
    assert "0.0" not in section


def test_dashboard_reports_a_missing_frame_as_not_recorded() -> None:
    html = build_html(_two_tick_journal(artifacts_dir="/does/not/exist"))

    assert "not recorded" in html
    assert "data:image/png" not in html
    assert "<img" not in html


def test_dashboard_embeds_a_recorded_frame_as_a_data_uri(tmp_path) -> None:
    from PIL import Image

    # Two real PNGs, written where the journal says its artifacts live.
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    Image.new("RGB", (2, 2), (10, 20, 30)).save(artifacts / "frame_000000.png")
    Image.new("RGB", (2, 2), (10, 20, 30)).save(artifacts / "trajectory_000000.png")
    journal = _journal(
        (
            _tick(
                0,
                source="jev",
                confidence=0.9,
                jev_called=True,
                jev_latency=5.0,
                frame_png="frame_000000.png",
                trajectory_png="trajectory_000000.png",
                probabilities={"NOOP": 1.0},
                choice="NOOP",
            ),
        ),
        run=_run_header(artifacts_dir=artifacts),
    )

    html = build_html(journal)

    assert html.count("data:image/png;base64,") == 2
    assert html.count("<img") == 2
    assert 'class="missing">not recorded</p></figure>' not in html


def test_dashboard_inlines_the_confidence_and_latency_series_as_svg() -> None:
    html = build_html(_two_tick_journal())

    assert "<svg" in html
    assert 'data-series="confidence"' in html
    assert 'data-series="latency"' in html
    # The no-call tick is a gap: one point in each chart, none for the skipped tick.
    assert html.count('<circle class="point"') == 2


def test_dashboard_escapes_a_script_tag_in_a_reason() -> None:
    injected = "danger <script>alert('x')</script>"
    html = build_html(_two_tick_journal(reason=injected))

    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_build_html_is_deterministic() -> None:
    journal = _two_tick_journal()

    assert build_html(journal) == build_html(journal)


def test_render_text_prints_sources_fallbacks_and_latency_stats() -> None:
    ticks = (
        _tick(0, source="fallback_no_key"),
        # These two observed their own calls, and what a call cost is
        # ``jev_call_ms``: the statistic used to read ``latencies_ms["jev"]``,
        # which is now the submit-and-take cost and would be microseconds.
        _tick(1, source="jev", jev_called=True, jev_call_ms=1.0, confidence=0.9),
        _tick(2, source="jev", jev_called=True, jev_call_ms=3.0, confidence=0.8),
    )
    text = render_text(_journal(ticks))

    assert "sources:" in text
    assert "fallback_no_key: 1" in text
    assert "jev: 2" in text
    assert "fallbacks (source fallback_*): 1" in text
    assert "confidence=no call" in text
    assert "jev latency over observed calls: min=1.000 median=2.000 max=3.000" in text


def test_render_text_says_so_when_no_tick_called_jev() -> None:
    text = render_text(_journal((_tick(0, source="fallback_no_key"),)))

    assert "jev latency over observed calls: no call was observed; not recorded" in text


# -- the retry, in both readers -----------------------------------------------------
# A tick whose wall clock reads 21 s and whose response reads 577 ms has ~20 s
# nobody can account for unless the attempt count is on the page.


def _retried_tick(number: int, *, jev_latency: float, attempts: int) -> dict[str, object]:
    return _tick(
        number,
        source="jev",
        action="right",
        jev_called=True,
        jev_latency=jev_latency,
        confidence=0.9,
        choice="right",
        probabilities={"right": 1.0},
        attempts=attempts,
    )


def test_render_text_names_the_retry_count_only_when_a_tick_retried() -> None:
    retried = (
        _retried_tick(0, jev_latency=577.0, attempts=2),
        _retried_tick(1, jev_latency=5.0, attempts=1),
    )

    text = render_text(_journal(retried))

    # Counted over the recorded answers, which is the set the numerator comes
    # from: ``decisions`` counts fired calls, and an answer can be applied by a
    # tick that fired nothing, so a ratio of one set over the other would mean
    # nothing.
    assert "retried jev calls: 1 of 2 recorded answers" in text
    # A journal with no retried tick says nothing about retries at all, rather
    # than printing a zero a reader has to interpret.
    assert "retr" not in render_text(_journal((retried[1],)))


def test_the_dashboard_tick_row_shows_the_attempt_count() -> None:
    ticks = (
        _retried_tick(0, jev_latency=577.0, attempts=2),
        _tick(1, source="fallback_no_key"),
    )

    html = build_html(_journal(ticks))

    assert "<th>jev attempts</th>" in html
    # Next to the latency it explains: 577 ms on the attempt that worked, two
    # attempts in total.
    assert "<td>577.000</td><td>2</td>" in html
    # And a tick that made no call still reads as no call in both cells, never as 1.
    assert "<td>no call</td><td>no call</td>" in html


# -- the call that left the tick, in both readers -----------------------------------
# Firing a call and applying its answer are now two facts about two ticks, and the
# trap this sets is ``latencies_ms["jev"]``: it still exists, it is still a Jev
# number, and it now measures the submit and the take rather than the call. A
# reader that keeps printing it reports a twenty-one-second call as two
# microseconds, which is the most flattering lie available.

#: What a tick spends on Jev now: submitting and taking, in microseconds.
JEV_PHASE_MS = 0.002
#: What the call itself cost -- the 21 s stall the live evaluation measured.
CALL_MS = 21271.52
#: The one attempt of that call that succeeded. The response records this, and it is
#: smaller than the call by everything the retry cost.
ATTEMPT_MS = 577.0

#: The keys a journal written before the call left the tick does not carry.
LEGACY_TICK_KEYS = ("asked_at_tick", "answer_age_ticks", "jev_call_ms")


def _legacy(journal: Journal) -> Journal:
    """The same run as an older build would have written it: without the new keys.

    Those journals exist, they still load, and every one of them is evidence for a
    run somebody made. What a reader must never do is fill the gap with a default:
    ``0`` applied answers is a claim about a run that never recorded one.
    """
    ticks = tuple(
        {key: value for key, value in tick.items() if key not in LEGACY_TICK_KEYS}
        for tick in journal.ticks
    )
    summary = {
        key: value
        for key, value in journal.summary.items()
        if key != "answers_applied"
    }
    return Journal(run=journal.run, ticks=ticks, summary=summary)


def _stale_journal() -> Journal:
    """Tick 0 fires a call; tick 1 applies the answer, one tick late."""
    ticks = (
        _tick(
            0,
            source="hold",
            action="right",
            reason="cadence not due on this frame",
            jev_called=True,
            jev_phase_ms=JEV_PHASE_MS,
        ),
        _tick(
            1,
            source="jev_stale",
            action="right",
            reason="Jev chose 'right'",
            confidence=0.9,
            probabilities={"right": 1.0},
            choice="right",
            jev_latency=CALL_MS,
            jev_phase_ms=JEV_PHASE_MS,
            asked_at_tick=0,
            answer_age_ticks=1,
        ),
    )
    return _journal(ticks)


def _discarded_journal() -> Journal:
    """A call fired on tick 0 and thrown away on tick 4 for being four ticks old.

    The staleness bound is three cadences, so tick 4 refuses to act on it -- and
    tick 4 is still the only tick that can say what the call cost, because it is
    the tick that observed it finish.
    """
    waiting = tuple(
        _tick(
            number,
            source="hold",
            action="right",
            reason="cadence not due on this frame",
            jev_phase_ms=JEV_PHASE_MS,
        )
        for number in range(1, 4)
    )
    ticks = (
        _tick(
            0,
            source="hold",
            action="right",
            reason="cadence not due on this frame",
            jev_called=True,
            jev_phase_ms=JEV_PHASE_MS,
        ),
        *waiting,
        _tick(
            4,
            source="hold",
            action="right",
            reason="cadence not due on this frame",
            jev_call_ms=CALL_MS,
            jev_phase_ms=JEV_PHASE_MS,
            degradations=("jev_stale_discarded",),
        ),
    )
    return _journal(ticks)


def _failed_journal() -> Journal:
    """A call fired on tick 0 that failed, observed and degraded on tick 1."""
    ticks = (
        _tick(
            0,
            source="hold",
            action="right",
            reason="cadence not due on this frame",
            jev_called=True,
            jev_phase_ms=JEV_PHASE_MS,
        ),
        _tick(
            1,
            source="fallback_degraded",
            action="right",
            reason="the endpoint timed out after 20.0s",
            jev_call_ms=CALL_MS,
            jev_phase_ms=JEV_PHASE_MS,
        ),
    )
    return _journal(ticks)


def test_render_text_reads_the_jev_latency_from_the_call_not_from_the_tick() -> None:
    text = render_text(_stale_journal())

    # The call cost 21 271.52 ms and the tick spent 0.002 ms submitting and taking
    # it. Only one of those is a Jev latency, and the tick's own number must not
    # appear anywhere a reader would read as one.
    assert f"jev_latency_ms={CALL_MS:.3f}" in text
    assert f"jev latency over observed calls: min={CALL_MS:.3f}" in text
    assert f"{JEV_PHASE_MS:.3f}" not in text


def test_render_text_separates_the_calls_fired_from_the_answers_applied() -> None:
    text = render_text(_stale_journal())

    assert "calls fired (jev_called true): 1" in text
    assert "answers applied: 1" in text
    assert "applied from an earlier tick: 1" in text
    assert "oldest applied answer age (ticks): 1" in text
    assert "answers discarded as stale: 0" in text
    # And per tick: the source says the answer was old, the age says how old.
    assert "source=jev_stale answer_age=1" in text
    assert "source=hold answer_age=no answer" in text


def test_render_text_shows_a_discarded_answer_as_a_call_that_was_paid_for() -> None:
    text = render_text(_discarded_journal())

    assert "answers discarded as stale: 1" in text
    assert "answers applied: 0" in text
    # The discarding tick made no *new* call, but it did observe one, and reading
    # it as "no call" would hide 21 s of work the run really paid for.
    assert f"answer_age=discarded (stale) confidence=not recorded risk=not recorded jev_called=false jev_latency_ms={CALL_MS:.3f}" in text


def test_render_text_shows_a_failed_call_with_what_it_cost() -> None:
    text = render_text(_failed_journal())

    assert "answers applied: 0" in text
    assert "calls fired (jev_called true): 1" in text
    assert "source=fallback_degraded answer_age=no answer" in text
    assert f"jev_latency_ms={CALL_MS:.3f}" in text
    assert f"jev latency over observed calls: min={CALL_MS:.3f}" in text


def test_render_text_reports_a_legacy_journal_as_not_recorded_never_as_zero(
    tmp_path,
) -> None:
    # A journal an older build really wrote: the sources it knew, and none of the
    # four keys this round added. Through a file, because "an old journal still
    # loads" is a claim about the reader and the bytes, not about a value built
    # in memory.
    path = _write_journal(tmp_path / "legacy.jsonl", _legacy(_two_tick_journal()))
    text = render_text(read_journal(path))

    assert "answers_applied: not recorded" in text
    assert "answers applied: not recorded" in text
    assert "answer_age=not recorded" in text
    assert "answers applied: 0" not in text
    assert "applied from an earlier tick" not in text
    # The response is still on the line, so its latency is still readable.
    assert "jev_latency_ms=5.000" in text


def test_the_dashboard_shows_the_answer_age_beside_the_source() -> None:
    html = build_html(_stale_journal())

    assert "<th>answer age (ticks)</th>" in html
    # A stale tick and a fresh one must not read the same: the source names it and
    # the age quantifies it, in the row and in the tick's own section.
    assert "<td>jev_stale</td><td>1</td>" in html
    assert "<th>answer age (ticks)</th><td>1</td>" in html
    fresh = build_html(_two_tick_journal())
    assert "<td>jev</td><td>0</td>" in fresh
    assert "jev_stale" not in fresh


def test_the_dashboard_reads_the_jev_latency_from_the_call() -> None:
    html = build_html(_stale_journal())

    assert f"<td>{CALL_MS:.3f}</td>" in html
    assert f"{JEV_PHASE_MS:.3f}" not in html


def test_the_dashboard_summary_counts_answers_applied_beside_decisions() -> None:
    html = build_html(_stale_journal())

    assert "<th>answers_applied</th><td>1</td>" in html
    assert "<th>decisions (calls fired)</th><td>1</td>" in html
    assert "<th>applied from an earlier tick</th><td>1</td>" in html
    assert "<th>oldest applied answer (ticks)</th><td>1</td>" in html


def test_the_dashboard_shows_a_discarded_answer_as_a_call_that_was_made() -> None:
    html = build_html(_discarded_journal())

    assert "<th>answers discarded as stale</th><td>1</td>" in html
    # The discarding tick: its age cell says why there is no answer, and its
    # latency cell says the call was still made and still cost 21 s.
    assert "<td>discarded (stale)</td>" in html
    assert f"<td>{CALL_MS:.3f}</td>" in html
    assert "jev_stale_discarded" in html
    # A tick that really made no call still reads as one, in the same table.
    assert "<td>no answer</td>" in html
    assert "<td>no call</td>" in html


def test_the_dashboard_shows_a_failed_call_with_what_it_cost() -> None:
    html = build_html(_failed_journal())

    assert "<th>answers_applied</th><td>0</td>" in html
    assert f"<td>{CALL_MS:.3f}</td>" in html
    # No response was recorded, so there is no attempt count to print -- and "1"
    # would be a number nobody measured.
    assert "<td>not recorded</td>" in html
    assert "the endpoint timed out after 20.0s" in html


def test_the_dashboard_shows_not_recorded_for_a_journal_without_the_new_keys() -> None:
    html = build_html(_legacy(_two_tick_journal()))

    assert "<th>answers_applied</th><td>not recorded</td>" in html
    assert "<th>answers applied (per tick)</th><td>not recorded</td>" in html
    assert "<th>applied from an earlier tick</th>" not in html
    # The age column is present for every tick and recorded for none of them.
    assert "<th>answer age (ticks)</th>" in html
    assert html.count("<td>not recorded</td>") >= 2


# -- why the run stopped, in both readers ------------------------------------------
# A run that stopped at 805 of 900 ticks is not "finished": the counts alone
# cannot separate a death from the flag from a hand on the keyboard.


def _died_journal() -> Journal:
    ticks = (
        _tick(0, source="jev", jev_called=True, jev_latency=1.0, confidence=0.9),
        _tick(1, source="jev", jev_called=True, jev_latency=1.0, confidence=0.9, terminated=True),
    )
    return _journal(ticks, termination="death")


def test_render_text_names_the_termination_and_keeps_the_counts() -> None:
    text = render_text(_died_journal())

    assert "termination: death" in text
    assert "ticks: 2" in text
    assert "frames: 2" in text


def test_render_text_reports_an_unrecorded_termination_as_not_recorded() -> None:
    journal = _died_journal()
    summary = dict(journal.summary)
    del summary["termination"]

    text = render_text(Journal(run=journal.run, ticks=journal.ticks, summary=summary))

    assert "termination: not recorded" in text
    assert "ticks: 2" in text


def test_the_dashboard_summary_shows_the_termination_alongside_the_counts() -> None:
    html = build_html(_died_journal())

    assert "<th>termination</th><td>death</td>" in html
    assert "<th>ticks</th><td>2</td>" in html


def test_the_dashboard_shows_not_recorded_for_an_older_journal() -> None:
    journal = _died_journal()
    summary = dict(journal.summary)
    del summary["termination"]

    html = build_html(Journal(run=journal.run, ticks=journal.ticks, summary=summary))

    assert "<th>termination</th><td>not recorded</td>" in html


# -- a credential in a recorded endpoint, in both readers ---------------------------
# ``Settings.redacted`` strips the userinfo before the header is written, so a
# journal this package produced has nothing left to strip. The readers strip again
# anyway: they are handed a *file*, and a header from an older build or an edited
# one must not be able to put a password on the page.


def _userinfo_journal() -> Journal:
    run = _run_header()
    settings = dict(run["settings"])  # type: ignore[arg-type]
    settings["jev_endpoint"] = ENDPOINT_WITH_USERINFO
    run["settings"] = settings
    return _journal((_tick(0, source="fallback_no_key"),), run=run)


def test_render_text_redacts_userinfo_from_a_recorded_endpoint() -> None:
    text = render_text(_userinfo_journal())

    assert URL_PASSWORD not in text
    assert "alice" not in text
    assert "endpoint: https://example.invalid/v1/systemone" in text


def test_the_dashboard_redacts_userinfo_from_a_recorded_endpoint() -> None:
    html = build_html(_userinfo_journal())

    assert URL_PASSWORD not in html
    assert "alice" not in html
    assert "<td>https://example.invalid/v1/systemone</td>" in html


def test_module_entry_point_delegates_to_main() -> None:
    import mario_jev.__main__ as entry

    assert entry.main is cli.main


def test_the_jet_latency_a_reader_shows_is_the_whole_call_not_the_attempt_that_worked() -> (
    None
):
    """A retried call has two honest figures, and the reader must show the larger.

    ``response["latency_ms"]`` is the attempt that succeeded; ``jev_call_ms`` is the
    dispatcher's measurement around the whole call, failed attempt and backoff
    included. Reporting the first as "the Jev latency" hides everything the retry
    cost -- for the call that motivated this schema, 20 117 ms of a 20 694 ms call.
    """
    journal = _journal(
        [
            _tick(
                1,
                source="jev",
                action="right",
                confidence=0.9,
                jev_called=True,
                attempts=2,
                jev_latency=ATTEMPT_MS,  # the successful attempt alone
                jev_call_ms=CALL_MS,  # the whole call, both attempts and the backoff
                asked_at_tick=1,
                answer_age_ticks=0,
                choice="right",
                probabilities={"right": 1.0},
            )
        ]
    )

    text = render_text(journal)

    assert f"jev_latency_ms={CALL_MS:.3f}" in text
    assert f"jev latency over observed calls: min={CALL_MS:.3f}" in text
    assert f"jev_latency_ms={ATTEMPT_MS:.3f}" not in text, (
        "the reader reported the successful attempt as the call's latency"
    )
