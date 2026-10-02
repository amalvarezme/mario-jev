"""The live page: one snapshot of the newest decision tick.

The page sits beside the game window and answers one question -- what is the
agent doing right now, and why? What is pinned here is that it answers from the
journal and nothing else: the frame is the newest tick's own recorded path, the
buttons come out of the action set by name, the answer's chosen candidate and the
applied action are each marked for the fact it stands for, a journal without its
trailer is ``in progress``, and a tick that never asked Jev says ``no call`` rather
than an invented score.

The journal builders are local, exactly as they are in the other test files: a
dozen lines each, and the test files here do not import one another. They follow
``tests/test_cli.py``'s ``_tick``/``_journal`` shape, narrowed to the keys this
page reads.
"""

from __future__ import annotations

import types
from pathlib import Path

from mario_jev.dashboard import build_live_html
from mario_jev.journal import Journal
from mario_jev.serve import Viewer, ViewerHandler


# -- crafted journals --------------------------------------------------------------


def _run_header(
    *, stage: str = "1-1", artifacts_dir: object | None = None
) -> dict[str, object]:
    run: dict[str, object] = {
        "kind": "run",
        "started_at": "2026-01-01T00:00:00+00:00",
        "stage": stage,
        "seed": 0,
        "settings": {
            "jev_endpoint": "https://example.invalid/v1/systemone",
            "jev_model": "jev-latest",
            "jev_available": True,
            "advisor_enabled": False,
        },
    }
    if artifacts_dir is not None:
        run["artifacts_dir"] = str(artifacts_dir)
    return run


def _choice_response(
    *,
    choice: str = "right",
    probabilities: dict[str, float] | None = None,
    confidence: float = 0.9,
) -> dict[str, object]:
    return {
        "model": "jev-test",
        "latency_ms": 1.0,
        "attempts": 1,
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "answers": {
            "best_action": {
                "type": "choice",
                "choice": choice,
                "confidence": confidence,
                "probabilities": (
                    {choice: 1.0} if probabilities is None else probabilities
                ),
            }
        },
    }


def _candidate(
    name: str, *, dx: int, outcome: str = "lands_safe", risk: float, final_x: int
) -> dict[str, object]:
    return {
        "action": name,
        "dx": dx,
        "outcome": outcome,
        "risk": risk,
        "final_x": final_x,
    }


def _tick(
    number: int,
    *,
    action: str = "NOOP",
    source: str = "fallback_no_key",
    reason: str = "recorded",
    response: dict[str, object] | None = None,
    candidates: tuple[dict[str, object], ...] = (),
    frame_png: object | None = None,
    jev_called: bool = False,
) -> dict[str, object]:
    return {
        "kind": "tick",
        "tick": number,
        "frame_index": number,
        "emulator_tick": number * 185,
        "state": {},
        "rollout": {"candidates": list(candidates)} if candidates else {},
        "questions": None,
        "response": response,
        "asked_at_tick": number if response is not None else None,
        "answer_age_ticks": 0 if response is not None else None,
        "jev_call_ms": 1.0 if response is not None else None,
        "decision": {
            "action": action,
            "source": source,
            "confidence": 0.9 if response is not None else None,
            "risk": 0.2 if response is not None else None,
            "reason": reason,
            "jev_called": jev_called,
        },
        "advisor": {"available": False, "reason": "disabled", "measured": False},
        "latencies_ms": {"rollout": 1.0, "jev": 0.0, "advisor": 0.0, "total": 2.0},
        "degradations": [],
        "terminated": False,
        "truncated": False,
        "frame_png": frame_png,
        "trajectory_png": None,
    }


def _summary(*, termination: str = "max_ticks") -> dict[str, object]:
    return {
        "kind": "summary",
        "ticks": 1,
        "frames": 1,
        "decisions": 1,
        "answers_applied": 1,
        "sources": {"jev": 1},
        "degradations": {},
        "elapsed_s": 1.0,
        "decision_rate_hz": 1.0,
        "termination": termination,
    }


def _journal(
    ticks: tuple[dict[str, object], ...],
    *,
    run: dict[str, object] | None = None,
    summary: dict[str, object] | None = None,
) -> Journal:
    return Journal(
        run=_run_header() if run is None else run,
        ticks=ticks,
        summary=summary,
    )


# -- the frame ----------------------------------------------------------------------


def test_the_newest_ticks_frame_is_referenced_and_the_route_resolves_it(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "frame_000000.png").write_bytes(b"\x89PNG\r\n\x1a\nfirst")
    (artifacts / "frame_000002.png").write_bytes(b"\x89PNG\r\n\x1a\nsecond")
    journal = _journal(
        (
            _tick(0, frame_png="frame_000000.png"),
            _tick(2, frame_png="frame_000002.png"),
        ),
        run=_run_header(artifacts_dir=artifacts),
    )

    html = build_live_html(journal)

    # The newest tick is the last one on disk, and it is its own ``frame_png``
    # that is referenced -- not a name reconstructed from the frame index.
    assert 'src="/frame_000002.png"' in html
    assert "/frame_000000.png" not in html
    # The reference is exactly the path the serve route strips its leading slash
    # from, so asking the handler to resolve it reaches the recorded file.
    stand_in = types.SimpleNamespace(
        viewer=Viewer(journal_path=Path("journal.jsonl"), artifacts_dir=artifacts)
    )
    assert ViewerHandler._resolve(stand_in, "/frame_000002.png") == (
        artifacts / "frame_000002.png"
    ).resolve()


def test_a_recorded_frame_that_is_not_on_disk_is_not_recorded(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    journal = _journal(
        (_tick(0, frame_png="frame_000000.png"),),
        run=_run_header(artifacts_dir=artifacts),
    )

    html = build_live_html(journal)

    assert "<img" not in html
    assert "not recorded" in html


def test_a_journal_with_no_tick_says_so_instead_of_showing_a_frame() -> None:
    html = build_live_html(_journal(()))

    assert "no tick has been recorded yet" in html
    assert "<img" not in html


# -- the buttons --------------------------------------------------------------------


def test_the_buttons_come_from_the_action_set_by_name() -> None:
    html = build_live_html(_journal((_tick(0, action="right+A+B"),)))

    # The applied action's buttons, marked on a pad built from the set itself.
    assert '<span class="pad-button pressed">right</span>' in html
    assert '<span class="pad-button pressed">A</span>' in html
    assert '<span class="pad-button pressed">B</span>' in html
    # A button the set can press but this action does not, left unmarked.
    assert '<span class="pad-button">left</span>' in html
    assert "right+A+B" in html


def test_an_action_name_outside_the_set_says_so_and_marks_nothing() -> None:
    html = build_live_html(_journal((_tick(0, action="teleport"),)))

    assert "teleport" in html
    assert "is not in the action set" in html
    assert 'class="pad-button pressed"' not in html


# -- the chosen candidate and the applied action ------------------------------------


def test_the_chosen_candidate_is_marked_in_the_probabilities_and_the_rollout_rows() -> None:
    response = _choice_response(
        choice="right", probabilities={"right": 1.0, "left": 0.0}
    )
    candidates = (
        _candidate("right", dx=12, risk=0.2, final_x=52),
        _candidate("left", dx=-3, outcome="no_progress", risk=0.5, final_x=37),
    )
    tick = _tick(
        0,
        action="right",
        source="jev",
        response=response,
        candidates=candidates,
        jev_called=True,
        frame_png="frame_000000.png",
    )

    html = build_live_html(_journal((tick,)))

    # Once in the probabilities, once in the rollout row: the answer's choice is
    # marked as such where both the answer and the candidates show it.
    assert '<tr class="chosen"><th>right (chosen)</th><td>1.0</td></tr>' in html
    # The applied row and Jev's pick are the same row here, so it is marked once
    # and says both facts instead of stacking two marks a reader must reconcile.
    assert (
        '<tr class="applied chosen"><th>right (applied, chosen)</th><td>12</td>'
        "<td>lands_safe</td><td>0.2</td><td>52</td></tr>" in html
    )
    # The candidate that was neither applied nor chosen is not marked.
    assert '<tr><th>left</th><td>-3</td>' in html


def test_an_applied_action_that_differs_from_jevs_pick_marks_each_row_for_its_fact() -> None:
    # A guardrail that applied ``right`` while Jev answered ``left``: the two rows
    # are different facts and each carries its own mark.
    response = _choice_response(
        choice="left", probabilities={"right": 0.4, "left": 0.6}
    )
    candidates = (
        _candidate("right", dx=12, risk=0.2, final_x=52),
        _candidate("left", dx=-3, outcome="no_progress", risk=0.5, final_x=37),
    )
    tick = _tick(
        0,
        action="right",
        source="guardrail_lethal",
        response=response,
        candidates=candidates,
        jev_called=True,
    )

    html = build_live_html(_journal((tick,)))

    # The applied row says what ran; Jev's row says what Jev answered.
    assert '<tr class="applied"><th>right (applied)</th><td>12</td>' in html
    assert '<tr class="chosen"><th>left (chosen)</th><td>-3</td>' in html
    # The scores still mark Jev's pick, because that is what the answer says.
    assert '<tr class="chosen"><th>left (chosen)</th><td>0.6</td></tr>' in html
    assert '<tr><th>right</th><td>0.4</td></tr>' in html
    # The buttons are the applied action's, not Jev's pick.
    assert '<span class="pad-button pressed">right</span>' in html
    assert '<span class="pad-button">left</span>' in html


def test_a_score_answer_shows_its_score_and_legend_and_a_noul_shows_its_note() -> None:
    response = _choice_response()
    answers = response["answers"]
    assert isinstance(answers, dict)
    answers["risk"] = {
        "type": "score",
        "score": 0.2,
        "confidence": 0.9,
        "legend": {"0": "safe", "1": "near-certain death"},
        "probabilities": {"0": 1.0},
    }
    answers["lethal_ahead"] = {"type": "noul", "noul": 0.1}

    html = build_live_html(_journal((_tick(0, response=response, jev_called=True),)))

    assert "score: 0.2" in html
    assert "<th>0</th><td>safe</td>" in html
    assert "<th>1</th><td>near-certain death</td>" in html
    # ``noul`` carries no confidence and no distribution, so it shows the
    # recorded note and nothing invented.
    assert "recorded note is 0.1" in html


# -- the status, from the trailer ---------------------------------------------------


def test_a_trailer_less_journal_reads_as_in_progress() -> None:
    html = build_live_html(_journal((_tick(0),), summary=None))

    assert "<td>in progress</td>" in html


def test_a_finished_journal_reads_its_recorded_termination() -> None:
    html = build_live_html(_journal((_tick(0),), summary=_summary(termination="death")))

    assert "<td>death</td>" in html
    assert "in progress" not in html


# -- a tick that never asked --------------------------------------------------------


def test_a_tick_with_no_response_says_no_call_instead_of_scores() -> None:
    html = build_live_html(_journal((_tick(0, jev_called=False),)))

    assert "no call" in html
    # No probabilities were recorded, so there is no score table to misread.
    assert 'class="scores"' not in html
    assert 'class="rollout"' not in html


# -- the page itself -----------------------------------------------------------------


def test_the_page_carries_the_refresh_meta_and_the_link_to_the_report() -> None:
    html = build_live_html(_journal((_tick(0),)))

    assert '<meta http-equiv="refresh" content="0.5">' in html
    assert '<a href="/">' in html


def test_the_page_says_this_is_the_tick_rate_not_60_fps() -> None:
    html = build_live_html(_journal((_tick(0),)))

    assert "decision tick rate, not 60 fps" in html


def test_a_hostile_reason_is_escaped() -> None:
    injected = "danger <script>alert('x')</script>"
    html = build_live_html(_journal((_tick(0, reason=injected),)))

    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_build_live_html_is_deterministic() -> None:
    journal = _journal((_tick(0, action="right"),))

    assert build_live_html(journal) == build_live_html(journal)
