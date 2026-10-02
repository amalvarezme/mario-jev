"""The live page: the decision panel on the left, the live game on the right.

The page answers one question -- what is the agent doing right now, and why? --
beside the game itself. What is pinned here is that the panel answers from the
journal and nothing else, and that the game is the frame the run publishes now,
never a recorded one: the panel is one ``#panel`` subtree the script can replace,
the game image points at the route ``serve`` answers, no recorded ``frame_png``
leaks into the page, the meta refresh stays for a browser without the script and
the script removes it, the buttons come out of the action set by name, the
answer's chosen candidate and the applied action are each marked for the fact it
stands for, a journal without its trailer is ``in progress``, and a tick that
never asked Jev says ``no call`` rather than an invented score.

The journal builders are local, exactly as they are in the other test files: a
dozen lines each, and the test files here do not import one another. They follow
``tests/test_cli.py``'s ``_tick``/``_journal`` shape, narrowed to the keys this
page reads.
"""

from __future__ import annotations

from pathlib import Path

from mario_jev import serve
from mario_jev.dashboard import build_live_html
from mario_jev.dashboard.live import (
    FRAME_POLL_MS,
    FRAME_SCALE,
    LIVE_FRAME_URL,
    REFRESH_SECONDS,
)
from mario_jev.emulator import SCREEN_SHAPE
from mario_jev.journal import Journal


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


# -- the board: the panel on the left, the game on the right -------------------------


def _between(html: str, start: str, end: str) -> str:
    """The text between the first ``start`` and the first ``end`` after it."""
    head = html.index(start) + len(start)
    return html[head : html.index(end, head)]


def test_the_panel_and_the_game_are_one_board_row_with_the_panel_first() -> None:
    html = build_live_html(_journal((_tick(0, action="right"),)))

    board = _between(html, '<div class="board">', "<noscript>")
    # One row, the decisions on the left and the game on the right: the panel
    # opens before the game column inside the same board.
    assert board.index('<div class="panel" id="panel">') < board.index(
        '<div class="game">'
    )
    assert html.count('<div class="board">') == 1


def test_the_panel_carries_its_id_and_holds_the_sections_in_order() -> None:
    html = build_live_html(_journal((_tick(0, action="right"),)))

    # ``id="panel"`` is the subtree the script re-fetches and replaces; the game
    # image must not sit inside it, or a panel refresh would reload the game.
    panel = _between(html, '<div class="panel" id="panel">', '<div class="game">')
    assert html.count('id="panel"') == 1
    assert "<img" not in panel
    headings = [
        "<h2>Right now</h2>",
        "<h2>Buttons</h2>",
        "<h2>Jev scores</h2>",
        "<h2>Rollout candidates</h2>",
        "<h2>Decision</h2>",
    ]
    positions = [panel.index(heading) for heading in headings]
    assert positions == sorted(positions)


def test_the_game_image_is_the_live_frame_at_twice_the_nes_size() -> None:
    html = build_live_html(_journal((_tick(0),)))

    game = html[html.index('<div class="game">') :]
    width = SCREEN_SHAPE[1] * FRAME_SCALE
    height = SCREEN_SHAPE[0] * FRAME_SCALE
    assert (
        f'<img id="live-frame-a" class="game-frame" src="{LIVE_FRAME_URL}" '
        f'width="{width}" height="{height}"' in game
    )
    # The swap target has no source of its own until the script gives it one.
    assert (
        f'<img id="live-frame-b" class="game-frame hidden" '
        f'width="{width}" height="{height}"' in game
    )


def test_the_live_frame_url_is_the_route_the_viewer_answers() -> None:
    # ``live`` cannot import ``serve`` (``serve`` imports it), so this pin is
    # what keeps the image and the route naming the same path.
    assert LIVE_FRAME_URL == serve.LIVE_FRAME_PATH


def test_no_recorded_frame_is_referenced_even_when_one_is_on_disk(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "frame_000000.png").write_bytes(b"\x89PNG\r\n\x1a\nfirst")
    (artifacts / "distinctive_frame_000002.png").write_bytes(
        b"\x89PNG\r\n\x1a\nnewest"
    )
    journal = _journal(
        (
            _tick(0, frame_png="frame_000000.png"),
            _tick(2, frame_png="distinctive_frame_000002.png"),
        ),
        run=_run_header(artifacts_dir=artifacts),
    )

    # The viewer calls the builder with ``artifacts_dir=``; the page accepts it
    # and does not consult it, so passing it changes nothing.
    html = build_live_html(journal, artifacts_dir=artifacts)

    assert "distinctive_frame_000002" not in html
    assert "frame_000000" not in html
    assert html == build_live_html(journal)


def test_a_journal_with_no_tick_says_so_and_still_shows_the_game() -> None:
    html = build_live_html(_journal(()))

    assert "no tick has been recorded yet" in html
    # The game is the run's newest published frame, not a journal fact, so an
    # empty journal still gets the live image.
    assert f'src="{LIVE_FRAME_URL}"' in html


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

    head = _between(html, "<head>", "</head>")
    assert f'<meta http-equiv="refresh" content="{REFRESH_SECONDS:g}">' in head
    assert '<a href="/">' in html


def test_the_script_runs_once_at_the_end_and_removes_the_refresh_meta() -> None:
    html = build_live_html(_journal((_tick(0),)))

    assert html.count("<script>") == 1
    script = _between(html, "<script>", "</script>")
    # The meta is the no-script fallback; a browser that runs the script must
    # drop it, or the whole-page reload would restart the polling every second.
    assert "document.querySelector('meta[http-equiv=\"refresh\"]')" in script
    assert "meta.parentNode.removeChild(meta)" in script
    assert "getElementById(\"panel\")" in script
    assert f'"{LIVE_FRAME_URL}"' in script
    # Emitted last in the body, after the markup it reaches for.
    assert _between(html, "</script>", "</body>").strip() == ""


def test_without_javascript_the_noscript_sentence_says_the_whole_page_reloads() -> None:
    html = build_live_html(_journal((_tick(0),)))

    noscript = _between(html, "<noscript>", "</noscript>")
    assert "the whole page" in noscript
    assert "reloads" in noscript
    assert f"{REFRESH_SECONDS:g} s" in noscript


def test_the_page_names_its_rates_and_does_not_claim_60_fps() -> None:
    html = build_live_html(_journal((_tick(0),)))

    assert "decision tick rate, not 60 fps" in html
    assert "this page is not 60 fps" in html
    assert f"about every {FRAME_POLL_MS} ms" in html
    # The old claim about the frame is gone: this page shows the frame the run is
    # publishing now, and the recorded per-tick frames stay in the report.
    assert "frame the journal recorded" not in html
    assert "publishing now" in html


def test_a_hostile_reason_is_escaped() -> None:
    injected = "danger <script>alert('x')</script>"
    html = build_live_html(_journal((_tick(0, reason=injected),)))

    # The page's own script is the one ``<script>`` element; the reason is text.
    assert "<script>alert(" not in html
    assert "&lt;script&gt;alert(" in html
    assert html.count("<script>") == 1


def test_build_live_html_is_deterministic() -> None:
    journal = _journal((_tick(0, action="right"),))

    assert build_live_html(journal) == build_live_html(journal)
