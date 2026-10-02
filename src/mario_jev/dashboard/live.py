"""The live page: one snapshot of what the agent is doing right now, and why.

``mario_jev live`` puts the game in a window and this page beside it. The window
is the 60 fps view; this page is the one thing the window cannot show -- the
decision the agent just made and the scores behind it. It is a **snapshot**, not
a stream: every request reads the journal as it is at that moment, builds the
page, and sends it. There is no JavaScript, no WebSocket and no polling loop, for
the same reason the static report has none -- the evidence is the journal, and a
page that keeps state of its own would be a second source of truth.

The page refreshes itself with ``<meta http-equiv="refresh" content="0.5">``.
Half a second is chosen, not measured: a full second lets a fast run finish
several decision ticks between refreshes, and a tenth of a second spends the
viewer rebuilding a page whose newest tick has not changed. Half a second keeps
the newest tick roughly current for a person sitting next to the game window. It
is deliberately not 60 fps and says so on the page: the tick rate is the rate the
agent decides, and the frame the page shows is the one frame the journal recorded
for that decision.

What the page answers
---------------------

* the stage, the newest tick, its frame index and the run's status -- the status
  is the trailer-derived rule the full report already uses: no summary trailer is
  ``in progress``, and a trailer is its recorded ``termination`` verbatim. It is
  never re-derived and never invented;
* the newest tick's frame, referenced through that tick's own ``frame_png`` path
  so the viewer's artifact route serves it;
* the buttons the applied action presses, read out of the action set by name --
  never by splitting the recorded name in the template;
* the Jev scores: every question the tick fired, its choice probabilities with
  the chosen one marked, a score answer's score and legend, and the recorded
  ``noul`` note;
* the rollout candidates with the row of the applied action and Jev's picked
  row each marked for the fact it stands for -- ``(applied)`` for what ran,
  ``(chosen)`` for what Jev answered, and both when one row is both;
* the decision itself: action, source, confidence, risk and reason.

Reuse and deliberate differences from the static report
-------------------------------------------------------

The escaping, the mapping coercion and the wording for "not recorded", "no call"
and "in progress" come from :mod:`mario_jev.dashboard.build`; this module does not
grow a second way to say the same thing. What it deliberately does *not* reuse is
the report's way of *transporting* a frame: :func:`build_html` embeds the PNG
bytes as a ``data:`` URI because that report is a self-contained file. This page
instead references the recorded name as a URL, because it is refreshed over and
over and because the serve route already knows how to serve one artifact by name.
The frame is the newest tick's own ``frame_png``; the name is never reconstructed
from the frame index.

The action set
--------------

The emulator always builds the default set today: ``_build_emulator`` passes no
``action_set``, and the journal header records none. This page therefore calls
:func:`~mario_jev.emulator.build_actions` with no argument. The day a run can use
another set, either the set's name has to be recorded in the header or the set has
to be passed in here -- the name recorded in ``decision.action`` is not enough to
choose the set it came from, and this page will not guess.

Write order
-----------

:class:`~mario_jev.journal.JournalWriter` writes the frame artifact **before** it
flushes the tick line that names it. A tick the reader can see therefore always
has its recorded frame already on disk, so no refresh can transiently reference a
file that is not there yet; no machinery is added for a window that the writer's
own ordering closes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from mario_jev.emulator import Action, build_actions
from mario_jev.journal import Journal

from .build import (
    IN_PROGRESS,
    NO_CALL,
    NOT_RECORDED,
    _answered,
    _best_action,
    _confidence_text,
    _decision,
    _escape,
    _frame_index,
    _mapping,
    _resolve_artifacts_dir,
    _risk_text,
    _rows,
    _source_text,
    _STYLE as _REPORT_STYLE,
    _tick_number,
)

__all__ = ["REFRESH_SECONDS", "build_live_html"]

#: How often the page reloads itself, in seconds. See the module docstring for why
#: it is half a second and not a frame.
REFRESH_SECONDS = 0.5

#: The live page's own title.
DEFAULT_TITLE = "mario-jev live"

#: The report's stylesheet, plus the few rules only this page needs. Reusing the
#: report's style keeps the two pages looking like the same artifact; the added
#: rules are the frame, the joypad pad and the answer line.
_LIVE_STYLE = _REPORT_STYLE + "\n" + """
.frame { image-rendering: pixelated; border: 1px solid #ddd; max-width: 100%; }
.pad { display: flex; gap: 8px; margin: 6px 0; }
.pad-button { border: 1px solid #bbb; border-radius: 4px; padding: 4px 10px;
              background: #f4f4f4; font-size: 13px; }
.pad-button.pressed { background: #fff1b0; border-color: #b58a00; font-weight: 600; }
.answer-line { font-size: 13px; margin: 4px 0; }
tr.applied th { background: #e6f0ff; }
h4 { margin: 12px 0 2px; }
""".strip()


# -- small helpers over the recorded tick -------------------------------------------


def _pad_buttons(actions: Sequence[Action]) -> list[str]:
    """Every button the set can press, in the order the set first presses them.

    ``NOOP`` is deliberately not a button and does not appear: it is the action
    that presses nothing (its bits are zero), and a pad of buttons is not a pad of
    actions. The order is first-seen across the set, so the same set always
    renders the same pad.
    """
    buttons: list[str] = []
    for action in actions:
        if action.bits == 0:
            continue
        for button in action.buttons:
            if button not in buttons:
                buttons.append(button)
    return buttons


def _chosen_name(tick: Mapping[str, object]) -> str | None:
    """Jev's picked candidate for this tick, by name, or ``None`` if it made none.

    The Jev ``best_action`` answer is what chose among the candidates the rollout
    table lists, so it is read straight from the response. It is a claim about
    the **answer**, never about what ran: the confidence gate and the guardrail
    can apply a different action, and the page marks the two facts apart for
    exactly that reason. A tick with no answer -- no credential, a degraded call,
    a tick that never fired one -- made no pick and carries no ``(chosen)`` mark
    in the rollout table; the scores section already shows whatever was recorded.
    """
    chosen, _ = _best_action(tick.get("response"))
    return chosen


def _applied_name(tick: Mapping[str, object]) -> str | None:
    """The action this tick applied, by name, or ``None`` if none was recorded.

    Read from ``decision.action``, which records what the loop actually ran after
    the confidence gate and the guardrail had their say. It is a claim about what
    **ran**, not about what Jev picked: the rollout table marks this row
    ``(applied)`` and Jev's picked row, when the two differ, ``(chosen)``, so a
    reader can never take the answered candidate for the executed one.
    """
    action = _decision(tick).get("action")
    return None if action is None else str(action)


def _status_text(journal: Journal) -> str:
    """The run's status, by the full report's trailer-derived rule.

    A journal with no summary trailer is a run still going, and reads ``in
    progress`` -- never ``not recorded``, which would describe a finished run that
    lost part of itself. A trailer is its recorded ``termination`` verbatim; a
    trailer missing that key is ``not recorded``, exactly as the report's summary
    table prints it.
    """
    if journal.summary is None:
        return IN_PROGRESS
    termination = _mapping(journal.summary).get("termination")
    return NOT_RECORDED if termination is None else str(termination)


def _probability_rows(value: object, chosen: object) -> str:
    """The probability table for one answer, with ``chosen`` marked."""
    probabilities = _mapping(value)
    if not probabilities:
        return f'<p class="missing">{NOT_RECORDED}: no probabilities were recorded</p>'
    rows = []
    for name, probability in probabilities.items():
        is_chosen = chosen is not None and str(name) == str(chosen)
        css = ' class="chosen"' if is_chosen else ""
        marker = " (chosen)" if is_chosen else ""
        rows.append(
            f"<tr{css}><th>{_escape(name)}{_escape(marker)}</th>"
            f"<td>{_escape(probability)}</td></tr>"
        )
    return '<table class="scores"><tbody>' + "".join(rows) + "</tbody></table>"


def _answer_block(question_id: str, answer: Mapping[str, object]) -> str:
    """One question's recorded answer, in the shape its type carries.

    The three types are rendered as three different facts rather than one table
    with holes: a choice has probabilities and a chosen name, a score has a
    fractional score and its legend, and ``noul`` has neither a confidence nor a
    distribution, so it shows the recorded ``noul`` note alone. An unknown type
    says so instead of guessing which keys matter.
    """
    kind = answer.get("type")
    heading = f"<h4>{_escape(question_id)}</h4>"
    if kind == "choice":
        chosen = answer.get("choice")
        confidence = answer.get("confidence")
        return (
            heading
            + f'<p class="answer-line">choice: {_escape(chosen)} '
            f'(confidence {_escape(confidence)})</p>'
            + _probability_rows(answer.get("probabilities"), chosen)
        )
    if kind == "score":
        legend = _mapping(answer.get("legend"))
        legend_rows = "".join(
            f"<tr><th>{_escape(level)}</th><td>{_escape(text)}</td></tr>"
            for level, text in legend.items()
        )
        legend_html = (
            f'<table class="scores legend"><tbody>{legend_rows}</tbody></table>'
            if legend_rows
            else f'<p class="missing">{NOT_RECORDED}: no legend was recorded</p>'
        )
        return (
            heading
            + f'<p class="answer-line">score: {_escape(answer.get("score"))} '
            f'(confidence {_escape(answer.get("confidence"))})</p>'
            + legend_html
            + _probability_rows(answer.get("probabilities"), None)
        )
    if kind == "noul":
        return (
            heading
            + '<p class="answer-line">noul: the recorded note is '
            f'{_escape(answer.get("noul"))}</p>'
        )
    return heading + f'<p class="missing">{NOT_RECORDED}: unknown answer type</p>'


# -- the page sections ---------------------------------------------------------------


def _now_section(journal: Journal, tick: Mapping[str, object] | None) -> str:
    if tick is None:
        tick_text: object = NOT_RECORDED
        frame_text: object = NOT_RECORDED
    else:
        tick_text = _tick_number(tick)
        frame_text = _frame_index(tick)
    pairs = [
        ("stage", journal.run.get("stage")),
        ("tick", tick_text),
        ("frame index", frame_text),
        ("status", _status_text(journal)),
    ]
    return f'<table class="facts"><tbody>{_rows(pairs)}</tbody></table>'


def _frame_section(base: Path | None, tick: Mapping[str, object] | None) -> str:
    """The newest tick's frame, referenced by its own recorded path.

    The reference is the viewer-independent URL ``/<frame_png>``: the serve route
    resolves exactly that shape against the artifacts directory and answers the
    PNG. A tick with no recorded frame, or one whose file is not on disk, is the
    words ``not recorded`` -- the report's own rule that a missing picture is
    text, never a placeholder that could be mistaken for a frame.
    """
    if tick is None:
        return (
            '<p class="missing">no tick has been recorded yet, so there is no '
            "frame to show</p>"
        )
    name = tick.get("frame_png")
    if name is None or str(name) == "":
        return f'<p class="missing">{NOT_RECORDED}: this tick recorded no frame</p>'
    if base is not None and not (base / str(name)).is_file():
        return (
            f'<p class="missing">{NOT_RECORDED}: the recorded frame is not on '
            "disk</p>"
        )
    label = f"frame tick {_tick_number(tick)}"
    return (
        f'<img class="frame" alt="{_escape(label)}" src="/{_escape(name)}">'
    )


def _buttons_section(tick: Mapping[str, object] | None) -> str:
    """The applied action, translated to button tokens by the action set itself.

    The name recorded in ``decision.action`` is looked up in the set the emulator
    builds; it is never split on ``+``. A name outside the set is said out loud
    rather than guessed at, because a name this page cannot place is a name whose
    buttons nobody here can know.
    """
    if tick is None:
        return '<p class="missing">no tick has been recorded yet</p>'
    action_value = _decision(tick).get("action")
    name = None if action_value is None else str(action_value)
    actions = build_actions()
    by_name = {action.name: action for action in actions}
    chosen = by_name.get(name) if name is not None else None

    pad = []
    for button in _pad_buttons(actions):
        pressed = chosen is not None and button in chosen.buttons
        css = ' class="pad-button pressed"' if pressed else ' class="pad-button"'
        pad.append(f"<span{css}>{_escape(button)}</span>")

    if name is None:
        note = f'<p class="missing">{NOT_RECORDED}: the tick recorded no action</p>'
    elif chosen is None:
        note = (
            "<p class=\"missing\">the applied action "
            f"<code>{_escape(name)}</code> is not in the action set, so its "
            "buttons cannot be read from it</p>"
        )
    else:
        tokens = "+".join(chosen.buttons) if chosen.bits else "nothing"
        note = (
            f'<p class="answer-line">pressed: <code>{_escape(tokens)}</code></p>'
        )
    return note + f'<div class="pad">{"".join(pad)}</div>'


def _scores_section(tick: Mapping[str, object] | None) -> str:
    """Every question the tick fired, with its recorded answer.

    A tick with no response says ``no call`` instead of an empty score table. A
    tick that observed a call but has no answer recorded yet says ``not
    recorded``; the two are different facts and the report already keeps them
    apart.
    """
    if tick is None:
        return f'<p class="missing">{NO_CALL}: no tick has been recorded yet</p>'
    response = _mapping(tick.get("response"))
    answers = _mapping(response.get("answers"))
    if not answers:
        if _answered(tick):
            return f'<p class="missing">{NOT_RECORDED}: no answers were recorded</p>'
        return (
            f'<p class="missing">{NO_CALL}: no Jev response was recorded for '
            "this tick</p>"
        )
    return "".join(
        _answer_block(str(question_id), _mapping(answer))
        for question_id, answer in answers.items()
    )


def _rollout_section(
    tick: Mapping[str, object] | None,
    applied: str | None,
    chosen: str | None,
) -> str:
    """The tick's rollout candidates, each marked for the fact it stands for.

    The applied row (``decision.action``) is marked ``(applied)`` because it is
    what ran; Jev's picked row is marked ``(chosen)`` because it is what the
    answer picked. A tick where the guardrail or the confidence gate applied
    something else shows both marks on two rows, and each says which fact it is;
    a tick where they agree marks one row once with ``(applied, chosen)`` rather
    than two marks a reader would have to reconcile.
    """
    if tick is None:
        return f'<p class="missing">{NOT_RECORDED}: no tick has been recorded yet</p>'
    candidates = _mapping(tick.get("rollout")).get("candidates")
    if (
        not isinstance(candidates, Sequence)
        or isinstance(candidates, (str, bytes))
        or not candidates
    ):
        return (
            f'<p class="missing">{NOT_RECORDED}: this tick recorded no rollout '
            "candidates</p>"
        )
    header = (
        "<tr><th>candidate</th><th>dx</th><th>outcome</th><th>risk</th>"
        "<th>final_x</th></tr>"
    )
    rows = []
    for candidate in candidates:
        item = _mapping(candidate)
        name = item.get("action")
        text = None if name is None else str(name)
        is_applied = applied is not None and text == applied
        is_chosen = chosen is not None and text == chosen
        classes = []
        if is_applied:
            classes.append("applied")
        if is_chosen:
            classes.append("chosen")
        css = f' class="{" ".join(classes)}"' if classes else ""
        if is_applied and is_chosen:
            marker = " (applied, chosen)"
        elif is_applied:
            marker = " (applied)"
        elif is_chosen:
            marker = " (chosen)"
        else:
            marker = ""
        rows.append(
            f"<tr{css}><th>{_escape(name)}{_escape(marker)}</th>"
            f"<td>{_escape(item.get('dx'))}</td>"
            f"<td>{_escape(item.get('outcome'))}</td>"
            f"<td>{_escape(item.get('risk'))}</td>"
            f"<td>{_escape(item.get('final_x'))}</td></tr>"
        )
    return (
        f'<table class="rollout"><thead>{header}</thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _decision_section(tick: Mapping[str, object] | None) -> str:
    """The tick's decision, with the report's own wording for missing values."""
    if tick is None:
        return f'<p class="missing">{NOT_RECORDED}: no tick has been recorded yet</p>'
    decision = _decision(tick)
    pairs = [
        ("action", decision.get("action")),
        ("source", _source_text(decision)),
        ("confidence", _confidence_text(tick)),
        ("risk", _risk_text(tick)),
        ("reason", decision.get("reason")),
    ]
    return f'<table class="facts"><tbody>{_rows(pairs)}</tbody></table>'


def build_live_html(
    journal: Journal,
    *,
    artifacts_dir: object | None = None,
) -> str:
    """Render the live page for ``journal`` as one HTML document.

    ``artifacts_dir`` overrides the directory recorded in the run header, exactly
    as :func:`~mario_jev.dashboard.build.build_html` uses it: the serve route
    passes the viewer's own directory, and a journal written without one records
    none. It is only consulted to tell a recorded frame from a missing one; the
    reference itself is the tick's recorded ``frame_png`` path.
    """
    base = _resolve_artifacts_dir(journal, artifacts_dir)
    tick = journal.ticks[-1] if journal.ticks else None
    applied = _applied_name(tick) if tick is not None else None
    chosen = _chosen_name(tick) if tick is not None else None
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f'<meta http-equiv="refresh" content="{REFRESH_SECONDS}">',
        f"<title>{_escape(DEFAULT_TITLE)}</title>",
        f"<style>{_LIVE_STYLE}</style>",
        "</head>",
        "<body>",
        f"<h1>{_escape(DEFAULT_TITLE)}</h1>",
        '<p class="note">This is a snapshot, refreshed every '
        f"{REFRESH_SECONDS} s: this is the decision tick rate, not 60 fps. The "
        "game itself runs in its own window at 60 fps; the frame below is the one "
        "frame the journal recorded for the newest decision tick.</p>",
        '<p><a href="/">the full report</a></p>',
        "<section><h2>Right now</h2>",
        f"{_now_section(journal, tick)}",
        "</section>",
        "<section><h2>Newest tick frame</h2>",
        f"{_frame_section(base, tick)}",
        "</section>",
        "<section><h2>Buttons</h2>",
        f"{_buttons_section(tick)}",
        "</section>",
        "<section><h2>Jev scores</h2>",
        f"{_scores_section(tick)}",
        "</section>",
        "<section><h2>Rollout candidates</h2>",
        f"{_rollout_section(tick, applied, chosen)}",
        "</section>",
        "<section><h2>Decision</h2>",
        f"{_decision_section(tick)}",
        "</section>",
        "</body>",
        "</html>",
    ]
    return "\n".join(parts)
