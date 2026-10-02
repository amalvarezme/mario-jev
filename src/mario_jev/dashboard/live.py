"""The live page: the game on the right, and one snapshot of the decisions beside it.

``mario_jev live`` plays the run and this page shows it. The game sits **inside**
the page now, beside the scores: the newest frame the run published is polled into
an image on the right, and the decision panel -- the buttons, the Jev scores, the
rollout candidates and the decision -- sits on the left in the same flex row. The
native game window is still there and is still the only 60 fps view; the page is
deliberately not, and says so where the game is.

JavaScript, and the change of mind
----------------------------------

G2 built this page with no JavaScript, on the plain reason that a page holding no
state of its own cannot become a second source of truth beside the journal. The
game inside the page needs some, and this module states what it is for rather than
slipping it in:

* the markup still carries ``<meta http-equiv="refresh">``, for a browser that
  never runs the script;
* on load the script **removes** that meta: a whole-page reload every second would
  restart the frame polling and blank the game for the length of a page load;
* it polls the frame URL with a cache-busting counter every ``FRAME_POLL_MS``,
  loading into a hidden second ``<img>`` and swapping only once the new frame has
  decoded, so one frame costs one request and the visible image never flickers and
  never shows a half-loaded frame;
* a failed request leaves the last decoded frame on screen instead of blanking it;
* it re-fetches the page every ``PANEL_REFRESH_MS`` and replaces only the panel
  subtree, so a panel refresh never touches the game image.

The script holds no state the journal does not: it moves pixels into an image and
copies rendered HTML into the panel, and the panel it copies is the same page this
module built. The fallback is the meta refresh, and a ``<noscript>`` sentence says
so: a browser without the script reloads the whole page, frame included, at that
slower rate.

What the page answers
---------------------

* the stage, the newest tick, its frame index and the run's status -- the status
  is the trailer-derived rule the full report already uses: no summary trailer is
  ``in progress``, and a trailer is its recorded ``termination`` verbatim. It is
  never re-derived and never invented;
* the live game: the newest frame the run published, polled into an image on the
  right, at twice the NES size and pixelated so the doubled pixels stay square;
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
the report's recorded frames: the per-tick PNGs, perception overlay and all, stay
where they belong, in the report at ``/``. This page shows the live frame the run
is publishing now -- the game, not a picture of a past tick -- and a polled image
of a recorded frame would only be a slow second copy of the report.

The action set
--------------

The emulator always builds the default set today: ``_build_emulator`` passes no
``action_set``, and the journal header records none. This page therefore calls
:func:`~mario_jev.emulator.build_actions` with no argument. The day a run can use
another set, either the set's name has to be recorded in the header or the set has
to be passed in here -- the name recorded in ``decision.action`` is not enough to
choose the set it came from, and this page will not guess.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from mario_jev.emulator import SCREEN_SHAPE, Action, build_actions
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
    _risk_text,
    _rows,
    _source_text,
    _STYLE as _REPORT_STYLE,
    _tick_number,
)

__all__ = [
    "FRAME_POLL_MS",
    "FRAME_SCALE",
    "LIVE_FRAME_URL",
    "PANEL_REFRESH_MS",
    "REFRESH_SECONDS",
    "build_live_html",
]

#: The path the game image points at: the newest frame the run published. It is
#: written here rather than imported from :mod:`mario_jev.serve` because ``serve``
#: imports this module, so the import back would be a cycle; a test pins the two
#: values equal instead. It must stay equal to ``serve.LIVE_FRAME_PATH``, which is
#: the route that answers it.
LIVE_FRAME_URL = "/live/frame.png"

#: The NES frame as the emulator defines it, and how many screen pixels each NES
#: pixel gets. The page shows the game at twice its native size so it can be read
#: without resizing a 256-pixel-wide image, and ``image-rendering: pixelated``
#: keeps the doubled pixels square instead of smearing them.
FRAME_HEIGHT, FRAME_WIDTH = SCREEN_SHAPE[0], SCREEN_SHAPE[1]
FRAME_SCALE = 2

#: How often the script asks for a new frame, in milliseconds. About 70 ms is
#: roughly 14 Hz: fast enough to look continuous beside the scores, and slow
#: enough that a slow viewer thread answers one request before the next is due.
FRAME_POLL_MS = 70

#: How often the script re-fetches the page and replaces the panel, in
#: milliseconds. About once a second: the panel is the decision tick rate, and a
#: faster refresh would spend the viewer rebuilding a page whose tick has not
#: changed.
PANEL_REFRESH_MS = 1000

#: How often the page reloads itself without JavaScript, in seconds. It is the
#: markup's ``<meta http-equiv="refresh">`` and the no-script fallback: the whole
#: page, the game image included, reloads at this slower rate. The script removes
#: the meta on load so that fallback cannot blank the polled game.
REFRESH_SECONDS = 1.0

#: The live page's own title.
DEFAULT_TITLE = "mario-jev live"

#: The one honest line under the game heading: the rates, and the refusal to claim
#: 60 fps for a page that polls.
_GAME_NOTE = (
    f"the game frame is polled about every {FRAME_POLL_MS} ms and the panel is "
    "re-fetched about once a second, so this page is not 60 fps; the native game "
    "window remains the 60 fps view."
)

#: What a browser without JavaScript gets, and says exactly: the whole page --
#: this frame included -- reloads at the meta-refresh rate instead of polling.
_NOSCRIPT_NOTE = (
    "JavaScript is off: the whole page, this frame included, reloads about every "
    f"{REFRESH_SECONDS:g} s at the meta-refresh rate instead."
)

#: The report's stylesheet, plus the rules this page needs. Reusing the report's
#: style keeps the two pages looking like the same artifact; the added rules are
#: the two-column board, the stacked game frames and the joypad pad.
_LIVE_STYLE = _REPORT_STYLE + "\n" + """
.board { display: flex; flex-wrap: wrap; gap: 24px; align-items: flex-start; }
.panel { flex: 1 1 320px; min-width: 280px; }
.game { flex: 1 1 520px; }
.frames { display: grid; max-width: 100%; }
.game-frame { grid-area: 1 / 1; image-rendering: pixelated; border: 1px solid #ddd;
              max-width: 100%; height: auto; }
.game-frame.hidden { visibility: hidden; }
.pad { display: flex; gap: 8px; margin: 6px 0; }
.pad-button { border: 1px solid #bbb; border-radius: 4px; padding: 4px 10px;
              background: #f4f4f4; font-size: 13px; }
.pad-button.pressed { background: #fff1b0; border-color: #b58a00; font-weight: 600; }
.answer-line { font-size: 13px; margin: 4px 0; }
tr.applied th { background: #e6f0ff; }
h4 { margin: 12px 0 2px; }
""".strip()

#: The whole of the page's JavaScript, inline. It is written with ``%``
#: placeholders because the script is mostly braces, and it prints the same rates
#: the constants above name. See the module docstring for what each half is for and
#: why the meta removal is not optional.
_LIVE_SCRIPT = """<script>
(function () {
  "use strict";
  var FRAME_POLL_MS = %(frame_poll_ms)d;
  var PANEL_REFRESH_MS = %(panel_refresh_ms)d;
  var FRAME_URL = "%(frame_url)s";

  // A whole-page reload every second would restart this polling loop and blank
  // the game for the length of a load. The markup keeps the meta for a browser
  // that never runs this script; this removes it for one that does.
  var meta = document.querySelector('meta[http-equiv="refresh"]');
  if (meta !== null && meta.parentNode !== null) {
    meta.parentNode.removeChild(meta);
  }

  var front = document.getElementById("live-frame-a");
  var back = document.getElementById("live-frame-b");
  var sequence = 0;

  function nextFrame() {
    // One request in flight at a time: the next poll is scheduled only after
    // this one has decoded or failed, so a slow response cannot race the swap.
    var incoming = back;
    incoming.onload = function () {
      incoming.onload = null;
      incoming.onerror = null;
      // Swap only once the new frame has decoded, so the visible image never
      // shows a half-loaded frame and never blinks.
      incoming.className = "game-frame";
      front.className = "game-frame hidden";
      var previous = front;
      front = incoming;
      back = previous;
      window.setTimeout(nextFrame, FRAME_POLL_MS);
    };
    incoming.onerror = function () {
      incoming.onload = null;
      incoming.onerror = null;
      // Leave the last decoded frame on screen: the incoming image was hidden
      // all along, and a failed request must not blank what is being watched.
      incoming.removeAttribute("src");
      window.setTimeout(nextFrame, FRAME_POLL_MS);
    };
    // The counter defeats any cache between the browser and the viewer; the
    // route answers no-store, and this also covers an intermediary's copy.
    incoming.src = FRAME_URL + "?n=" + (sequence++);
  }

  function refreshPanel() {
    // Replace only the panel subtree; the game image is never touched, so a
    // panel refresh cannot disturb the frame being polled.
    fetch(window.location.href, { cache: "no-store" })
      .then(function (response) { return response.text(); })
      .then(function (text) {
        var parsed = new DOMParser().parseFromString(text, "text/html");
        var fresh = parsed.getElementById("panel");
        var current = document.getElementById("panel");
        if (fresh !== null && current !== null) {
          current.innerHTML = fresh.innerHTML;
        }
        window.setTimeout(refreshPanel, PANEL_REFRESH_MS);
      })
      .catch(function () {
        // A failed panel fetch leaves the panel it has; the next try is soon.
        window.setTimeout(refreshPanel, PANEL_REFRESH_MS);
      });
  }

  nextFrame();
  window.setTimeout(refreshPanel, PANEL_REFRESH_MS);
})();
</script>""" % {
    "frame_poll_ms": FRAME_POLL_MS,
    "panel_refresh_ms": PANEL_REFRESH_MS,
    "frame_url": LIVE_FRAME_URL,
}


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


def _game_section() -> str:
    """The live game: the frame the run publishes now, polled into an image.

    Two ``<img>`` elements share one grid cell and the script swaps between them:
    the newest frame loads into the hidden one and becomes visible only once it
    has decoded, so the picture never blinks and one frame is one request. The
    first image carries the frame URL in the markup, which gives a browser
    without JavaScript a frame to show, at the meta-refresh rate.

    The image is twice the NES size because a 256-pixel-wide frame is unreadable
    on a normal display, and ``pixelated`` keeps the doubled pixels square rather
    than smearing them; the reported URL is a constant pinned to the serve route
    by a test, never imported from it.
    """
    width = FRAME_WIDTH * FRAME_SCALE
    height = FRAME_HEIGHT * FRAME_SCALE
    return (
        '<div class="frames">'
        f'<img id="live-frame-a" class="game-frame" src="{LIVE_FRAME_URL}" '
        f'width="{width}" height="{height}" alt="the live game frame">'
        f'<img id="live-frame-b" class="game-frame hidden" '
        f'width="{width}" height="{height}" alt="">'
        "</div>"
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

    ``artifacts_dir`` is accepted and not consulted. The viewer calls every page
    builder the same way, ``live(journal, artifacts_dir=...)``, so the parameter
    stays for that uniform call; this page references no recorded artifact -- the
    game is the frame the run publishes now -- so there is nothing for it to
    resolve.

    The panel is one ``id="panel"`` subtree because that is exactly what the
    script re-fetches and replaces; the game sits outside it, so a panel refresh
    can never reload the image being polled. The script comes last in the body
    so every element it reaches for already exists when it runs.
    """
    del artifacts_dir  # accepted for the viewer's uniform call; see above
    tick = journal.ticks[-1] if journal.ticks else None
    applied = _applied_name(tick) if tick is not None else None
    chosen = _chosen_name(tick) if tick is not None else None
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f'<meta http-equiv="refresh" content="{REFRESH_SECONDS:g}">',
        f"<title>{_escape(DEFAULT_TITLE)}</title>",
        f"<style>{_LIVE_STYLE}</style>",
        "</head>",
        "<body>",
        f"<h1>{_escape(DEFAULT_TITLE)}</h1>",
        '<p class="note">The panel on the left is the newest decision tick, '
        "refreshed about once a second: this is the decision tick rate, not 60 "
        "fps. The game on the right is the frame the run is publishing now, not a "
        "frame recorded in the journal; the recorded per-tick frames stay in the "
        "full report.</p>",
        '<p><a href="/">the full report</a></p>',
        '<div class="board">',
        '<div class="panel" id="panel">',
        "<section><h2>Right now</h2>",
        f"{_now_section(journal, tick)}",
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
        "</div>",
        '<div class="game">',
        "<section><h2>The game</h2>",
        f"{_game_section()}",
        f'<p class="note">{_escape(_GAME_NOTE)}</p>',
        f'<noscript><p class="note">{_escape(_NOSCRIPT_NOTE)}</p></noscript>',
        "</section>",
        "</div>",
        "</div>",
        _LIVE_SCRIPT,
        "</body>",
        "</html>",
    ]
    return "\n".join(parts)
