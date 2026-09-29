"""The evidence dashboard: one static, self-contained HTML file.

There is no server and no web framework here. :func:`build_html` is a pure
function of a :class:`~mario_jev.journal.Journal` (plus the PNG artifacts on
disk): it reads no clock, opens no socket and touches no global, so the same
journal always renders byte-identical HTML. That purity is not a nicety — it is
what makes the report reviewable evidence rather than a snapshot of when it was
opened.

Three honesty rules shape every decision in this module, and they are the point
of the artifact rather than a formatting preference:

* **A tick that made no Jev call is shown as "no call".** It is never rendered as
  a zero confidence or a zero latency, and in the time series it is a *gap* in the
  line rather than a point at ``0``.
* **The Jev latency shown is the call's, never the tick's.** With the call off the
  tick, ``latencies_ms["jev"]`` is the few microseconds the tick spent submitting
  and taking — a 21 s call lands on a tick whose Jev phase reads 0.002 ms — so
  every latency on this page comes from the *call*: ``jev_call_ms``, the
  dispatcher's measurement around the whole thing, and only failing that
  ``response["latency_ms"]``, which is the single attempt that succeeded and is all
  a journal written before ``jev_call_ms`` existed has to offer. The order matters:
  for a call that retried, the response's own figure is the *smaller* one — a
  20 694 ms call whose second attempt took 577 ms would be reported as 577 ms, which
  is the undercount this journal format was extended to stop making. A tick that
  observed no call has no latency to show, and says ``no call`` or ``not recorded``
  instead of a number that would read as a fast one.
* **A missing or unreadable picture is text, not a picture.** A tick with no
  ``frame_png``, or one whose file cannot be read, is labelled "not recorded" in
  words. No placeholder is ever drawn that could be mistaken for a real frame.
* **Numbers are derived from what was recorded, or they are not shown.** The
  source counts, the fallback count and the fallback rate are counted from the
  recorded ``decision.source``/``decision.jev_called`` values; the recorded
  summary fields are printed verbatim. Nothing is estimated. A journal with no
  trailer yet is a run still in progress and says so in place of the summary
  counts; it is never labelled ``not recorded``, which would describe a finished
  run that had lost part of itself.

Every string that came out of the journal is HTML-escaped before it reaches the
page, because a reason is data and data must not be able to inject markup.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Mapping, Sequence

from mario_jev.config import redact_url_userinfo
from mario_jev.journal import Journal
# The one recorded degradation this module has to recognise by name: a tick that
# threw an answer away still paid for the call, and must not read as one that
# never asked. Imported rather than repeated, so the page and the loop can never
# disagree about the spelling.
from mario_jev.policy import JEV_STALE_DISCARDED

__all__ = ["build_html", "render_text", "write_html"]

#: A fallback source is one whose recorded name carries this prefix.
FALLBACK_PREFIX = "fallback_"
#: The default page title and the terminal report's heading.
DEFAULT_TITLE = "mario-jev run report"
#: The words used when no call was made, and when a value was not recorded.
NO_CALL = "no call"
NOT_RECORDED = "not recorded"
#: The words for a summary trailer that has not been written yet. A run still
#: going has not recorded its counts -- and ``not recorded`` would describe a
#: finished run that lost part of itself, which is a different fact.
IN_PROGRESS = "in progress"
#: The words for a tick that applied no answer, and for one that threw away the
#: answer it took. Both are recorded facts, and neither is "no call": the run may
#: well have had a call in flight, or have just paid for one it refused to act on.
NO_ANSWER = "no answer"
DISCARDED = "discarded (stale)"
#: The label used for a decision whose source was not recorded at all.
UNRECORDED_SOURCE = "(unrecorded)"


# -- small, total readers over plain journal data ---------------------------------


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _escape(value: object) -> str:
    """Journal text, escaped for HTML text and attributes alike."""
    return html.escape("" if value is None else str(value), quote=True)


def _decision(tick: Mapping[str, object]) -> Mapping[str, object]:
    return _mapping(tick.get("decision"))


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _frame_index(tick: Mapping[str, object]) -> int:
    if "frame_index" in tick:
        return _as_int(tick["frame_index"])
    return _as_int(tick.get("tick"))


def _tick_number(tick: Mapping[str, object]) -> int:
    return _as_int(tick.get("tick"))


def _source_text(decision: Mapping[str, object]) -> str:
    source = decision.get("source")
    return UNRECORDED_SOURCE if source is None else str(source)


def _degradations(tick: Mapping[str, object]) -> tuple[str, ...]:
    """The tick's recorded degradations as plain strings, or an empty tuple."""
    recorded = tick.get("degradations")
    if isinstance(recorded, Sequence) and not isinstance(recorded, (str, bytes)):
        return tuple(str(item) for item in recorded)
    return ()


def _call_latency(tick: Mapping[str, object]) -> float | None:
    """What the *call* this tick observed cost, or ``None`` if it observed none.

    ``jev_call_ms`` first: it is the dispatcher's measurement around the whole call,
    so it includes every attempt and the backoff between them. ``response["latency_ms"]``
    is only the attempt that succeeded, and for a call that retried the two differ by
    the entire failed attempt -- a 20 694 ms call whose second attempt took 577 ms is
    one such case, and reporting 577 as "the Jev latency" hides 20 s. The response's
    own figure is therefore the *fallback*, and only because a journal written before
    ``jev_call_ms`` existed has nothing else to offer.

    ``latencies_ms["jev"]`` is deliberately not consulted — it is this tick's
    submit-and-take cost, and reporting it as the Jev latency would turn a 21 s call
    into 0.002 ms.
    """
    value = tick.get("jev_call_ms")
    if value is None:
        value = _mapping(tick.get("response")).get("latency_ms")
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _answered(tick: Mapping[str, object]) -> bool:
    """Whether this tick has any Jev call of its own to speak about.

    True when it fired one, when it applied an answer, or when it observed a call
    finish -- including one that failed or whose answer it discarded.
    ``decision.jev_called`` alone is no longer this question: it means "this tick
    *fired* a call", and a tick can apply or pay for a call it did not fire.
    Anything that would otherwise print ``no call`` over a tick that did ask asks
    this instead.
    """
    return (
        bool(_decision(tick).get("jev_called"))
        or tick.get("response") is not None
        or _call_latency(tick) is not None
    )


def _confidence_text(tick: Mapping[str, object]) -> str:
    value = _decision(tick).get("confidence")
    if value is not None:
        return f"{float(value):.3f}"  # type: ignore[arg-type]
    return NOT_RECORDED if _answered(tick) else NO_CALL


def _jev_latency_text(tick: Mapping[str, object]) -> str:
    value = _call_latency(tick)
    if value is not None:
        return f"{value:.3f}"
    if _decision(tick).get("jev_called"):
        # A call was fired and had not come back by the end of this tick. Its cost
        # is recorded on the tick that observes it, not invented here.
        return NOT_RECORDED
    return NO_CALL


def _answer_age_text(tick: Mapping[str, object]) -> str:
    """How old the answer this tick applied was, in ticks, or why there is no number.

    Four different facts, and none of them is ``0``: a journal written before the
    staleness fields existed says ``not recorded``; a tick that applied a fresh
    answer says ``0``; a tick that applied none says ``no answer``; and a tick that
    took an answer and threw it away says so, because it paid for that call.
    """
    if "answer_age_ticks" not in tick:
        return NOT_RECORDED
    value = tick.get("answer_age_ticks")
    if value is not None:
        return str(_as_int(value))
    if JEV_STALE_DISCARDED in _degradations(tick):
        return DISCARDED
    return NO_ANSWER


def _attempts_text(tick: Mapping[str, object]) -> str:
    """How many transport calls this tick's Jev answer cost, in words when unknown.

    A tick that made no call reads as ``no call``, never as ``1``: one attempt and
    no attempt are different facts, and a run's whole point can be that it never
    asked. A tick that observed a call which failed has no attempt count on the
    record at all, and says ``not recorded`` rather than borrowing one.
    """
    value = _mapping(tick.get("response")).get("attempts")
    if value is not None:
        return str(_as_int(value, default=1))
    return NOT_RECORDED if _answered(tick) else NO_CALL


def _risk_text(tick: Mapping[str, object]) -> str:
    decision = _decision(tick)
    value = decision.get("risk")
    if value is None:
        return NOT_RECORDED
    return f"{float(value):.3f}"  # type: ignore[arg-type]


def _best_action(response: object) -> tuple[str | None, Mapping[str, object]]:
    """The recorded ``best_action`` choice and its probabilities, when present."""
    answers = _mapping(_mapping(response).get("answers"))
    best = _mapping(answers.get("best_action"))
    choice = best.get("choice")
    probabilities = _mapping(best.get("probabilities"))
    return (None if choice is None else str(choice)), probabilities


# -- the derived numbers, counted from the recorded values ------------------------


def _derived(journal: Journal) -> dict[str, object]:
    """The counts this module is willing to stand behind.

    Every value is a count over the recorded decisions. ``sources`` is built in
    tick order, so its iteration order is deterministic, and ``fallbacks`` follows
    the recorded source names rather than a second definition of "fallback".
    """
    sources: dict[str, int] = {}
    decisions = 0
    fallbacks = 0
    for tick in journal.ticks:
        decision = _decision(tick)
        source = _source_text(decision)
        sources[source] = sources.get(source, 0) + 1
        if source.startswith(FALLBACK_PREFIX):
            fallbacks += 1
        if decision.get("jev_called"):
            decisions += 1
    total = len(journal.ticks)
    return {
        "ticks": total,
        "decisions": decisions,
        "sources": sources,
        "fallbacks": fallbacks,
        "fallback_rate": (fallbacks / total) if total else 0.0,
    }


def _staleness(journal: Journal) -> dict[str, int] | None:
    """How many answers were applied, how late, and how many were thrown away.

    ``None`` means the journal predates these fields entirely — no tick line
    carries ``answer_age_ticks`` — and the readers print ``not recorded`` for it.
    Counting zeros over an old file would be a fact nobody recorded: a run from
    before this change applied answers, it just never wrote down when they were
    asked for.
    """
    if not any("answer_age_ticks" in tick for tick in journal.ticks):
        return None
    applied = 0
    stale = 0
    oldest = 0
    discarded = 0
    for tick in journal.ticks:
        if JEV_STALE_DISCARDED in _degradations(tick):
            discarded += 1
        value = tick.get("answer_age_ticks")
        if value is None:
            continue
        age = _as_int(value)
        applied += 1
        oldest = max(oldest, age)
        if age > 0:
            stale += 1
    return {
        "applied": applied,
        "stale": stale,
        "oldest": oldest,
        "discarded": discarded,
    }


def _retried_ticks(journal: Journal) -> int:
    """How many ticks recorded a Jev answer that took more than one attempt.

    Counted over the recorded answers, wherever they landed: the attempt count is
    a fact about the call, and the tick that applies an answer is the one that
    writes it down, which need not be the tick that fired it.
    """
    retried = 0
    for tick in journal.ticks:
        value = _mapping(tick.get("response")).get("attempts")
        if value is not None and _as_int(value, default=1) > 1:
            retried += 1
    return retried


def _latency_stats(journal: Journal) -> tuple[float, float, float] | None:
    """``(min, median, max)`` over the calls this run observed, or ``None``.

    Over the *calls*, not over the ticks that fired one: a call fired on tick 4 and
    observed on tick 6 costs what it cost, and tick 6 is the only line that can say
    so. A tick that fired and is still waiting contributes nothing rather than a
    zero.
    """
    values: list[float] = []
    for tick in journal.ticks:
        value = _call_latency(tick)
        if value is None:
            continue
        values.append(value)
    if not values:
        return None
    values.sort()
    count = len(values)
    if count % 2:
        median = values[count // 2]
    else:
        median = (values[count // 2 - 1] + values[count // 2]) / 2.0
    return values[0], median, values[-1]


def _series(
    journal: Journal,
    selector,
) -> list[tuple[int, float | None]]:
    """One ``(frame_index, value-or-None)`` point per tick, in order.

    ``selector`` returns the value for a tick, or ``None`` when nothing was
    recorded. A ``None`` is a genuine gap in the series, never a zero.
    """
    points: list[tuple[int, float | None]] = []
    for tick in journal.ticks:
        points.append((_frame_index(tick), selector(tick)))
    return points


def _confidence_point(tick: Mapping[str, object]) -> float | None:
    value = _decision(tick).get("confidence")
    return None if value is None else float(value)  # type: ignore[arg-type]


def _latency_point(tick: Mapping[str, object]) -> float | None:
    return _call_latency(tick)


# -- the inline SVG charts ---------------------------------------------------------


def _svg_chart(
    *,
    chart_id: str,
    title: str,
    series: Sequence[tuple[int, float | None]],
    y_min: float,
    y_max: float,
    y_format: str = "{:.2f}",
) -> str:
    """One inline SVG line chart, with gaps where no value was recorded."""
    width, height = 680, 190
    left, right, top, bottom = 62, width - 16, 32, height - 34

    xs = [x for x, _ in series] or [0]
    x_lo = min(xs)
    x_hi = max(xs)
    if x_hi <= x_lo:
        x_hi = x_lo + 1
    if y_max <= y_min:
        y_max = y_min + 1.0

    def px(x: float) -> float:
        return left + (x - x_lo) / (x_hi - x_lo) * (right - left)

    def py(y: float) -> float:
        return bottom - (y - y_min) / (y_max - y_min) * (bottom - top)

    parts: list[str] = [
        f'<svg class="chart" data-series="{_escape(chart_id)}" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        'role="img" xmlns="http://www.w3.org/2000/svg">'
    ]
    parts.append(
        f'<text class="chart-title" x="{left}" y="18">{_escape(title)}</text>'
    )
    # Horizontal gridlines and their labels.
    for value in (y_min, (y_min + y_max) / 2.0, y_max):
        y = py(value)
        parts.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="axis-label" x="{left - 8}" y="{y + 3:.1f}" '
            f'text-anchor="end">{_escape(y_format.format(value))}</text>'
        )
    parts.append(
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>'
    )
    parts.append(
        f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>'
    )
    parts.append(
        f'<text class="axis-label" x="{left}" y="{bottom + 16}" '
        f'text-anchor="middle">{x_lo}</text>'
    )
    parts.append(
        f'<text class="axis-label" x="{right}" y="{bottom + 16}" '
        f'text-anchor="middle">{x_hi}</text>'
    )
    parts.append(
        f'<text class="axis-title" x="{(left + right) / 2:.1f}" y="{height - 6}" '
        f'text-anchor="middle">frame index</text>'
    )

    present = [(x, y) for x, y in series if y is not None]
    if not present:
        parts.append(
            f'<text class="chart-note" x="{left}" y="{(top + bottom) / 2:.1f}">'
            "no Jev calls were recorded; the series is empty</text>"
        )
        parts.append("</svg>")
        return "".join(parts)

    # One polyline per run of consecutive recorded points: a gap breaks the line.
    run: list[tuple[float, float]] = []
    for x, y in series:
        if y is None:
            _emit_segment(parts, run, px, py)
            run = []
            continue
        run.append((x, y))
    _emit_segment(parts, run, px, py)

    for x, y in present:
        parts.append(
            f'<circle class="point" cx="{px(x):.1f}" cy="{py(y):.1f}" r="3"/>'
        )

    parts.append("</svg>")
    return "".join(parts)


def _emit_segment(parts: list[str], run, px, py) -> None:
    if not run:
        return
    coordinates = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in run)
    parts.append(f'<polyline class="series" points="{coordinates}"/>')


# -- images ------------------------------------------------------------------------


def _resolve_artifacts_dir(
    journal: Journal,
    artifacts_dir: object | None,
) -> Path | None:
    """Where this journal's PNGs live: the argument, else the recorded directory."""
    if artifacts_dir is not None:
        return Path(artifacts_dir)
    recorded = journal.run.get("artifacts_dir")
    if recorded is None or str(recorded) == "":
        return None
    return Path(str(recorded))


def _embed_png(base: Path | None, name: object) -> str | None:
    """A ``data:`` URI for one recorded PNG, or ``None`` when it is not readable."""
    if base is None or name is None or str(name) == "":
        return None
    try:
        data = (base / str(name)).read_bytes()
    except OSError:
        return None
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _figure(base: Path | None, name: object, label: str) -> str:
    uri = _embed_png(base, name)
    caption = _escape(label)
    if uri is None:
        return (
            f'<figure class="artifact"><figcaption>{caption}</figcaption>'
            f'<p class="missing">{NOT_RECORDED}</p></figure>'
        )
    return (
        f'<figure class="artifact"><figcaption>{caption}</figcaption>'
        f'<img alt="{caption}" src="{uri}"></figure>'
    )


# -- HTML sections -----------------------------------------------------------------


def _rows(pairs: Sequence[tuple[str, object]]) -> str:
    return "".join(
        f"<tr><th>{_escape(key)}</th><td>{_escape(value)}</td></tr>"
        for key, value in pairs
    )


def _counts_table(counts: Mapping[str, object]) -> str:
    if not counts:
        return f"<p class=\"missing\">{NOT_RECORDED}</p>"
    return "<table class=\"counts\"><tbody>" + "".join(
        f"<tr><th>{_escape(key)}</th><td>{_escape(value)}</td></tr>"
        for key, value in counts.items()
    ) + "</tbody></table>"


def _endpoint(settings: Mapping[str, object]) -> object:
    """The recorded endpoint with any ``user:password@`` userinfo removed.

    :attr:`~mario_jev.config.Settings.redacted` already strips the userinfo before
    the header is written, so a journal this package produced carries nothing to
    strip. This redacts again anyway, because a reader is handed a *file* and not
    a guarantee: a header written by an older build, or edited by hand, must not
    be able to put a password on the page. Same function, so the two can never
    disagree about what a printed endpoint looks like.
    """
    recorded = settings.get("jev_endpoint")
    if not isinstance(recorded, str):
        return recorded
    return redact_url_userinfo(recorded)


def _run_facts(journal: Journal) -> str:
    run = journal.run
    settings = _mapping(run.get("settings"))
    credential = settings.get("jev_available")
    pairs = [
        ("stage", run.get("stage")),
        ("seed", run.get("seed")),
        ("started_at", run.get("started_at")),
        ("model", settings.get("jev_model")),
        ("endpoint", _endpoint(settings)),
        ("credential configured", "configured" if credential else "not configured"),
        ("advisor enabled", "enabled" if settings.get("advisor_enabled") else "disabled"),
        ("artifacts_dir", run.get("artifacts_dir")),
    ]
    return (
        "<section><h2>Run facts</h2>"
        f"<table class=\"facts\"><tbody>{_rows(pairs)}</tbody></table></section>"
    )


def _summary_section(journal: Journal, derived: Mapping[str, object]) -> str:
    # ``termination`` leads: a run that stopped at 805 of 900 ticks is read from
    # the top of this table, and the tick count alone cannot say whether that was
    # a death, the flag, or a hand on the keyboard. The counts stay where they
    # were; the reason is added in front of them, not instead of them.
    derived_pairs = [
        ("ticks counted", derived["ticks"]),
        ("decisions (jev_called true)", derived["decisions"]),
        ("fallbacks (source fallback_*)", derived["fallbacks"]),
        ("fallback_rate", f"{float(derived['fallback_rate']):.4f}"),
    ]
    staleness = _staleness(journal)
    if staleness is None:
        derived_pairs.append(("answers applied (per tick)", NOT_RECORDED))
    else:
        derived_pairs += [
            ("answers applied (per tick)", staleness["applied"]),
            ("applied from an earlier tick", staleness["stale"]),
            ("oldest applied answer (ticks)", staleness["oldest"]),
            ("answers discarded as stale", staleness["discarded"]),
        ]
    if journal.summary is None:
        # No trailer yet, so there are no recorded counts to print. The words are
        # "in progress" and not "not recorded": nothing was lost, the run simply
        # has not finished. The derived block below still stands, because it is
        # counted from the ticks that are really on disk.
        recorded = (
            "<h3>Recorded</h3>"
            f'<p class="missing">{IN_PROGRESS}: this run has not written its '
            "summary trailer yet</p>"
        )
    else:
        summary = _mapping(journal.summary)
        rows = [
            ("termination", summary.get("termination", NOT_RECORDED)),
            ("ticks", summary.get("ticks")),
            ("frames", summary.get("frames")),
            ("decisions (calls fired)", summary.get("decisions")),
            # Beside ``decisions``, never instead of it: one counts the calls this run
            # fired and the other the answers it acted on, and with the call off the
            # tick they are different numbers.
            ("answers_applied", summary.get("answers_applied", NOT_RECORDED)),
            ("decision_rate_hz", summary.get("decision_rate_hz")),
            ("elapsed_s", summary.get("elapsed_s")),
        ]
        recorded = (
            "<h3>Recorded</h3>"
            f"<table class=\"facts\"><tbody>{_rows(rows)}</tbody></table>"
            "<h3>Recorded sources</h3>"
            f"{_counts_table(_mapping(summary.get('sources')))}"
            "<h3>Recorded degradations</h3>"
            f"{_counts_table(_mapping(summary.get('degradations')))}"
        )
    return (
        "<section><h2>Summary</h2>"
        f"{recorded}"
        "<h3>Derived from the recorded sources and jev_called</h3>"
        f"<table class=\"facts\"><tbody>{_rows(derived_pairs)}</tbody></table>"
        "</section>"
    )


def _charts_section(journal: Journal) -> str:
    confidence = _series(journal, _confidence_point)
    latency = _series(journal, _latency_point)
    latency_max = max((y for _, y in latency if y is not None), default=0.0)
    latency_ceiling = latency_max * 1.1 if latency_max > 0.0 else 1.0
    return (
        "<section><h2>Confidence and Jev latency over time</h2>"
        f"{_svg_chart(chart_id='confidence', title='confidence over frame index', series=confidence, y_min=0.0, y_max=1.0, y_format='{:.2f}')}"
        f"{_svg_chart(chart_id='latency', title='Jev latency (ms) over frame index', series=latency, y_min=0.0, y_max=latency_ceiling, y_format='{:.1f}')}"
        "<p class=\"note\">A tick that observed no Jev call is a gap in the series, "
        "never a point at zero. The latency plotted is the call's own, on the tick "
        "that observed it finish.</p>"
        "</section>"
    )


def _advisor_html(advisor: object) -> str:
    block = _mapping(advisor)
    if not block:
        return f"<p>advisor: {NOT_RECORDED}</p>"
    if not block.get("available"):
        measured = "measured" if block.get("measured") else "not measured"
        return (
            f"<p>advisor: no advice ({_escape(block.get('reason'))}, "
            f"{measured})</p>"
        )
    top = _mapping(block.get("top"))
    moves = block.get("moves")
    move_rows = ""
    if isinstance(moves, Sequence) and not isinstance(moves, (str, bytes)):
        move_rows = "".join(
            f"<tr><th>{_escape(_mapping(move).get('name'))}</th>"
            f"<td>{_escape(_mapping(move).get('probability'))}</td></tr>"
            for move in moves
        )
    return (
        "<p>advisor: available</p>"
        f"<p>advisor top: {_escape(top.get('name'))} "
        f"({_escape(top.get('probability'))})</p>"
        f"<table class=\"scores\"><tbody>{move_rows}</tbody></table>"
    )


def _scores_html(tick: Mapping[str, object], chosen: str | None, probabilities: Mapping[str, object]) -> str:
    if not probabilities:
        if _answered(tick):
            return f"<p class=\"missing\">{NOT_RECORDED}</p>"
        return f"<p class=\"missing\">{NO_CALL}: no Jev response was recorded</p>"
    rows = []
    for name, probability in probabilities.items():
        marker = " (chosen)" if chosen is not None and str(name) == chosen else ""
        css = ' class="chosen"' if marker else ""
        rows.append(
            f"<tr{css}><th>{_escape(name)}{_escape(marker)}</th>"
            f"<td>{_escape(probability)}</td></tr>"
        )
    return "<table class=\"scores\"><tbody>" + "".join(rows) + "</tbody></table>"


def _tick_table(journal: Journal) -> str:
    header = (
        "<tr><th>tick</th><th>frame</th><th>emulator tick</th><th>action</th>"
        "<th>source</th><th>answer age (ticks)</th><th>confidence</th><th>risk</th>"
        "<th>jev called</th><th>jev latency (ms)</th><th>jev attempts</th></tr>"
    )
    rows = []
    for tick in journal.ticks:
        decision = _decision(tick)
        rows.append(
            "<tr>"
            f"<td>{_tick_number(tick)}</td>"
            f"<td>{_frame_index(tick)}</td>"
            f"<td>{_as_int(tick.get('emulator_tick'))}</td>"
            f"<td>{_escape(decision.get('action'))}</td>"
            f"<td>{_escape(_source_text(decision))}</td>"
            # Beside the source, because ``jev_stale`` says an answer was old and
            # only this column says how old, and because a discarded answer shows
            # up here rather than vanishing into an unread degradations list.
            f"<td>{_escape(_answer_age_text(tick))}</td>"
            f"<td>{_escape(_confidence_text(tick))}</td>"
            f"<td>{_escape(_risk_text(tick))}</td>"
            f"<td>{_escape('yes' if decision.get('jev_called') else 'no')}</td>"
            # Next to the latency, because the latency alone cannot explain a tick
            # that spent 21 s to record a 577 ms answer.
            f"<td>{_escape(_jev_latency_text(tick))}</td>"
            f"<td>{_escape(_attempts_text(tick))}</td>"
            "</tr>"
        )
    if not rows:
        return "<section><h2>Ticks</h2><p class=\"missing\">no ticks were recorded</p></section>"
    return (
        "<section><h2>Ticks</h2>"
        f"<table class=\"ticks\"><thead>{header}</thead><tbody>{''.join(rows)}</tbody></table>"
        "</section>"
    )


def _tick_sections(journal: Journal, base: Path | None) -> str:
    blocks = []
    for tick in journal.ticks:
        number = _tick_number(tick)
        decision = _decision(tick)
        chosen, probabilities = _best_action(tick.get("response"))
        pairs = [
            ("action", decision.get("action")),
            ("source", _source_text(decision)),
            ("answer age (ticks)", _answer_age_text(tick)),
            ("confidence", _confidence_text(tick)),
            ("risk", _risk_text(tick)),
            ("jev called", "yes" if decision.get("jev_called") else "no"),
            ("jev latency (ms)", _jev_latency_text(tick)),
            ("jev attempts", _attempts_text(tick)),
            ("reason", decision.get("reason")),
        ]
        degradations = _degradations(tick)
        if degradations:
            pairs.append(("degradations", ", ".join(degradations)))
        blocks.append(
            f"<section class=\"tick\" id=\"tick-{number}\">"
            f"<h3>Tick {number} (frame {_frame_index(tick)})</h3>"
            f"<table class=\"facts\"><tbody>{_rows(pairs)}</tbody></table>"
            "<h4>Jev decision scores (best_action)</h4>"
            f"{_scores_html(tick, chosen, probabilities)}"
            "<h4>Advisor</h4>"
            f"{_advisor_html(tick.get('advisor'))}"
            "<h4>Artifacts</h4>"
            "<div class=\"artifacts\">"
            f"{_figure(base, tick.get('frame_png'), f'frame tick {number}')}"
            f"{_figure(base, tick.get('trajectory_png'), f'trajectory tick {number}')}"
            "</div>"
            "</section>"
        )
    return "".join(blocks)


_STYLE = """
body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
       margin: 24px; color: #1c1c1c; background: #ffffff; }
h1 { font-size: 22px; } h2 { font-size: 18px; margin-top: 28px; }
h3 { font-size: 15px; margin-top: 20px; } h4 { font-size: 14px; color: #555; }
table { border-collapse: collapse; margin: 6px 0 12px; }
th, td { border: 1px solid #ddd; padding: 3px 8px; font-size: 13px;
         text-align: left; vertical-align: top; }
table.facts th { background: #f4f4f4; width: 190px; }
table.scores th.chosen, tr.chosen th { background: #fff1b0; }
svg.chart { display: block; margin: 8px 0; background: #fafafa;
            border: 1px solid #eee; }
svg.chart .grid { stroke: #e1e1e1; stroke-width: 1; }
svg.chart .axis { stroke: #969696; stroke-width: 1; }
svg.chart .series { fill: none; stroke: #467fb4; stroke-width: 2; }
svg.chart .point { fill: #c2410c; }
svg.chart text { font-family: inherit; fill: #1c1c1c; font-size: 12px; }
svg.chart .chart-title { font-size: 13px; font-weight: 600; }
svg.chart .axis-label, svg.chart .axis-title, svg.chart .chart-note { fill: #6e6e6e; }
.missing, .note { color: #8a5a00; font-size: 13px; }
.artifacts { display: flex; flex-wrap: wrap; gap: 16px; }
figure.artifact { margin: 0; }
figure.artifact img { max-width: 420px; image-rendering: pixelated;
                      border: 1px solid #ddd; }
""".strip()


def build_html(
    journal: Journal,
    *,
    artifacts_dir: object | None = None,
    title: str = DEFAULT_TITLE,
) -> str:
    """Render the whole report as one self-contained HTML document.

    Pure: the same journal and the same artifacts on disk always produce the same
    bytes. ``artifacts_dir`` overrides the directory recorded in the run header; a
    relative recorded directory is resolved against the process's current
    directory, which is the only ambient input and is explicit in the header.
    """
    base = _resolve_artifacts_dir(journal, artifacts_dir)
    derived = _derived(journal)
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{_escape(title)}</title>",
        f"<style>{_STYLE}</style>",
        "</head>",
        "<body>",
        f"<h1>{_escape(title)}</h1>",
        _run_facts(journal),
        _summary_section(journal, derived),
        _charts_section(journal),
        _tick_table(journal),
        _tick_sections(journal, base),
        "</body>",
        "</html>",
    ]
    return "\n".join(parts)


def write_html(
    path: object,
    journal: Journal,
    *,
    artifacts_dir: object | None = None,
) -> Path:
    """Write :func:`build_html` to ``path`` and return the path it wrote."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        build_html(journal, artifacts_dir=artifacts_dir),
        encoding="utf-8",
    )
    return target


# -- the terminal report -----------------------------------------------------------


def render_text(journal: Journal) -> str:
    """The same facts as the dashboard, as plain text for the terminal.

    A tick that made no Jev call prints ``no call`` for its confidence and its
    Jev latency; the Jev latency statistics cover only the calls this run actually
    observed, and say so when there were none. Every latency here is the *call's*,
    taken from the applied answer or from ``jev_call_ms`` — never from
    ``latencies_ms["jev"]``, which since the call left the tick measures only the
    submit and the take.

    Firing a call and applying an answer are printed as the two different facts
    they are: ``decisions`` counts the calls this run fired, ``answers_applied``
    the answers it acted on, and the block between them says how many of those
    answers came from an earlier tick and how many were thrown away for being too
    old. A journal written before those fields existed says ``not recorded`` there
    rather than zero.

    The retry line is printed only when some tick's answer cost more than one
    transport attempt. Read with the header's ``jev_timeout_s``, it is what turns
    a tick that spent 21 s to record a 577 ms answer into two attempts against a
    20 s deadline rather than unaccounted time.

    A journal without its trailer -- a run still in progress -- prints
    ``in progress`` where the summary counts would be, and never ``not recorded``:
    those counts are not lost, they do not exist yet. Everything derived from the
    ticks really on disk is printed unchanged.
    """
    run = journal.run
    settings = _mapping(run.get("settings"))
    summary = _mapping(journal.summary)
    derived = _derived(journal)

    lines: list[str] = [DEFAULT_TITLE, "", "run:"]
    lines.append(f"  stage: {run.get('stage')}")
    lines.append(f"  seed: {run.get('seed')}")
    lines.append(f"  started_at: {run.get('started_at')}")
    lines.append(f"  model: {settings.get('jev_model')}")
    lines.append(f"  endpoint: {_endpoint(settings)}")
    credential = "configured" if settings.get("jev_available") else "not configured"
    lines.append(f"  credential configured: {credential}")
    recorded_dir = run.get("artifacts_dir")
    lines.append(f"  artifacts_dir: {recorded_dir if recorded_dir else NOT_RECORDED}")

    lines += ["", "summary (recorded):"]
    if journal.summary is None:
        # A live journal has not written its trailer yet, so there are no recorded
        # counts. The words are "in progress", never "not recorded": nothing was
        # lost, the run has not finished.
        lines.append(f"  {IN_PROGRESS}: this run has not written its summary trailer yet")
    else:
        for key in (
            "termination",
            "ticks",
            "frames",
            "decisions",
            "answers_applied",
            "decision_rate_hz",
            "elapsed_s",
        ):
            value = summary.get(key)
            lines.append(f"  {key}: {value if value is not None else NOT_RECORDED}")
        sources = _mapping(summary.get("sources"))
        lines.append("  sources:")
        if sources:
            for key in sorted(sources):
                lines.append(f"    {key}: {sources[key]}")
        else:
            lines.append(f"    {NOT_RECORDED}")
        degradations = _mapping(summary.get("degradations"))
        lines.append("  degradations:")
        if degradations:
            for key in sorted(degradations):
                lines.append(f"    {key}: {degradations[key]}")
        else:
            lines.append("    none recorded")

    lines += ["", "derived from recorded sources and jev_called:"]
    lines.append(f"  ticks counted: {derived['ticks']}")
    lines.append(f"  decisions (jev_called true): {derived['decisions']}")
    lines.append(f"  fallbacks (source fallback_*): {derived['fallbacks']}")
    lines.append(f"  fallback_rate: {float(derived['fallback_rate']):.4f}")

    # Fired and applied are not the same fact any more, and a reader who only saw
    # ``decisions`` would read a run that asked ten times and acted seven as one
    # that decided ten times.
    staleness = _staleness(journal)
    lines += ["", "jev calls, fired versus applied:"]
    lines.append(f"  calls fired (jev_called true): {derived['decisions']}")
    if staleness is None:
        lines.append(
            f"  answers applied: {NOT_RECORDED} (this journal carries no "
            "answer_age_ticks; it predates the staleness fields)"
        )
    else:
        lines.append(f"  answers applied: {staleness['applied']}")
        lines.append(f"  applied from an earlier tick: {staleness['stale']}")
        lines.append(f"  oldest applied answer age (ticks): {staleness['oldest']}")
        lines.append(f"  answers discarded as stale: {staleness['discarded']}")

    lines += ["", "ticks:"]
    if journal.ticks:
        for tick in journal.ticks:
            decision = _decision(tick)
            lines.append(
                f"  tick {_tick_number(tick)} frame {_frame_index(tick)} "
                f"emulator_tick {_as_int(tick.get('emulator_tick'))} "
                f"action={decision.get('action')} "
                f"source={_source_text(decision)} "
                f"answer_age={_answer_age_text(tick)} "
                f"confidence={_confidence_text(tick)} "
                f"risk={_risk_text(tick)} "
                f"jev_called={'true' if decision.get('jev_called') else 'false'} "
                f"jev_latency_ms={_jev_latency_text(tick)}"
            )
    else:
        lines.append("  none recorded")

    retried = _retried_ticks(journal)
    if retried:
        # Printed only when there was one. A "0" here would be a number a reader
        # has to interpret; silence is the same fact without the ambiguity. The
        # denominator is the recorded answers, which is the set the numerator was
        # counted over -- ``decisions`` counts fired calls, and a ratio of one set
        # over another would be a number that means nothing.
        answers = sum(1 for tick in journal.ticks if tick.get("response") is not None)
        lines.append("")
        lines.append(
            f"retried jev calls: {retried} of {answers} recorded answers "
            "reported more than one transport attempt"
        )

    stats = _latency_stats(journal)
    lines.append("")
    if stats is None:
        lines.append(
            "jev latency over observed calls: no call was observed; not recorded"
        )
    else:
        low, median, high = stats
        lines.append(
            "jev latency over observed calls: "
            f"min={low:.3f} median={median:.3f} max={high:.3f}"
        )
    return "\n".join(lines)
