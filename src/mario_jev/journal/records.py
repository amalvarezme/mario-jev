"""The evidence journal's fixed schema, expressed as pure serializers.

The journal is the artifact a human reads to decide whether to believe the
agent, so its shape is pinned here rather than improvised at each call site.
Every value these functions return is plain JSON data: ``dict``, ``list``,
``str``, ``int``, ``float``, ``bool`` and ``None``. That is what makes
``read(write(x)) == x`` hold, and it is why ``degradations`` is a list and not
the ``tuple`` the policy carries.

Nothing here reads the clock, the filesystem or the network. The header
timestamp is *injected* by the caller, so a test can pin it and never sleep.

A note on the ``advisor`` block
-------------------------------

The policy records an absent advisor as ``None`` inside the tick projection, and
a ``None`` advice alone cannot say *why* no advice arrived. The tick line
therefore carries a second, typed block at the top level with one of exactly
three shapes:

* ``{"available": true, ...}`` with the advisor's own vocabulary, when advice
  reached the tick;
* ``{"available": false, "reason": <measured>, "measured": true}`` when the
  advisor was enabled and recorded its own absence — the reason is one of the
  advisor's constants, read from the absence it measured;
* ``{"available": false, "reason": "disabled", "measured": false}`` when the
  advisor was disabled, so nothing was ever measured.

A reader never has to guess whether a second opinion is present, and a run that
had the advisor enabled never gets told it was disabled.

``available`` is the observable fact, and it is what separates "an answer" from
"an empty one" (``{"available": true, "moves": []}``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from mario_jev.advisor import REASON_DISABLED, AdvisorAbsence, AdvisorAdvice
from mario_jev.config import Settings
from mario_jev.policy import RunSummary, TickRecord, advisor_view

#: The run header. Exactly one per journal, always the first line.
KIND_RUN = "run"
#: One decision tick. Zero or more per journal, never after the trailer.
KIND_TICK = "tick"
#: The summary trailer. Exactly one per journal, always the last line.
KIND_SUMMARY = "summary"
#: The closed set of line kinds. A kind outside it is a bug, not a new state.
KINDS: frozenset[str] = frozenset({KIND_RUN, KIND_TICK, KIND_SUMMARY})


class JournalFormatError(Exception):
    """A journal line that cannot be read as the fixed schema.

    ``line`` is the 1-based line number the reader was on when it gave up, so a
    malformed file names its own location instead of failing anonymously. For a
    journal that ends without its trailer, the number is the last line present:
    that is where the reader discovered the file had ended.
    """

    def __init__(self, line: int, reason: str) -> None:
        self.line = int(line)
        self.reason = str(reason)
        super().__init__(f"journal line {self.line}: {self.reason}")


@dataclass(frozen=True, slots=True)
class Journal:
    """A whole, validated journal: one header, its ticks, one trailer."""

    run: dict[str, object]
    ticks: tuple[dict[str, object], ...]
    summary: dict[str, object]


def _utc_isoformat(moment: datetime) -> str:
    """An aware timestamp rendered as UTC ISO-8601.

    A naive timestamp is refused rather than assumed: guessing a timezone would
    put a claim in the journal that nobody observed.
    """
    if not isinstance(moment, datetime):
        raise TypeError(f"started_at must be a datetime, got {type(moment).__name__}")
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("started_at must be timezone-aware; a naive timestamp is ambiguous")
    return moment.astimezone(timezone.utc).isoformat()


def _source_value(source: object) -> str:
    """The plain string value of a decision source, enum or bare string."""
    value = getattr(source, "value", None)
    return str(value) if value is not None else str(source)


def run_record(
    settings: Settings,
    *,
    started_at: datetime,
    artifacts_dir: object | None = None,
) -> dict[str, object]:
    """The journal header for one run.

    ``settings`` is recorded through :attr:`Settings.redacted`, never the raw
    object, so the credential cannot ride along inside the header. The key
    ``artifacts_dir`` is present only when artifacts were enabled for the run.
    """
    record: dict[str, object] = {
        "kind": KIND_RUN,
        "started_at": _utc_isoformat(started_at),
        "stage": str(settings.stage),
        "seed": int(settings.seed),
        "settings": dict(settings.redacted),
    }
    if artifacts_dir is not None:
        record["artifacts_dir"] = str(artifacts_dir)
    return record


def advisor_record(
    advice: AdvisorAdvice | None,
    absence: AdvisorAbsence | None = None,
) -> dict[str, object]:
    """The typed advisor block for one tick, in exactly three shapes.

    ``available`` is the observable fact that separates "no advice" from "an
    answer"; the answer itself reuses the policy's own projection verbatim. An
    advice whose distribution is empty is still ``available``: the advisor *was*
    asked, and the empty ``moves`` list says it returned nothing.

    When there is no advice, ``absence`` is the advisor's *measured* reason for
    being absent — the reason it recorded itself, from its own closed set — and
    it is reported as ``measured: true``. With no advice and no measured absence,
    the advisor was disabled, and the block says so with ``measured: false``
    rather than inventing a reason nobody observed.
    """
    if advice is not None:
        return {"available": True, **(advisor_view(advice) or {})}
    if absence is not None:
        return {"available": False, "reason": absence.reason, "measured": True}
    return {"available": False, "reason": REASON_DISABLED, "measured": False}


def tick_record(
    record: TickRecord,
    *,
    frame_png: object | None = None,
    trajectory_png: object | None = None,
) -> dict[str, object]:
    """One tick line, in the fixed schema.

    ``state`` is exactly :attr:`TickRecord.journal_state`, ``rollout`` is exactly
    :meth:`RolloutTable.as_dict`, and ``response`` is exactly
    :meth:`JevResponse.as_dict`: the journal records what those produced, it does
    not re-render it. ``frame_png`` and ``trajectory_png`` are relative artifact
    paths, or ``None`` when no artifact was written for this tick.

    ``terminated`` and ``truncated`` are the post-step facts the record carries.
    They are the only fields on the line that describe the world *after* the
    tick's action ran, which is what lets a reader find the tick that ended the
    run instead of inferring it from the line count.

    ``questions`` is what this tick **fired** and ``response`` what it
    **applied**, and they need not describe the same call. The three keys between
    them close that gap: ``asked_at_tick`` is the tick that fired the applied
    answer, ``answer_age_ticks`` is ``tick - asked_at_tick``, and ``jev_call_ms``
    is what the *call* cost, on the tick that observed it finish -- applied,
    discarded or failed. All three are ``null`` where there is nothing to report,
    and ``jev_call_ms`` is deliberately **not** ``latencies_ms["jev"]``: that one
    is the few microseconds this tick spent submitting and taking, and folding a
    21 s call into it would break the arithmetic a reader does over the phases.

    The keys are, in order: ``kind``, ``tick``, ``frame_index``,
    ``emulator_tick``, ``state``, ``rollout``, ``questions``, ``response``,
    ``asked_at_tick``, ``answer_age_ticks``, ``jev_call_ms``, ``decision``,
    ``advisor``, ``latencies_ms``, ``degradations``, ``terminated``,
    ``truncated``, ``frame_png`` and ``trajectory_png``.
    """
    decision = record.decision
    return {
        "kind": KIND_TICK,
        "tick": int(record.tick),
        "frame_index": int(record.frame_index),
        "emulator_tick": int(record.emulator_tick),
        "state": record.journal_state,
        "rollout": record.rollout.as_dict(),
        "questions": record.jev_questions,
        "response": record.jev_response,
        "asked_at_tick": (
            None if record.asked_at_tick is None else int(record.asked_at_tick)
        ),
        "answer_age_ticks": (
            None if record.answer_age_ticks is None else int(record.answer_age_ticks)
        ),
        "jev_call_ms": (
            None if record.jev_call_ms is None else float(record.jev_call_ms)
        ),
        "decision": {
            "action": str(decision.action),
            "source": _source_value(decision.source),
            "confidence": None if decision.confidence is None else float(decision.confidence),
            "risk": None if decision.risk is None else float(decision.risk),
            "reason": str(decision.reason),
            "jev_called": bool(decision.jev_called),
        },
        "advisor": advisor_record(record.advisor, record.advisor_absence),
        "latencies_ms": {
            str(name): float(value) for name, value in record.latencies_ms.items()
        },
        "degradations": [str(item) for item in record.degradations],
        "terminated": bool(record.terminated),
        "truncated": bool(record.truncated),
        "frame_png": None if frame_png is None else str(frame_png),
        "trajectory_png": None if trajectory_png is None else str(trajectory_png),
    }


def summary_record(summary: RunSummary) -> dict[str, object]:
    """The summary trailer for one run.

    ``sources`` and ``degradations`` are plain objects keyed by string names, so
    the counts are readable without importing the enum that produced them.
    ``termination`` is the run's own classified reason for stopping, one member
    of :data:`~mario_jev.policy.TERMINATIONS`, written verbatim: the summary was
    refused at construction if it were anything else.

    ``decisions`` counts the ticks that **fired** a call and ``answers_applied``
    the ticks that **applied** an answer. They are different numbers once the call
    is off the tick, so both are written: a run can fire ten calls and act on
    seven of them, and no arithmetic over the other keys can recover that.

    The keys are, in order: ``kind``, ``ticks``, ``frames``, ``decisions``,
    ``answers_applied``, ``sources``, ``degradations``, ``elapsed_s``,
    ``decision_rate_hz`` and ``termination``.
    """
    return {
        "kind": KIND_SUMMARY,
        "ticks": int(summary.ticks),
        "frames": int(summary.frames),
        "decisions": int(summary.decisions),
        "answers_applied": int(summary.answers_applied),
        "sources": {
            _source_value(source): int(count) for source, count in summary.sources.items()
        },
        "degradations": {
            str(name): int(count) for name, count in summary.degradations.items()
        },
        "elapsed_s": float(summary.elapsed_s),
        "decision_rate_hz": float(summary.decision_rate_hz),
        "termination": str(summary.termination),
    }


__all__ = [
    "KINDS",
    "KIND_RUN",
    "KIND_SUMMARY",
    "KIND_TICK",
    "Journal",
    "JournalFormatError",
    "advisor_record",
    "run_record",
    "summary_record",
    "tick_record",
]
