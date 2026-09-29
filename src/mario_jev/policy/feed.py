"""The recorded-answer feed: replay a journal's Jev answers with no socket and no thread.

Why this exists
---------------

A live run has two sources of unreproducibility. One is the horizon's cost model,
which ``Settings.rollout_cost_source`` removes. The other is *when* an answer is
applied: :meth:`~mario_jev.policy.dispatch.JevDispatcher.take` never blocks, so
the tick that takes an answer is decided by the worker thread rather than by the
state. Two runs fed the same answers at different wall-clock moments therefore
execute different actions, and one of them may even discard an answer as
``jev_stale``.

A journal already carries everything needed to remove that source. Per tick it
records the response that was applied (``response``), how many ticks old that
answer was (``answer_age_ticks``) and what the call cost (``jev_call_ms``).
:class:`RecordedAnswerFeed` hands those recorded answers back through the same
four members the loop uses on the dispatcher -- :attr:`in_flight`,
:meth:`request`, :meth:`take` and :meth:`close` -- in the order the calls were
fired, each after its recorded delay in ticks, and only to a build that is asking
the question the record answers.

Why a fed run cannot exhibit the dispatcher race
------------------------------------------------

There is no worker and no future here. :meth:`RecordedAnswerFeed.request` records
the firing tick synchronously and :meth:`RecordedAnswerFeed.take` returns the
answer once the recorded number of ticks has elapsed. Nothing a fed run does
depends on thread scheduling, so the dispatcher race is not *removed* -- it is
*absent*. That is exactly why this module must never be used to claim the live
race is fixed: the live path is still
:class:`~mario_jev.policy.dispatch.JevDispatcher` with its one worker, and a fed
run proves nothing about it. The feed is a measuring instrument for comparing
builds, not a repair of the concurrent path.

This module deliberately constructs no :class:`~mario_jev.jev.JevClient`, opens no
socket and starts no thread. It reads recorded responses and returns
:class:`~mario_jev.policy.dispatch.Taken` values.

Positional, not temporal
------------------------

The feed is keyed by **call order**: the k-th fired call receives the k-th
recorded response. A build that fires on slightly different ticks therefore still
receives the transcript's answers in order, which is what makes an A/B comparison
possible. The delay, though, is absolute: the k-th answer lands
``answer_age_ticks`` ticks after *this* run fired its k-th call. A build whose
cadence differs can therefore see a recorded delay -- and the loop's staleness
bound -- differently, and the loop's own rules decide what to do with it.

What a transcript does not hold
-------------------------------

The transcript is built from the tick lines that **applied** an answer, because
only those carry a ``response``. A call that failed transiently, or whose answer
the recording discarded as ``jev_stale``, left a ``jev_call_ms`` and a
degradation but no response body -- there is nothing to replay. Such a call has
no slot in the transcript, so the k-th recorded response is the k-th answer the
recording actually applied, not the k-th call it fired. That is the positional
model's own limitation, and it is why a transcript is a fixture and a stale one
is a trap: the count of fired calls in the replaying build has to be read next to
the count of applied answers, and a divergence is a fact the harness must state
rather than quietly absorb.

What a fed run's numbers are not
--------------------------------

``jev_call_ms`` on a fed tick is the *recorded* latency, not a measurement of this
run. The loop stamps every tick of a fed run with
:data:`~mario_jev.policy.loop.TRANSCRIPT_ANSWERS` so no line can be read as a live
measurement of the replaying machine.

Only the question it answers
----------------------------

A recorded answer is evidence about one question, and the feed hands it over only
to a build that is asking that question. Before a call is fired, the feed reads the
record the way the live client reads a reply -- the same parser, the same expected
``type`` per asked id, the same choice criteria -- and refuses the call when the
record cannot answer what this build asked: an asked id the record does not answer,
an answer of a different ``type``, a chosen candidate the question never offered,
or a scored legend naming a level the question never asked about. A record that
answers *extra* questions the build did not ask is **not** a mismatch: it answers
the same question whoever asked it.

Refusing is the point. Delivering on a mismatch would journal this build's
``questions`` beside a ``response`` that answered someone else's -- evidence whose
origin is not established, presented as this build's own reasoning -- and an A/B
comparison built on it would look valid while comparing nothing. A refused call
consumes nothing, so a later tick that asks the recorded question can still use the
record; it sets :attr:`RecordedAnswerFeed.mismatch`, which the loop records as
:data:`~mario_jev.policy.loop.TRANSCRIPT_MISMATCH`. Running out stays
:data:`~mario_jev.policy.loop.ANSWERS_EXHAUSTED`, so neither signal can be read as
the other.

Running out, and answering someone else's question
--------------------------------------------------

Two different things can stop the feed from answering, and the journal keeps them
apart. *Running out* is :data:`~mario_jev.policy.loop.ANSWERS_EXHAUSTED`: the run
fired more calls than the transcript holds. *A mismatch* is
:data:`~mario_jev.policy.loop.TRANSCRIPT_MISMATCH`: the record cannot answer the
question this build asked. A reader must never have to guess which one a fed run
hit, so they are named separately, recorded separately and counted separately.
If the run fires **fewer** calls, the unused tail is simply never asked for and
the feed reports nothing: an answer nobody requested is not a fact about the run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from mario_jev.jev import (
    JevProtocolError,
    JevResponse,
    ScoreAnswer,
    parse_response,
)
from mario_jev.state import MarioState

from .dispatch import Taken

#: What :attr:`RecordedAnswerFeed.source_kind` reports. The loop treats any
#: ``source_kind`` other than ``"live"`` as recorded and marks the run.
TRANSCRIPT_SOURCE = "transcript"

__all__ = [
    "TRANSCRIPT_SOURCE",
    "RecordedAnswer",
    "RecordedAnswerFeed",
    "recorded_answers_from_ticks",
]


@dataclass(frozen=True, slots=True)
class RecordedAnswer:
    """One response read off a journal tick line, with its recorded cost and age.

    ``response`` is exactly :meth:`~mario_jev.jev.JevResponse.as_dict` as the
    recording wrote it. ``delay_ticks`` is the recording's ``answer_age_ticks`` --
    how many ticks after it was fired the answer was applied -- and ``call_ms`` is
    its ``jev_call_ms``. A line that carries a response always carries both
    numbers, because the loop sets them together; :func:`recorded_answers_from_ticks`
    refuses a line that has one without the others rather than defaulting them.
    """

    response: Mapping[str, object]
    delay_ticks: int
    call_ms: float


@dataclass(frozen=True, slots=True)
class _Pending:
    """A fired call waiting for its recorded delay to elapse."""

    response: JevResponse
    call_ms: float
    asked_at_tick: int
    deliver_at_tick: int


def _record_requested(record: Mapping[str, object]) -> dict[str, str]:
    """The question ids and answer ``type``s a recorded body carries.

    Read on the record's own terms -- each answer names its own ``type`` -- so a
    transcript can be validated before any build has asked it anything.
    """
    answers = record.get("answers")
    if not isinstance(answers, Mapping):
        raise ValueError("a recorded response must carry an 'answers' object")
    requested: dict[str, str] = {}
    for question_id, answer in answers.items():
        if not isinstance(answer, Mapping):
            raise ValueError(
                f"recorded answer {question_id!r} must be an object, "
                f"got {type(answer).__name__}"
            )
        answer_type = answer.get("type")
        if not isinstance(answer_type, str):
            raise ValueError(
                f"recorded answer {question_id!r} must carry a string 'type'"
            )
        requested[str(question_id)] = answer_type
    return requested


def _record_accounting(record: Mapping[str, object]) -> tuple[float, int]:
    """The record's own ``latency_ms`` and ``attempts``, refused unless well formed."""
    latency_ms = record.get("latency_ms")
    attempts = record.get("attempts")
    if not isinstance(latency_ms, (int, float)) or isinstance(latency_ms, bool):
        raise ValueError("a recorded response must carry a numeric 'latency_ms'")
    if not isinstance(attempts, int) or isinstance(attempts, bool):
        raise ValueError("a recorded response must carry an integer 'attempts'")
    return float(latency_ms), attempts


def _response_from_record(record: Mapping[str, object]) -> JevResponse:
    """Rebuild the typed :class:`JevResponse` a journal line recorded.

    The recorded body is :meth:`~mario_jev.jev.JevResponse.as_dict`, which is
    shaped enough like the parsed reply that the committed parser accepts it:
    ``model``, ``answers`` and ``usage`` are read from it, while ``latency_ms``
    and ``attempts`` are passed explicitly. This reads the record **on its own
    terms** -- the id -> type map comes from the answers themselves -- because at
    this point no build has asked anything, and its only job is to refuse a
    malformed transcript before the run starts. Reusing
    :func:`~mario_jev.jev.parse_response` means the record is validated by the
    same code that produced it, not by a second parser that could drift from it.
    The parse that binds the record to the questions a build is actually asking
    is :func:`_bind_to_questions`, and it happens per call, in ``request``.
    """
    latency_ms, attempts = _record_accounting(record)
    return parse_response(
        record,
        requested=_record_requested(record),
        latency_ms=latency_ms,
        attempts=attempts,
    )


def _asked_expectations(
    questions: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """How the live client reads a question set, mirrored exactly.

    ``JevClient._parse`` turns a question set into an id -> ``type`` map plus, for
    every ``choice`` question keyed by criteria, the candidate names it offered.
    The feed asks the same of a record, so the two paths agree by construction
    about what a question expects.

    Raises:
        JevProtocolError: a question is not an object, or does not carry a string
            ``type``. Then nothing can be bound to it, and the call is refused.
    """
    requested: dict[str, str] = {}
    allowed_choices: dict[str, list[str]] = {}
    for question_id, question in questions.items():
        if not isinstance(question, Mapping):
            raise JevProtocolError(
                f"question {question_id!r} must be an object, "
                f"got {type(question).__name__}"
            )
        question_type = question.get("type")
        if not isinstance(question_type, str):
            raise JevProtocolError(
                f"question {question_id!r} must carry a string 'type'"
            )
        requested[str(question_id)] = question_type
        if question_type != "choice":
            continue
        criteria = question.get("criteria")
        if isinstance(criteria, Mapping):
            allowed_choices[str(question_id)] = list(criteria)
    return requested, allowed_choices


def _asked_levels(question: Mapping[str, object]) -> set[str] | None:
    """The level names a ``score`` question offers, or ``None`` if it offers none.

    ``build_questions`` writes a score scale as a list of level names, and a
    recorded answer's ``legend`` names the levels it used. Only a list or tuple of
    strings can be compared, so anything else is left to the ``type`` check rather
    than guessed at.
    """
    criteria = question.get("criteria")
    if not isinstance(criteria, (list, tuple)):
        return None
    names = {item for item in criteria if isinstance(item, str)}
    return names or None


def _check_scored_levels(
    response: JevResponse, questions: Mapping[str, Mapping[str, object]]
) -> None:
    """Refuse a ``score`` answer whose legend names levels the build did not ask about.

    The live parser checks a ``choice`` against the offered candidates but says
    nothing about a ``score``'s legend, so the containment is checked here. It is a
    containment and not an equality: a recording's legend may cover fewer levels
    than this build offers, and only a level that was neither offered nor recorded
    is a mismatch.

    Raises:
        JevProtocolError: a recorded legend names a level the question never
            offered. The answer is to a different scale and cannot be handed over.
    """
    for question_id, question in questions.items():
        answer = response.answers.get(question_id)
        if not isinstance(answer, ScoreAnswer):
            continue
        offered = _asked_levels(question)
        if offered is None:
            continue
        unknown = sorted(level for level in answer.legend.values() if level not in offered)
        if unknown:
            raise JevProtocolError(
                f"question {question_id!r}: the recorded legend names levels that "
                "were not asked: " + ", ".join(repr(level) for level in unknown)
            )


def _bind_to_questions(
    record: Mapping[str, object],
    questions: Mapping[str, Mapping[str, object]],
    latency_ms: float,
    attempts: int,
) -> JevResponse:
    """Read a recorded body as the answer to ``questions``, or raise.

    The record is read the way the live client reads a reply: the same parser, the
    same expected ``type`` per asked id, the same choice criteria. So the returned
    response answers exactly the ids this build asked -- no more, and never fewer --
    and anything that stops it is a mismatch.

    Raises:
        JevProtocolError: the asked ids and the recorded answers do not line up --
            an asked id the record does not answer, an answer of a different
            ``type``, a chosen candidate the question never offered, a scored
            legend naming an unasked level, or a question that is not readable.
    """
    requested, allowed_choices = _asked_expectations(questions)
    response = parse_response(
        record,
        requested=requested,
        allowed_choices=allowed_choices,
        latency_ms=latency_ms,
        attempts=attempts,
    )
    _check_scored_levels(response, questions)
    return response


def recorded_answers_from_ticks(
    ticks: Sequence[Mapping[str, object]],
) -> tuple[RecordedAnswer, ...]:
    """The transcript a journal's tick lines carry, in order.

    Accepts the ``ticks`` of a journal (``read_journal(path).ticks``) rather than
    the journal object, so this module -- which lives in ``mario_jev.policy`` --
    does not import ``mario_jev.journal`` and reopen the cycle: the journal
    already imports the policy layer.

    A tick line whose ``response`` is ``None`` applied no answer and contributes
    nothing. A line that carries a response but is missing its
    ``answer_age_ticks`` or ``jev_call_ms`` is a malformed transcript and raises
    ``ValueError`` rather than defaulting either number: a fabricated latency is
    worse than a refused transcript.
    """
    answers: list[RecordedAnswer] = []
    for line in ticks:
        response = line.get("response")
        if response is None:
            continue
        if not isinstance(response, Mapping):
            raise ValueError(
                f"a tick line's response must be an object, "
                f"got {type(response).__name__}"
            )
        delay = line.get("answer_age_ticks")
        call_ms = line.get("jev_call_ms")
        if not isinstance(delay, int) or isinstance(delay, bool):
            raise ValueError(
                "a tick line that applied an answer must carry an integer "
                "'answer_age_ticks'"
            )
        if not isinstance(call_ms, (int, float)) or isinstance(call_ms, bool):
            raise ValueError(
                "a tick line that applied an answer must carry a numeric 'jev_call_ms'"
            )
        answers.append(
            RecordedAnswer(
                response=response,
                delay_ticks=delay,
                call_ms=float(call_ms),
            )
        )
    return tuple(answers)


class RecordedAnswerFeed:
    """A journal's recorded answers, served in order; no socket, no client, no thread.

    The loop must call :meth:`take` exactly once per tick, which is what it
    already does on every step. The feed counts those calls to know the current
    tick, because ``take`` deliberately takes no argument -- changing it would
    change the dispatcher's contract, and both sources share one. That count is
    the feed's only notion of time; there is no clock anywhere in this class.
    """

    __slots__ = (
        "_answers",
        "_closed",
        "_exhausted",
        "_mismatched",
        "_next",
        "_pending",
        "_responses",
        "_takes",
    )

    def __init__(self, answers: Sequence[RecordedAnswer]) -> None:
        self._answers = tuple(answers)
        # Parse every recorded body once, now, on its own terms, so a malformed
        # transcript fails at construction rather than mid-run from inside
        # ``take``. The parse that binds a record to the questions a build is
        # asking happens in ``request``, once there is a build to ask them.
        self._responses = tuple(
            _response_from_record(answer.response) for answer in self._answers
        )
        self._next = 0
        self._pending: _Pending | None = None
        # How many ``take`` calls have been made. The loop calls ``take`` once per
        # step, so this is the current tick, counted without a clock.
        self._takes = 0
        self._exhausted = False
        self._mismatched = False
        self._closed = False

    @classmethod
    def from_ticks(
        cls, ticks: Sequence[Mapping[str, object]]
    ) -> "RecordedAnswerFeed":
        """Build a feed from a journal's tick lines, in order."""
        return cls(recorded_answers_from_ticks(ticks))

    @property
    def in_flight(self) -> bool:
        """Whether a fired call is still waiting for its recorded delay."""
        return self._pending is not None

    @property
    def source_kind(self) -> str:
        """``"transcript"``: this source's answers are recorded, not measured."""
        return TRANSCRIPT_SOURCE

    @property
    def exhausted(self) -> bool:
        """Whether a fired call could not be answered because the transcript ran out."""
        return self._exhausted

    @property
    def mismatch(self) -> bool:
        """Whether a fired call was refused because the record answers another question.

        Set when the record cannot answer what this build is asking -- a different
        id, a different ``type``, a candidate the question never offered, or a
        scored level it never named. It is a different fact from
        :attr:`exhausted`, and the loop records it as a different degradation, so
        "the transcript ran out" can never be read as "the transcript answers
        someone else".
        """
        return self._mismatched

    def request(
        self,
        questions: Mapping[str, Mapping[str, object]],
        state: MarioState,
        *,
        tick: int,
    ) -> bool:
        """Fire one call and answer whether it was fired.

        ``state`` is accepted only to match the dispatcher's signature; the feed
        never sends it anywhere, because there is nobody to ask. ``questions`` is
        not decoration: the recorded answer is bound to them before it is handed
        over, and a record that cannot answer them refuses the call.

        ``False`` means the call was refused, and which refusal it was is read off
        the feed afterwards:

        * the feed was closed, or a call was already in flight -- neither flag;
        * the transcript ran out -- :attr:`exhausted` is set;
        * the record answers a different question -- :attr:`mismatch` is set.

        The last two leave the record **unconsumed**, so a later tick that asks the
        recorded question can still use it.
        """
        if self._closed or self._pending is not None:
            return False
        if self._next >= len(self._answers):
            self._exhausted = True
            return False
        index = self._next
        answer = self._answers[index]
        recorded = self._responses[index]
        try:
            response = _bind_to_questions(
                answer.response, questions, recorded.latency_ms, recorded.attempts
            )
        except JevProtocolError:
            # The record answers a different question than the one this build is
            # asking. Handing it over would journal this build's ``questions``
            # beside a ``response`` that answered someone else's, which is exactly
            # the provenance this feed exists to keep. Refuse, keep the record for
            # a tick that does ask the recorded question, and say so.
            self._mismatched = True
            return False
        self._next += 1
        self._pending = _Pending(
            response=response,
            call_ms=float(answer.call_ms),
            asked_at_tick=int(tick),
            deliver_at_tick=int(tick) + answer.delay_ticks,
        )
        return True

    def take(self) -> Taken | None:
        """The answer whose recorded delay has elapsed, or ``None``; never blocks."""
        current_tick = self._takes
        self._takes += 1
        pending = self._pending
        if pending is None or current_tick < pending.deliver_at_tick:
            return None
        self._pending = None
        return Taken(
            response=pending.response,
            error=None,
            asked_at_tick=pending.asked_at_tick,
            call_ms=pending.call_ms,
        )

    def close(self) -> None:
        """Refuse further calls. Idempotent; there is no worker to join."""
        self._closed = True
