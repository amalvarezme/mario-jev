"""The execution loop: one planner, one cadence, and every tick recorded.

One :class:`ExecutionLoop` owns one :class:`~mario_jev.trajectory.TrajectoryPlanner`
for the whole run. That is a requirement rather than an optimisation: the planner
calibrates the machine on its first call and sizes later horizons from what it
measured, so a fresh planner per tick would pay the full calibration every frame.

Each :meth:`ExecutionLoop.step` does the same five things, in order:

1. read the loop's own game-frame counter and the raw ``emulator.tick``;
2. build the typed state from the frame, RAM and ``info``;
3. run the rollout sweep every tick, because the trajectories are local and cheap;
4. decide with :func:`~mario_jev.policy.decision.select_action`, *submitting* a Jev
   call only when the cadence is due, a credential is configured, there are
   candidates and nothing is already in flight, and applying whatever answer has
   finished by now;
5. execute the decided action with the real emulator, then advance the loop's
   game-frame counter by exactly one.

The tick never waits for Jev
----------------------------

Step 4 used to call ``JevClient.ask`` **in series inside the tick**, and the
measured cost was the whole world stopping: 900 frames in 184 s (4.9 game frames
per second), and one tick that spent 21 271.52 ms inside Jev out of a 21 375.84 ms
tick because a single attempt timed out. The call now goes to a
:class:`~mario_jev.policy.dispatch.JevDispatcher`, which submits it to one worker
and returns immediately; the tick fires at most one call, takes whatever has
finished, and executes its frame either way.

That splits one fact into three, so the record carries three:

* this tick **fired** a call -- ``TickDecision.jev_called``, counted by
  :attr:`RunSummary.decisions`;
* this tick **applied** an answer, and how old it was --
  :attr:`TickRecord.asked_at_tick` and :attr:`TickRecord.answer_age_ticks`,
  counted by :attr:`RunSummary.answers_applied`;
* what the **call itself** cost -- :attr:`TickRecord.jev_call_ms`.

An answer older than ``STALE_LIMIT_CADENCES * decision_cadence_frames`` ticks is
discarded and the tick records :data:`JEV_STALE_DISCARDED`; it is not cancelled,
because nothing can cancel it (see :mod:`mario_jev.policy.dispatch`). An applied
answer that was asked on an earlier tick is recorded as
:attr:`~mario_jev.policy.decision.DecisionSource.JEV_STALE` rather
than ``jev``, so "Jev decided this" never silently means "Jev decided this three
ticks ago".

The loop owns the frame index because the planner does not. The rollout sweep steps
the same emulator while enumerating candidates and never rewinds its counter, so
``emulator.tick`` jumps by the whole rollout cost on every tick; keying the cadence
off that counter would fire on an arbitrary subset of ticks rather than every
``decision_cadence_frames`` game frames. ``frame_index`` therefore starts at ``0``
and counts *executed* frames, while the raw ``emulator.tick`` is recorded alongside
it as ``emulator_tick`` so the planner's inflation stays visible.

Timing is **recorded, never repaired**: ``timing_overrun`` is appended when the
rollout's own ``elapsed_ms`` exceeds ``rollout_budget_ms * OVERSHOOT_FACTOR``, and
``JEV_RETRIED`` when the tick's successful response reports more than one
transport attempt. The loop holds one planner across ticks; the limit stays
visible in the journal.

``latencies_ms`` keeps meaning *what this tick spent in each phase*, so the phases
-- everything in the mapping except ``total`` itself -- sum to at most
``latencies_ms["total"]`` on every record. (``total`` is a member of the same
mapping, so a sum over *its* values would be counting it twice; the invariant is
about the phases it bounds.)
``latencies_ms["jev"]`` is therefore the time the tick spent in Jev code --
building questions, submitting, taking -- and **not** what the call cost; the
call's own duration is ``jev_call_ms``. The two are separate on purpose: a 21 s
call now lands on a tick whose ``total`` is a couple of milliseconds, and folding
it back into ``latencies_ms`` would break the arithmetic the evaluation reads.

The one exception is the measured stall, and it is recorded too. The loop counts
consecutive frames in which the state's ``x_max`` does not advance, and after
``STALL_FRAMES`` of them it asks the planner for a horizon of at least
``STALL_MIN_HORIZON``, appending ``STALL_ESCALATION`` to that tick's
``degradations``. The counter resets the moment progress resumes, so the
escalation lasts exactly as long as the blindness does. It is not a repair of the
timing: the tick knowingly spends past the budget, and both the horizon it used
and what it cost are on the recorded table.

What can fail, and what is allowed to
-------------------------------------

Two tick conditions degrade to a recorded source: no Jev credential, recorded as
``FALLBACK_NO_KEY``, and a transient client failure, recorded as
``FALLBACK_DEGRADED``. An absent advisor is not a decision source at all: it is
recorded as ``advisor=None`` alongside the measured ``advisor_absence`` it holds
(when the advisor was enabled; a disabled advisor has no measurement), and the
tick proceeds unchanged. ``JevClient`` holds
no session, so ``close()`` releases nothing and a closed client still serves; the
client cannot be absent either, because ``__init__`` substitutes a default. In this
module, exactly one exception is handled, and only where a finished call is taken:
:class:`~mario_jev.jev.JevUnavailable`. It becomes ``degraded_reason``, which turns
into ``FALLBACK_DEGRADED``. ``JevAuthError``, ``JevRequestError`` and
``JevProtocolError`` are **not** handled here: a rejected credential, a malformed
request or a drifted contract aborts the run, because an endless fallback would
make a run that never once consulted Jev look Jev-driven.

With the call off the tick, both are attributed to the tick that **observes** them
rather than the tick that fired them -- the failure is not lost, only dated
honestly. The fatal ones are re-raised out of that tick, and ``run()`` closes the
dispatcher on every exit path, including that one, so no worker outlives the run.

Advisor boundary (stated plainly)
---------------------------------

The optional PlayJev advisor is **never** sent to Jev. ``JevClient.ask`` takes a
:class:`~mario_jev.state.MarioState` and projects it itself through the frozen
five-key ``to_jev_state()``, and that client is a committed, separately verified
slice; reopening it to accept a state override is out of scope here. The advisor
block is therefore an input to the **recorded** tick projection only, which this
policy layer forms in :func:`~mario_jev.policy.decision.journal_state` and exposes
as :attr:`TickRecord.journal_state`. The payload that reaches Jev is exactly
``state.to_jev_state()``, with no advisor key.

The same boundary applies to what is journalled about the Jev exchange. The
committed client builds and sends the real request in one step and hands back only
the parsed response, so this layer cannot capture the wire request. It records only
what it genuinely built — the four questions from
:func:`~mario_jev.jev.build_questions`, as :attr:`TickRecord.jev_questions` — and
the client's parsed :class:`~mario_jev.jev.JevResponse` as the sanctioned record of
the reply. The journal carries no fabricated header and no request mirror.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Callable, Mapping

import numpy as np

from mario_jev.advisor import AdvisorAbsence, AdvisorAdvice, PlayJevAdvisor
from mario_jev.config import Settings
from mario_jev.emulator import MarioEmulator, StepResult
from mario_jev.jev import JevClient, JevResponse, JevUnavailable, build_questions
from mario_jev.perception import FramePerception, detect
from mario_jev.state import MarioState, MarioStateBuilder, is_death, is_goal
from mario_jev.trajectory import RolloutTable, TrajectoryPlanner

# Imported from the module rather than the package: the horizon constants belong
# to the planner, and the package's curated ``__all__`` does not re-export them.
from mario_jev.trajectory.planner import STALL_MIN_HORIZON

from .decision import (
    DecisionCadence,
    DecisionSource,
    TickDecision,
    journal_state,
    select_action,
)
from .dispatch import JevDispatcher

#: How far past the rollout budget a sweep may go before the tick is recorded as
#: overrunning. The loop does not try to repair the overrun; it records it.
OVERSHOOT_FACTOR = 3.0

#: How many consecutive frames without ``x_max`` advancing count as a stall.
#: Measured from the recorded 1-1 run: the agent reached ``x_max = 724`` by frame
#: 271 and then stood still for 600 consecutive frames against a 48 px wall, so
#: anything that lasts half a second of game time here is not a pause in the
#: play, it is the loop having nothing to see.
STALL_FRAMES = 30

#: The degradation recorded on a tick that escalated its horizon because of a
#: stall. It is a degradation and not a feature: the tick knowingly spent past
#: the budget it was given, and the journal has to show that it did.
STALL_ESCALATION = "stall_escalation"

#: The degradation recorded on a tick whose Jev call succeeded only after a retry.
#: It is what makes a stalled tick readable after the fact. In the measured run a
#: tick spent 21 272 ms of wall clock while its recorded response reported 577 ms,
#: and its ``degradations`` list was empty because the tick did eventually
#: succeed; ~20.7 s had no entry anywhere. Two recorded facts close that gap and
#: only together: the run header's ``jev_timeout_s`` says a failed attempt could
#: cost 20 s, and this tick's response says the call took two attempts. One
#: timed-out attempt plus the 577 ms one that worked is the missing time.
JEV_RETRIED = "jev_retried"

#: How many decision cadences an answer may be old and still be applied. The
#: bound is ``STALE_LIMIT_CADENCES * settings.decision_cadence_frames`` ticks, so
#: it scales with how often the loop asks rather than with a wall clock the tick
#: does not consult. Three is read off the measurement: the median round trip was
#: 416 ms against a tick of roughly 200 ms, so an answer normally lands within a
#: couple of ticks, and one three cadences old describes a world the agent left.
STALE_LIMIT_CADENCES = 3

#: The degradation recorded on a tick that took an answer older than the staleness
#: bound and threw it away. It is recorded rather than passed over in silence
#: because the call was still made and still cost what it cost -- ``jev_call_ms``
#: says how much -- and a run where this appears often is a run whose cadence asks
#: faster than the model answers. The *source* of a stale answer that was applied is
#: not here: it is :attr:`~mario_jev.policy.decision.DecisionSource.JEV_STALE`, in
#: the closed set that owns every ``decision.source`` value.
JEV_STALE_DISCARDED = "jev_stale_discarded"

#: The closed set of reasons a run can stop. It lives here because the loop is
#: what observes the stop; the journal and the readers only copy the
#: classification out. A value outside this set is a bug, not a new state, and
#: :class:`RunSummary` refuses it at construction.
TERMINATIONS: frozenset[str] = frozenset(
    {"max_ticks", "goal", "death", "truncated", "episode_ended"}
)

__all__ = [
    "JEV_RETRIED",
    "JEV_STALE_DISCARDED",
    "OVERSHOOT_FACTOR",
    "STALE_LIMIT_CADENCES",
    "STALL_ESCALATION",
    "STALL_FRAMES",
    "TERMINATIONS",
    "ExecutionLoop",
    "RunSummary",
    "TickRecord",
    "classify_termination",
]


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def classify_termination(
    *, ended: bool, truncated: bool, info: Mapping[str, object]
) -> str:
    """Why a run stopped, as exactly one member of :data:`TERMINATIONS`.

    The episode's own signals are read through :func:`~mario_jev.state.is_goal`
    and :func:`~mario_jev.state.is_death`, never re-derived here: those two own
    the info keys, and a second copy of the key names would be a second contract
    to keep in step.

    The order is part of the contract rather than an implementation detail. A
    goal and a death cannot both be true, but a step may report ``truncated``
    alongside either, so the concrete signals are asked first and the catch-alls
    last: a death that also truncated is recorded as a death. ``episode_ended``
    is the honest name for an episode that ended with neither signal set -- it
    means "it ended and the info said nothing more", not "nobody looked".
    """
    if not ended:
        return "max_ticks"
    if is_goal(info):
        return "goal"
    if is_death(info):
        return "death"
    if truncated:
        return "truncated"
    return "episode_ended"


@dataclass(frozen=True, slots=True)
class TickRecord:
    """Everything one tick produced, in one value.

    ``frame_index`` is the loop's own game-frame counter for this tick: it starts at
    ``0`` and advances by exactly one per executed frame, so it measures game frames
    rather than the emulator's raw counter. ``emulator_tick`` is the raw
    ``emulator.tick`` read at the start of the tick; the planner's rollout sweep
    steps that counter without rewinding it, so ``emulator_tick`` jumps by the
    rollout cost while ``frame_index`` does not. The ``DecisionCadence`` is keyed
    off ``frame_index``, never ``emulator_tick``.

    ``frame`` is the tick's own frame, read **before** the rollout sweep, and
    ``perception`` is exactly what :func:`~mario_jev.perception.detect` concluded
    from it. Both are carried here rather than re-derived downstream, because the
    observer runs after the action was executed and the emulator's *current* frame
    is then one frame past the one the tick described; reconstructing the
    association after the fact would be a guess. Holding the pixels is safe:
    :meth:`MarioEmulator.frame` returns a fresh C-contiguous copy whose memory is
    not shared with the emulator, so a later step cannot mutate what this record
    holds. Both fields are ``compare=False``; a ``TickRecord`` is a frozen
    ``eq=True`` dataclass, and comparing two records must not reach for numpy's
    ambiguous array truth value. Being ``compare=False`` also keeps them out of
    ``__hash__``.

    ``jev_questions`` is what this tick asked Jev, exactly as
    :func:`~mario_jev.jev.build_questions` built it and handed it to the client, or
    ``None`` when no call was made. It is a reconstruction of the ask, not a capture
    of the wire request: the committed client exposes no request handle, so the
    journal records only the questions the policy genuinely built. The state that
    accompanied them is on ``state``, whose :meth:`MarioState.to_jev_state` is the
    frozen projection that was sent; ``JevResponse.as_dict()`` remains the
    sanctioned record of the reply in ``jev_response``.

    With the call off the tick, ``jev_questions`` is what this tick **fired** and
    ``jev_response`` is what this tick **applied**, and they need not describe the
    same call: a tick can fire one and apply an earlier one, or do only one of the
    two. ``asked_at_tick`` closes that gap -- it is the ``tick`` on which the
    applied answer was asked, and ``answer_age_ticks`` is ``tick -
    asked_at_tick``. Both are ``None`` on a tick that applied no answer, which
    includes a tick that discarded a stale one; the discard is on ``degradations``
    as :data:`JEV_STALE_DISCARDED` instead.

    ``jev_call_ms`` is what the *call* cost, measured in the worker around
    ``client.ask``, on the tick that observed it finish. It is set whenever a call
    was taken -- applied, discarded or failed -- because the cost was paid either
    way. It is deliberately **not** in ``latencies_ms``: that mapping always holds
    at least ``rollout``, ``jev``, ``advisor`` and ``total``, and it means what
    *this tick* spent, so the phases it holds -- everything except ``total``
    itself -- sum to at most ``total`` on every record. ``latencies_ms["jev"]`` is the few microseconds of
    submitting and taking; a 21 s call that arrives on a 2 ms tick belongs in
    ``jev_call_ms`` and nowhere else, because a tick's ``total`` does not contain
    it and pretending otherwise would break the arithmetic downstream reads.

    ``advisor`` is the advice that reached the tick, or ``None``. When it is
    ``None``, ``advisor_absence`` tells the journal *why*: the advisor was enabled
    and measured itself absent (the reason is the advisor's own, from its closed
    set), or the advisor was disabled and no measurement exists at all. It is only
    set while the advisor is enabled, and it is ``compare=False`` for the same
    reason the pixels are.

    ``terminated`` and ``truncated`` are the *post-step* facts, copied from the
    :class:`~mario_jev.emulator.StepResult` this tick's action produced. Every
    other field on this record describes the world the tick decided *from*, read
    before the action ran; these two describe what the action did to it, which is
    why a record whose ``state`` still shows two lives can nevertheless be the
    tick that ended the episode. They default to ``False`` because that is the
    honest reading for a tick on which the episode did not end, which is every
    tick but at most one, and because the journal's own record builders construct
    records that never ran an emulator at all.
    """

    tick: int
    frame_index: int
    emulator_tick: int
    frame: np.ndarray = field(compare=False)
    perception: FramePerception = field(compare=False)
    state: MarioState
    rollout: RolloutTable
    jev_questions: dict[str, object] | None
    jev_response: dict[str, object] | None
    decision: TickDecision
    advisor: AdvisorAdvice | None
    advisor_absence: AdvisorAbsence | None = field(compare=False)
    latencies_ms: Mapping[str, float]
    degradations: tuple[str, ...]
    # These three sit after ``degradations`` rather than beside ``jev_response``
    # only because a dataclass puts its defaulted fields last. ``None`` is the
    # honest reading for a tick that applied no answer and observed no call, which
    # is most ticks, and it is the same reason ``terminated`` defaults to ``False``.
    asked_at_tick: int | None = None
    answer_age_ticks: int | None = None
    jev_call_ms: float | None = None
    terminated: bool = False
    truncated: bool = False

    @property
    def journal_state(self) -> dict[str, object]:
        """The recorded state projection: frozen Jev state plus the advisor block.

        This is what the journal and dashboard read. It is *not* what is sent to
        Jev; the model receives :meth:`MarioState.to_jev_state`'s five keys.
        """
        return journal_state(self.state, self.advisor)


@dataclass(frozen=True, slots=True)
class RunSummary:
    """The measured shape of one run, counted rather than claimed.

    ``ticks`` is the number of completed steps and ``frames`` the number of game
    frames the loop advanced (one per step).

    ``decisions`` counts ticks that **fired** a Jev call -- ticks where
    ``TickDecision.jev_called`` is true -- and ``decision_rate_hz`` is
    ``decisions / elapsed_s`` (``0.0`` when ``elapsed_s <= 0``), so it is the rate
    at which the loop *asked*. Both kept that meaning exactly when the call moved
    off the tick, which is the point: every number already published in
    ``docs/EVALUATION.md`` was computed from them, and silently redefining a field
    would invalidate a report rather than update it.

    ``answers_applied`` is the other meaning, and it is a different number:
    the count of ticks that **applied** an answer, fresh or stale -- exactly the
    records whose ``answer_age_ticks`` is not ``None``. It can be lower than
    ``decisions`` (a call failed, or its answer was discarded as stale, or the run
    ended with one still in flight) and it is never higher, because at most one
    call is in flight and every applied answer was fired first. It has no default,
    for the same reason ``termination`` has none: a construction that forgets it
    must fail rather than quietly report zero applied answers.

    ``termination`` says *why* the run stopped, as one member of
    :data:`TERMINATIONS`, and it is what turns ``ticks: 805`` against a 900-tick
    budget from a mystery into a fact. It is enforced at construction, the way
    :class:`~mario_jev.advisor.AdvisorAbsence` enforces its own reason: a value
    outside the set is a bug, and it must not survive to the journal. It has no
    default on purpose -- a default would quietly label every construction that
    forgot it, and a wrong reason in the evidence file is worse than none.

    Be exact about what that check buys. :func:`classify_termination` is the single
    producer of this field in the run path, and the check enforces **set
    membership, not truth**: ``termination="goal"`` on a run that hit its tick cap
    is a member of :data:`TERMINATIONS` and is accepted. This is a plain dataclass
    and public construction is legitimate (the journal readers rebuild summaries),
    so no validator here could tell a wrong-but-valid reason from a right one --
    only the classifier that watched the run can. Read a ``termination`` as "what
    the producer concluded", and trust it exactly as far as you trust the producer.
    """

    ticks: int
    frames: int
    decisions: int
    answers_applied: int
    sources: Mapping[DecisionSource, int]
    degradations: Mapping[str, int]
    elapsed_s: float
    decision_rate_hz: float
    termination: str

    def __post_init__(self) -> None:
        if self.termination not in TERMINATIONS:
            # The message names the whole closed set: a caller who got this wrong
            # needs to know what is allowed, and a reader of the traceback should
            # not have to open this module to find out.
            allowed = ", ".join(sorted(TERMINATIONS))
            raise ValueError(
                f"unknown termination: {self.termination!r}; must be one of: {allowed}"
            )


class ExecutionLoop:
    """The tick loop over one emulator, one planner, one client and one advisor.

    Every collaborator can be injected. With none injected the loop builds the real
    ones from ``settings``: a :class:`MarioEmulator` on ``settings.stage``, a
    :class:`TrajectoryPlanner`, a :class:`JevClient`, a :class:`MarioStateBuilder`,
    a :class:`~mario_jev.policy.dispatch.JevDispatcher` over that client, and —
    only when ``settings.advisor_enabled`` — a :class:`PlayJevAdvisor`.

    The dispatcher is injectable for the same reason the client is: it carries the
    ``runner`` seam, so a test can drive a call to completion by hand instead of
    waiting on a thread. ``run()`` closes whichever dispatcher it was given.
    """

    __slots__ = (
        "_action_index",
        "_advisor",
        "_builder",
        "_cadence",
        "_client",
        "_dispatcher",
        "_emulator",
        "_end_info",
        "_ended",
        "_frame_index",
        "_last_action",
        "_last_x_max",
        "_planner",
        "_settings",
        "_stall_frames",
        "_tick",
    )

    def __init__(
        self,
        settings: Settings,
        *,
        emulator: MarioEmulator | None = None,
        planner: TrajectoryPlanner | None = None,
        client: JevClient | None = None,
        builder: MarioStateBuilder | None = None,
        advisor: PlayJevAdvisor | None = None,
        dispatcher: JevDispatcher | None = None,
    ) -> None:
        if emulator is None:
            emulator = MarioEmulator(
                stage=settings.stage,
                seed=settings.seed,
                real_time=settings.real_time,
                target_fps=settings.target_fps,
            )
            emulator.reset()
        self._settings = settings
        self._emulator = emulator
        self._planner = (
            planner if planner is not None else TrajectoryPlanner.from_settings(settings)
        )
        self._client = (
            client if client is not None else JevClient.from_settings(settings)
        )
        self._builder = builder if builder is not None else MarioStateBuilder()
        self._dispatcher = (
            dispatcher if dispatcher is not None else JevDispatcher(self._client)
        )
        if advisor is None and settings.advisor_enabled:
            advisor = PlayJevAdvisor.from_settings(settings)
        self._advisor = advisor
        self._cadence = DecisionCadence(settings.decision_cadence_frames)
        self._action_index = {
            name: index for index, name in enumerate(emulator.action_names)
        }
        self._tick = 0
        # The loop's own game-frame counter. It starts at 0 and advances once per
        # executed frame, independent of the emulator counter the planner inflates.
        self._frame_index = 0
        self._last_action = "NOOP"
        # Stall detection lives here because only the loop sees the history: the
        # planner is handed one savestate and cannot know whether the last thirty
        # frames went anywhere. ``None`` means nothing has been observed yet, and
        # a first observation is not a stalled one.
        self._last_x_max: int | None = None
        self._stall_frames = 0
        self._ended = False
        # The info the ending step reported, kept so ``run`` can classify the
        # stop through the state layer's own predicates. ``None`` means no step
        # has ended the episode, which is a different thing from an empty info.
        self._end_info: Mapping[str, object] | None = None

    def step(self) -> TickRecord:
        """Advance exactly one tick and return its record.

        Raises:
            RuntimeError: the episode already ended. ``nes_py`` refuses a step past
                a done episode, so the loop refuses one first rather than letting
                the emulator raise inside the sweep.
        """
        if self._ended:
            raise RuntimeError(
                "the episode already ended; the loop must not step past "
                "terminated/truncated"
            )

        settings = self._settings
        emulator = self._emulator
        started = time.perf_counter()

        # 1. The loop owns the frame index: the planner steps the emulator's counter
        #    without rewinding it, so that counter cannot measure game frames. The
        #    raw emulator tick is read here too, before the sweep moves it, and
        #    recorded separately so the inflation stays visible. The frame is read
        #    here so it is the one the state and the advisor both describe.
        frame_index = self._frame_index
        emulator_tick = int(emulator.tick)
        frame = emulator.frame()
        latencies: dict[str, float] = {}

        # 2. Typed state from the frame, RAM and info.
        perception = detect(frame, tick=frame_index)
        state = self._builder.observe(perception, emulator.ram(), emulator.info)

        # 3. The sweep runs every tick; the trajectories are local, and the state
        #    must carry them before anything asks the model about them. A measured
        #    stall -- ``STALL_FRAMES`` frames in which the state's own ``x_max``
        #    never advanced -- puts a floor under the horizon for this tick only:
        #    at a wall the budget's horizon shows every candidate as
        #    ``no_progress`` at one identical risk, and a table with no signal in
        #    it is decided by the tie-break rather than by the measurement.
        stalled = self._observe_progress(state.progress.x_max)
        rollout_started = time.perf_counter()
        if stalled:
            table = self._planner.plan(emulator, min_horizon=STALL_MIN_HORIZON)
        else:
            table = self._planner.plan(emulator)
        latencies["rollout"] = _elapsed_ms(rollout_started)
        state = state.with_candidates(table.assessments)

        degradations: list[str] = []
        if table.elapsed_ms > settings.rollout_budget_ms * OVERSHOOT_FACTOR:
            degradations.append("timing_overrun")
        # Recorded on the loop's own measurement, not on the planner's answer:
        # this tick asked for more frames than the budget allotted. Whether those
        # frames cost anything beyond the budget's own choice is the planner's
        # fact, and it is on ``table.escalated``.
        #
        # Only when there was something to plan. The loop cannot know the table is
        # empty before asking, so ``plan()`` still receives the floor -- but a table
        # with no candidates has no horizon to lengthen, and claiming an escalation
        # would describe a cost that could not have been paid. On that tick
        # ``fallback_no_candidates`` already says what happened.
        if stalled and table.candidates:
            degradations.append(STALL_ESCALATION)

        # The advisor is optional input, never an authority, and never on the
        # critical path: absent or failing, the tick proceeds unchanged.
        advice: AdvisorAdvice | None = None
        advisor_started = time.perf_counter()
        if settings.advisor_enabled and self._advisor is not None and self._advisor.available:
            advice = self._advisor.advise(frame, emulator.action_names)
        latencies["advisor"] = _elapsed_ms(advisor_started)
        # The measured absence, when there is one. It is carried only while the
        # advisor is enabled: a disabled advisor was never measured, and writing
        # its absence as a measured reason would state a fact nobody observed.
        advisor_absence = (
            self._advisor.absence
            if settings.advisor_enabled and self._advisor is not None
            else None
        )

        # 4. Jev, in two independent halves that share a tick but not a call.
        #
        #    Fire, only on a cadence tick with a credential, something to ask about
        #    and nothing already in flight. ``build_questions`` refuses an empty
        #    candidate set, so the candidates are part of the gate rather than an
        #    afterthought, and a refused fire builds no questions at all -- the
        #    record's ``jev_questions is None`` is how the tick says it did not ask.
        cadence_due = self._cadence.due(frame_index)
        jev_questions: dict[str, object] | None = None
        response: JevResponse | None = None
        degraded_reason: str | None = None
        asked_at_tick: int | None = None
        answer_age_ticks: int | None = None
        jev_call_ms: float | None = None
        fired = False
        jev_started = time.perf_counter()
        if (
            cadence_due
            and settings.jev_available
            and table.candidates
            and not self._dispatcher.in_flight
        ):
            questions = build_questions(state)
            # Recorded only because the policy built them; the client exposes no
            # request handle, so this is the ask, not a capture of the wire request.
            jev_questions = questions
            fired = self._dispatcher.request(questions, state, tick=self._tick)

        #    Take, on every tick, and never block. What comes back was asked on
        #    this tick or an earlier one, and the tick that observes it is the tick
        #    that must account for it.
        taken = self._dispatcher.take()
        if taken is not None:
            # The call cost what it cost whether or not its answer is used, and
            # this is the only tick that can report it.
            jev_call_ms = taken.call_ms
            if taken.error is not None:
                if isinstance(taken.error, JevUnavailable):
                    # The only failure the loop absorbs, recorded on the tick that
                    # saw it rather than the one that fired it.
                    degraded_reason = str(taken.error)
                else:
                    # Auth, request and protocol errors abort the run. The worker
                    # carried the exception back instead of raising it out of a
                    # thread; re-raising it here is what keeps the committed
                    # contract that those never degrade.
                    raise taken.error
            elif taken.response is not None:
                # A successful call that took more than one transport attempt paid
                # for a failed one first, and nothing else records that cost. It is
                # read off the taken response, so a tick that took no call has no
                # attempt count at all rather than a defaulted 1.
                if taken.response.attempts > 1:
                    degradations.append(JEV_RETRIED)
                age = self._tick - taken.asked_at_tick
                if age <= STALE_LIMIT_CADENCES * settings.decision_cadence_frames:
                    response = taken.response
                    asked_at_tick = taken.asked_at_tick
                    answer_age_ticks = age
                else:
                    # Discarded, not cancelled: the call ran to completion and this
                    # tick simply refuses to act on a world the agent has left.
                    degradations.append(JEV_STALE_DISCARDED)
        latencies["jev"] = _elapsed_ms(jev_started)

        # ``select_action`` decides on a *frame*, and with the call off the tick the
        # frames that carry a decision are the frames an answer lands on, not the
        # frames a call is fired on. So the rule is told a decision is due when
        # there is an answer to apply (however old, however failed) -- and, when no
        # credential is configured, on the cadence itself, which is the one branch
        # that decides without an answer. Every other tick holds: no answer yet,
        # one in flight, or one discarded as stale.
        answer_available = response is not None or degraded_reason is not None
        decision = select_action(
            table,
            response,
            settings,
            cadence_due=answer_available or (cadence_due and not settings.jev_available),
            last_action=self._last_action,
            degraded_reason=degraded_reason,
        )
        # Two corrections the pure rule cannot make, because it sees one tick and
        # these are facts about two. ``jev_called`` means "this tick fired a call",
        # which is what ``RunSummary.decisions`` counts; the rule infers it from
        # having consulted an answer, and an answer can be consulted a tick or
        # three after it was asked for. And an answer from an earlier tick is
        # ``DecisionSource.JEV_STALE`` rather than ``JEV`` -- only where the rule
        # chose ``JEV``, since ``guardrail_lethal`` and
        # ``fallback_below_confidence`` already name something more specific and
        # carry the age in the record either way.
        source = decision.source
        if source is DecisionSource.JEV and answer_age_ticks:
            source = DecisionSource.JEV_STALE
        decision = replace(decision, source=source, jev_called=fired)
        self._last_action = decision.action

        # 5. Execute. A decided name the emulator does not know is a KeyError here,
        #    which is a bug: the table's action set is the emulator's action set.
        result: StepResult = emulator.step(
            emulator.action(self._action_index[decision.action])
        )
        # Exactly one executed game frame per step, counted after the action ran.
        self._frame_index += 1
        latencies["total"] = _elapsed_ms(started)
        if result.terminated or result.truncated:
            self._ended = True
            # Kept as its own mapping: ``emulator.info`` may move on, and the
            # classification must read the info of the step that actually ended.
            self._end_info = dict(result.info)

        record = TickRecord(
            tick=self._tick,
            frame_index=frame_index,
            emulator_tick=emulator_tick,
            frame=frame,
            perception=perception,
            state=state,
            rollout=table,
            jev_questions=jev_questions,
            jev_response=None if response is None else response.as_dict(),
            decision=decision,
            advisor=advice,
            advisor_absence=advisor_absence,
            latencies_ms=latencies,
            degradations=tuple(degradations),
            asked_at_tick=asked_at_tick,
            answer_age_ticks=answer_age_ticks,
            jev_call_ms=jev_call_ms,
            terminated=bool(result.terminated),
            truncated=bool(result.truncated),
        )
        self._tick += 1
        return record

    def _observe_progress(self, x_max: int) -> bool:
        """Record this tick's ``x_max`` and answer whether the run has stalled.

        ``x_max`` comes from the state the tick already built, which reads it from
        ``info['x_pos_max']`` -- the same source, so the loop cannot disagree with
        its own record about whether Mario advanced. The first tick observes
        without judging: there is no earlier frame to compare it to.

        One rule, one place. There is no second escalation path and no retry: when
        progress resumes the counter resets and the next tick plans on its budget
        again, and while it does not, every tick escalates the same way.
        """
        if self._last_x_max is not None and x_max <= self._last_x_max:
            self._stall_frames += 1
        else:
            self._stall_frames = 0
        self._last_x_max = x_max
        return self._stall_frames >= STALL_FRAMES

    def run(
        self,
        *,
        max_ticks: int | None = None,
        observer: Callable[[TickRecord], None] | None = None,
    ) -> RunSummary:
        """Step until the episode ends or ``max_ticks`` is reached.

        ``max_ticks`` overrides ``settings.max_ticks`` when given; ``None`` on both
        means "until the episode ends". ``observer`` is called with every record,
        including the terminating one. The run stops on the first
        ``terminated``/``truncated`` step and never steps past it.

        The summary says *why* it stopped, classified by
        :func:`classify_termination` from the ending step's own info. Reaching
        ``limit`` is ``"max_ticks"``; anything else is the episode's own signal.

        The dispatcher is closed on **every** exit path, including the one where a
        :class:`~mario_jev.jev.JevAuthError` or
        :class:`~mario_jev.jev.JevRequestError` aborts the run, so no worker
        outlives the run. Closing joins, so a run that aborts while a call is still
        running waits for it; there is nothing to cancel it with, and a lingering
        thread would be worse than a bounded wait.
        """
        limit = self._settings.max_ticks if max_ticks is None else max_ticks
        started = time.perf_counter()
        ticks = 0
        decisions = 0
        answers_applied = 0
        last: TickRecord | None = None
        sources: Counter[DecisionSource] = Counter()
        degradations: Counter[str] = Counter()

        try:
            while limit is None or ticks < limit:
                record = self.step()
                last = record
                ticks += 1
                sources[record.decision.source] += 1
                degradations.update(record.degradations)
                # Two counters, two meanings: a call fired, and an answer applied.
                if record.decision.jev_called:
                    decisions += 1
                if record.answer_age_ticks is not None:
                    answers_applied += 1
                if observer is not None:
                    observer(record)
                if self._ended:
                    break

            elapsed_s = time.perf_counter() - started
            rate = 0.0 if elapsed_s <= 0 else decisions / elapsed_s
            termination = classify_termination(
                ended=self._ended,
                truncated=last is not None and last.truncated,
                info={} if self._end_info is None else self._end_info,
            )
            return RunSummary(
                ticks=ticks,
                frames=ticks,
                decisions=decisions,
                answers_applied=answers_applied,
                sources=sources,
                degradations=degradations,
                elapsed_s=elapsed_s,
                decision_rate_hz=rate,
                termination=termination,
            )
        finally:
            self._dispatcher.close()
