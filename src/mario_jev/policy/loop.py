"""The execution loop: one planner, one cadence, and every tick recorded.

One :class:`ExecutionLoop` owns one :class:`~mario_jev.trajectory.TrajectoryPlanner`
for the whole run. That is a requirement rather than an optimisation: the planner
calibrates the machine on its first call and sizes later horizons from what it
measured, so a fresh planner per tick would pay the full calibration every frame.

Each :meth:`ExecutionLoop.step` does the same five things, in order:

1. read the loop's own game-frame counter and the raw ``emulator.tick``;
2. build the typed state from the frame, RAM and ``info``;
3. run the rollout sweep every tick, because the trajectories are local and cheap;
4. decide with :func:`~mario_jev.policy.decision.select_action`, consulting Jev only
   when the cadence is due, a credential is configured, and there are candidates;
5. execute the decided action with the real emulator, then advance the loop's
   game-frame counter by exactly one.

The loop owns the frame index because the planner does not. The rollout sweep steps
the same emulator while enumerating candidates and never rewinds its counter, so
``emulator.tick`` jumps by the whole rollout cost on every tick; keying the cadence
off that counter would fire on an arbitrary subset of ticks rather than every
``decision_cadence_frames`` game frames. ``frame_index`` therefore starts at ``0``
and counts *executed* frames, while the raw ``emulator.tick`` is recorded alongside
it as ``emulator_tick`` so the planner's inflation stays visible.

Timing is **recorded, never repaired**: ``timing_overrun`` is appended when the
rollout's own ``elapsed_ms`` exceeds ``rollout_budget_ms * OVERSHOOT_FACTOR``. The
loop holds one planner across ticks; the limit stays visible in the journal.

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
module, exactly one exception is caught, and only around the client call:
:class:`~mario_jev.jev.JevUnavailable`. It becomes ``degraded_reason``, which turns
into ``FALLBACK_DEGRADED``. ``JevAuthError``, ``JevRequestError`` and
``JevProtocolError`` are **not** caught here: a rejected credential, a malformed
request or a drifted contract aborts the run, because an endless fallback would
make a run that never once consulted Jev look Jev-driven.

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
from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np

from mario_jev.advisor import AdvisorAbsence, AdvisorAdvice, PlayJevAdvisor
from mario_jev.config import Settings
from mario_jev.emulator import MarioEmulator, StepResult
from mario_jev.jev import JevClient, JevUnavailable, build_questions
from mario_jev.perception import FramePerception, detect
from mario_jev.state import MarioState, MarioStateBuilder
from mario_jev.trajectory import RolloutTable, TrajectoryPlanner

from .decision import (
    DecisionCadence,
    DecisionSource,
    TickDecision,
    journal_state,
    select_action,
)

#: How far past the rollout budget a sweep may go before the tick is recorded as
#: overrunning. The loop does not try to repair the overrun; it records it.
OVERSHOOT_FACTOR = 3.0

__all__ = [
    "OVERSHOOT_FACTOR",
    "ExecutionLoop",
    "RunSummary",
    "TickRecord",
]


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


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
    sanctioned record of the reply in ``jev_response``. ``latencies_ms`` always
    holds at least ``rollout``, ``jev``, ``advisor`` and ``total``.

    ``advisor`` is the advice that reached the tick, or ``None``. When it is
    ``None``, ``advisor_absence`` tells the journal *why*: the advisor was enabled
    and measured itself absent (the reason is the advisor's own, from its closed
    set), or the advisor was disabled and no measurement exists at all. It is only
    set while the advisor is enabled, and it is ``compare=False`` for the same
    reason the pixels are.
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
    frames the loop advanced (one per step). ``decisions`` counts ticks where
    ``TickDecision.jev_called`` is true, so a run that never consulted Jev cannot
    overstate its rate. ``decision_rate_hz`` is ``decisions / elapsed_s``, or
    ``0.0`` when ``elapsed_s <= 0``.
    """

    ticks: int
    frames: int
    decisions: int
    sources: Mapping[DecisionSource, int]
    degradations: Mapping[str, int]
    elapsed_s: float
    decision_rate_hz: float


class ExecutionLoop:
    """The tick loop over one emulator, one planner, one client and one advisor.

    Every collaborator can be injected. With none injected the loop builds the real
    ones from ``settings``: a :class:`MarioEmulator` on ``settings.stage``, a
    :class:`TrajectoryPlanner`, a :class:`JevClient`, a :class:`MarioStateBuilder`,
    and — only when ``settings.advisor_enabled`` — a :class:`PlayJevAdvisor`.
    """

    __slots__ = (
        "_action_index",
        "_advisor",
        "_builder",
        "_cadence",
        "_client",
        "_emulator",
        "_ended",
        "_frame_index",
        "_last_action",
        "_planner",
        "_settings",
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
        self._ended = False

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
        #    must carry them before anything asks the model about them.
        rollout_started = time.perf_counter()
        table = self._planner.plan(emulator)
        latencies["rollout"] = _elapsed_ms(rollout_started)
        state = state.with_candidates(table.assessments)

        degradations: list[str] = []
        if table.elapsed_ms > settings.rollout_budget_ms * OVERSHOOT_FACTOR:
            degradations.append("timing_overrun")

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

        # 4. Jev, only on a cadence tick with a credential and something to ask
        #    about. ``build_questions`` refuses an empty candidate set, so the
        #    candidates are part of the gate rather than an afterthought.
        cadence_due = self._cadence.due(frame_index)
        jev_questions: dict[str, object] | None = None
        response = None
        degraded_reason: str | None = None
        jev_started = time.perf_counter()
        if cadence_due and settings.jev_available and table.candidates:
            questions = build_questions(state)
            # Recorded only because the policy built them; the client exposes no
            # request handle, so this is the ask, not a capture of the wire request.
            jev_questions = questions
            try:
                response = self._client.ask(questions, state)
            except JevUnavailable as exc:
                # The only catch in the loop. Auth, request and protocol errors are
                # deliberately allowed to propagate and abort the run.
                degraded_reason = str(exc)
        latencies["jev"] = _elapsed_ms(jev_started)

        decision = select_action(
            table,
            response,
            settings,
            cadence_due=cadence_due,
            last_action=self._last_action,
            degraded_reason=degraded_reason,
        )
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
        )
        self._tick += 1
        return record

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
        """
        limit = self._settings.max_ticks if max_ticks is None else max_ticks
        started = time.perf_counter()
        ticks = 0
        decisions = 0
        sources: Counter[DecisionSource] = Counter()
        degradations: Counter[str] = Counter()

        while limit is None or ticks < limit:
            record = self.step()
            ticks += 1
            sources[record.decision.source] += 1
            degradations.update(record.degradations)
            if record.decision.jev_called:
                decisions += 1
            if observer is not None:
                observer(record)
            if self._ended:
                break

        elapsed_s = time.perf_counter() - started
        rate = 0.0 if elapsed_s <= 0 else decisions / elapsed_s
        return RunSummary(
            ticks=ticks,
            frames=ticks,
            decisions=decisions,
            sources=sources,
            degradations=degradations,
            elapsed_s=elapsed_s,
            decision_rate_hz=rate,
        )
