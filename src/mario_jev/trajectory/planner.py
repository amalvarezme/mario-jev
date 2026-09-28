"""Enumeration of candidate trajectories over the real emulator.

Each candidate action is rolled forward from the *same* savestate, and the
emulator is put back afterwards, so the sweep answers "what would each action do
from here" rather than "what does a sequence of actions do".

Why savestates rather than a hand-written physics model: ``dump_state`` /
``load_state`` are real savestates, measured at several hundred load-and-step
frames per second on this machine (the plan recorded 350 fps under load), so the
exact answer is affordable. A reduced physics model would have to re-derive the
walking acceleration, the run speed, the jump arc and the tile collisions, and
every one of those re-derivations would be a subtly wrong answer that looks
plausible.

The cost of the sweep is measured, not assumed, but it is measured *once*: the
first :meth:`TrajectoryPlanner.plan` on an instance calibrates the machine, and
every later call reuses that measurement to pick a horizon without paying for a
calibration pass it does not need. The remembered measurement is not sticky — a
sweep that overruns the budget runs the bounded correction pass and the fresh
measurement replaces the remembered one, so the model can only ever be corrected
towards the machine's real behaviour.

Four measured behaviours shape this module, and each one has a test:

1. ``nes_py`` raises a ``ValueError`` ("cannot step in a done environment") on the
   first step after the episode ends. The installed message wraps ``reset`` in
   backticks, which is why the test matches on ``"done environment"`` and not on
   the whole sentence. The rollout therefore has to break on
   ``terminated``/``truncated`` — and it has to, because a candidate that dies
   mid-rollout is the signal this planner exists to produce.
2. ``load_state`` clears that done flag, so a candidate that died does not stop
   the next one from running. There is no ``reset()`` anywhere in here.
3. ``load_state`` refreshes the wrapper's cached ``info`` itself. The trap is the
   *caller* keeping a copy from before the load; this module re-reads ``info``
   after every ``load_state`` and never reuses one across it.
4. The frame counter is not rewound by ``load_state``, and this module does not
   try to rewind it: a rollout is not gameplay, and the tick is how the journal
   stays honest about how much work a tick really did.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

from mario_jev.config import Settings
from mario_jev.emulator import RAM_ADDRESSES, Action, MarioEmulator, build_actions
from mario_jev.perception import TerrainProfile, detect
from mario_jev.perception.palette import PLAYER_BOX_SIZE
from mario_jev.state import (
    INFO_X_POS,
    CandidateAssessment,
    HAZARD_OUTCOMES,
    TerminalOutcome,
    is_death,
    is_goal,
    nearest_enemy_gap,
)

from .risk import risk_cost

#: How many ``load_state`` calls the calibration times, and how many steady
#: frames it steps after the first one. Small on purpose: the calibration has to
#: be far cheaper than the sweep it sizes, and only the first ``plan()`` on a
#: planner instance pays for it at all.
CALIBRATION_LOADS: int = 3
CALIBRATION_STEADY_STEPS: int = 3

#: Floor for the measured per-frame cost. A clock that resolves to zero must not
#: turn into a division by zero — it means "too fast to measure", not "free".
MIN_PER_FRAME_MS: float = 0.001

#: The sweep runs at most twice: once at the horizon the cost model predicted,
#: and once more if the measured sweep overran the budget and a smaller horizon
#: can still fit.
MAX_SWEEP_PASSES: int = 2


class CostSource(StrEnum):
    """Where the cost model that sized one sweep came from.

    A closed two-value set, because the journal has to record a fact rather than
    an inference: either this call measured the machine itself, or it reused the
    measurement its predecessor on the same planner left behind.
    """

    #: This call ran the calibration pass and sized the sweep from it.
    CALIBRATED = "calibrated"
    #: This call skipped the calibration pass and reused the remembered cost.
    REUSED = "reused"


def choose_horizon(
    *,
    budget_ms: float,
    candidate_count: int,
    base_ms: float,
    per_frame_ms: float,
    min_frames: int,
    max_frames: int,
) -> int:
    """How many frames each candidate gets, from measured cost.

    Pure by construction: it takes the measured numbers rather than the clock, so
    the arithmetic can be tested exactly while the planner still measures a real
    machine at run time.
    """
    if candidate_count < 1:
        raise ValueError(f"candidate_count must be >= 1, got {candidate_count}")
    if budget_ms <= 0:
        raise ValueError(f"budget_ms must be > 0, got {budget_ms}")
    if min_frames < 1:
        raise ValueError(f"min_frames must be >= 1, got {min_frames}")
    if max_frames < min_frames:
        raise ValueError(
            f"max_frames must be >= min_frames, got {max_frames} < {min_frames}"
        )

    per_frame = max(float(per_frame_ms), MIN_PER_FRAME_MS)
    base = max(0.0, float(base_ms))
    per_candidate_ms = float(budget_ms) / candidate_count
    frames = int((per_candidate_ms - base) / per_frame)
    return max(int(min_frames), min(int(max_frames), frames))


def remeasured_per_frame_ms(
    *,
    sweep_ms: float,
    horizon: int,
    candidate_count: int,
    base_ms: float,
) -> float:
    """Recover the per-frame cost from a sweep that actually ran.

    The model is ``n * (base + horizon * per_frame)`` for ``n`` candidates, so
    once the fixed term is subtracted the measured total gives the per-frame
    term directly. The fixed term is carried over rather than re-derived: one
    sweep cannot isolate ``base`` from ``per_frame``, and ``base`` is the small
    term, so the honest move is to keep the measured one and re-measure the term
    that dominates the horizon. Pure by construction, so the arithmetic can be
    checked exactly without a clock.
    """
    if candidate_count < 1:
        raise ValueError(f"candidate_count must be >= 1, got {candidate_count}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    per_candidate_ms = float(sweep_ms) / candidate_count
    per_frame = (per_candidate_ms - max(0.0, float(base_ms))) / horizon
    return max(MIN_PER_FRAME_MS, per_frame)


def resolve_outcome(
    *,
    reached_goal: bool,
    died: bool,
    over_gap: bool,
    advanced: bool,
) -> TerminalOutcome:
    """Resolve a rollout end to exactly one outcome, in the contracted precedence.

    ``reaches_goal`` wins over a death bit on the same frame, then a death splits
    on whether Mario's hitbox was over a hole column when it happened.
    """
    if reached_goal:
        return TerminalOutcome.REACHES_GOAL
    if died:
        return TerminalOutcome.FALLS_IN_GAP if over_gap else TerminalOutcome.HITS_ENEMY
    if not advanced:
        return TerminalOutcome.NO_PROGRESS
    return TerminalOutcome.LANDS_SAFE


def died_over_gap(
    screen_x: int,
    terrain: TerrainProfile,
    *,
    size: int = PLAYER_BOX_SIZE,
) -> bool:
    """Whether Mario's sprite box covered a hole column when he died.

    ``screen_x`` is the on-screen sprite column (RAM ``0x03AD``) and the terrain
    columns are frame columns, so the two are directly comparable. Columns outside
    the frame are not holes: :meth:`TerrainProfile.is_hole` returns ``False``
    there, which is the honest answer for "off the edge of what we measured".
    """
    return any(terrain.is_hole(column) for column in range(screen_x, screen_x + size))


@dataclass(frozen=True, slots=True)
class CandidateRollout:
    """One action, rolled forward from the tick's savestate."""

    action: Action
    horizon: int
    #: Frames actually stepped. Less than :attr:`horizon` when the episode ended first.
    frames: int
    start_x: int
    final_x: int
    x_max: int
    outcome: TerminalOutcome
    risk: float
    #: Pixels between Mario's box and the nearest enemy at the end of the rollout.
    final_enemy_distance: int | None
    terminated: bool
    truncated: bool
    #: Whether the death (if any) happened over a hole column.
    over_gap: bool = False

    @property
    def name(self) -> str:
        return self.action.name

    @property
    def dx(self) -> int:
        """How far the rollout moved Mario, in in-level pixels."""
        return self.final_x - self.start_x

    @property
    def died(self) -> bool:
        return self.outcome in HAZARD_OUTCOMES

    def to_assessment(self) -> CandidateAssessment:
        """The four fields Jev is shown for this candidate."""
        return CandidateAssessment(
            action=self.name,
            dx=self.dx,
            outcome=self.outcome,
            risk=self.risk,
            final_x=self.final_x,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "action": self.name,
            "action_index": int(self.action.index),
            "horizon": int(self.horizon),
            "frames": int(self.frames),
            "start_x": int(self.start_x),
            "final_x": int(self.final_x),
            "x_max": int(self.x_max),
            "dx": int(self.dx),
            "outcome": self.outcome.value,
            "risk": round(float(self.risk), 4),
            "final_enemy_distance": (
                None if self.final_enemy_distance is None else int(self.final_enemy_distance)
            ),
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
            "over_gap": bool(self.over_gap),
        }


@dataclass(frozen=True, slots=True)
class RolloutTable:
    """The whole sweep, including how it was sized and what it cost.

    ``elapsed_ms`` is the measured wall clock of the entire ``plan()`` call, and
    ``sweep_ms`` is the measured wall clock of the rollouts alone. Both are
    reported rather than one: the journal wants the real per-tick cost, and the
    budget is about the sweep that the horizon was chosen for.
    """

    tick: int
    start_x: int
    horizon: int
    budget_ms: float
    elapsed_ms: float
    sweep_ms: float
    base_ms: float
    per_frame_ms: float
    calibration_frames: int
    #: 1 normally; 2 when the measured sweep forced a smaller horizon.
    passes: int
    #: Whether the cost model above was measured on this call or remembered from
    #: the previous one on the same planner instance. Recorded, never inferred.
    cost_source: CostSource
    #: Total emulator steps issued by the whole ``plan()`` call. The frame counter
    #: moves by exactly this much.
    steps_taken: int
    candidates: tuple[CandidateRollout, ...]

    @property
    def assessments(self) -> tuple[CandidateAssessment, ...]:
        """The candidates as the state projection wants them."""
        return tuple(candidate.to_assessment() for candidate in self.candidates)

    @property
    def best(self) -> CandidateRollout:
        """Lowest risk, and among equals the one that advanced furthest."""
        if not self.candidates:
            raise ValueError("the table has no candidates")
        return min(self.candidates, key=lambda candidate: (candidate.risk, -candidate.dx))

    def by_name(self, name: str) -> CandidateRollout:
        for candidate in self.candidates:
            if candidate.name == name:
                return candidate
        raise KeyError(f"no candidate for action {name!r}")

    def as_dict(self) -> dict[str, object]:
        return {
            "tick": int(self.tick),
            "start_x": int(self.start_x),
            "horizon": int(self.horizon),
            "budget_ms": round(float(self.budget_ms), 3),
            "elapsed_ms": round(float(self.elapsed_ms), 3),
            "sweep_ms": round(float(self.sweep_ms), 3),
            "base_ms": round(float(self.base_ms), 4),
            "per_frame_ms": round(float(self.per_frame_ms), 4),
            "calibration_frames": int(self.calibration_frames),
            "passes": int(self.passes),
            "cost_source": str(self.cost_source),
            "steps_taken": int(self.steps_taken),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class _Calibration:
    """What the machine cost for one savestate load and one frame."""

    base_ms: float
    per_frame_ms: float
    frames: int


@dataclass(frozen=True, slots=True)
class _CostModel:
    """The ``base + horizon * per_frame`` model the next horizon is sized from.

    Deliberately separate from :class:`_Calibration`: a calibration also carries
    the emulator steps it cost, while a model remembered from a previous call
    costs this call nothing to obtain.
    """

    base_ms: float
    per_frame_ms: float


class TrajectoryPlanner:
    """Rolls every candidate action forward from the current tick's savestate."""

    __slots__ = (
        "_actions",
        "_budget_ms",
        "_calibration_action",
        "_max_frames",
        "_min_frames",
        "_remembered",
    )

    def __init__(
        self,
        *,
        budget_ms: float,
        min_frames: int,
        max_frames: int,
        actions: Sequence[Action] | None = None,
    ) -> None:
        if budget_ms <= 0:
            raise ValueError(f"budget_ms must be > 0, got {budget_ms}")
        if min_frames < 1:
            raise ValueError(f"min_frames must be >= 1, got {min_frames}")
        if max_frames < min_frames:
            raise ValueError(
                f"max_frames must be >= min_frames, got {max_frames} < {min_frames}"
            )
        selected = tuple(actions) if actions is not None else build_actions()
        if not selected:
            raise ValueError("the planner needs at least one candidate action")

        self._budget_ms = float(budget_ms)
        self._min_frames = int(min_frames)
        self._max_frames = int(max_frames)
        self._actions = selected
        #: What the previous ``plan()`` measured, or ``None`` before the first
        #: one. ``None`` is the only trigger for a calibration pass.
        self._remembered: _CostModel | None = None
        # Calibration must not be allowed to die mid-measurement, so it steps the
        # action that does nothing when the set has one.
        self._calibration_action = next(
            (action for action in selected if action.name == "NOOP"), selected[0]
        )

    @classmethod
    def from_settings(
        cls, settings: Settings, *, actions: Sequence[Action] | None = None
    ) -> "TrajectoryPlanner":
        """Build a planner from the rollout settings in ``config``."""
        return cls(
            budget_ms=settings.rollout_budget_ms,
            min_frames=settings.rollout_min_frames,
            max_frames=settings.rollout_max_frames,
            actions=actions,
        )

    @property
    def actions(self) -> tuple[Action, ...]:
        return self._actions

    @property
    def budget_ms(self) -> float:
        return self._budget_ms

    @property
    def horizon_bounds(self) -> tuple[int, int]:
        return (self._min_frames, self._max_frames)

    # -- the sweep ---------------------------------------------------------------

    def plan(self, emulator: MarioEmulator) -> RolloutTable:
        """Roll every candidate action forward and leave the emulator unmoved."""
        started = time.perf_counter()
        snapshot = emulator.dump_state()
        tick = int(emulator.tick)
        start_x = int(dict(emulator.info)[INFO_X_POS])

        # The whole sweep runs with pacing suspended: every step below is a
        # simulated future rolled forward from the snapshot, so it must not
        # consume the real timeline. The loop's own `step()` keeps its pacing
        # untouched -- only the rollout is exempt, and the schedule is reset on
        # the way out so the next real frame starts a fresh period.
        with emulator.pacer.suspended():
            # Warm start: only the first call on an instance pays for a calibration.
            # Later calls size their horizon from what the machine cost last time,
            # which is what keeps the steady-state tick at roughly one sweep.
            if self._remembered is None:
                calibration = self._calibrate(emulator, snapshot)
                cost = _CostModel(
                    base_ms=calibration.base_ms, per_frame_ms=calibration.per_frame_ms
                )
                calibration_frames = calibration.frames
                cost_source = CostSource.CALIBRATED
            else:
                cost = self._remembered
                calibration_frames = 0
                cost_source = CostSource.REUSED
            # Seed the next call's model with this call's, before the sweep can refute
            # it. A calibrated call hands its measurement on; a reused call keeps the
            # one it inherited.
            self._remembered = cost

            horizon = choose_horizon(
                budget_ms=self._budget_ms,
                candidate_count=len(self._actions),
                base_ms=cost.base_ms,
                per_frame_ms=cost.per_frame_ms,
                min_frames=self._min_frames,
                max_frames=self._max_frames,
            )

            sweep_ms, candidates, sweep_steps = self._sweep(emulator, snapshot, horizon)
            steps_taken = calibration_frames + sweep_steps
            passes = 1

            # The horizon above comes from a short calibration or a remembered model,
            # so it can under-load a busy machine. One bounded correction: if the
            # measured sweep overran the budget, re-run at the horizon the measurement
            # says fits, and let the fresh measurement replace the remembered model so
            # the next call does not repeat the mistake.
            if (
                passes < MAX_SWEEP_PASSES
                and sweep_ms > self._budget_ms
                and horizon > self._min_frames
            ):
                tighter = max(
                    self._min_frames, int(horizon * self._budget_ms / sweep_ms)
                )
                if tighter < horizon:
                    overran_ms = sweep_ms
                    overran_horizon = horizon
                    sweep_ms, candidates, extra_steps = self._sweep(
                        emulator, snapshot, tighter
                    )
                    steps_taken += extra_steps
                    horizon = tighter
                    passes += 1
                    self._remembered = self._learn_from_overrun(
                        sweep_ms=overran_ms,
                        horizon=overran_horizon,
                        refuted_horizon=tighter,
                        base_ms=cost.base_ms,
                    )

        # Put the emulator back exactly where it was found. The frame counter is
        # deliberately *not* rewound: `load_state` does not rewind it, and a
        # rollout is not gameplay.
        emulator.load_state(snapshot)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        return RolloutTable(
            tick=tick,
            start_x=start_x,
            horizon=horizon,
            budget_ms=self._budget_ms,
            elapsed_ms=elapsed_ms,
            sweep_ms=sweep_ms,
            base_ms=cost.base_ms,
            per_frame_ms=cost.per_frame_ms,
            calibration_frames=calibration_frames,
            passes=passes,
            cost_source=cost_source,
            steps_taken=steps_taken,
            candidates=tuple(candidates),
        )

    def _learn_from_overrun(
        self,
        *,
        sweep_ms: float,
        horizon: int,
        refuted_horizon: int,
        base_ms: float,
    ) -> _CostModel:
        """The cost model a sweep that overran the budget leaves behind.

        The per-frame term comes from the measurement, but it is floored at the
        cost that reproduces ``refuted_horizon``: the machine already refused a
        horizon, so the model replacing it must not hand the same frames back
        through a rounding error. The fixed term is carried over because a single
        sweep cannot re-isolate it.
        """
        per_frame = remeasured_per_frame_ms(
            sweep_ms=sweep_ms,
            horizon=horizon,
            candidate_count=len(self._actions),
            base_ms=base_ms,
        )
        per_candidate_ms = self._budget_ms / len(self._actions)
        if per_candidate_ms > base_ms:
            per_frame = max(per_frame, (per_candidate_ms - base_ms) / refuted_horizon)
        return _CostModel(base_ms=base_ms, per_frame_ms=per_frame)

    def _calibrate(self, emulator: MarioEmulator, snapshot: object) -> _Calibration:
        """Measure what a savestate load and a candidate frame cost on this machine.

        The three costs are measured separately because they are not the same
        thing: on this machine a load is tens of microseconds, the *first* step
        after a load is reliably about twice a steady step, and the steady steps
        are what dominate a long rollout. Timing them as one average is what makes
        a horizon estimate drift — measured here at 26 frames when the sweep then
        overran the 120 ms budget at that horizon.
        """
        noop = self._calibration_action
        load_total = 0.0
        first_total = 0.0
        steady_total = 0.0
        steady_frames = 0
        frames = 0

        for _ in range(CALIBRATION_LOADS):
            load_started = time.perf_counter()
            emulator.load_state(snapshot)
            load_total += time.perf_counter() - load_started

            first_started = time.perf_counter()
            result = emulator.step(noop)
            first_total += time.perf_counter() - first_started
            frames += 1
            if result.terminated or result.truncated:
                continue

            steady_started = time.perf_counter()
            for _ in range(CALIBRATION_STEADY_STEPS):
                result = emulator.step(noop)
                frames += 1
                steady_frames += 1
                if result.terminated or result.truncated:
                    break
            steady_total += time.perf_counter() - steady_started

        load_ms = load_total / CALIBRATION_LOADS * 1000.0
        first_ms = first_total / CALIBRATION_LOADS * 1000.0
        steady_ms = steady_total / max(1, steady_frames) * 1000.0

        # One candidate costs `load + first + (H - 1) * steady`, which is the
        # two-parameter model `base + H * per_frame` with
        # `base = load + first - steady` and `per_frame = steady`.
        return _Calibration(
            base_ms=max(0.0, load_ms + first_ms - steady_ms),
            per_frame_ms=steady_ms,
            frames=frames,
        )

    def _sweep(
        self,
        emulator: MarioEmulator,
        snapshot: object,
        horizon: int,
    ) -> tuple[float, tuple[CandidateRollout, ...], int]:
        started = time.perf_counter()
        rollouts: list[CandidateRollout] = []
        steps = 0

        for action in self._actions:
            # Before *every* candidate, and always this snapshot: the previous
            # candidate may have ended the episode, and load_state clears that
            # done flag, so the next candidate starts from a live emulator.
            emulator.load_state(snapshot)
            # `info` is refreshed by load_state. It is re-read here, never carried
            # in from before the load.
            start_x = int(dict(emulator.info)[INFO_X_POS])
            x = start_x
            x_max = start_x
            frames = 0
            terminated = False
            truncated = False
            reached_goal = False
            died = False

            for _ in range(horizon):
                result = emulator.step(action)
                frames += 1
                steps += 1
                info = result.info
                x = int(info[INFO_X_POS])
                if x > x_max:
                    x_max = x
                # Break on the episode ending. Stepping once more raises
                # ValueError("cannot step in a done environment! call 'reset'"),
                # and a candidate dying mid-rollout is the signal we are here for.
                if result.terminated or result.truncated:
                    terminated = bool(result.terminated)
                    truncated = bool(result.truncated)
                    reached_goal = is_goal(info)
                    died = is_death(info)
                    break

            final_ram = emulator.ram()
            final_screen_x = int(final_ram.byte(RAM_ADDRESSES["player_x_offset"]))
            over_gap = False
            if died:
                over_gap = died_over_gap(final_screen_x, detect(emulator.frame()).terrain)

            outcome = resolve_outcome(
                reached_goal=reached_goal,
                died=died,
                over_gap=over_gap,
                advanced=x_max > start_x,
            )
            distance, _ = nearest_enemy_gap(
                screen_x=final_screen_x,
                enemies=final_ram.active_enemies(),
            )
            rollouts.append(
                CandidateRollout(
                    action=action,
                    horizon=horizon,
                    frames=frames,
                    start_x=start_x,
                    final_x=x,
                    x_max=x_max,
                    outcome=outcome,
                    risk=risk_cost(outcome, final_enemy_distance=distance),
                    final_enemy_distance=distance,
                    terminated=terminated,
                    truncated=truncated,
                    over_gap=over_gap,
                )
            )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return elapsed_ms, tuple(rollouts), steps


__all__ = [
    "CALIBRATION_LOADS",
    "CALIBRATION_STEADY_STEPS",
    "MAX_SWEEP_PASSES",
    "MIN_PER_FRAME_MS",
    "CandidateRollout",
    "CostSource",
    "RolloutTable",
    "TrajectoryPlanner",
    "choose_horizon",
    "died_over_gap",
    "remeasured_per_frame_ms",
    "resolve_outcome",
]
