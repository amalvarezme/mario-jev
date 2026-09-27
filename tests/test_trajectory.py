"""The rollout planner: real savestates, a measured horizon, and checkable risk.

The live cases here are all built on measured scenarios rather than on a hopeful
walk across the level:

* from the 1-1 start, ``right+B`` runs into the first Goomba at frame 106;
* from 60 frames of ``right+B``, the same collision happens 46 frames later.

That second setup is what makes the interesting tests possible: a rollout that
dies *inside* its horizon, so the planner has to stop on ``terminated`` and still
run the next candidate afterwards.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from mario_jev.config import Settings
from mario_jev.emulator import MarioEmulator, build_actions
from mario_jev.perception import GROUND, GROUND_HIGHLIGHT, SKY, detect
from mario_jev.state import (
    INFO_X_POS,
    HAZARD_OUTCOMES,
    MarioStateBuilder,
    TerminalOutcome,
)
from mario_jev.trajectory import (
    CALIBRATION_LOADS,
    CALIBRATION_STEADY_STEPS,
    FALLS_IN_GAP_RISK,
    GOAL_RISK,
    HITS_ENEMY_RISK,
    MAX_SWEEP_PASSES,
    MIN_PER_FRAME_MS,
    NO_PROGRESS_RISK,
    SAFE_MARGIN_PX,
    CostSource,
    TrajectoryPlanner,
    choose_horizon,
    died_over_gap,
    remeasured_per_frame_ms,
    resolve_outcome,
    risk_cost,
)
# The exact payload ``_calibrate`` returns, imported so a test can inject a cost
# model the way the planner consumes one instead of making the machine lie.
from mario_jev.trajectory.planner import _Calibration

HEIGHT, WIDTH = 240, 256
FLOOR_TOP = 208

#: The action that runs right, and the one that jumps while running. Indices come
#: from the ``simple`` set the package defines, pinned in ``test_emulator.py``.
RUN_RIGHT = 3  # right+B
JUMP_RIGHT = 4  # right+A+B

#: Measured: ``right+B`` for this many frames from the 1-1 start leaves Mario at
#: x=177 with the first Goomba's slot active and closing.
GOOMBA_APPROACH_FRAMES = 60
#: Measured: from there, ``right+B`` dies 46 frames later — inside a 48-frame
#: horizon, which is the whole point of this scenario.
DEATH_FRAME = 46


# -- pure: horizon selection -------------------------------------------------------


def test_choose_horizon_turns_measured_cost_into_a_horizon() -> None:
    # The measured cost of this machine, from the plan: ~0.5 ms of fixed cost plus
    # ~1.1 ms per frame per candidate, over the 7 candidate actions.
    def horizon(budget_ms: float) -> int:
        return choose_horizon(
            budget_ms=budget_ms,
            candidate_count=7,
            base_ms=0.5,
            per_frame_ms=1.1,
            min_frames=6,
            max_frames=48,
        )

    # 7 * (0.5 + 15 * 1.1) = 119 ms, which is the 120 ms budget.
    assert horizon(120.0) == 15
    assert horizon(400.0) == 48  # clamped at the top: 51 frames would be 400 ms
    assert horizon(40.0) == 6  # 4.7 frames -> clamped up to the minimum
    assert horizon(0.1) == 6
    assert horizon(50.0) == 6  # 50 ms buys exactly the minimum horizon

    # More budget never buys fewer frames.
    bought = [horizon(budget) for budget in (0.1, 40.0, 80.0, 120.0, 200.0, 400.0)]
    assert bought == sorted(bought)


def test_choose_horizon_clamps_a_zero_measurement_instead_of_dividing_by_it() -> None:
    # A clock too fast to resolve must not become "infinitely many frames".
    assert (
        choose_horizon(
            budget_ms=120.0,
            candidate_count=7,
            base_ms=0.0,
            per_frame_ms=0.0,
            min_frames=6,
            max_frames=48,
        )
        == 48
    )


def test_choose_horizon_rejects_impossible_inputs() -> None:
    with pytest.raises(ValueError):
        choose_horizon(
            budget_ms=0.0,
            candidate_count=7,
            base_ms=0.5,
            per_frame_ms=1.1,
            min_frames=6,
            max_frames=48,
        )
    with pytest.raises(ValueError):
        choose_horizon(
            budget_ms=120.0,
            candidate_count=0,
            base_ms=0.5,
            per_frame_ms=1.1,
            min_frames=6,
            max_frames=48,
        )
    with pytest.raises(ValueError):
        choose_horizon(
            budget_ms=120.0,
            candidate_count=7,
            base_ms=0.5,
            per_frame_ms=1.1,
            min_frames=0,
            max_frames=48,
        )
    with pytest.raises(ValueError):
        choose_horizon(
            budget_ms=120.0,
            candidate_count=7,
            base_ms=0.5,
            per_frame_ms=1.1,
            min_frames=48,
            max_frames=6,
        )


def test_the_planner_takes_its_budget_from_the_settings_it_is_given() -> None:
    settings = Settings()

    planner = TrajectoryPlanner.from_settings(settings)

    assert planner.budget_ms == settings.rollout_budget_ms == 120.0
    assert planner.horizon_bounds == (
        settings.rollout_min_frames,
        settings.rollout_max_frames,
    ) == (6, 48)
    assert [action.name for action in planner.actions] == [
        action.name for action in build_actions("simple")
    ]


def test_the_planner_rejects_impossible_bounds() -> None:
    with pytest.raises(ValueError):
        TrajectoryPlanner(budget_ms=0.0, min_frames=6, max_frames=48)
    with pytest.raises(ValueError):
        TrajectoryPlanner(budget_ms=120.0, min_frames=0, max_frames=48)
    with pytest.raises(ValueError):
        TrajectoryPlanner(budget_ms=120.0, min_frames=48, max_frames=6)
    with pytest.raises(ValueError):
        TrajectoryPlanner(budget_ms=120.0, min_frames=1, max_frames=1, actions=())


# -- pure: outcomes -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("reached_goal", "died", "over_gap", "advanced", "expected"),
    [
        # The precedence, in order: the goal beats even a death bit on the frame.
        (True, True, True, True, TerminalOutcome.REACHES_GOAL),
        (True, False, False, True, TerminalOutcome.REACHES_GOAL),
        (False, True, True, True, TerminalOutcome.FALLS_IN_GAP),
        (False, True, False, True, TerminalOutcome.HITS_ENEMY),
        # Dying in a pit without having advanced is still a pit death.
        (False, True, True, False, TerminalOutcome.FALLS_IN_GAP),
        (False, False, False, False, TerminalOutcome.NO_PROGRESS),
        (False, False, False, True, TerminalOutcome.LANDS_SAFE),
    ],
)
def test_resolve_outcome_follows_the_contracted_precedence(
    reached_goal: bool,
    died: bool,
    over_gap: bool,
    advanced: bool,
    expected: TerminalOutcome,
) -> None:
    assert (
        resolve_outcome(
            reached_goal=reached_goal, died=died, over_gap=over_gap, advanced=advanced
        )
        is expected
    )


def test_died_over_gap_splits_a_death_on_the_column_under_mario() -> None:
    frame = np.tile(np.array(SKY, dtype=np.uint8), (HEIGHT, WIDTH, 1))
    frame[FLOOR_TOP:, :] = np.array(GROUND, dtype=np.uint8)
    frame[FLOOR_TOP, :] = np.array(GROUND_HIGHLIGHT, dtype=np.uint8)
    frame[FLOOR_TOP:, 120:160] = np.array(SKY, dtype=np.uint8)

    terrain = detect(frame).terrain

    assert terrain.hole_columns == tuple(range(120, 160))
    # Standing over the pit, or with a hand's width of sprite still over it.
    assert died_over_gap(120, terrain) is True
    assert died_over_gap(159, terrain) is True
    # On solid floor just left or right of it.
    assert died_over_gap(104, terrain) is False
    assert died_over_gap(160, terrain) is False


def test_died_over_gap_uses_the_whole_sprite_not_one_column() -> None:
    frame = np.tile(np.array(SKY, dtype=np.uint8), (HEIGHT, WIDTH, 1))
    frame[FLOOR_TOP:, :] = np.array(GROUND, dtype=np.uint8)
    frame[FLOOR_TOP, :] = np.array(GROUND_HIGHLIGHT, dtype=np.uint8)
    # A single-column pit: Mario's front edge hangs over it while his body is on
    # the floor, which is exactly the case a single-column check would miss.
    frame[FLOOR_TOP:, 131] = np.array(SKY, dtype=np.uint8)

    terrain = detect(frame).terrain

    assert terrain.hole_columns == (131,)
    assert died_over_gap(120, terrain) is True
    assert died_over_gap(132, terrain) is False


# -- pure: risk ---------------------------------------------------------------------


def test_risk_orderings() -> None:
    # The ordering the plan fixes, plus the two comparisons that make the enemy
    # distance term mean something.
    pit = risk_cost(TerminalOutcome.FALLS_IN_GAP)
    enemy = risk_cost(TerminalOutcome.HITS_ENEMY)
    idle = risk_cost(TerminalOutcome.NO_PROGRESS)
    close = risk_cost(TerminalOutcome.LANDS_SAFE, final_enemy_distance=0)
    near = risk_cost(TerminalOutcome.LANDS_SAFE, final_enemy_distance=SAFE_MARGIN_PX // 2)
    far = risk_cost(TerminalOutcome.LANDS_SAFE, final_enemy_distance=SAFE_MARGIN_PX)
    clear = risk_cost(TerminalOutcome.LANDS_SAFE)
    goal = risk_cost(TerminalOutcome.REACHES_GOAL)

    assert pit > enemy > idle > close > near > far > goal
    assert far == clear  # beyond the safety margin, the distance stops mattering
    assert goal == GOAL_RISK == 0.0
    assert min(pit, enemy, idle, close, near, far, clear) > goal

    for value in (pit, enemy, idle, close, near, far, clear, goal):
        assert 0.0 <= value <= 1.0


def test_every_hazard_outranks_every_survival_however_close_the_enemy_is() -> None:
    worst_survival = risk_cost(TerminalOutcome.LANDS_SAFE, final_enemy_distance=0)
    best_survival = risk_cost(TerminalOutcome.LANDS_SAFE)
    idle = risk_cost(TerminalOutcome.NO_PROGRESS)

    for outcome in HAZARD_OUTCOMES:
        hazard = risk_cost(outcome)
        assert hazard > idle > worst_survival > best_survival

    assert risk_cost(TerminalOutcome.REACHES_GOAL) < best_survival


def test_the_named_risk_constants_are_the_ones_the_tests_assert() -> None:
    # Guards against someone changing a constant and leaving the ordering test
    # green because it only compared the function to itself.
    assert FALLS_IN_GAP_RISK == 0.9
    assert HITS_ENEMY_RISK == 0.8
    assert NO_PROGRESS_RISK == 0.6
    assert GOAL_RISK == 0.0
    assert risk_cost(TerminalOutcome.FALLS_IN_GAP) == FALLS_IN_GAP_RISK
    assert risk_cost(TerminalOutcome.HITS_ENEMY) == HITS_ENEMY_RISK
    assert risk_cost(TerminalOutcome.NO_PROGRESS) == NO_PROGRESS_RISK


def test_risk_is_a_pure_function_of_its_arguments() -> None:
    # No clock, no emulator: the same arguments always produce the same number, so
    # a journal can be replayed and a human can check the ordering by hand.
    source = inspect.getsource(risk_cost)
    assert "perf_counter" not in source
    assert "time." not in source
    assert risk_cost(TerminalOutcome.LANDS_SAFE, final_enemy_distance=8) == risk_cost(
        TerminalOutcome.LANDS_SAFE, final_enemy_distance=8
    )


# -- live: the scenarios -------------------------------------------------------------


@pytest.fixture
def approaching_goomba(emulator_factory):
    """Mario running at the first Goomba, with the collision inside one horizon."""
    emulator = emulator_factory()
    emulator.reset(seed=0)
    for _ in range(GOOMBA_APPROACH_FRAMES):
        emulator.step(RUN_RIGHT)
    # The measured precondition, asserted so a change in the emulator or in the
    # setup cannot turn the tests below into assertions about nothing.
    assert emulator.info[INFO_X_POS] == 177
    return emulator


def test_stepping_a_finished_episode_raises_which_is_why_the_sweep_breaks(
    emulator_factory,
) -> None:
    # The reason invariant 1 exists, pinned directly: this is what a naive
    # `for _ in range(horizon)` loop hits the first time a candidate dies.
    emulator = emulator_factory()
    emulator.reset(seed=0)
    terminal = None
    for _ in range(200):
        result = emulator.step(RUN_RIGHT)
        if result.terminated or result.truncated:
            terminal = result
            break
    assert terminal is not None

    with pytest.raises(ValueError, match="done environment"):
        emulator.step(RUN_RIGHT)


def test_a_rollout_that_kills_mario_reports_a_hazard_instead_of_raising(
    approaching_goomba: MarioEmulator,
) -> None:
    planner = TrajectoryPlanner(budget_ms=5000.0, min_frames=48, max_frames=48)

    table = planner.plan(approaching_goomba)

    dying = table.by_name("right+B")
    assert dying.outcome is TerminalOutcome.HITS_ENEMY
    assert dying.outcome in HAZARD_OUTCOMES
    assert dying.frames == DEATH_FRAME == 46  # stopped early, inside the horizon
    assert dying.horizon == 48
    assert dying.terminated is True
    assert dying.truncated is False
    assert dying.over_gap is False  # measured: the floor under him is solid there
    assert dying.final_enemy_distance == 0
    assert dying.dx == 138
    assert dying.risk == HITS_ENEMY_RISK == 0.8
    assert dying.died is True


def test_the_live_table_orders_the_risks_the_way_the_formula_says(
    approaching_goomba: MarioEmulator,
) -> None:
    # Every kind of ending on one scene: a death, survivals that end on top of the
    # Goomba, and a retreat that ends far from it.
    planner = TrajectoryPlanner(budget_ms=5000.0, min_frames=48, max_frames=48)

    table = planner.plan(approaching_goomba)

    outcomes = {candidate.outcome for candidate in table.candidates}
    assert outcomes <= set(TerminalOutcome)
    assert TerminalOutcome.HITS_ENEMY in outcomes
    assert TerminalOutcome.LANDS_SAFE in outcomes
    assert all(0.0 <= candidate.risk <= 1.0 for candidate in table.candidates)

    # Death costs more than any survival, and the survival that ends on top of the
    # Goomba (distance 0) costs more than the one that ends far from it.
    assert table.by_name("right+B").risk == HITS_ENEMY_RISK
    assert table.by_name("right+B").risk > table.by_name("right+A").risk
    assert table.by_name("right+A").risk > table.by_name("left").risk
    assert table.by_name("right+A").final_enemy_distance == 0
    assert table.by_name("left").final_enemy_distance == 141


def test_the_candidate_after_a_terminated_one_still_runs(
    approaching_goomba: MarioEmulator,
) -> None:
    # Invariant 2: `load_state` clears the wrapper's done flag, so the sweep
    # continues. Without that, this candidate would raise instead of rolling.
    planner = TrajectoryPlanner(budget_ms=5000.0, min_frames=48, max_frames=48)
    table = planner.plan(approaching_goomba)

    names = [candidate.name for candidate in table.candidates]
    dying_index = names.index("right+B")
    following = table.candidates[dying_index + 1]

    assert dying_index < len(table.candidates) - 1, "the dying candidate cannot be last"
    assert following.name == "right+A+B"
    assert following.frames == table.horizon
    assert following.terminated is False
    assert following.outcome is not TerminalOutcome.HITS_ENEMY

    # And the emulator itself is still live: the sweep's last act is restoring the
    # snapshot it started from, which also clears the ended-episode state.
    with_goomba_ahead = approaching_goomba.info[INFO_X_POS]
    result = approaching_goomba.step(0)
    assert result.terminated is False
    assert approaching_goomba.info[INFO_X_POS] >= with_goomba_ahead


def test_a_full_sweep_at_the_1_1_start_reports_legal_outcomes_and_risks(
    emulator: MarioEmulator,
) -> None:
    planner = TrajectoryPlanner(budget_ms=5000.0, min_frames=8, max_frames=8)

    table = planner.plan(emulator)

    assert table.horizon == 8
    assert table.passes == 1
    assert [candidate.name for candidate in table.candidates] == list(
        emulator.action_names
    )
    assert len(table.candidates) == 7

    for candidate in table.candidates:
        assert candidate.outcome in set(TerminalOutcome)
        assert 0.0 <= candidate.risk <= 1.0
        assert candidate.frames == 8
        assert candidate.terminated is False
        assert candidate.truncated is False

    # Nothing hazards Mario in the first eight frames of 1-1: no enemy slot is even
    # active yet, and there is no pit on the first screen.
    assert not any(
        candidate.outcome in HAZARD_OUTCOMES for candidate in table.candidates
    )

    # Measured: the level start is a standstill, so the action set separates into
    # "advanced", "did nothing" and "walked backwards".
    assert table.by_name("NOOP").dx == 0
    assert table.by_name("NOOP").outcome is TerminalOutcome.NO_PROGRESS
    assert table.by_name("A").dx == 0
    assert table.by_name("left").dx == -3
    assert table.by_name("right+B").dx == 3
    assert table.by_name("right").dx == 2
    assert table.by_name("right+B").risk == 0.2

    # The best candidate is the lowest-risk one, and among equals the one that
    # advanced furthest — measured at this horizon: running right.
    assert table.best.name == "right+B"
    assert table.best.risk == min(candidate.risk for candidate in table.candidates)


def test_the_sweep_produces_exactly_the_model_contract(emulator: MarioEmulator) -> None:
    planner = TrajectoryPlanner(budget_ms=5000.0, min_frames=8, max_frames=8)
    table = planner.plan(emulator)
    builder = MarioStateBuilder()
    state = builder.observe(
        detect(emulator.frame(), tick=emulator.tick), emulator.ram(), emulator.info
    )

    payload = json.loads(json.dumps(state.with_candidates(table.assessments).to_jev_state()))

    assert set(payload["candidates"]) == set(emulator.action_names)
    for entry in payload["candidates"].values():
        assert set(entry) == {"dx", "outcome", "risk", "final_x"}
        assert entry["outcome"] in {outcome.value for outcome in TerminalOutcome}
        assert 0.0 <= entry["risk"] <= 1.0
        assert isinstance(entry["final_x"], int)
    # The player's absolute x and the candidates' final_x are the same quantity.
    assert payload["player"]["x"] == table.start_x
    assert payload["candidates"]["right+B"]["final_x"] == 43


def test_the_table_reports_the_horizon_and_the_measured_cost(emulator: MarioEmulator) -> None:
    planner = TrajectoryPlanner(budget_ms=5000.0, min_frames=8, max_frames=8)

    table = planner.plan(emulator)
    journal = table.as_dict()

    assert table.horizon == 8
    assert table.budget_ms == 5000.0
    assert table.sweep_ms > 0.0
    assert table.elapsed_ms >= table.sweep_ms
    assert table.per_frame_ms > 0.0
    assert table.base_ms >= 0.0
    assert journal["horizon"] == 8
    assert journal["sweep_ms"] > 0.0
    assert journal["elapsed_ms"] > 0.0
    assert journal["tick"] == 0
    assert len(journal["candidates"]) == 7


def test_the_planner_leaves_the_emulator_unmoved_and_does_not_rewind_the_tick(
    emulator: MarioEmulator,
) -> None:
    start_x = emulator.info[INFO_X_POS]
    start_tick = emulator.tick
    planner = TrajectoryPlanner(budget_ms=5000.0, min_frames=10, max_frames=10)

    table = planner.plan(emulator)

    assert table.passes == 1
    assert table.calibration_frames == CALIBRATION_LOADS * (1 + CALIBRATION_STEADY_STEPS)
    assert table.steps_taken == table.calibration_frames + 7 * 10
    # The frame counter is not rewound, and it is not advanced by anything except
    # the steps the planner actually took.
    assert emulator.tick == start_tick + table.steps_taken
    # The position, and therefore the state the tick describes, is exactly restored.
    assert emulator.info[INFO_X_POS] == start_x
    assert emulator.ram().player_x_in_level == 0

    # ... and the emulator is still usable, not stuck in a finished episode.
    result = emulator.step(1)
    assert result.tick == start_tick + table.steps_taken + 1


def test_the_horizon_shrinks_under_a_tiny_budget_and_adapts_to_a_large_one(
    emulator: MarioEmulator,
) -> None:
    # Invariant 5: every timing test runs unpaced. The `emulator` fixture already
    # builds with real_time=False, which is what makes these numbers mean anything.
    assert emulator.pacer.enabled is False

    tiny = TrajectoryPlanner(budget_ms=0.1, min_frames=6, max_frames=48)
    tiny_table = tiny.plan(emulator)

    assert tiny_table.horizon == 6  # clamped up to the minimum, never below it
    assert tiny_table.per_frame_ms > 0.0
    assert tiny_table.sweep_ms > 0.0
    assert tiny_table.elapsed_ms >= tiny_table.sweep_ms

    horizons = []
    for budget in (0.1, 40.0, 120.0, 400.0):
        planner = TrajectoryPlanner(budget_ms=budget, min_frames=6, max_frames=48)
        table = planner.plan(emulator)
        horizons.append(table.horizon)

        assert 6 <= table.horizon <= 48
        assert table.budget_ms == budget
        assert table.per_frame_ms > 0.0 and table.sweep_ms > 0.0
        # The horizon is exactly what this run's own measurements imply, or smaller
        # when the second pass had to correct an overrunning sweep.
        implied = choose_horizon(
            budget_ms=budget,
            candidate_count=len(planner.actions),
            base_ms=table.base_ms,
            per_frame_ms=table.per_frame_ms,
            min_frames=6,
            max_frames=48,
        )
        if table.passes == 1:
            assert table.horizon == implied
        else:
            assert table.passes == 2
            assert table.horizon < implied
            assert table.sweep_ms > 0.0

    assert horizons[0] == 6
    assert horizons[-1] >= horizons[0]


def test_a_huge_budget_buys_the_maximum_horizon(emulator: MarioEmulator) -> None:
    planner = TrajectoryPlanner(budget_ms=10_000_000.0, min_frames=6, max_frames=48)

    table = planner.plan(emulator)

    assert table.horizon == 48
    assert table.passes == 1
    assert all(candidate.frames == 48 for candidate in table.candidates)


# -- warm start: the cost model is measured once, then remembered --------------------


def test_remeasured_per_frame_ms_recovers_the_cost_the_sweep_just_measured() -> None:
    # The model is n * (base + horizon * per_frame). One sweep cannot separate the
    # two terms, but it can re-measure the per-frame one once base is subtracted.
    # 7 candidates * (2.0 + 10 * 0.9) = 77 ms.
    assert remeasured_per_frame_ms(
        sweep_ms=77.0, horizon=10, candidate_count=7, base_ms=2.0
    ) == pytest.approx(0.9)

    # A clock that resolves to nothing cannot turn into a zero-cost model.
    assert (
        remeasured_per_frame_ms(
            sweep_ms=0.0, horizon=10, candidate_count=7, base_ms=0.0
        )
        == MIN_PER_FRAME_MS
    )

    with pytest.raises(ValueError):
        remeasured_per_frame_ms(sweep_ms=1.0, horizon=0, candidate_count=7, base_ms=0.0)
    with pytest.raises(ValueError):
        remeasured_per_frame_ms(sweep_ms=1.0, horizon=10, candidate_count=0, base_ms=0.0)


def test_the_first_plan_calibrates_and_the_second_reuses_that_measurement(
    emulator: MarioEmulator,
) -> None:
    # A fixed horizon keeps this test about the cost source alone: the machine can
    # be as slow or as fast as it likes without ever triggering a correction.
    planner = TrajectoryPlanner(budget_ms=5000.0, min_frames=10, max_frames=10)

    first = planner.plan(emulator)
    tick_after_first = emulator.tick
    second = planner.plan(emulator)

    # The first call paid for the calibration and says so.
    assert first.cost_source is CostSource.CALIBRATED
    assert first.calibration_frames == CALIBRATION_LOADS * (1 + CALIBRATION_STEADY_STEPS)

    # The second call skipped it: no calibration frames, the same measured cost,
    # and the honest source value rather than an inference.
    assert second.cost_source is CostSource.REUSED
    assert second.calibration_frames == 0
    assert second.base_ms == first.base_ms
    assert second.per_frame_ms == first.per_frame_ms
    assert second.horizon == first.horizon == 10

    # The frame counter moves by exactly the sweep steps, so nothing was measured
    # behind the caller's back.
    assert second.steps_taken == 7 * second.horizon
    assert emulator.tick == tick_after_first + second.steps_taken

    # The journal records the source rather than a value a reader has to guess at.
    assert second.as_dict()["cost_source"] == str(CostSource.REUSED)
    assert {str(source) for source in CostSource} == {"calibrated", "reused"}


def test_a_remembered_cost_the_machine_contradicts_is_replaced_and_corrected(
    emulator: MarioEmulator, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Inject the cost instead of slowing the machine down: this calibration claims
    # a frame costs microseconds, so the planner predictably picks the maximum
    # horizon and the real machine predictably overruns the budget.
    injected = _Calibration(base_ms=0.0, per_frame_ms=MIN_PER_FRAME_MS, frames=0)
    monkeypatch.setattr(
        TrajectoryPlanner, "_calibrate", lambda self, emulator, snapshot: injected
    )

    planner = TrajectoryPlanner(budget_ms=120.0, min_frames=6, max_frames=48)
    overran = planner.plan(emulator)

    assert overran.cost_source is CostSource.CALIBRATED
    assert overran.passes == MAX_SWEEP_PASSES == 2  # the machine refuted it
    # The injected cost implies the maximum horizon; the correction pulled it down,
    # and the table reports the model that was used rather than the repair.
    assert overran.horizon < 48
    assert overran.base_ms == 0.0
    assert overran.per_frame_ms == MIN_PER_FRAME_MS

    # The fresh measurement - not the injected one - is what the next call uses.
    # base is carried over because one sweep cannot re-isolate it; per_frame is
    # re-measured, so the absurd injected value is gone.
    adaptive = planner.plan(emulator)

    assert adaptive.cost_source is CostSource.REUSED
    assert adaptive.base_ms == 0.0
    assert adaptive.per_frame_ms > MIN_PER_FRAME_MS
    # A horizon the machine already refused is not reused as-is: whatever this
    # repaired model predicts is at or below the horizon the overrun forced.
    implied_next = choose_horizon(
        budget_ms=120.0,
        candidate_count=7,
        base_ms=adaptive.base_ms,
        per_frame_ms=adaptive.per_frame_ms,
        min_frames=6,
        max_frames=48,
    )
    assert implied_next <= overran.horizon
    assert adaptive.horizon <= overran.horizon


def test_a_horizon_that_overran_the_budget_is_not_sticky(
    emulator: MarioEmulator, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Same injection, but the assertion is about the direction of the correction:
    # an overrun must lower the next horizon rather than keep predicting from a
    # cost the machine already contradicted.
    injected = _Calibration(base_ms=0.0, per_frame_ms=MIN_PER_FRAME_MS, frames=0)
    monkeypatch.setattr(
        TrajectoryPlanner, "_calibrate", lambda self, emulator, snapshot: injected
    )

    planner = TrajectoryPlanner(budget_ms=120.0, min_frames=6, max_frames=48)
    overran = planner.plan(emulator)
    assert overran.passes == 2  # evidence that the budget was actually overrun
    assert overran.horizon < 48

    after = planner.plan(emulator)

    assert after.cost_source is CostSource.REUSED
    assert after.horizon <= overran.horizon

    # And it stays bounded on later calls rather than drifting back up.
    for _ in range(3):
        later = planner.plan(emulator)
        assert later.cost_source is CostSource.REUSED
        assert later.horizon <= overran.horizon


def test_steady_state_reuses_the_cost_and_keeps_each_sweep_inside_the_budget(
    emulator: MarioEmulator,
) -> None:
    # The point of the change on a live machine: the calibration is paid once, and
    # the calls after it are one sweep each, sized from what the last call cost.
    # The budget carries headroom over the maximum horizon so the assertion is
    # about the warm start rather than about this machine's load at this instant.
    planner = TrajectoryPlanner(budget_ms=600.0, min_frames=6, max_frames=48)

    tables = [planner.plan(emulator) for _ in range(4)]

    assert tables[0].cost_source is CostSource.CALIBRATED
    assert tables[0].calibration_frames == CALIBRATION_LOADS * (1 + CALIBRATION_STEADY_STEPS)
    first_overhead_ms = tables[0].elapsed_ms - tables[0].sweep_ms

    for table in tables[1:]:
        assert table.cost_source is CostSource.REUSED
        assert table.calibration_frames == 0
        # Warm start is only an improvement if the sweep it sizes still fits.
        assert table.sweep_ms <= planner.budget_ms
        # And a steady-state call is one sweep plus the bookkeeping around it, not
        # a sweep plus the calibration the first call already paid for.
        assert table.elapsed_ms - table.sweep_ms <= first_overhead_ms
