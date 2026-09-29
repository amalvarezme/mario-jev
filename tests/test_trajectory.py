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
import math
from pathlib import Path

import numpy as np
import pytest

from mario_jev.config import Settings
from mario_jev.emulator import MarioEmulator, build_actions
from mario_jev.journal import read_journal
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
    RolloutTable,
    TrajectoryPlanner,
    choose_horizon,
    died_over_gap,
    remeasured_per_frame_ms,
    resolve_outcome,
    risk_cost,
)
# The exact payload ``_calibrate`` returns, imported so a test can inject a cost
# model the way the planner consumes one instead of making the machine lie. The
# module itself is imported too, so the recovery test can replace the pure helper
# where the planner looks it up and prove the relaxation really calls it.
from mario_jev.trajectory import planner as planner_module
from mario_jev.trajectory.planner import (
    COST_RECOVERY,
    STALL_MAX_HORIZON,
    STALL_MIN_HORIZON,
    _Calibration,
)

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

#: The recorded live run, and the frame of it at which the agent stood still
#: against the wall. ``artifacts/`` is gitignored, so the test that reads this
#: says how to produce the file rather than skipping when it is absent.
JOURNAL_NAME = "eval-1-1.jsonl"
#: Measured: replaying the recorded actions for this many frames puts Mario at
#: x=722, the state whose rollout carried no signal.
STALL_TICK = 420
STALL_X = 722


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

    # The second call skipped it: no calibration frames, the cost model the first
    # call left behind, and the honest source value rather than an inference.
    assert second.cost_source is CostSource.REUSED
    assert second.calibration_frames == 0
    assert second.base_ms == first.base_ms
    # Not necessarily the same number, and deliberately so: a reused model is
    # carried *and* relaxed one COST_RECOVERY step towards what the sweep measured
    # on every sweep that fits the budget, which the first call's sweep did. It can
    # only ever be lower, never higher — raising belongs to the overrun path.
    # "Reused" records that this call did not calibrate, not that the numbers are
    # frozen, which is why the source is recorded rather than inferred from them.
    assert (
        first.per_frame_ms * COST_RECOVERY
        <= second.per_frame_ms
        <= first.per_frame_ms
    )
    assert second.horizon == first.horizon == 10

    # The frame counter moves by exactly the sweep steps, so nothing was measured
    # behind the caller's back.
    assert second.steps_taken == 7 * second.horizon
    assert emulator.tick == tick_after_first + second.steps_taken

    # The journal records the source rather than a value a reader has to guess at.
    assert second.as_dict()["cost_source"] == str(CostSource.REUSED)
    assert CostSource.FIXED.value == "fixed"
    assert {str(source) for source in CostSource} == {"calibrated", "reused", "fixed"}


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

    # And it stays bounded on later calls rather than jumping back to a cost the
    # machine already contradicted. Bounded is not frozen: every sweep that fits
    # relaxes the model one COST_RECOVERY step back towards what that sweep just
    # measured, so the horizon does recover — at a rate, never in a jump. The
    # bound below *is* that rate: after k relaxations the model is still at least
    # COST_RECOVERY**k of the one the overrun left behind (raising can only push
    # it back up), so the frames it buys are at most that much more than the
    # corrected call itself chose. The 48 the machine refused stay out of reach.
    for relaxations in range(1, 4):
        later = planner.plan(emulator)
        assert later.cost_source is CostSource.REUSED
        assert later.horizon < 48
        assert later.horizon <= math.floor(
            (after.horizon + 1) / COST_RECOVERY**relaxations
        )


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


# -- the sweep is a simulated future, so it must not consume the real timeline ------


def _counting_sleeps(monkeypatch) -> list[float]:
    """Count ``env``'s sleeps instead of performing them.

    The pacer reaches ``time.sleep`` through the ``time`` module that ``env``
    imported, so replacing that attribute observes exactly the sleeps a paced
    emulator would have taken, deterministically and without real waiting.
    """
    import mario_jev.emulator.env as env_module

    calls: list[float] = []
    monkeypatch.setattr(env_module.time, "sleep", calls.append)
    return calls


def test_the_sweep_does_not_pace_the_simulated_future(
    emulator_factory, monkeypatch
) -> None:
    # A paced emulator sleeps once per frame, so a sweep that is *not* suspended
    # issues roughly one sleep per emulator step. With the suspension in place
    # ``plan()`` must issue no sleeps at all, which is what the factor of 4 leaves
    # room for: it fails loudly for pacing (~1 sleep/step) while tolerating a few
    # stray sleeps if a future change ever introduces one deliberately.
    sleeps = _counting_sleeps(monkeypatch)
    emulator = emulator_factory(real_time=True)
    emulator.reset(seed=0)
    planner = TrajectoryPlanner(budget_ms=120.0, min_frames=6, max_frames=48)

    table = planner.plan(emulator)

    assert table.steps_taken > 0
    assert len(sleeps) < table.steps_taken / 4


def test_the_unpaced_sweep_is_pinned_to_the_recorded_result(
    emulator: MarioEmulator,
) -> None:
    # Recorded from the code before the suspension was added, with real_time=False.
    # The suspension must not change a single frame of the sweep itself.
    planner = TrajectoryPlanner(budget_ms=5000.0, min_frames=8, max_frames=8)

    table = planner.plan(emulator)

    assert table.tick == 0
    assert table.horizon == 8
    assert table.calibration_frames == 12
    assert table.steps_taken == 68
    assert table.passes == 1
    assert table.start_x == 40
    assert {
        candidate.name: (candidate.outcome.value, candidate.dx, candidate.frames)
        for candidate in table.candidates
    } == {
        "NOOP": ("no_progress", 0, 8),
        "right": ("lands_safe", 2, 8),
        "right+A": ("lands_safe", 2, 8),
        "right+B": ("lands_safe", 3, 8),
        "right+A+B": ("lands_safe", 2, 8),
        "A": ("no_progress", 0, 8),
        "left": ("no_progress", -3, 8),
    }


def test_the_sweep_cost_does_not_depend_on_pacing(emulator_factory) -> None:
    # Coarse sanity check only: this compares two wall-clock measurements, so it
    # can be disturbed by machine load. The deterministic pin is
    # ``test_the_sweep_does_not_pace_the_simulated_future`` above, which counts
    # sleeps rather than seconds. Both plans run the same stage at the same tick
    # with the same fixed horizon, so removing the pacing should bring them close.
    def elapsed_ms(real_time: bool) -> float:
        emulator = emulator_factory(real_time=real_time)
        emulator.reset(seed=0)
        planner = TrajectoryPlanner(budget_ms=5000.0, min_frames=8, max_frames=8)
        return planner.plan(emulator).elapsed_ms

    paced = elapsed_ms(True)
    unpaced = elapsed_ms(False)

    assert paced < unpaced * 2


# -- the stall escalation ------------------------------------------------------------
#
# The defect these tests describe is measured, not hypothesised. In the recorded
# 1-1 run the agent reached x_max=724 by frame 271 and then stood still for 600
# frames against a 48 px wall: at its horizon of 8 frames every candidate came
# back ``no_progress`` at an identical risk of 0.6, so ``table.best`` was decided
# by the dx tie-break and landed on NOOP. Replaying that run to the stalled
# savestate and re-planning at fixed horizons shows where the escape becomes
# visible: nothing at 8, 12 or 16, two ``lands_safe`` candidates at 24, and the
# same two at dx=+40/+55 by 48. The rollout was blind, and it was blind by budget.


def test_min_horizon_none_reproduces_the_recorded_sweep_exactly(
    emulator: MarioEmulator,
) -> None:
    # The regression requirement, stated as an assertion: passing ``min_horizon``
    # explicitly as ``None`` must leave the sweep byte-for-byte what it was before
    # the parameter existed. These are the numbers
    # ``test_the_unpaced_sweep_is_pinned_to_the_recorded_result`` pins for the
    # call with no keyword at all.
    planner = TrajectoryPlanner(budget_ms=5000.0, min_frames=8, max_frames=8)

    table = planner.plan(emulator, min_horizon=None)

    assert table.horizon == 8
    assert table.escalated is False
    assert table.calibration_frames == 12
    assert table.steps_taken == 68
    assert table.passes == 1
    assert table.start_x == 40
    assert {
        candidate.name: (candidate.outcome.value, candidate.dx, candidate.frames)
        for candidate in table.candidates
    } == {
        "NOOP": ("no_progress", 0, 8),
        "right": ("lands_safe", 2, 8),
        "right+A": ("lands_safe", 2, 8),
        "right+B": ("lands_safe", 3, 8),
        "right+A+B": ("lands_safe", 2, 8),
        "A": ("no_progress", 0, 8),
        "left": ("no_progress", -3, 8),
    }


def test_a_floor_above_the_budgets_choice_raises_the_horizon_and_is_recorded(
    emulator: MarioEmulator,
) -> None:
    # A budget this small buys the minimum horizon and nothing more, which is the
    # blind end of the measured table.
    planner = TrajectoryPlanner(budget_ms=0.1, min_frames=6, max_frames=48)

    unescalated = planner.plan(emulator)
    escalated = planner.plan(emulator, min_horizon=STALL_MIN_HORIZON)

    assert unescalated.horizon == 6
    assert unescalated.escalated is False
    assert escalated.horizon == STALL_MIN_HORIZON
    assert escalated.escalated is True
    # The frames were really stepped, so the overspend is measurable and not just
    # asserted: 7 candidates at 24 frames each.
    assert escalated.steps_taken == 7 * STALL_MIN_HORIZON


def test_a_floor_the_budget_had_already_cleared_is_not_an_escalation(
    emulator: MarioEmulator,
) -> None:
    # Nothing was spent beyond what the budget itself chose, so nothing is
    # recorded as spent beyond it — even though a floor was asked for.
    planner = TrajectoryPlanner(budget_ms=5000.0, min_frames=32, max_frames=32)

    table = planner.plan(emulator, min_horizon=STALL_MIN_HORIZON)

    assert table.horizon == 32
    assert table.escalated is False


def test_the_escalated_horizon_stops_at_the_hard_cap(emulator: MarioEmulator) -> None:
    # A floor is a request, not a mandate: escalation may buy frames the budget
    # did not, but never more than STALL_MAX_HORIZON of them.
    planner = TrajectoryPlanner(budget_ms=0.1, min_frames=6, max_frames=8)

    table = planner.plan(emulator, min_horizon=1000)

    assert STALL_MAX_HORIZON == 96
    assert table.horizon == 96
    assert table.escalated is True


def test_a_floor_below_the_minimum_cannot_push_the_horizon_under_it(
    emulator: MarioEmulator,
) -> None:
    # The clamp holds at the bottom too: `min_horizon` can only ever raise the
    # horizon, never lower it past the planner's own minimum.
    planner = TrajectoryPlanner(budget_ms=0.1, min_frames=12, max_frames=48)

    table = planner.plan(emulator, min_horizon=3)

    assert table.horizon == 12
    assert table.escalated is False


def test_as_dict_carries_escalated(emulator: MarioEmulator) -> None:
    # The journal serialises the table through as_dict(), so a fact the table
    # holds but the dict drops is a fact the evidence never gets.
    planner = TrajectoryPlanner(budget_ms=0.1, min_frames=6, max_frames=48)

    plain = planner.plan(emulator).as_dict()
    escalated = planner.plan(emulator, min_horizon=STALL_MIN_HORIZON).as_dict()

    assert plain["escalated"] is False
    assert escalated["escalated"] is True
    # It survives the trip through JSON, which is how the journal stores it.
    assert json.loads(json.dumps(escalated))["escalated"] is True


def test_the_escalated_plan_sees_the_escape_the_recorded_run_was_blind_to(
    emulator_factory,
) -> None:
    """The behavioural proof, on the real emulator at the real stalled state.

    The recorded run's own actions are replayed to the frame where it stalled,
    and the two plans are run from that exact savestate: one at the horizon the
    run actually used, one escalated to STALL_MIN_HORIZON.

    The un-escalated side is pinned to the horizon the *journal* recorded at that
    tick rather than to whatever the budget would buy here, and deliberately so:
    on a machine fast enough, 120 ms buys 23-24 frames at this state and the
    escape is already visible, so a default-bounds plan would assert nothing and
    would assert it differently on every machine. The blindness being reproduced
    is the recorded one.
    """
    journal_path = Path(__file__).resolve().parent.parent / "artifacts" / JOURNAL_NAME
    if not journal_path.exists():
        # Not a skip. The evidence is reproducible in one command, and a silent
        # skip here would quietly retire the only test that proves the fix does
        # anything at all.
        pytest.fail(
            f"the recorded run {journal_path} is missing (artifacts/ is gitignored). "
            "Produce it with: "
            "scripts/jev-env.sh .venv/bin/python -m mario_jev run --fast --max-ticks 900"
        )

    journal = read_journal(journal_path)
    recorded = journal.ticks[STALL_TICK]
    recorded_horizon = int(recorded["rollout"]["horizon"])
    actions = [tick["decision"]["action"] for tick in journal.ticks]

    emulator = emulator_factory()
    emulator.reset(seed=0)
    names = emulator.action_names
    for action in actions[:STALL_TICK]:
        emulator.step(names.index(action))
    # The replay landed where the run recorded it landing; without this the two
    # plans below would be assertions about some other state entirely.
    assert emulator.info[INFO_X_POS] == STALL_X

    blind = TrajectoryPlanner(
        budget_ms=5000.0, min_frames=recorded_horizon, max_frames=recorded_horizon
    ).plan(emulator)
    escalated = TrajectoryPlanner(budget_ms=0.1, min_frames=6, max_frames=48).plan(
        emulator, min_horizon=STALL_MIN_HORIZON
    )

    assert blind.horizon == recorded_horizon
    assert escalated.horizon == STALL_MIN_HORIZON
    assert escalated.escalated is True

    blind_risks = {candidate.risk for candidate in blind.candidates}
    assert all(
        candidate.outcome is TerminalOutcome.NO_PROGRESS for candidate in blind.candidates
    ), (
        f"at horizon {blind.horizon} the recorded run saw no way out; this plan saw "
        f"{[(c.name, c.outcome.value, c.dx) for c in blind.candidates]}"
    )
    assert len(blind_risks) == 1, (
        f"at horizon {blind.horizon} every candidate carried the same risk, so "
        f"`best` fell to the dx tie-break; this plan carried {sorted(blind_risks)}"
    )

    escapes = [
        candidate
        for candidate in escalated.candidates
        if candidate.outcome is TerminalOutcome.LANDS_SAFE and candidate.dx > 0
    ]
    assert escapes, (
        f"escalating from {blind.horizon} to {escalated.horizon} frames must make the "
        "running jump visible; it showed "
        f"{[(c.name, c.outcome.value, c.dx) for c in escalated.candidates]}"
    )
    assert min(candidate.risk for candidate in escapes) < min(blind_risks), (
        f"the escape found at horizon {escalated.horizon} must be measurably safer "
        f"than anything at horizon {blind.horizon}: "
        f"{sorted({c.risk for c in escalated.candidates})} against {sorted(blind_risks)}"
    )
    assert escalated.best.dx > 0


# -- the cost model recovers ---------------------------------------------------------
#
# The defect these tests describe is measured, and it is measured in the runs'
# own journals. The remembered cost only ever went up: ``eval-after4`` climbed
# 0.6896 -> 0.855 -> 1.71 and settled at its maximum of 2.9191 ms/frame for the
# rest of its 900 ticks, ``eval-1-1`` climbed 0.693 -> 1.1954 and settled at
# 2.0919, and both spent 899 of 900 ticks on a reused model. The same journals'
# ``elapsed_ms / steps_taken`` puts the machine's real cost at a p50 of 1.3048
# and 1.2408 ms, so the memory sat at about 2.2x the truth. At 120 ms over 7
# candidates that is a horizon of 6 and 7 frames where the real cost pays for 13:
# the run was blind by budget, and the budget was set by a worst moment it never
# recovered from.


#: The recorded inflation, and the machine under it. Taken from ``eval-after4``'s
#: last tick (``base_ms`` 0.0428, ``per_frame_ms`` 2.9191) and from that same
#: journal's median ``elapsed_ms / steps_taken``.
INFLATED_PER_FRAME_MS = 2.9191
RECORDED_BASE_MS = 0.0428
REAL_PER_FRAME_MS = 1.3048
#: Measured from the trace below: 27 fitting ticks to undo the recorded
#: inflation, which is the number ``COST_RECOVERY``'s docstring justifies itself
#: with and the same order as the loop's own ``STALL_FRAMES = 30``.
RECOVERY_TICKS = 27


def _injected_calibration(
    monkeypatch: pytest.MonkeyPatch, *, per_frame_ms: float, base_ms: float
) -> None:
    """Start the planner from a chosen cost model instead of from this machine."""
    injected = _Calibration(base_ms=base_ms, per_frame_ms=per_frame_ms, frames=0)
    monkeypatch.setattr(
        TrajectoryPlanner, "_calibrate", lambda self, emulator, snapshot: injected
    )


def _simulated_machine(
    monkeypatch: pytest.MonkeyPatch,
    *,
    per_frame_ms: float,
    base_ms: float,
    candidate_count: int = 7,
) -> None:
    """Make the sweep's *cost* arithmetic instead of wall clock.

    Everything else in ``plan()`` still runs against the real emulator: the
    savestate is dumped and restored, the horizon is chosen from the remembered
    model, the correction pass fires or does not, and the table is built. Only
    the cost of the rollouts is simulated, and deliberately: a recovery trace
    measured against this machine's load at this instant would be a different
    trace on every run and would therefore pin nothing.
    """

    def sweep(self, emulator, snapshot, horizon):  # type: ignore[no-untyped-def]
        sweep_ms = candidate_count * (base_ms + horizon * per_frame_ms)
        return sweep_ms, (), candidate_count * horizon

    monkeypatch.setattr(TrajectoryPlanner, "_sweep", sweep)


def _recovery_trace(
    emulator: MarioEmulator, monkeypatch: pytest.MonkeyPatch, *, ticks: int
) -> list[tuple[float, int]]:
    """The recorded inflation, put on a machine that really costs 1.3048 ms/frame."""
    _injected_calibration(
        monkeypatch, per_frame_ms=INFLATED_PER_FRAME_MS, base_ms=RECORDED_BASE_MS
    )
    _simulated_machine(
        monkeypatch, per_frame_ms=REAL_PER_FRAME_MS, base_ms=RECORDED_BASE_MS
    )
    planner = TrajectoryPlanner(budget_ms=120.0, min_frames=6, max_frames=48)
    tables = [planner.plan(emulator) for _ in range(ticks)]
    # Nothing here may overrun: this is the fitting path, and the correction pass
    # firing would mean the trace below is about the wrong mechanism.
    assert all(table.passes == 1 for table in tables)
    assert all(table.sweep_ms <= planner.budget_ms for table in tables)
    return [(table.per_frame_ms, table.horizon) for table in tables]


def test_a_fitting_sweep_pulls_the_remembered_cost_back_down(
    emulator: MarioEmulator, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = [per_frame for per_frame, _ in _recovery_trace(emulator, monkeypatch, ticks=40)]

    # The first tick is sized by the model it inherited: the recorded inflation.
    assert trace[0] == pytest.approx(INFLATED_PER_FRAME_MS)
    # The second is exactly one COST_RECOVERY step below it, so the relaxation is
    # the rate the constant names rather than some other rate that also falls.
    assert trace[1] == pytest.approx(INFLATED_PER_FRAME_MS * COST_RECOVERY)
    # It falls, tick after tick, and it stops at what the machine really costs.
    assert all(later <= earlier for earlier, later in zip(trace, trace[1:]))
    assert trace[-1] == pytest.approx(REAL_PER_FRAME_MS)
    reached = next(
        index
        for index, value in enumerate(trace)
        if value == pytest.approx(REAL_PER_FRAME_MS)
    )
    assert reached == RECOVERY_TICKS
    # The floor is the measurement, so the recovery stops at the machine and does
    # not carry on past it into a model that would overrun on the next tick.
    assert min(trace) == pytest.approx(REAL_PER_FRAME_MS)


def test_a_fitting_sweep_whose_measurement_is_higher_leaves_the_memory_alone(
    emulator: MarioEmulator, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Remembered 1.0 ms/frame on a machine that really costs 3.0, with a budget
    # large enough that the slow sweep still fits inside it. The relaxation sees a
    # measurement *above* the memory, and raising is not its job: the overrun path
    # owns that, and it cannot fire here because nothing overran.
    _injected_calibration(monkeypatch, per_frame_ms=1.0, base_ms=0.0)
    _simulated_machine(monkeypatch, per_frame_ms=3.0, base_ms=0.0)
    planner = TrajectoryPlanner(budget_ms=5000.0, min_frames=8, max_frames=8)

    tables = [planner.plan(emulator) for _ in range(5)]

    assert all(table.sweep_ms <= planner.budget_ms for table in tables)  # all fitted
    assert all(table.passes == 1 for table in tables)  # so no correction ran
    assert [table.per_frame_ms for table in tables] == [1.0] * 5
    assert planner._remembered is not None
    assert planner._remembered.per_frame_ms == 1.0
    assert planner._remembered.base_ms == 0.0


def test_the_relaxed_cost_converges_instead_of_drifting(
    emulator: MarioEmulator, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = _recovery_trace(emulator, monkeypatch, ticks=60)
    costs = [per_frame for per_frame, _ in trace]
    horizons = [horizon for _, horizon in trace]

    steps = [earlier - later for earlier, later in zip(costs, costs[1:])]
    # Every step is a fall, and each one is smaller than the last: the change is a
    # fraction of a value that is itself shrinking, so it cannot oscillate.
    assert all(step >= 0.0 for step in steps)
    assert all(
        later <= earlier + 1e-12 for earlier, later in zip(steps, steps[1:])
    )
    # And it is a fixed point, not a slow drift: the tail has zero total variation
    # in both the cost and the horizon it buys.
    tail = costs[RECOVERY_TICKS:]
    assert sum(abs(later - earlier) for earlier, later in zip(tail, tail[1:])) == 0.0
    assert len(set(horizons[RECOVERY_TICKS:])) == 1


def test_an_overrun_raises_the_memory_before_any_relaxation_can_lower_it(
    emulator: MarioEmulator, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A model that claims a frame costs microseconds, on the machine the journals
    # measured. The planner takes the maximum horizon, the sweep overruns, and the
    # correction pass raises the memory in one step -- before any fitting sweep
    # gets the chance to relax it.
    _injected_calibration(
        monkeypatch, per_frame_ms=MIN_PER_FRAME_MS, base_ms=RECORDED_BASE_MS
    )
    _simulated_machine(
        monkeypatch, per_frame_ms=REAL_PER_FRAME_MS, base_ms=RECORDED_BASE_MS
    )
    planner = TrajectoryPlanner(budget_ms=120.0, min_frames=6, max_frames=48)

    overran = planner.plan(emulator)

    assert overran.passes == MAX_SWEEP_PASSES == 2
    assert overran.horizon == 13  # 48 frames refuted, 13 is what the budget fits
    assert overran.per_frame_ms == MIN_PER_FRAME_MS  # the model that sized it
    assert planner._remembered is not None
    raised = planner._remembered.per_frame_ms
    # The raise happened, and the refuted_horizon floor still bites: the model is
    # left slightly *above* the cost the sweep measured, which is what stops a
    # rounding error handing back the 48 frames the machine just refused.
    assert raised > MIN_PER_FRAME_MS
    assert raised == pytest.approx((120.0 / 7 - RECORDED_BASE_MS) / 13)
    assert raised > REAL_PER_FRAME_MS

    # Only now may the fitting sweeps lower it, and only as far as the machine.
    after = [planner.plan(emulator) for _ in range(5)]

    assert all(table.passes == 1 for table in after)
    assert after[0].per_frame_ms == pytest.approx(raised)  # sized by the raise
    assert planner._remembered.per_frame_ms == pytest.approx(REAL_PER_FRAME_MS)
    assert planner._remembered.per_frame_ms < raised  # the fall came after it
    # The horizon the machine refused is not handed back by the recovery.
    assert all(table.horizon <= overran.horizon for table in after)


def test_the_relaxation_calls_the_pure_helper_rather_than_re_implementing_it(
    emulator: MarioEmulator, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A sentinel between one COST_RECOVERY step (2.8315) and the remembered value
    # (2.9191), so the relaxation cannot produce it by accident: if a future edit
    # forks the arithmetic instead of calling the pure helper, this number cannot
    # come out.
    sentinel_ms = 2.9
    assert INFLATED_PER_FRAME_MS * COST_RECOVERY < sentinel_ms < INFLATED_PER_FRAME_MS
    calls: list[dict[str, float | int]] = []

    def sentinel(
        *, sweep_ms: float, horizon: int, candidate_count: int, base_ms: float
    ) -> float:
        calls.append(
            {
                "sweep_ms": sweep_ms,
                "horizon": horizon,
                "candidate_count": candidate_count,
                "base_ms": base_ms,
            }
        )
        return sentinel_ms

    _injected_calibration(
        monkeypatch, per_frame_ms=INFLATED_PER_FRAME_MS, base_ms=RECORDED_BASE_MS
    )
    _simulated_machine(
        monkeypatch, per_frame_ms=REAL_PER_FRAME_MS, base_ms=RECORDED_BASE_MS
    )
    monkeypatch.setattr(planner_module, "remeasured_per_frame_ms", sentinel)
    planner = TrajectoryPlanner(budget_ms=120.0, min_frames=6, max_frames=48)

    first = planner.plan(emulator)
    second = planner.plan(emulator)

    # The helper is what the relaxation calls — once per fitting sweep, with the
    # sweep that just ran and the fixed term carried in rather than re-derived.
    assert len(calls) == 2
    assert calls[0] == {
        "sweep_ms": 7 * (RECORDED_BASE_MS + first.horizon * REAL_PER_FRAME_MS),
        "horizon": first.horizon,
        "candidate_count": 7,
        "base_ms": RECORDED_BASE_MS,
    }
    # And its answer is what the memory became, rather than the value the real
    # arithmetic would have produced from the same sweep.
    assert second.per_frame_ms == pytest.approx(sentinel_ms)
    assert second.per_frame_ms != pytest.approx(INFLATED_PER_FRAME_MS * COST_RECOVERY)
    # The fixed term is carried over, never re-derived from the one sweep.
    assert second.base_ms == RECORDED_BASE_MS


def test_the_recovered_horizon_climbs_back_towards_what_the_real_cost_pays_for(
    emulator: MarioEmulator, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The consequence the whole change exists for, in the units the run was blind
    # in. At 120 ms over 7 candidates the recorded inflation buys 6 frames and the
    # machine's real cost buys 13 -- the ~40% of horizon the run never got back.
    def implied(per_frame_ms: float) -> int:
        return choose_horizon(
            budget_ms=120.0,
            candidate_count=7,
            base_ms=RECORDED_BASE_MS,
            per_frame_ms=per_frame_ms,
            min_frames=6,
            max_frames=48,
        )

    assert implied(INFLATED_PER_FRAME_MS) == 6
    assert implied(REAL_PER_FRAME_MS) == 13

    horizons = [horizon for _, horizon in _recovery_trace(emulator, monkeypatch, ticks=40)]

    assert horizons[0] == 6
    assert horizons[-1] == 13
    assert all(later >= earlier for earlier, later in zip(horizons, horizons[1:]))
    assert horizons.index(13) == RECOVERY_TICKS
    # Pinned rather than described: this is the measured climb, tick by tick.
    assert horizons[:28] == (
        [6] * 6 + [7] * 5 + [8] * 4 + [9] * 3 + [10] * 3 + [11] * 3 + [12] * 3 + [13]
    )


# -- the deterministic cost source --------------------------------------------------
#
# The property T1 exists to pin: with the clock made absurd, the horizon and the
# table are unchanged under ``"fixed"`` and move under ``"measured"``. The clock
# is not slowed and nothing sleeps -- the planner's ``time`` module is replaced
# with an advancing fake, the same "inject the cost" move the warm-start tests
# make with ``_calibrate`` and ``_sweep``.


class _AdvancingClock:
    """A stand-in for the planner's ``time`` module.

    Only ``perf_counter`` is read by the planner. Each read advances the clock by
    ``step_ms``, so a tiny step models a fast machine and a huge one models a
    machine far slower than any figure the plan would configure.
    """

    def __init__(self, step_ms: float) -> None:
        self._now = 0.0
        self._step_s = float(step_ms) / 1000.0

    def perf_counter(self) -> float:
        now = self._now
        self._now += self._step_s
        return now


def _decision_view(table: RolloutTable) -> dict[str, object]:
    """The table a decision is made from, without the two genuine measurements.

    ``elapsed_ms`` and ``sweep_ms`` come from the clock under both sources: they
    are real facts about the tick, which is why the design keeps them. Removing
    exactly those two lets the test compare every field a clock could have moved
    a decision with, and says out loud that the measurements are the only things
    allowed to differ.
    """
    view = table.as_dict()
    del view["elapsed_ms"]
    del view["sweep_ms"]
    return view


def test_a_fixed_source_planner_takes_the_configured_figures_from_the_settings(
    emulator: MarioEmulator,
) -> None:
    settings = Settings(
        rollout_budget_ms=120.0,
        rollout_cost_source="fixed",
        rollout_fixed_base_ms=0.5,
        rollout_fixed_per_frame_ms=1.1,
    )

    planner = TrajectoryPlanner.from_settings(settings)
    table = planner.plan(emulator)

    assert table.cost_source is CostSource.FIXED
    assert table.base_ms == 0.5
    assert table.per_frame_ms == 1.1
    assert table.calibration_frames == 0
    assert table.passes == 1
    assert table.horizon == choose_horizon(
        budget_ms=120.0,
        candidate_count=len(planner.actions),
        base_ms=0.5,
        per_frame_ms=1.1,
        min_frames=6,
        max_frames=48,
    )
    # The measured facts are still recorded, even though they decided nothing.
    assert table.elapsed_ms > 0.0
    assert table.sweep_ms > 0.0
    assert table.as_dict()["cost_source"] == "fixed"


def test_the_default_planner_sizes_its_horizon_from_a_measurement(
    emulator: MarioEmulator,
) -> None:
    planner = TrajectoryPlanner.from_settings(Settings())

    table = planner.plan(emulator)

    assert table.cost_source is CostSource.CALIBRATED
    assert table.calibration_frames == CALIBRATION_LOADS * (1 + CALIBRATION_STEADY_STEPS)


def test_a_fixed_source_keeps_one_horizon_across_ticks_and_remembers_nothing(
    emulator: MarioEmulator,
) -> None:
    planner = TrajectoryPlanner(
        budget_ms=120.0,
        min_frames=6,
        max_frames=48,
        cost_source="fixed",
        fixed_base_ms=0.5,
        fixed_per_frame_ms=1.1,
    )

    tables = [planner.plan(emulator) for _ in range(3)]

    assert all(table.cost_source is CostSource.FIXED for table in tables)
    assert len({table.horizon for table in tables}) == 1
    assert len({(table.base_ms, table.per_frame_ms) for table in tables}) == 1
    # No measurement is ever handed to the next tick: the fixed model is enough.
    assert planner._remembered is None


def test_a_fixed_source_ignores_an_absurd_clock_while_a_measured_source_does_not(
    emulator_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The observable property, with the negative control that keeps it honest."""

    def plan(*, cost_source: str, step_ms: float) -> RolloutTable:
        monkeypatch.setattr(planner_module, "time", _AdvancingClock(step_ms))
        emulator = emulator_factory()
        emulator.reset(seed=0)
        if cost_source == "fixed":
            planner = TrajectoryPlanner(
                budget_ms=120.0,
                min_frames=6,
                max_frames=48,
                cost_source="fixed",
                fixed_base_ms=0.5,
                fixed_per_frame_ms=1.1,
            )
        else:
            planner = TrajectoryPlanner(budget_ms=120.0, min_frames=6, max_frames=48)
        return planner.plan(emulator)

    fixed_fast = plan(cost_source="fixed", step_ms=0.003)
    fixed_absurd = plan(cost_source="fixed", step_ms=3000.0)

    # Every field a clock could have moved the decision with is identical.
    assert fixed_fast.cost_source is CostSource.FIXED
    assert fixed_absurd.cost_source is CostSource.FIXED
    assert _decision_view(fixed_fast) == _decision_view(fixed_absurd)
    assert fixed_fast.horizon == fixed_absurd.horizon
    assert fixed_fast.passes == fixed_absurd.passes == 1
    # The two genuine measurements are the only fields that move, which is why the
    # comparison above excludes exactly them.
    assert fixed_fast.sweep_ms != fixed_absurd.sweep_ms
    assert fixed_fast.elapsed_ms != fixed_absurd.elapsed_ms

    measured_fast = plan(cost_source="measured", step_ms=0.003)
    measured_absurd = plan(cost_source="measured", step_ms=3000.0)

    # The same manipulations visibly move the measured horizon: the test above is
    # not vacuous.
    assert measured_fast.cost_source is CostSource.CALIBRATED
    assert measured_absurd.cost_source is CostSource.CALIBRATED
    assert measured_fast.horizon != measured_absurd.horizon


def test_a_fixed_source_never_makes_the_calibration_a_measured_source_must(
    emulator: MarioEmulator, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The raising half of the absurd-clock property. Two clock reads can move a
    # measured horizon -- the calibration, and the sweep that decides both the
    # correction pass and the learning branches -- and under the fixed source
    # neither decides anything, so a calibration that raises cannot disturb the
    # table. Under the measured source the same raise is fatal.
    def exploding_calibrate(self, emulator, snapshot):  # type: ignore[no-untyped-def]
        raise RuntimeError("the clock exploded")

    monkeypatch.setattr(TrajectoryPlanner, "_calibrate", exploding_calibrate)

    fixed = TrajectoryPlanner(
        budget_ms=120.0,
        min_frames=6,
        max_frames=48,
        cost_source="fixed",
        fixed_base_ms=0.5,
        fixed_per_frame_ms=1.1,
    )
    table = fixed.plan(emulator)

    assert table.cost_source is CostSource.FIXED
    assert table.horizon == 15
    assert fixed._remembered is None

    measured = TrajectoryPlanner(budget_ms=120.0, min_frames=6, max_frames=48)
    with pytest.raises(RuntimeError, match="exploded"):
        measured.plan(emulator)


def test_a_fixed_source_takes_one_pass_and_never_corrects_a_measured_overrun(
    emulator: MarioEmulator, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Make the recorded sweep cost more than the budget without changing the
    # rollouts, and start the measured side from an injected cheapest-possible
    # model so it predictably picks the maximum horizon and then overruns. The
    # measured planner corrects; the fixed one deliberately does not, because the
    # correction is measurement-driven and a fixed horizon does not follow one.
    injected = _Calibration(base_ms=0.0, per_frame_ms=MIN_PER_FRAME_MS, frames=0)
    monkeypatch.setattr(
        TrajectoryPlanner, "_calibrate", lambda self, emulator, snapshot: injected
    )
    real_sweep = TrajectoryPlanner._sweep

    def overrunning_sweep(self, emulator, snapshot, horizon):  # type: ignore[no-untyped-def]
        sweep_ms, candidates, steps = real_sweep(self, emulator, snapshot, horizon)
        return 10_000.0, candidates, steps

    monkeypatch.setattr(TrajectoryPlanner, "_sweep", overrunning_sweep)

    fixed = TrajectoryPlanner(
        budget_ms=120.0,
        min_frames=6,
        max_frames=48,
        cost_source="fixed",
        fixed_base_ms=0.5,
        fixed_per_frame_ms=1.1,
    )
    fixed_table = fixed.plan(emulator)

    assert fixed_table.sweep_ms == 10_000.0
    assert fixed_table.passes == 1
    assert fixed_table.horizon == 15  # the constants' own choice, not a correction
    assert fixed._remembered is None

    measured = TrajectoryPlanner(budget_ms=120.0, min_frames=6, max_frames=48)
    measured_table = measured.plan(emulator)

    assert measured_table.cost_source is CostSource.CALIBRATED
    assert measured_table.passes == MAX_SWEEP_PASSES == 2


def test_the_planner_rejects_a_cost_source_it_cannot_honour() -> None:
    # The settings layer validates its own view; the planner validates what it is
    # constructed with, because a planner can be built without going through
    # ``Settings`` at all. Each message names the offending field, so a caller can
    # fix it rather than guess.
    with pytest.raises(ValueError, match="cost_source"):
        TrajectoryPlanner(
            budget_ms=120.0, min_frames=6, max_frames=48, cost_source="guessed"
        )

    with pytest.raises(ValueError, match="fixed_base_ms"):
        TrajectoryPlanner(
            budget_ms=120.0,
            min_frames=6,
            max_frames=48,
            cost_source="fixed",
            fixed_base_ms=0.0,
            fixed_per_frame_ms=1.1,
        )

    with pytest.raises(ValueError, match="fixed_per_frame_ms"):
        TrajectoryPlanner(
            budget_ms=120.0,
            min_frames=6,
            max_frames=48,
            cost_source="fixed",
            fixed_base_ms=0.5,
            fixed_per_frame_ms=math.inf,
        )
