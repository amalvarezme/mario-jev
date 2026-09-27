"""Typed state: the relative features, the JSON contract, and the info keys behind them.

Synthetic frames carry most of the weight here for the same reason they do in
``test_perception.py``: the game will not produce a pit directly in front of Mario
on demand, and the terrain readings that matter are the ones on either side of a
boundary. Live frames cover everything the emulator can be asked for — the start
of 1-1, a jump, a walk, a death — because those are the readings the loop will
actually run on.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from mario_jev.emulator import RAM_ADDRESSES, EnemySlot, MarioEmulator
from mario_jev.perception import GROUND, GROUND_HIGHLIGHT, SKY, TILE, detect
from mario_jev.perception.palette import PLAYER_BOX_SIZE
from mario_jev.state import (
    DEATH_INFO_KEYS,
    GOAL_INFO_KEYS,
    HAZARD_OUTCOMES,
    INFO_CLEAR,
    INFO_DEATH,
    INFO_FLAG_GET,
    INFO_IS_DEAD,
    INFO_IS_DYING,
    INFO_LIFE,
    INFO_PLAYER_STATE,
    INFO_STAGE,
    INFO_TIME,
    INFO_WORLD,
    INFO_X_POS,
    INFO_X_POS_MAX,
    INFO_Y_PIXEL,
    MIN_JUMP_WINDOW_FRAMES,
    OUTCOME_PRECEDENCE,
    PLAYER_STATE_DYING,
    SURVIVAL_OUTCOMES,
    CandidateAssessment,
    MarioStateBuilder,
    TerminalOutcome,
    is_death,
    is_goal,
    to_jev_state,
)

HEIGHT, WIDTH = 240, 256
FLOOR_TOP = 208
PLAYER_X_ADDRESS = RAM_ADDRESSES["player_x_offset"]
PLAYER_Y_ADDRESS = RAM_ADDRESSES["player_y_on_screen"]

#: The measured frame at which ``right+B`` from the 1-1 start runs into the first
#: Goomba. The game is deterministic for a fixed seed and action sequence.
ENEMY_CONTACT_FRAME = 106


# -- synthetic frames, built the way test_perception.py builds them ----------------


def blank_frame(colour: tuple[int, int, int] = SKY) -> np.ndarray:
    return np.tile(np.array(colour, dtype=np.uint8), (HEIGHT, WIDTH, 1))


def frame_with_floor(gap_columns: range | None = None) -> np.ndarray:
    frame = blank_frame(SKY)
    frame[FLOOR_TOP:, :] = np.array(GROUND, dtype=np.uint8)
    frame[FLOOR_TOP, :] = np.array(GROUND_HIGHLIGHT, dtype=np.uint8)
    if gap_columns is not None:
        frame[FLOOR_TOP:, gap_columns] = np.array(SKY, dtype=np.uint8)
    return frame


class FakeRam:
    """The two RAM reads the state needs, without an emulator."""

    def __init__(
        self,
        *,
        x: int = 40,
        y: int = 176,
        enemies: tuple[EnemySlot, ...] = (),
    ) -> None:
        self._bytes = {PLAYER_X_ADDRESS: x, PLAYER_Y_ADDRESS: y}
        self._enemies = tuple(enemies)

    def byte(self, address: int) -> int:
        return self._bytes[address]

    def active_enemies(self) -> tuple[EnemySlot, ...]:
        return self._enemies


def goomba(*, drawn_x: int, slot: int = 0, x_screen: int = 215, y: int = 176) -> EnemySlot:
    """An active enemy. ``x_screen`` is deliberately the misleading ``0x0087`` block.

    Measured earlier in this project: for the first Goomba, ``0x0087`` read 215
    while the drawn sprite sat near column 129 and ``0x03AE`` read 129. Tests that
    leave ``x_screen`` at the wrong value fail loudly if anything starts reading it.
    """
    return EnemySlot(
        slot=slot,
        type_id=1,
        x_level=1,
        x_screen=x_screen,
        x_offset=drawn_x,
        y=y,
    )


def info_for(
    *,
    x_pos: int = 40,
    x_pos_max: int | None = None,
    lives: int = 2,
    time_left: int = 400,
    world: int = 1,
    stage: int = 1,
) -> dict[str, object]:
    return {
        INFO_X_POS: x_pos,
        INFO_X_POS_MAX: x_pos if x_pos_max is None else x_pos_max,
        INFO_LIFE: lives,
        INFO_TIME: time_left,
        INFO_WORLD: world,
        INFO_STAGE: stage,
    }


# -- the contract ------------------------------------------------------------------


def test_the_player_sprite_is_one_tile() -> None:
    # The frame columns and the sprite columns are the same space, so these two
    # constants have to agree for any of the gap arithmetic to mean anything.
    assert PLAYER_BOX_SIZE == TILE == 16


def test_the_outcome_vocabulary_is_exactly_the_contracted_five() -> None:
    assert {outcome.value for outcome in TerminalOutcome} == {
        "reaches_goal",
        "falls_in_gap",
        "hits_enemy",
        "no_progress",
        "lands_safe",
    }
    assert [outcome.value for outcome in OUTCOME_PRECEDENCE] == [
        "reaches_goal",
        "falls_in_gap",
        "hits_enemy",
        "no_progress",
        "lands_safe",
    ]
    assert HAZARD_OUTCOMES == frozenset(
        {TerminalOutcome.FALLS_IN_GAP, TerminalOutcome.HITS_ENEMY}
    )
    assert SURVIVAL_OUTCOMES == frozenset(
        {TerminalOutcome.NO_PROGRESS, TerminalOutcome.LANDS_SAFE}
    )


def test_an_invented_outcome_or_an_out_of_range_risk_is_rejected() -> None:
    with pytest.raises(ValueError):
        CandidateAssessment(action="right", dx=1, outcome="wins", risk=0.0, final_x=41)

    with pytest.raises(ValueError):
        CandidateAssessment(
            action="right", dx=1, outcome=TerminalOutcome.LANDS_SAFE, risk=1.5, final_x=41
        )

    # The plain string is accepted and normalised to the enum.
    assessment = CandidateAssessment(
        action="right", dx=1, outcome="lands_safe", risk=0.2, final_x=41
    )
    assert assessment.outcome is TerminalOutcome.LANDS_SAFE


def test_to_jev_state_is_json_serializable_and_has_exactly_the_contracted_keys() -> None:
    builder = MarioStateBuilder()
    state = builder.observe(detect(frame_with_floor(), tick=0), FakeRam(), info_for())
    state = state.with_candidates(
        [
            CandidateAssessment(
                action="right+B",
                dx=13,
                outcome=TerminalOutcome.LANDS_SAFE,
                risk=0.2,
                final_x=53,
            )
        ]
    )

    payload = json.loads(json.dumps(state.to_jev_state()))

    assert set(payload) == {
        "player",
        "nearest_enemy",
        "terrain_ahead",
        "progress",
        "candidates",
    }
    assert set(payload["player"]) == {"x", "y", "on_ground", "facing"}
    assert set(payload["nearest_enemy"]) == {"distance_px", "approaching"}
    assert set(payload["terrain_ahead"]) == {"gap_columns", "gap_distance_px", "wall_height"}
    assert set(payload["progress"]) == {"world", "x_max", "lives", "time_left"}
    assert set(payload["candidates"]) == {"right+B"}
    assert set(payload["candidates"]["right+B"]) == {"dx", "outcome", "risk", "final_x"}

    # The module-level function is the same projection as the method.
    assert to_jev_state(state) == payload


def test_every_projected_value_is_a_plain_python_type() -> None:
    # The readings travel through numpy on their way here and json.dumps refuses
    # numpy scalars, so "it serializes" is a real assertion, not a formality.
    builder = MarioStateBuilder()
    state = builder.observe(detect(frame_with_floor(), tick=0), FakeRam(), info_for())
    payload = state.to_jev_state()

    def walk(value: object) -> None:
        assert isinstance(value, (bool, int, float, str, dict)) or value is None, repr(value)
        if isinstance(value, dict):
            for item in value.values():
                walk(item)

    walk(payload)


# -- frames_to_jump_window: the sharpest feature -----------------------------------


def test_frames_to_jump_window_is_none_when_no_hole_is_ahead() -> None:
    builder = MarioStateBuilder()
    builder.observe(detect(frame_with_floor(), tick=0), FakeRam(), info_for(x_pos=100))
    state = builder.observe(
        detect(frame_with_floor(), tick=1), FakeRam(), info_for(x_pos=102)
    )

    assert state.speed_px_per_tick == pytest.approx(2.0)
    assert state.terrain_ahead.gap_distance_px is None
    assert state.terrain_ahead.gap_columns == 0
    assert state.terrain_ahead.frames_to_jump_window is None


def test_frames_to_jump_window_is_none_when_the_speed_is_zero() -> None:
    pit_frame = frame_with_floor(gap_columns=range(160, 200))

    # No previous tick at all: the speed is unknown, so there is no arrival time
    # even though the pit is right there.
    first = MarioStateBuilder().observe(
        detect(pit_frame, tick=0), FakeRam(x=100), info_for()
    )
    assert first.terrain_ahead.gap_distance_px == 44
    assert first.terrain_ahead.frames_to_jump_window is None

    # Two ticks with the same position: measured speed 0, still no arrival time.
    builder = MarioStateBuilder()
    builder.observe(detect(pit_frame, tick=0), FakeRam(x=100), info_for(x_pos=100))
    state = builder.observe(detect(pit_frame, tick=1), FakeRam(x=100), info_for(x_pos=100))

    assert state.speed_px_per_tick == 0.0
    assert state.terrain_ahead.gap_distance_px == 44
    assert state.terrain_ahead.frames_to_jump_window is None


def test_frames_to_jump_window_is_the_frames_left_before_the_front_edge_arrives() -> None:
    pit_frame = frame_with_floor(gap_columns=range(160, 200))
    builder = MarioStateBuilder()
    # Mario's drawn column is 100, so his front edge is at 116 and the pit starts
    # at 160: 44 pixels away at a measured 2 px/frame.
    builder.observe(detect(pit_frame, tick=0), FakeRam(x=100), info_for(x_pos=100))
    state = builder.observe(detect(pit_frame, tick=1), FakeRam(x=100), info_for(x_pos=102))

    window = state.terrain_ahead.frames_to_jump_window

    assert window == 22
    assert isinstance(window, int) and window > 0
    assert state.terrain_ahead.gap_distance_px == 44
    assert state.terrain_ahead.gap_columns == 40


def test_the_jump_window_never_reads_as_zero() -> None:
    # A pit that starts under Mario's front edge is "now", and "now" is one frame:
    # the value is a countdown, and a countdown of zero has already passed.
    pit_frame = frame_with_floor(gap_columns=range(116, 140))
    builder = MarioStateBuilder()
    builder.observe(detect(pit_frame, tick=0), FakeRam(x=100), info_for(x_pos=100))
    state = builder.observe(detect(pit_frame, tick=1), FakeRam(x=100), info_for(x_pos=102))

    assert state.terrain_ahead.gap_distance_px == 0
    assert state.terrain_ahead.frames_to_jump_window == MIN_JUMP_WINDOW_FRAMES == 1


# -- relative features ---------------------------------------------------------------


def test_wall_height_is_whole_tiles_because_raised_ground_is_built_from_them() -> None:
    builder = MarioStateBuilder()

    flat = builder.observe(
        detect(frame_with_floor(), tick=0), FakeRam(x=80), info_for(x_pos=80)
    )
    assert flat.terrain_ahead.wall_height == 0

    # A one-pixel stub against the floor is what a sprite or a seam produces, not
    # a step: the live 1-1 start measures exactly this and must read as flat.
    stubbed = frame_with_floor()
    stubbed[FLOOR_TOP - 1, 130] = np.array(GROUND, dtype=np.uint8)
    noise = MarioStateBuilder().observe(
        detect(stubbed, tick=0), FakeRam(x=80), info_for(x_pos=80)
    )
    assert noise.terrain_ahead.wall_height == 0

    # A platform 16 px up is one tile of wall.
    raised = frame_with_floor()
    raised[FLOOR_TOP - 16 : FLOOR_TOP, 100:121] = np.array(GROUND, dtype=np.uint8)
    step = MarioStateBuilder().observe(
        detect(raised, tick=0), FakeRam(x=80), info_for(x_pos=80)
    )
    assert step.terrain_ahead.wall_height == 16


def test_the_nearest_enemy_comes_from_the_0x03ae_block_not_the_0x0087_block() -> None:
    state = MarioStateBuilder().observe(
        detect(frame_with_floor(), tick=0),
        FakeRam(x=40, enemies=(goomba(drawn_x=129, x_screen=215),)),
        info_for(),
    )

    # Mario's box ends at column 40 + 15 = 55 and the Goomba's starts at 129, so 74
    # pixels of empty space. Reading the wrong block (0x0087 = 215) would say 159.
    assert PLAYER_BOX_SIZE == 16
    assert state.nearest_enemy.distance_px == 129 - (40 + PLAYER_BOX_SIZE - 1) == 74
    assert state.nearest_enemy.distance_px != 215 - 55
    assert state.nearest_enemy.slot == 0


def test_an_enemy_that_overlaps_mario_has_zero_distance_not_a_negative_one() -> None:
    state = MarioStateBuilder().observe(
        detect(frame_with_floor(), tick=0),
        FakeRam(x=100, enemies=(goomba(drawn_x=104),)),
        info_for(),
    )

    assert state.nearest_enemy.distance_px == 0


def test_an_enemy_behind_mario_is_not_the_nearest_enemy_ahead() -> None:
    state = MarioStateBuilder().observe(
        detect(frame_with_floor(), tick=0),
        FakeRam(x=100, enemies=(goomba(drawn_x=40),)),
        info_for(),
    )

    assert state.nearest_enemy.distance_px is None
    assert state.nearest_enemy.slot is None


def test_approaching_is_measured_from_the_enemy_column_and_is_unknown_on_first_sight() -> None:
    frame = frame_with_floor()
    builder = MarioStateBuilder()

    walking_at_mario = []
    for tick, drawn_x in enumerate((200, 190, 180)):
        ram = FakeRam(x=40, enemies=(goomba(drawn_x=drawn_x),))
        walking_at_mario.append(
            builder.observe(detect(frame, tick=tick), ram, info_for(x_pos=40 + tick))
        )

    # First sighting: no previous column to difference against, so the honest
    # answer is "not known to be approaching" rather than a guess.
    assert walking_at_mario[0].nearest_enemy.approaching is False
    assert walking_at_mario[1].nearest_enemy.approaching is True
    assert walking_at_mario[2].nearest_enemy.approaching is True

    walking_away = []
    for tick, drawn_x in enumerate((180, 190, 200), start=3):
        ram = FakeRam(x=40, enemies=(goomba(drawn_x=drawn_x),))
        walking_away.append(
            builder.observe(detect(frame, tick=tick), ram, info_for(x_pos=40 + tick))
        )

    assert walking_away[-1].nearest_enemy.approaching is False


# -- the info keys, read from the live package -----------------------------------------


def test_the_info_keys_the_state_depends_on_are_real_and_mean_what_we_claim(
    emulator: MarioEmulator,
) -> None:
    info = emulator.info
    ram = emulator.ram()

    for key in (
        INFO_X_POS,
        INFO_X_POS_MAX,
        INFO_Y_PIXEL,
        INFO_LIFE,
        INFO_TIME,
        INFO_WORLD,
        INFO_STAGE,
        INFO_CLEAR,
        INFO_FLAG_GET,
        INFO_DEATH,
        INFO_IS_DYING,
        INFO_IS_DEAD,
        INFO_PLAYER_STATE,
    ):
        assert key in info, key

    # x_pos is the 16-bit absolute position, NOT the 0x03AD screen column. The two
    # read the same value at the level start, which is exactly why this is pinned.
    assert info[INFO_X_POS] == ram.byte(0x006D) * 0x100 + ram.byte(0x0086)
    assert info[INFO_X_POS] == ram.byte(0x03AD)  # only true while 0x006D is still 0
    assert ram.byte(0x006D) == 0

    assert info[INFO_Y_PIXEL] == ram.byte(0x03B8)
    assert info[INFO_LIFE] == ram.byte(0x075A)
    # `_time` is `_read_mem_range(0x07f8, 3)`: three decimal digits on the HUD.
    assert info[INFO_TIME] == (
        ram.byte(0x07F8) * 100 + ram.byte(0x07F9) * 10 + ram.byte(0x07FA)
    )
    assert info[INFO_WORLD] == ram.byte(0x075F) + 1
    assert info[INFO_STAGE] == ram.byte(0x075C) + 1

    # `clear` and `flag_get` are the same reading under two names.
    assert info[INFO_CLEAR] == info[INFO_FLAG_GET] is False
    # `death` is exactly `is_dying or is_dead`.
    assert info[INFO_DEATH] == (info[INFO_IS_DYING] or info[INFO_IS_DEAD]) is False
    assert is_death(info) is False
    assert is_goal(info) is False
    for key in DEATH_INFO_KEYS:
        assert key in info
    for key in GOAL_INFO_KEYS:
        assert key in info


def test_the_goal_and_death_keys_are_pinned_to_the_installed_package_contract() -> None:
    # Reaching the flag requires finishing the level, which no test can do in a
    # second, so the *meaning* of the goal keys is pinned to the installed source
    # instead of being assumed. Same for `death`.
    from gym_super_mario_bros.smb_env import SuperMarioBrosEnv

    normalized = inspect.getsource(SuperMarioBrosEnv._normalized_info)
    base = inspect.getsource(SuperMarioBrosEnv._get_info)
    flag = inspect.getsource(SuperMarioBrosEnv._flag_get.fget)

    assert "clear=self._flag_get" in normalized
    assert "death = self._is_dying or self._is_dead" in normalized
    assert "flag_get=self._flag_get" in base
    assert "self._is_world_over or self._is_stage_over" in flag


def test_a_live_death_sets_exactly_the_keys_the_planner_reads(
    emulator_factory,
) -> None:
    emulator = emulator_factory()
    emulator.reset(seed=0)

    terminal = None
    for frame in range(200):
        result = emulator.step(3)  # right+B: measured to run into the first Goomba
        if result.terminated or result.truncated:
            terminal = (frame, result)
            break

    assert terminal is not None, "no death within 200 frames, so this asserts nothing"
    frame, result = terminal
    assert frame + 1 == ENEMY_CONTACT_FRAME
    assert result.terminated is True
    assert result.truncated is False

    info = result.info
    assert info[INFO_DEATH] is True
    assert info[INFO_IS_DYING] is True
    assert info[INFO_IS_DEAD] is False
    assert info[INFO_PLAYER_STATE] == PLAYER_STATE_DYING == 0x0B
    assert emulator.ram().byte(0x000E) == PLAYER_STATE_DYING
    assert info[INFO_CLEAR] is False
    assert is_death(info) is True
    assert is_goal(info) is False


# -- the live start, the walk, the jump -------------------------------------------------


def test_a_live_state_at_the_1_1_start(emulator: MarioEmulator) -> None:
    state = MarioStateBuilder().observe(
        detect(emulator.frame(), tick=emulator.tick), emulator.ram(), emulator.info
    )

    # Measured at the start: no enemy slot is active yet, the ground is flat and
    # there is no pit in the first four tiles.
    assert state.nearest_enemy.distance_px is None
    assert state.nearest_enemy.approaching is False
    assert state.terrain_ahead.gap_columns == 0
    assert state.terrain_ahead.gap_distance_px is None
    assert state.terrain_ahead.wall_height == 0
    assert state.terrain_ahead.frames_to_jump_window is None
    assert state.terrain_ahead.vertical_clearance == 0
    assert state.player.on_ground is True
    # The measured relation from the perception layer: the feet rest one pixel
    # above the floor top at row 208.
    assert state.player.feet_row == detect(emulator.frame()).terrain.ground_line - 1 == 207
    assert state.player.x == 40
    assert state.player.y == 176
    assert state.player.screen_x == 40
    assert state.player.facing == "right"
    assert state.progress.world == "1-1"
    assert state.progress.x_max == 40
    assert state.progress.lives == 2
    assert state.progress.time_left == 400
    assert state.speed_px_per_tick == 0.0


def test_a_live_walk_measures_a_positive_speed_and_the_facing(emulator: MarioEmulator) -> None:
    builder = MarioStateBuilder()
    builder.observe(detect(emulator.frame(), tick=emulator.tick), emulator.ram(), emulator.info)
    for _ in range(30):
        emulator.step(1)
    state = builder.observe(
        detect(emulator.frame(), tick=emulator.tick), emulator.ram(), emulator.info
    )

    assert state.speed_px_per_tick > 0.0
    assert state.player.facing == "right"
    assert state.player.x > 40
    assert state.terrain_ahead.frames_to_jump_window is None  # still no hole ahead

    # ... and walking left reports the other facing. The skid means the position
    # keeps rising for a few frames first, so the facing is asserted after enough
    # frames for the turn to be a measurement rather than a guess.
    for _ in range(40):
        emulator.step(6)
    left = builder.observe(
        detect(emulator.frame(), tick=emulator.tick), emulator.ram(), emulator.info
    )
    assert left.speed_px_per_tick < 0.0
    assert left.player.facing == "left"
    assert left.player.x < left.progress.x_max


def test_a_live_jump_shows_up_as_vertical_clearance(emulator: MarioEmulator) -> None:
    builder = MarioStateBuilder()
    builder.observe(detect(emulator.frame(), tick=emulator.tick), emulator.ram(), emulator.info)

    airborne = []
    for _ in range(12):
        emulator.step(5)  # A: jump on the spot
        airborne.append(
            builder.observe(
                detect(emulator.frame(), tick=emulator.tick), emulator.ram(), emulator.info
            )
        )

    highest = max(state.terrain_ahead.vertical_clearance for state in airborne)
    assert highest > 0
    # The first jump frame is still on the ground — measured, and worth pinning:
    # a state reader that assumed "jump pressed means airborne" would be wrong for
    # exactly one frame.
    assert airborne[0].player.on_ground is True
    assert airborne[-1].player.on_ground is False
    assert any(state.player.on_ground is False for state in airborne[1:])

    # Landing is measured too, and the clearance goes back to zero.
    for _ in range(40):
        emulator.step(0)
        state = builder.observe(
            detect(emulator.frame(), tick=emulator.tick), emulator.ram(), emulator.info
        )
        if state.player.on_ground:
            break
    assert state.player.on_ground is True
    assert state.terrain_ahead.vertical_clearance == 0
