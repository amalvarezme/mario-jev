"""The emulator boundary, tested against the packages it wraps.

Several tests here exist because a plausible assumption turned out to be wrong
during development, and the point of pinning them is that they cannot quietly
become true again.
"""

from __future__ import annotations

import time

import pytest

from mario_jev.emulator import (
    BUTTON_BITS,
    FramePacer,
    MarioEmulator,
    SCREEN_SHAPE,
    bits_for,
    build_actions,
    parse_stage,
)

# -- action set ------------------------------------------------------------------


def test_action_bits_match_the_package_joypad_map() -> None:
    from nes_py.wrappers.joypad_space import JoypadSpace

    assert BUTTON_BITS == JoypadSpace._button_map


def test_simple_action_set_matches_the_package_simple_movement() -> None:
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

    built = build_actions("simple")

    assert [list(action.buttons) for action in built] == SIMPLE_MOVEMENT
    assert [action.name for action in built] == [
        "NOOP",
        "right",
        "right+A",
        "right+B",
        "right+A+B",
        "A",
        "left",
    ]


def test_action_bits_are_or_combinations_of_their_buttons() -> None:
    built = build_actions("simple")
    right_a = built[2]

    assert right_a.bits == BUTTON_BITS["right"] | BUTTON_BITS["A"] == 0x81
    assert right_a.moves_right and right_a.jumps
    assert not right_a.moves_left and not right_a.runs


def test_unknown_action_set_is_rejected_with_the_known_sets() -> None:
    with pytest.raises(KeyError, match="simple"):
        build_actions("does-not-exist")


def test_unknown_button_is_rejected() -> None:
    with pytest.raises(KeyError, match="teleport"):
        bits_for(["teleport"])


@pytest.mark.parametrize(
    ("stage", "expected"),
    [("1-1", (1, 1)), ("2-3", (2, 3)), (" 8-4 ", (8, 4))],
)
def test_stage_parsing(stage: str, expected: tuple[int, int]) -> None:
    assert parse_stage(stage) == expected


@pytest.mark.parametrize("stage", ["1", "1-", "-1", "one-one", "1-2-3"])
def test_malformed_stage_is_rejected(stage: str) -> None:
    with pytest.raises(ValueError):
        parse_stage(stage)


# -- frames ----------------------------------------------------------------------


def test_frame_shape_dtype_and_contiguity(emulator: MarioEmulator) -> None:
    frame = emulator.frame()

    assert frame.shape == SCREEN_SHAPE == (240, 256, 3)
    assert frame.dtype.name == "uint8"
    assert frame.flags["C_CONTIGUOUS"]


def test_captured_frame_is_stable_across_a_step(emulator: MarioEmulator) -> None:
    # The emulator hands out a live view by default; frame() must not.
    first = emulator.frame()
    emulator.step(1)
    observed_live = emulator.screen_view()

    assert first.shape == observed_live.shape
    # first must be an independent buffer: writing it cannot affect the emulator.
    first[:] = 0
    assert emulator.screen_view().any()


def test_reset_returns_the_first_frame(emulator: MarioEmulator) -> None:
    result = emulator.reset(seed=0)

    assert result.tick == 0
    assert result.done is False
    assert result.reward == 0.0
    assert result.frame.shape == SCREEN_SHAPE


# -- RAM -------------------------------------------------------------------------


def test_info_exposes_the_keys_the_loop_depends_on(emulator: MarioEmulator) -> None:
    # These come from _get_info plus the _normalized_info/_reward_info merge, and
    # they are what the state layer reads. A silent rename must break a test.
    for key in ("x_pos", "y_pixel", "life", "time", "status", "coins", "score", "stage"):
        assert key in emulator.info, key

    assert len(emulator.info) == 42


def test_x_pos_is_a_sixteen_bit_absolute_position(emulator: MarioEmulator) -> None:
    emulator.step(1)
    ram = emulator.ram()

    assert emulator.info["x_pos"] == ram.byte(0x006D) * 0x100 + ram.byte(0x0086)


def test_y_pixel_is_the_screen_y_address(emulator: MarioEmulator) -> None:
    emulator.step(1)
    ram = emulator.ram()

    assert emulator.info["y_pixel"] == ram.byte(0x03B8)


@pytest.mark.slow
def test_x_pos_diverges_from_the_screen_offset_once_the_page_byte_is_set(
    emulator: MarioEmulator,
) -> None:
    # The correction this test protects: x_pos is 0x6D*0x100 + 0x86, NOT 0x03AD.
    # The two agree only while 0x6D is still zero, which is why the wrong mapping
    # looked correct when only the level start was inspected.
    for _ in range(400):
        emulator.step(1)
        ram = emulator.ram()
        if ram.byte(0x006D) != 0:
            break
    else:
        pytest.fail("never crossed the 0x6D page boundary, so the claim is untested")

    assert ram.byte(0x006D) != 0
    assert emulator.info["x_pos"] == ram.byte(0x006D) * 0x100 + ram.byte(0x0086)
    assert emulator.info["x_pos"] != ram.byte(0x03AD)


def test_ram_view_is_detached_from_later_steps(emulator: MarioEmulator) -> None:
    before = emulator.ram()
    for _ in range(10):
        emulator.step(1)
    after = emulator.ram()

    assert before.player_x_on_screen != after.player_x_on_screen
    assert before.player_x_on_screen == 40


def test_ram_view_rejects_out_of_range_reads(emulator: MarioEmulator) -> None:
    ram = emulator.ram()

    assert len(ram) == 2048
    with pytest.raises(IndexError):
        ram.byte(0x2000)
    with pytest.raises(IndexError):
        ram.enemy(99)


def test_enemy_slots_cover_five_entries(emulator: MarioEmulator) -> None:
    ram = emulator.ram()

    assert len(ram.enemies()) == 5
    # Inactive slots are the ones with a zero type byte; the accessor must agree
    # with that definition rather than inventing a different one.
    assert all(slot.is_active == (slot.type_id != 0) for slot in ram.enemies())


# -- savestates ------------------------------------------------------------------


def test_savestate_round_trip_restores_the_position(emulator: MarioEmulator) -> None:
    for _ in range(30):
        emulator.step(2)
    snapshot = emulator.dump_state()
    x_at_snapshot = emulator.info["x_pos"]

    for _ in range(20):
        emulator.step(0)

    emulator.load_state(snapshot)

    assert emulator.info["x_pos"] == x_at_snapshot
    assert emulator.ram().player_x_in_level == emulator.ram().byte(0x006D)


def test_load_state_refreshes_the_cached_info(emulator: MarioEmulator) -> None:
    # Regression: restoring the emulator is not enough. The gym wrapper derives
    # its readings from RAM at call time, so a stale cached info would make every
    # rollout candidate look like it started from the wrong state.
    for _ in range(30):
        emulator.step(2)
    snapshot = emulator.dump_state()
    expected = emulator.info["x_pos"]

    for _ in range(40):
        emulator.step(1)
    assert emulator.info["x_pos"] != expected

    emulator.load_state(snapshot)

    assert emulator.info["x_pos"] == expected


def test_rollout_does_not_advance_the_frame_counter(emulator: MarioEmulator) -> None:
    emulator.step(1)
    tick = emulator.tick
    snapshot = emulator.dump_state()

    for _ in range(10):
        emulator.step(2)
    emulator.load_state(snapshot)

    # A rollout is not gameplay; rewinding the tick would corrupt the real trace.
    assert emulator.tick == tick + 10


def test_the_adapter_exposes_no_file_backed_savestate(emulator: MarioEmulator) -> None:
    # Guard against re-introducing dead code. nes_py's NativeStateSnapshot has no
    # __bytes__, no __reduce__ and no buffer protocol, and load_state rejects
    # anything that is not that exact type, so a save-to-disk API cannot work.
    assert not hasattr(emulator, "save_state")
    assert not hasattr(emulator, "load_state_from_file")

    snapshot = emulator.dump_state()
    with pytest.raises(TypeError):
        memoryview(snapshot)


# -- lifecycle and errors --------------------------------------------------------


def test_step_rejects_an_out_of_range_index(emulator: MarioEmulator) -> None:
    with pytest.raises(IndexError):
        emulator.step(len(emulator.actions))


def test_step_rejects_a_bool(emulator: MarioEmulator) -> None:
    # bool is an int subclass; accepting it would silently map True to action 1.
    with pytest.raises(TypeError):
        emulator.step(True)


def test_step_rejects_a_float(emulator: MarioEmulator) -> None:
    with pytest.raises(TypeError):
        emulator.step(1.0)


def test_the_emulator_is_a_context_manager_and_closes(emulator_factory) -> None:
    with emulator_factory() as emulator:
        emulator.reset(seed=0)
        emulator.step(1)

    with pytest.raises(RuntimeError, match="closed"):
        emulator.step(1)


# -- pacing ----------------------------------------------------------------------


def test_pacer_is_a_no_op_when_disabled() -> None:
    pacer = FramePacer(60.0, enabled=False)

    assert pacer.wait() == 0.0
    assert pacer.wait() == 0.0


def test_pacer_sleeps_roughly_one_period_per_frame() -> None:
    pacer = FramePacer(200.0, enabled=True)  # 5 ms per frame
    pacer.wait()  # first call only arms the deadline

    start = time.monotonic()
    slept = pacer.wait()
    elapsed = time.monotonic() - start

    assert slept > 0.0
    # Tolerant bound: this asserts pacing happens, not that sleep is exact.
    assert elapsed >= 0.001
    assert pacer.last_lag_s == 0.0


def test_pacer_reports_lag_instead_of_skipping_frames() -> None:
    pacer = FramePacer(1000.0, enabled=True)  # 1 ms per frame
    pacer.wait()
    time.sleep(0.02)  # fall far behind the 1 ms deadline

    slept = pacer.wait()

    assert slept == 0.0
    assert pacer.last_lag_s > 0.0
    assert pacer.total_lag_s >= pacer.last_lag_s


def test_pacer_rejects_a_non_positive_rate() -> None:
    with pytest.raises(ValueError):
        FramePacer(0.0)


# -- suspension ------------------------------------------------------------------


@pytest.fixture
def sleep_calls(monkeypatch) -> list[float]:
    """Count the pacer's sleeps instead of performing them.

    ``env`` imports the ``time`` module and calls ``time.sleep`` through it, so
    replacing that attribute counts every sleep the pacer asks for while keeping
    the test instant. The clock behind ``time.monotonic`` is left alone.
    """
    import mario_jev.emulator.env as env_module

    calls: list[float] = []
    monkeypatch.setattr(env_module.time, "sleep", calls.append)
    return calls


def test_suspension_suppresses_sleeps_and_resumes_on_a_fresh_schedule(
    sleep_calls: list[float],
) -> None:
    pacer = FramePacer(200.0, enabled=True)  # 5 ms per frame
    pacer.wait()  # arm the deadline

    with pacer.suspended():
        assert pacer.wait() == 0.0
        assert pacer.wait() == 0.0
        assert sleep_calls == []

    # The outermost exit resets the schedule, so the next real frame arms a fresh
    # period instead of inheriting a deadline that expired while suspended.
    assert pacer.wait() == 0.0
    assert sleep_calls == []


def test_pacing_still_sleeps_outside_a_suspension(sleep_calls: list[float]) -> None:
    pacer = FramePacer(200.0, enabled=True)  # 5 ms per frame
    pacer.wait()  # arm the deadline

    slept = pacer.wait()

    assert slept > 0.0
    assert len(sleep_calls) == 1


def test_an_inner_suspension_exit_does_not_re_enable_pacing(
    sleep_calls: list[float],
) -> None:
    pacer = FramePacer(200.0, enabled=True)
    pacer.wait()  # arm the deadline

    with pacer.suspended():
        with pacer.suspended():
            assert pacer.wait() == 0.0
        # The inner exit must not re-enable pacing: an outer suspension is still
        # in force, so the very next wait must still perform no sleep.
        assert pacer.wait() == 0.0

    assert sleep_calls == []


@pytest.mark.parametrize("enabled", [True, False])
def test_suspension_preserves_the_enabled_state(enabled: bool) -> None:
    # A caller must never gain pacing it did not ask for, and a disabled pacer
    # must come back disabled.
    pacer = FramePacer(60.0, enabled=enabled)

    with pacer.suspended():
        pass

    assert pacer.enabled is enabled


def test_enabled_stays_a_read_only_property() -> None:
    pacer = FramePacer(60.0, enabled=True)

    with pytest.raises(AttributeError):
        pacer.enabled = False  # type: ignore[misc]


def test_reset_still_clears_the_schedule_and_the_lag() -> None:
    pacer = FramePacer(1000.0, enabled=True)  # 1 ms per frame
    pacer.wait()
    time.sleep(0.02)  # fall far behind, so there is lag to clear
    pacer.wait()
    assert pacer.last_lag_s > 0.0

    pacer.reset()

    assert pacer.last_lag_s == 0.0
    assert pacer.total_lag_s == 0.0
    assert pacer.wait() == 0.0  # a fresh deadline is armed without sleeping
