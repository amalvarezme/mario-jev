"""The emulator boundary, tested against the packages it wraps.

Several tests here exist because a plausible assumption turned out to be wrong
during development, and the point of pinning them is that they cannot quietly
become true again.
"""

from __future__ import annotations

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
from mario_jev.emulator.env import MAX_DEFICIT_PERIODS

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

#: 64 fps, not 60. ``1/64`` is exact in binary floating point and so is every
#: multiple of it used below, which lets the schedule assertions be exact
#: equalities instead of tolerances that could hide real drift.
VIRTUAL_FPS = 64.0
VIRTUAL_PERIOD = 1.0 / VIRTUAL_FPS


class VirtualClock:
    """A monotonic clock that only moves when a test says so.

    Every pacing property below is pinned against this clock, and **no test in
    this file asserts a real elapsed interval**. That is deliberate: a loaded
    machine turns a requested 16.67 ms ``time.sleep`` into ~100 ms, so a
    wall-clock assertion would measure the machine rather than the pacer and
    would be a flake by construction.

    ``env`` imports the ``time`` module and calls ``time.monotonic`` and
    ``time.sleep`` through it, so replacing those two attributes is enough to put
    the pacer on this clock.

    The :func:`clock` fixture that installs it is ``autouse`` for this whole file
    rather than requested per test, so "no pacer in this file ever reaches the real
    clock" is a property of the module instead of a habit each test has to
    remember. Nothing here needs the real clock: the emulator fixtures build with
    ``real_time=False``, so their pacer never sleeps either way.
    """

    def __init__(self, overshoots: list[float] | None = None) -> None:
        self.now = 0.0
        #: Every duration the pacer asked to sleep, in order.
        self.sleeps: list[float] = []
        #: Extra time each sleep overshoots by, consumed one entry per sleep.
        #: Empty means sleeps land exactly on the deadline.
        self.overshoots: list[float] = list(overshoots or ())

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        overshoot = self.overshoots.pop(0) if self.overshoots else 0.0
        self.now += seconds + overshoot

    def advance(self, seconds: float) -> None:
        """Simulate work: time passes without the pacer having slept for it."""
        self.now += seconds


@pytest.fixture(autouse=True)
def clock(monkeypatch) -> VirtualClock:
    """Put ``env``'s clock on a :class:`VirtualClock` for every test in this file.

    ``autouse`` on purpose. A test that only wants to know that a *disabled* pacer
    does not sleep still calls ``time.monotonic`` through ``env``, and if it were
    the one test left unpatched then the file's claim would be "almost every pacing
    test drives the virtual clock", which is not a property anyone can rely on.
    Tests that need to move time or inspect the sleeps request ``clock`` by name
    and get this same instance.
    """
    import mario_jev.emulator.env as env_module

    virtual = VirtualClock()
    monkeypatch.setattr(env_module.time, "monotonic", virtual.monotonic)
    monkeypatch.setattr(env_module.time, "sleep", virtual.sleep)
    return virtual


def test_pacer_is_a_no_op_when_disabled(clock: VirtualClock) -> None:
    pacer = FramePacer(60.0, enabled=False)

    assert pacer.wait() == 0.0
    assert pacer.wait() == 0.0
    # Named explicitly so the no-op claim covers the sleeping too, not only the
    # returned zeros: a disabled pacer that slept and returned 0.0 would pass the
    # two assertions above.
    assert clock.sleeps == []


def test_pacer_sleeps_one_period_per_frame(clock: VirtualClock) -> None:
    pacer = FramePacer(VIRTUAL_FPS, enabled=True)
    pacer.wait()  # first call only arms the deadline

    slept = pacer.wait()

    assert slept == VIRTUAL_PERIOD
    assert clock.sleeps == [VIRTUAL_PERIOD]
    assert pacer.last_lag_s == 0.0


def test_the_schedule_does_not_drift_when_every_sleep_is_exact(
    clock: VirtualClock,
) -> None:
    # 100 waits on a clock where ``sleep(d)`` advances exactly ``d``: the average
    # interval must be the period itself, to the last bit, and nothing may be
    # reported as lag.
    pacer = FramePacer(VIRTUAL_FPS, enabled=True)
    pacer.wait()  # arms the deadline without moving the clock
    start = clock.now

    for _ in range(100):
        pacer.wait()

    elapsed = clock.now - start
    assert elapsed == 100 * VIRTUAL_PERIOD
    assert elapsed / 100 == VIRTUAL_PERIOD
    assert clock.sleeps == [VIRTUAL_PERIOD] * 100
    assert pacer.total_lag_s == 0.0
    assert pacer.last_lag_s == 0.0


def test_an_overshooting_sleep_is_repaid_not_forgiven(clock: VirtualClock) -> None:
    """One late step must not turn into a permanently slower rate.

    Re-anchoring the deadline on ``now`` would forgive the overshoot and add it
    to the period for good. Advancing the absolute schedule instead repays it by
    skipping the next sleeps, so the *average* interval returns to the period.

    The observed sequence, in periods, with the one sleep overshooting by 2.5
    periods (the nine waits after the arming call)::

        returned:  3.5  0.0  0.0  0.5  1.0  1.0  1.0  1.0  1.0
        requested: 1.0   --   --  0.5  1.0  1.0  1.0  1.0  1.0
        last_lag:  0.0  1.5  0.5  0.0  0.0  0.0  0.0  0.0  0.0

    The first wait asked for one period and ``sleep`` took 3.5, and ``wait()``
    returns the **measured** 3.5 rather than the 1.0 it requested -- that is the
    whole difference between the two rows.

    The two ``0.0`` sleeps are the outstanding deficit being repaid: those
    wake-ups happen with no sleep at all while the schedule is still behind.
    Skipping those sleeps skips no game frame -- ``step()`` still executes
    exactly one frame per call either way.
    """
    clock.overshoots = [2.5 * VIRTUAL_PERIOD]
    pacer = FramePacer(VIRTUAL_FPS, enabled=True)
    pacer.wait()  # arms the deadline
    start = clock.now

    slept = [pacer.wait() for _ in range(9)]

    assert slept == [
        3.5 * VIRTUAL_PERIOD,
        0.0,
        0.0,
        0.5 * VIRTUAL_PERIOD,
        VIRTUAL_PERIOD,
        VIRTUAL_PERIOD,
        VIRTUAL_PERIOD,
        VIRTUAL_PERIOD,
        VIRTUAL_PERIOD,
    ]
    # The requested duration is still one period: the return value is measured
    # around the sleep, and the schedule was still advanced by one period.
    assert clock.sleeps[0] == VIRTUAL_PERIOD
    # The property that matters: the average interval is the period again.
    elapsed = clock.now - start
    assert elapsed == 9 * VIRTUAL_PERIOD
    assert elapsed / 9 == VIRTUAL_PERIOD
    # And the lag stays visible rather than being smoothed away.
    assert pacer.total_lag_s == 2.0 * VIRTUAL_PERIOD
    assert pacer.last_lag_s == 0.0


def test_the_deficit_stops_growing_once_it_passes_the_bound(
    clock: VirtualClock,
) -> None:
    """A permanently slow machine must not drift arbitrarily far behind.

    Every step here does 1.5 periods of work, so the pacer can never catch up.
    The deficit is repaid one period at a time until it is further behind than
    ``MAX_DEFICIT_PERIODS``, at which point the schedule re-anchors and the
    deficit starts over. The observed ``last_lag_s`` sequence, in periods::

        0.5  1.0  1.5  2.0  2.5  3.0  3.5  4.0  4.5  0.5  1.0  1.5  2.0  2.5

    and the matching ``total_lag_s``, which only ever grows::

        0.5  1.5  3.0  5.0  7.5  10.5  14.0  18.0  22.5  23.0  24.0  25.5 ...

    The drop from 4.5 back to 0.5 is the bound doing its job: 4.5 > 4 periods, so
    that step re-anchors. The lag it reported is still charged to the total.
    """
    work = 1.5 * VIRTUAL_PERIOD
    pacer = FramePacer(VIRTUAL_FPS, enabled=True)
    clock.advance(work)
    pacer.wait()  # arms the deadline

    lags: list[float] = []
    totals: list[float] = []
    for _ in range(14):
        clock.advance(work)
        pacer.wait()
        lags.append(pacer.last_lag_s)
        totals.append(pacer.total_lag_s)

    in_periods = [lag / VIRTUAL_PERIOD for lag in lags]
    assert in_periods == [
        0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 0.5, 1.0, 1.5, 2.0, 2.5
    ]
    # (a) the machine's inability to keep up stays visible and keeps growing.
    assert all(later > earlier for earlier, later in zip(totals, totals[1:]))
    assert totals[-1] == 30.0 * VIRTUAL_PERIOD
    # (b) the deficit re-anchors instead of running away: it never exceeds the
    # bound plus the one step of work that discovered it.
    assert max(lags) == 4.5 * VIRTUAL_PERIOD
    assert max(lags) <= (MAX_DEFICIT_PERIODS + 1) * VIRTUAL_PERIOD
    # (c) a machine this far behind never sleeps at all, so no pacer sleep can
    # delay one of its steps.
    assert clock.sleeps == []

    # And when the work stops, the outstanding deficit is repaid with skipped and
    # shortened sleeps -- never with a sleep longer than one period, which is what
    # would delay a later step.
    clock.sleeps.clear()
    tail = [pacer.wait() for _ in range(5)]
    assert tail == [
        0.0,
        0.0,
        0.5 * VIRTUAL_PERIOD,
        VIRTUAL_PERIOD,
        VIRTUAL_PERIOD,
    ]
    assert all(slept <= VIRTUAL_PERIOD for slept in clock.sleeps)


def test_a_step_that_overran_the_period_reports_how_late_it_was(
    clock: VirtualClock,
) -> None:
    # ``wait()`` called 2.5 periods after the deadline was armed: the step's own
    # work already ate 2.5 periods, so the step is 1.5 periods late and must say
    # so instead of reporting zero.
    pacer = FramePacer(VIRTUAL_FPS, enabled=True)
    pacer.wait()  # arms the deadline
    clock.advance(2.5 * VIRTUAL_PERIOD)

    assert pacer.wait() == 0.0
    assert pacer.last_lag_s == 1.5 * VIRTUAL_PERIOD

    # The remaining 0.5 period of deficit is still outstanding, which is how the
    # repayment is visible: the next wake-up is also sleep-free and still late.
    assert pacer.wait() == 0.0
    assert pacer.last_lag_s == 0.5 * VIRTUAL_PERIOD
    assert pacer.total_lag_s == 2.0 * VIRTUAL_PERIOD
    assert clock.sleeps == []


def test_pacer_reports_lag_instead_of_hiding_it(clock: VirtualClock) -> None:
    pacer = FramePacer(VIRTUAL_FPS, enabled=True)
    pacer.wait()
    clock.advance(20 * VIRTUAL_PERIOD)  # fall far behind the deadline

    slept = pacer.wait()

    assert slept == 0.0
    assert pacer.last_lag_s > 0.0
    assert pacer.total_lag_s >= pacer.last_lag_s


def test_wait_returns_the_measured_sleep_and_charges_an_oversleep_once(
    clock: VirtualClock,
) -> None:
    """The four facts about an oversleep, in one place.

    ``time.sleep`` guarantees *at least* the requested duration, so a wait can ask
    for one period and be gone for 2.5. This pins what the pacer says about that:

    (a) ``wait()`` returns the measured sleep, not the request;
    (b) ``last_lag_s`` is ``0.0`` on the wait whose sleep overshot -- it read the
        clock before sleeping, so nothing was late when it looked;
    (c) the *next* wait reports exactly the resulting deficit as its own
        ``last_lag_s``;
    (d) that deficit is charged to ``total_lag_s`` exactly once, by the wait that
        discovered it, and by no other wait.

    Be exact about (c), because the obvious reading is wrong: the deficit is not
    the overshoot, it is ``overshoot - period``. The sleep was entitled to consume
    one period, so a 2.5-period sleep against a 1-period request lands 1.5 periods
    past the *next* deadline. An overshoot under one period is absorbed by the next
    (shorter) sleep and is correctly reported as no lag at all.
    """
    clock.overshoots = [2.5 * VIRTUAL_PERIOD]
    pacer = FramePacer(VIRTUAL_FPS, enabled=True)
    pacer.wait()  # arms the deadline

    # (a) one period was requested; 3.5 periods were actually spent.
    returned = pacer.wait()
    assert clock.sleeps == [VIRTUAL_PERIOD]
    assert returned == 3.5 * VIRTUAL_PERIOD
    assert returned != clock.sleeps[0]

    # (b) that wait reports no lag: it slept, and it measured before sleeping.
    assert pacer.last_lag_s == 0.0
    assert pacer.total_lag_s == 0.0

    # (c) the deficit surfaces on the following wait, to the exact amount, and
    # (d) it is charged there and nowhere else: the wait that overslept charged
    # nothing at all, and the discovering wait's charge equals what it observed.
    assert pacer.wait() == 0.0
    assert pacer.last_lag_s == 1.5 * VIRTUAL_PERIOD
    assert pacer.total_lag_s == 1.5 * VIRTUAL_PERIOD

    # The repayment is not a second charge of the 1.5: the schedule advanced one
    # period, so what is still outstanding is 0.5, and that is what this wait
    # observes and adds. ``total_lag_s`` is the integral of observed lateness, and
    # it is the round-A arithmetic this fix deliberately did not move.
    assert pacer.wait() == 0.0
    assert pacer.last_lag_s == 0.5 * VIRTUAL_PERIOD
    assert pacer.total_lag_s == 2.0 * VIRTUAL_PERIOD

    # Repaid in full: the next waits sleep again, and nothing further is charged.
    assert pacer.wait() == 0.5 * VIRTUAL_PERIOD
    assert pacer.wait() == VIRTUAL_PERIOD
    assert pacer.last_lag_s == 0.0
    assert pacer.total_lag_s == 2.0 * VIRTUAL_PERIOD


def test_an_oversleep_under_one_period_is_absorbed_not_reported(
    clock: VirtualClock,
) -> None:
    # The negative case for the rule above: the sleep took 1.5 periods, which is
    # still short of the next deadline at 2.0, so the next sleep is simply shorter
    # and no lag is reported. Reporting one here would make every ordinary
    # scheduling jitter look like a machine that cannot keep up.
    #
    # 0.5 rather than a friendlier-looking 0.4: only multiples of 1/64 are exact in
    # binary floating point, and this file asserts exact equalities on purpose.
    clock.overshoots = [0.5 * VIRTUAL_PERIOD]
    pacer = FramePacer(VIRTUAL_FPS, enabled=True)
    pacer.wait()  # arms the deadline

    assert pacer.wait() == 1.5 * VIRTUAL_PERIOD
    assert pacer.wait() == 0.5 * VIRTUAL_PERIOD
    assert clock.sleeps == [VIRTUAL_PERIOD, 0.5 * VIRTUAL_PERIOD]
    assert pacer.last_lag_s == 0.0
    assert pacer.total_lag_s == 0.0


def test_pacer_rejects_a_non_positive_rate() -> None:
    with pytest.raises(ValueError):
        FramePacer(0.0)


# -- suspension ------------------------------------------------------------------


@pytest.fixture
def sleep_calls(clock: VirtualClock) -> list[float]:
    """Every sleep the pacer asked for, recorded on the virtual clock."""
    return clock.sleeps


def test_suspension_suppresses_sleeps_and_resumes_on_a_fresh_schedule(
    sleep_calls: list[float],
) -> None:
    pacer = FramePacer(VIRTUAL_FPS, enabled=True)
    pacer.wait()  # arm the deadline

    with pacer.suspended():
        assert pacer.wait() == 0.0
        assert pacer.wait() == 0.0
        assert sleep_calls == []

    # The outermost exit resets the schedule, so the next real frame arms a fresh
    # period instead of inheriting a deadline that expired while suspended.
    assert pacer.wait() == 0.0
    assert sleep_calls == []


def test_a_suspended_step_leaves_the_schedule_untouched(clock: VirtualClock) -> None:
    # A suspended step is a simulated future: it must neither sleep nor move the
    # deadline, even when real time passes while the sweep runs.
    pacer = FramePacer(VIRTUAL_FPS, enabled=True)
    pacer.wait()  # arms the deadline
    armed = pacer._next_deadline

    with pacer.suspended():
        clock.advance(10 * VIRTUAL_PERIOD)
        pacer.wait()
        pacer.wait()
        assert pacer._next_deadline == armed
        assert pacer.last_lag_s == 0.0
        assert clock.sleeps == []


def test_pacing_still_sleeps_outside_a_suspension(sleep_calls: list[float]) -> None:
    pacer = FramePacer(VIRTUAL_FPS, enabled=True)
    pacer.wait()  # arm the deadline

    slept = pacer.wait()

    assert slept == VIRTUAL_PERIOD
    assert sleep_calls == [VIRTUAL_PERIOD]


def test_an_inner_suspension_exit_does_not_re_enable_pacing(
    sleep_calls: list[float],
) -> None:
    pacer = FramePacer(VIRTUAL_FPS, enabled=True)
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


def test_reset_still_clears_the_schedule_and_the_lag(clock: VirtualClock) -> None:
    pacer = FramePacer(VIRTUAL_FPS, enabled=True)
    pacer.wait()
    clock.advance(20 * VIRTUAL_PERIOD)  # fall far behind, so there is lag to clear
    pacer.wait()
    assert pacer.last_lag_s > 0.0

    pacer.reset()

    assert pacer.last_lag_s == 0.0
    assert pacer.total_lag_s == 0.0
    assert pacer.wait() == 0.0  # a fresh deadline is armed without sleeping
