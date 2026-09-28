"""The emulator boundary.

Everything above this module works with three things and nothing else: an RGB
frame, a RAM snapshot, and an integer joypad action. That boundary is what lets
the perception layer be tested against synthetic frames and lets the rollout
planner use real savestates instead of an approximated physics model.

Two measured facts shape this adapter:

* ``nes_py`` hands out a *zero-copy view* for the default observation mode, so
  :meth:`MarioEmulator.frame` returns an explicit contiguous copy. Anything that
  keeps a frame across a step must use that method, not the live view.
* ``dump_state`` / ``load_state`` are real savestates. The rollout planner
  depends on them, and they were measured at roughly 350 fps for a
  load-state-and-step loop on this machine.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import numpy as np

from .actions import DEFAULT_ACTION_SET, Action, build_actions
from .ram import RamView

#: Height, width and channel count of the NES frame, as the package defines it.
SCREEN_SHAPE: tuple[int, int, int] = (240, 256, 3)

#: Observation mode that returns a C-contiguous copy rather than a live view.
_COPY_MODE = "rgb_array_contiguous"

#: How far behind its absolute schedule :class:`FramePacer` will let the deficit
#: grow before it stops trying to repay it. A machine this far behind is not
#: going to catch up, and pretending otherwise only makes the next steps burst:
#: every later wake-up would find the deadline long past and return without
#: sleeping, so the deficit is written off once instead of being carried.
#: How far behind the absolute schedule may fall before the deficit is written off
#: instead of repaid, in periods. Four is a **judgement, not a measurement**: at 60
#: fps it is 66 ms. Smaller, and a briefly-loaded machine would write off a deficit
#: it could still repay; larger, and one stall hands the loop a longer run of
#: sleep-free frames. There is nothing here to measure on a machine whose own
#: ``time.sleep`` cannot reliably deliver the period.
MAX_DEFICIT_PERIODS = 4


@dataclass(frozen=True, slots=True)
class StepResult:
    """The outcome of one emulator frame."""

    frame: np.ndarray
    info: Mapping[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    tick: int

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class FramePacer:
    """Wall-clock pacing so the emulator runs at roughly real time.

    The schedule is *absolute*: each deadline is the previous deadline plus one
    period, never ``now`` plus one period. That distinction is the whole point.
    Re-anchoring on ``now`` would forgive the time a slow step overran by, which
    permanently adds that step's work to the period and makes the effective rate
    ``period + work`` instead of ``period``. Advancing the deadline by one period
    instead repays the deficit over the following steps.

    A deficit is repaid by *skipping sleeps*, and skipping a sleep never skips a
    game frame: the emulator still executes exactly one frame per
    :meth:`MarioEmulator.step` call, so the game sees the same frame sequence
    either way. Only the sleeping changes.

    The lag is reported and accumulated, never hidden, so a machine that
    genuinely cannot keep up stays visible in :attr:`last_lag_s` and
    :attr:`total_lag_s` even while the schedule tries to repay it. Repayment is
    bounded by :data:`MAX_DEFICIT_PERIODS`.

    When the lag is observed, and why it matters
    --------------------------------------------

    A deficit is discovered at the **head** of a wait: :meth:`wait` reads the
    clock once on entry and compares it to the deadline it armed last time. Every
    way of falling behind therefore surfaces one wait later than it happened,
    including ``time.sleep`` itself overshooting -- and it does overshoot, because
    the OS only guarantees *at least* the requested duration.

    The consequence is precise and is not a rounding detail:

    * :attr:`last_lag_s` is ``0.0`` on the wait whose own sleep overshot. That
      wait slept, so nothing was late *when it looked*; the overshoot had not
      happened yet.
    * The **next** wait finds the clock past its deadline and reports that as its
      own ``last_lag_s``. The amount is ``overshoot - period``, not the overshoot:
      the sleep was entitled to consume one period, so a 2.5-period sleep against
      a one-period request lands 1.5 periods past the following deadline. An
      overshoot shorter than one period is absorbed by the next (shorter) sleep and
      is correctly reported as no lag at all.
    * :attr:`total_lag_s` charges that deficit exactly once, on the wait that
      discovers it. The wait that overslept charges nothing, and no oversleep is
      counted twice.

    Two consequences for a reader. ``last_lag_s`` read per frame is a one-wait
    lagged signal. ``total_lag_s`` is the *integral of observed lateness over
    waits*, not the sum of distinct stalls: while a deficit is being repaid, each
    sleep-free wait observes and charges what is still outstanding, so one
    1.5-period deficit repaid over two waits totals 2.0 periods. That is the
    intended "how far behind did this run get" reading, and it is why the total is
    the number to trust rather than the per-frame one.
    """

    __slots__ = (
        "_period",
        "_enabled",
        "_next_deadline",
        "_lag_total",
        "_last_lag",
        "_suspend_depth",
    )

    def __init__(self, target_fps: float, enabled: bool = True) -> None:
        if target_fps <= 0:
            raise ValueError("target_fps must be > 0")
        self._period = 1.0 / target_fps
        self._enabled = enabled
        self._next_deadline: float | None = None
        self._lag_total = 0.0
        self._last_lag = 0.0
        self._suspend_depth = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def last_lag_s(self) -> float:
        """How far behind schedule the previous frame finished."""
        return self._last_lag

    @property
    def total_lag_s(self) -> float:
        return self._lag_total

    def reset(self) -> None:
        self._next_deadline = None
        self._lag_total = 0.0
        self._last_lag = 0.0

    @contextmanager
    def suspended(self) -> Iterator[None]:
        """Run without pacing, then resume on a fresh schedule.

        Used by the rollout sweep: its steps are simulated futures and must not
        consume the real timeline. On exit the schedule is reset, so resuming does
        not hand the next real frame a deadline that expired while the sweep ran --
        the game is effectively paused while the agent thinks, which is what the
        pause really is.

        The suspension is nestable and is counted, not toggled: an inner exit
        leaves an outer suspension in force. ``enabled`` is deliberately left
        alone throughout, so a pacer that was disabled comes back disabled.

        The honest consequence is that the lag accounting no longer charges the
        sweep to the real frame's schedule. That is intended, not a regression: a
        suspended step is not part of the real timeline, so there is no schedule
        for it to be behind. ``last_lag_s`` and ``total_lag_s`` therefore describe
        the real frames only, and the first real frame after the sweep starts a
        fresh period from the moment it is requested.

        Be precise about what that costs: the outermost exit calls :meth:`reset`,
        which clears ``total_lag_s`` as well as the deadline. A caller must not
        read it as a total since the run began -- it is the total **since the last
        resume**. That is the right meaning here (the sweep was a pause, not a
        frame the game missed), but it is a side effect of the reset rather than a
        separate decision, so it is stated rather than implied.
        """
        self._suspend_depth += 1
        try:
            yield
        finally:
            self._suspend_depth -= 1
            if self._suspend_depth == 0:
                # Start over: a deadline set before the sweep is in the past by
                # the time pacing resumes, and carrying it over would make the
                # first real frame look late for a delay the game never had.
                self.reset()

    def wait(self) -> float:
        """Sleep until the next frame is due.

        Returns the **measured** duration of this call's sleep, in seconds: the
        clock is read either side of ``time.sleep`` and the difference is
        returned, so an oversleeping OS is reported as what it cost rather than as
        what was asked for. A wait that did not sleep returns ``0.0``.

        The measurement is deliberately *only* a return value. The schedule is
        still advanced by exactly one period and the deficit the oversleep created
        is still discovered, and charged once, at the head of the following wait
        (see the class docstring). Subtracting the measured overshoot here as well
        would charge it twice.
        """
        if self._suspend_depth:
            # A suspended step is a simulated future: it must not touch the real
            # schedule, and it must not sleep at all.
            return 0.0

        now = time.monotonic()
        if self._next_deadline is None:
            self._next_deadline = now + self._period
            self._last_lag = 0.0
            return 0.0

        if not self._enabled:
            self._next_deadline = now + self._period
            return 0.0

        remaining = self._next_deadline - now
        if remaining > 0:
            before = time.monotonic()
            time.sleep(remaining)
            slept = time.monotonic() - before
            self._next_deadline += self._period
            self._last_lag = 0.0
            return slept

        # Behind schedule: do not sleep, and record the lag honestly.
        deficit = -remaining
        self._last_lag = deficit
        self._lag_total += deficit
        if deficit > MAX_DEFICIT_PERIODS * self._period:
            # Too far behind to repay. Write the deficit off in one place, which
            # bounds the run-away: without this a permanently slow machine would
            # drift arbitrarily far behind and every later wake-up would be a
            # sleep-free burst. The lag already charged above is not erased.
            self._next_deadline = now + self._period
        else:
            # Repay, do not forgive: advance the absolute schedule exactly as
            # the happy path does, so the following steps skip their sleeps
            # until the schedule is met again.
            self._next_deadline += self._period
        return 0.0


def parse_stage(stage: str) -> tuple[int, int]:
    """Turn ``"1-1"`` into ``(1, 1)``."""
    parts = stage.strip().split("-")
    if len(parts) != 2:
        raise ValueError(f"stage must look like '1-1', got {stage!r}")
    try:
        world, level = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"stage must look like '1-1', got {stage!r}") from exc
    return world, level


class MarioEmulator:
    """A thin, honest wrapper over ``gym_super_mario_bros``.

    The wrapper deliberately uses the ROM bundled with the package
    (``gym_super_mario_bros/_roms/super-mario-bros.nes``). There is no custom-ROM
    path: adding one would mean reimplementing the package's own RAM-to-``info``
    derivation for no gain, and a configuration option that cannot work is worse
    than no option at all.
    """

    def __init__(
        self,
        stage: str = "1-1",
        action_set: str = DEFAULT_ACTION_SET,
        *,
        seed: int | None = None,
        real_time: bool = True,
        target_fps: float = 60.0,
        render_mode: str | None = None,
    ) -> None:
        # Imported lazily so the rest of the package stays importable (and
        # testable) without the native emulator.
        from gym_super_mario_bros.smb_env import SuperMarioBrosEnv

        self.stage = stage
        self._target = parse_stage(stage)
        self.actions: tuple[Action, ...] = build_actions(action_set)
        self.action_set = action_set
        self._seed = seed
        self._nes = SuperMarioBrosEnv(target=self._target, render_mode=render_mode)
        self._last_info: Mapping[str, Any] = {}
        self._tick = -1
        self._pacer = FramePacer(target_fps, enabled=real_time)
        self._closed = False

    # -- lifecycle -----------------------------------------------------------

    def reset(self, seed: int | None = None) -> StepResult:
        """Reset the episode and return the first frame."""
        self._require_open()
        effective_seed = self._seed if seed is None else seed
        screen, info = self._nes.reset(seed=effective_seed)
        self._pacer.reset()
        self._tick = 0
        self._last_info = dict(info)
        return StepResult(
            frame=np.ascontiguousarray(screen),
            info=self._last_info,
            reward=0.0,
            terminated=False,
            truncated=False,
            tick=self._tick,
        )

    def step(self, action: Action | int) -> StepResult:
        """Advance exactly one frame with the given action."""
        self._require_open()
        bits, _ = self._resolve_action(action)
        screen, reward, terminated, truncated, info = self._nes.step(bits)
        self._tick += 1
        self._last_info = dict(info)
        self._pacer.wait()
        return StepResult(
            frame=np.ascontiguousarray(screen),
            info=self._last_info,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            tick=self._tick,
        )

    def close(self) -> None:
        if not self._closed:
            self._nes.close()
            self._closed = True

    def __enter__(self) -> "MarioEmulator":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("emulator is closed")

    # -- actions -------------------------------------------------------------

    def _resolve_action(self, action: Action | int) -> tuple[int, int]:
        if isinstance(action, Action):
            return action.bits, action.index
        if isinstance(action, bool) or not isinstance(action, int):
            raise TypeError(f"action must be an Action or int, got {type(action).__name__}")
        if not 0 <= action < len(self.actions):
            raise IndexError(
                f"action index {action} is outside the {len(self.actions)}-action set"
            )
        return self.actions[action].bits, action

    @property
    def action_names(self) -> tuple[str, ...]:
        return tuple(action.name for action in self.actions)

    def action(self, index: int) -> Action:
        return self.actions[index]

    # -- observation ---------------------------------------------------------

    def frame(self) -> np.ndarray:
        """A fresh C-contiguous RGB copy of the current frame.

        Use this — not :meth:`screen_view` — whenever the frame must survive
        another emulator step.
        """
        self._require_open()
        copied = self._nes.observation(_COPY_MODE)
        return np.ascontiguousarray(copied)

    def screen_view(self) -> np.ndarray:
        """The emulator's live screen buffer. Mutated in place by every step."""
        self._require_open()
        return self._nes.screen

    def ram(self) -> RamView:
        """A detached RAM snapshot, valid across later steps."""
        self._require_open()
        return RamView(self._nes.ram)

    @property
    def info(self) -> Mapping[str, Any]:
        return self._last_info

    @property
    def tick(self) -> int:
        return self._tick

    # -- savestates ----------------------------------------------------------

    def dump_state(self) -> Any:
        """An opaque snapshot of the whole emulator, for rollout planning."""
        self._require_open()
        return self._nes.dump_state()

    def load_state(self, snapshot: Any) -> None:
        """Restore a snapshot produced by :meth:`dump_state`.

        The frame counter intentionally does *not* rewind: a rollout is not
        gameplay, and rewinding the tick would corrupt the real-time trace.

        The cached ``info`` is refreshed from live RAM afterwards. That step is
        required, not cosmetic: the gym wrapper derives ``x_pos``, ``y_pixel``
        and the rest from RAM at call time, so without it ``info`` would keep
        describing the frame we just rolled *forward* from.
        """
        self._require_open()
        self._nes.load_state(snapshot)
        self._last_info = dict(self._nes._get_info())

    # -- pacing diagnostics --------------------------------------------------

    @property
    def pacer(self) -> FramePacer:
        return self._pacer


# Note on disk savestates: ``nes_py``'s ``NativeStateSnapshot`` is fully opaque.
# It exposes no ``__bytes__``, no ``__reduce__``, and no buffer protocol, and
# ``load_state`` rejects anything that is not that exact type, so a snapshot
# cannot be written to disk or reloaded from one. A file-backed savestate would
# therefore be dead code, and it is deliberately not implemented.
