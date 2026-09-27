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
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .actions import DEFAULT_ACTION_SET, Action, build_actions
from .ram import RamView

#: Height, width and channel count of the NES frame, as the package defines it.
SCREEN_SHAPE: tuple[int, int, int] = (240, 256, 3)

#: Observation mode that returns a C-contiguous copy rather than a live view.
_COPY_MODE = "rgb_array_contiguous"


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

    When the caller falls behind, this reports the lag instead of trying to
    catch up by skipping frames: skipping would silently change the game speed
    and make every measurement meaningless.
    """

    __slots__ = ("_period", "_enabled", "_next_deadline", "_lag_total", "_last_lag")

    def __init__(self, target_fps: float, enabled: bool = True) -> None:
        if target_fps <= 0:
            raise ValueError("target_fps must be > 0")
        self._period = 1.0 / target_fps
        self._enabled = enabled
        self._next_deadline: float | None = None
        self._lag_total = 0.0
        self._last_lag = 0.0

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

    def wait(self) -> float:
        """Sleep until the next frame is due. Returns the seconds actually slept."""
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
            time.sleep(remaining)
            self._next_deadline += self._period
            self._last_lag = 0.0
            return remaining

        # Behind schedule: do not sleep, and record the lag honestly.
        self._last_lag = -remaining
        self._lag_total += self._last_lag
        self._next_deadline = now + self._period
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
