"""One observation -> :class:`MarioState`, with the speed measured across ticks.

The builder is *stateful* on purpose. Two of the relative features the plan asks
for cannot be read from a single frame:

* ``frames_to_jump_window`` needs ``x_pos`` measured at two successive ticks;
* ``approaching`` needs the same enemy's on-screen column at two successive ticks.

A stateless function would have to take the previous sample as an argument and
would let a caller forget it. Holding the previous sample here makes "observed
once" the only way to build a state, and :meth:`MarioStateBuilder.reset` makes
the next sample the first one again.

Nothing here detects enemies: they come from RAM, exactly as the perception layer
documents. The frame is used only for terrain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from mario_jev.emulator import RAM_ADDRESSES, EnemySlot
from mario_jev.perception import TILE, FramePerception, ram_player_box
from mario_jev.perception.palette import PLAYER_BOX_SIZE

from .mario_state import (
    INFO_LIFE,
    INFO_STAGE,
    INFO_TIME,
    INFO_WORLD,
    INFO_X_POS,
    INFO_X_POS_MAX,
    MarioState,
    NearestEnemy,
    PlayerFeatures,
    Progress,
    TerrainAhead,
)

#: How far ahead of Mario's front edge the terrain readings look, in pixels.
#: Four tiles: long enough to see a pit before a jump window closes, short enough
#: that the reading still describes what is in front of Mario.
DEFAULT_LOOKAHEAD_PX: int = 64

#: A gap that begins this frame is reported as one frame away, never zero: the
#: value is a countdown, and a countdown that has reached zero has already ended.
MIN_JUMP_WINDOW_FRAMES: int = 1


class RamLike(Protocol):
    """What this module needs from a RAM view.

    Typed loosely, like :func:`mario_jev.perception.audit`, so a test can hand in
    a two-method stub instead of a live emulator.
    """

    def byte(self, address: int) -> int: ...

    def active_enemies(self) -> tuple[EnemySlot, ...]: ...


def sprite_gap_px(a_x: int, b_x: int, *, size: int = PLAYER_BOX_SIZE) -> int:
    """Empty pixels between two 16-px sprite boxes at on-screen columns.

    Zero means the boxes touch or overlap, which is the only honest reading: a
    negative "distance" would be a collision, not a distance.
    """
    a_right = a_x + size - 1
    b_right = b_x + size - 1
    return max(0, max(a_x - b_right, b_x - a_right))


def nearest_enemy_gap(
    screen_x: int,
    enemies: tuple[EnemySlot, ...],
    *,
    size: int = PLAYER_BOX_SIZE,
) -> tuple[int | None, EnemySlot | None]:
    """The closest active enemy that is not behind Mario.

    Returns ``(None, None)`` when no active enemy overlaps or sits ahead. Ties are
    broken by RAM slot so the answer is deterministic.
    """
    ahead = tuple(enemy for enemy in enemies if int(enemy.drawn_x) + size - 1 >= screen_x)
    if not ahead:
        return None, None
    nearest = min(ahead, key=lambda e: (sprite_gap_px(screen_x, int(e.drawn_x), size=size), e.slot))
    return sprite_gap_px(screen_x, int(nearest.drawn_x), size=size), nearest


@dataclass(frozen=True, slots=True)
class _Sample:
    """The previous observation, kept only to difference against."""

    tick: int
    x_pos: int
    screen_x: int
    facing: str
    enemy_x: Mapping[int, int]


class MarioStateBuilder:
    """Builds :class:`MarioState` from successive observations."""

    __slots__ = ("_lookahead_px", "_previous")

    def __init__(self, *, lookahead_px: int = DEFAULT_LOOKAHEAD_PX) -> None:
        if lookahead_px < 1:
            raise ValueError(f"lookahead_px must be >= 1, got {lookahead_px}")
        self._lookahead_px = int(lookahead_px)
        self._previous: _Sample | None = None

    @property
    def lookahead_px(self) -> int:
        return self._lookahead_px

    def reset(self) -> None:
        """Forget the previous sample, e.g. after the episode restarts."""
        self._previous = None

    def observe(
        self,
        perception: FramePerception,
        ram: RamLike,
        info: Mapping[str, Any],
    ) -> MarioState:
        """Turn one tick's perception, RAM snapshot and ``info`` into typed state."""
        tick = int(perception.tick)
        x_pos = int(info[INFO_X_POS])
        screen_x = int(ram.byte(RAM_ADDRESSES["player_x_offset"]))
        y_pixel = int(ram.byte(RAM_ADDRESSES["player_y_on_screen"]))
        terrain = perception.terrain

        box = ram_player_box(screen_x, y_pixel)
        feet_row = int(box[3])

        previous = self._previous
        speed = 0.0
        if previous is not None:
            elapsed_ticks = tick - previous.tick
            if elapsed_ticks > 0:
                speed = (x_pos - previous.x_pos) / elapsed_ticks

        # Facing comes from the measured velocity. While Mario stands still there
        # is no velocity to read, so the last known facing is carried forward
        # rather than invented.
        if speed > 0.0:
            facing = "right"
        elif speed < 0.0:
            facing = "left"
        elif previous is not None:
            facing = previous.facing
        else:
            facing = "right"

        front_edge = screen_x + PLAYER_BOX_SIZE
        gap_distance = terrain.hole_distance(front_edge, self._lookahead_px)
        gap_columns = terrain.hole_width_ahead(front_edge, self._lookahead_px)
        # Raised ground in Super Mario Bros. is built from 16-px blocks, so a
        # sub-tile offset is not a step: it is noise from the sprites, seams and
        # props the frame draws against the floor (measured: the flat 1-1 start
        # reports a raw offset of 1 px). Flooring to whole tiles is what makes the
        # reading about the level rather than about the last sprite that passed.
        wall_height = (terrain.surface_offset(front_edge, self._lookahead_px) // TILE) * TILE

        frames_to_jump_window: int | None = None
        if gap_distance is not None and speed > 0.0:
            frames_to_jump_window = max(
                MIN_JUMP_WINDOW_FRAMES, math.ceil(gap_distance / speed)
            )

        center = screen_x + PLAYER_BOX_SIZE // 2
        surface_row: int | None = None
        if 0 <= center < terrain.width:
            surface_row = terrain.floor_row[center]

        vertical_clearance = 0
        on_ground = False
        if surface_row is not None:
            vertical_clearance = max(0, int(surface_row) - feet_row - 1)
            on_ground = feet_row >= int(surface_row) - 1

        active = tuple(ram.active_enemies())
        distance, nearest = nearest_enemy_gap(screen_x, active)

        approaching = False
        if nearest is not None and previous is not None:
            before = previous.enemy_x.get(int(nearest.slot))
            if before is not None:
                moved = int(nearest.drawn_x) - before
                approaching = moved * (screen_x - int(nearest.drawn_x)) > 0

        self._previous = _Sample(
            tick=tick,
            x_pos=x_pos,
            screen_x=screen_x,
            facing=facing,
            enemy_x={int(enemy.slot): int(enemy.drawn_x) for enemy in active},
        )

        return MarioState(
            tick=tick,
            player=PlayerFeatures(
                x=x_pos,
                y=y_pixel,
                screen_x=screen_x,
                feet_row=feet_row,
                on_ground=on_ground,
                facing=facing,
            ),
            nearest_enemy=NearestEnemy(
                distance_px=distance,
                approaching=approaching,
                slot=None if nearest is None else int(nearest.slot),
            ),
            terrain_ahead=TerrainAhead(
                gap_columns=gap_columns,
                gap_distance_px=gap_distance,
                wall_height=wall_height,
                frames_to_jump_window=frames_to_jump_window,
                vertical_clearance=vertical_clearance,
                lookahead_px=self._lookahead_px,
            ),
            progress=Progress(
                world=f"{int(info[INFO_WORLD])}-{int(info[INFO_STAGE])}",
                x_max=int(info[INFO_X_POS_MAX]),
                lives=int(info[INFO_LIFE]),
                time_left=int(info[INFO_TIME]),
            ),
            speed_px_per_tick=float(speed),
        )


__all__ = [
    "DEFAULT_LOOKAHEAD_PX",
    "MIN_JUMP_WINDOW_FRAMES",
    "MarioStateBuilder",
    "RamLike",
    "nearest_enemy_gap",
    "sprite_gap_px",
]
