"""Frame -> detections.

Everything here is deterministic and local: no network, no model, no randomness.
Given the same frame it produces the same detections, which is what makes the
recorded evidence replayable.

What this layer detects, and what it deliberately does not
---------------------------------------------------------
It detects:

* **the player**, from his red pixels, as a bounding box plus a confidence;
* **the terrain**, from the floor palette: the floor surface row per column and
  which columns are pits.

It does **not** detect enemies, and the reason is measured rather than assumed.
At world 1-1 the Goomba's cap is drawn with the palette entries ``(228, 92, 16)``
and ``(252, 160, 68)`` — the *same* entries used by ground blocks, bricks and
coins. A colour-only classifier therefore cannot separate an enemy from level
furniture. A first attempt keyed on the brown entry ``(172, 124, 0)`` and
produced exactly the wrong answer: in sampled frames that entry belongs to
**Mario's own shoes and buttons** (measured: 46 brown pixels at columns 114..125,
which is inside Mario's box), while the visible Goomba was not isolated at all.
Shipping that would have been a detector that looks like it works.

Enemies therefore come from RAM, which is exact, free, and available every frame.
The player's position is taken from the frame *and* cross-checked against RAM by
:func:`audit`, so the disagreement rate is a measured number.

Measured structure of an overworld frame (world 1-1, this emulator)
-------------------------------------------------------------------
* The fixed HUD occupies the top two tile rows (0..31) and contains a red coin
  icon, which is why the HUD is excluded before looking for Mario.
* The floor occupies rows **208..239**. Row 208 is the highlight line; below it
  blocks alternate ground/highlight/black. Every column has floor unless it is a pit.
* The player's sprite box derived from RAM is ``[x, x+16) x [y+16, y+32)``. With
  ``0x03AD = 112`` and ``0x03B8 = 176`` that is rows 192..207, so the player's
  feet sit exactly on row 207, one pixel above the floor top at 208.
* The floor blocks are separated by **black seams**. A naive "contiguous from the
  bottom edge" scan stops on the first seam and reports the whole floor as a
  single row at 239 — which is what happened before black was counted as solid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from .palette import (
    BLACK,
    GROUND,
    GROUND_HIGHLIGHT,
    HUD_ROWS,
    MARIO_RED,
    PLAYER_BOX_DX,
    PLAYER_BOX_DY,
    TILE,
    dominant_colour,
    mask_for,
)

#: The floor's own colours.
FLOOR_COLOURS: Final[tuple[tuple[int, int, int], ...]] = (GROUND, GROUND_HIGHLIGHT)

#: What counts as solid when scanning up from the bottom edge. The black seam
#: between floor blocks must be included, or the scan stops on the first seam.
SOLID_COLOURS: Final[tuple[tuple[int, int, int], ...]] = (*FLOOR_COLOURS, BLACK)

#: Fewest red pixels that still count as "Mario is on screen".
MIN_MARIO_RED_PIXELS: Final[int] = 8

#: Red-pixel count across a walk cycle, measured over 260 live frames: it ranges
#: 37..55 with a median of 45, because the walk animation occludes different parts
#: of the sprite. A test that pinned 45 would be asserting one pose, not the sprite.
MARIO_RED_PIXELS_RANGE: Final[tuple[int, int]] = (37, 55)

#: Median red-pixel count, used as the confidence denominator.
EXPECTED_MARIO_RED_PIXELS: Final[int] = 45


@dataclass(frozen=True, slots=True)
class TerrainProfile:
    """The floor and the pits."""

    background: tuple[int, int, int]
    #: Top row of the contiguous solid run from the bottom, per column.
    #: ``None`` means the column has no floor at all — a pit.
    floor_row: tuple[int | None, ...]
    #: The row the level's floor generally sits at, i.e. the modal floor row.
    ground_line: int
    #: Columns with no floor: a pit.
    hole_columns: tuple[int, ...]

    @property
    def width(self) -> int:
        return len(self.floor_row)

    def is_hole(self, x: int) -> bool:
        if not 0 <= x < self.width:
            return False
        return self.floor_row[x] is None

    def hole_width_ahead(self, x: int, lookahead: int) -> int:
        """Width of the contiguous pit starting at or after ``x``.

        Returns 0 when there is no pit within ``lookahead`` columns.
        """
        for column in range(max(0, x), min(self.width, x + lookahead)):
            if self.floor_row[column] is None:
                run = 0
                probe = column
                while probe < self.width and self.floor_row[probe] is None:
                    run += 1
                    probe += 1
                return run
        return 0

    def hole_distance(self, x: int, lookahead: int) -> int | None:
        """Columns from ``x`` to the nearest pit edge, or ``None`` if clear."""
        for column in range(max(0, x), min(self.width, x + lookahead)):
            if self.floor_row[column] is None:
                return column - x
        return None

    def surface_offset(self, x: int, lookahead: int) -> int:
        """How far above the ground line the highest floor within reach sits.

        Positive means raised ground (a step up); 0 means level ground.
        """
        end = min(self.width, x + lookahead)
        rows = [
            self.floor_row[column]
            for column in range(max(0, x), end)
            if self.floor_row[column] is not None
        ]
        if not rows:
            return 0
        return self.ground_line - min(rows)

    def as_dict(self) -> dict[str, object]:
        return {
            "background": list(self.background),
            "ground_line": self.ground_line,
            "hole_columns": list(self.hole_columns),
            "hole_count": len(self.hole_columns),
        }


@dataclass(frozen=True, slots=True)
class FramePerception:
    """Everything the perception layer concluded about one frame."""

    tick: int
    background: tuple[int, int, int]
    terrain: TerrainProfile
    mario_box: tuple[int, int, int, int] | None
    mario_red_pixels: int
    mario_confidence: float

    @property
    def mario_detected(self) -> bool:
        return self.mario_box is not None

    def as_dict(self) -> dict[str, object]:
        return {
            "tick": self.tick,
            "background": list(self.background),
            "terrain": self.terrain.as_dict(),
            "mario_box": list(self.mario_box) if self.mario_box else None,
            "mario_red_pixels": self.mario_red_pixels,
            "mario_confidence": round(self.mario_confidence, 4),
        }


@dataclass(frozen=True, slots=True)
class PerceptionAudit:
    """Frame-derived readings against RAM ground truth.

    This is the honesty surface of the vision layer: it is recorded every tick so
    the disagreement rate is a measured number rather than a claim.
    """

    mario_detected_from_frame: bool
    mario_box_from_frame: tuple[int, int, int, int] | None
    mario_box_from_ram: tuple[int, int, int, int] | None
    mario_inside_ram_box: bool | None
    offset: tuple[int, int] | None
    feet_on_ground: bool | None
    ram_active_enemies: int

    @property
    def agreed(self) -> bool | None:
        return self.mario_inside_ram_box

    def as_dict(self) -> dict[str, object]:
        return {
            "mario_detected_from_frame": self.mario_detected_from_frame,
            "mario_box_from_frame": (
                list(self.mario_box_from_frame) if self.mario_box_from_frame else None
            ),
            "mario_box_from_ram": (
                list(self.mario_box_from_ram) if self.mario_box_from_ram else None
            ),
            "mario_inside_ram_box": self.mario_inside_ram_box,
            "offset": list(self.offset) if self.offset else None,
            "feet_on_ground": self.feet_on_ground,
            "ram_active_enemies": self.ram_active_enemies,
        }


def _solid_mask(frame: np.ndarray) -> np.ndarray:
    mask = np.zeros(frame.shape[:2], dtype=bool)
    for colour in SOLID_COLOURS:
        mask |= mask_for(frame, colour)
    return mask


def _floor_profile(frame: np.ndarray) -> tuple[tuple[int | None, ...], tuple[int, ...]]:
    """Top of the contiguous solid run from the bottom edge, per column."""
    height, width = frame.shape[0], frame.shape[1]
    solid = _solid_mask(frame)

    missing = ~solid[::-1]
    first_gap = np.argmax(missing, axis=0)
    has_gap = missing.any(axis=0)
    run_length = np.where(has_gap, first_gap, height)

    floor_row: list[int | None] = []
    holes: list[int] = []
    for column in range(width):
        length = int(run_length[column])
        if length == 0:
            floor_row.append(None)
            holes.append(column)
        else:
            floor_row.append(height - length)
    return tuple(floor_row), tuple(holes)


def _bbox_of(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.where(mask.any(axis=1))[0]
    if rows.size == 0:
        return None
    columns = np.where(mask.any(axis=0))[0]
    return (int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max()))


def detect(frame: np.ndarray, *, tick: int = -1) -> FramePerception:
    """Characterize one frame."""
    array = np.ascontiguousarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) frame, got {array.shape}")

    background = dominant_colour(array)
    floor_row, hole_columns = _floor_profile(array)

    ground_line = 0
    known_floors = [row for row in floor_row if row is not None]
    if known_floors:
        values, counts = np.unique(np.array(known_floors), return_counts=True)
        ground_line = int(values[counts.argmax()])

    terrain = TerrainProfile(
        background=background,
        floor_row=floor_row,
        ground_line=ground_line,
        hole_columns=hole_columns,
    )

    # Mario is found by his red pixels in the gameplay rows. The bounding box of
    # the red mask is more robust than one connected component, because his
    # outline is black and can split the sprite into pieces.
    red = mask_for(array, MARIO_RED)
    red[:HUD_ROWS, :] = False
    red_pixels = int(red.sum())

    mario_box = _bbox_of(red) if red_pixels >= MIN_MARIO_RED_PIXELS else None
    confidence = 0.0 if mario_box is None else min(1.0, red_pixels / EXPECTED_MARIO_RED_PIXELS)

    return FramePerception(
        tick=tick,
        background=background,
        terrain=terrain,
        mario_box=mario_box,
        mario_red_pixels=red_pixels,
        mario_confidence=confidence,
    )


def ram_player_box(
    player_x_offset: int, player_y: int, size: int = TILE
) -> tuple[int, int, int, int]:
    """The player's sprite box implied by its RAM readings.

    Measured relation: the sprite occupies ``[x, x + 16)`` horizontally and
    ``[y + 16, y + 32)`` vertically. The game draws the player 16 px below its
    stored Y, which is why the vertical offset is not zero.
    """
    x0 = player_x_offset + PLAYER_BOX_DX
    y0 = player_y + PLAYER_BOX_DY
    return (x0, y0, x0 + size - 1, y0 + size - 1)


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def audit(perception: FramePerception, ram: object) -> PerceptionAudit:
    """Compare the frame-derived readings with the RAM ground truth.

    ``ram`` is a :class:`mario_jev.emulator.ram.RamView`; it is typed loosely here
    to keep the perception layer independent of the emulator package.
    """
    x_offset = ram.byte(0x03AD)  # type: ignore[attr-defined]
    y_pixel = ram.byte(0x03B8)  # type: ignore[attr-defined]
    ram_box = ram_player_box(x_offset, y_pixel)
    active = len(ram.active_enemies())  # type: ignore[attr-defined]

    # Independent consistency check that needs no RAM at all: a player standing
    # on level ground must have its RAM box sitting directly on the floor line.
    feet_on_ground: bool | None = None
    if perception.terrain.ground_line > 0:
        feet_on_ground = ram_box[3] < perception.terrain.ground_line

    if perception.mario_box is None:
        return PerceptionAudit(
            mario_detected_from_frame=False,
            mario_box_from_frame=None,
            mario_box_from_ram=ram_box,
            mario_inside_ram_box=None,
            offset=None,
            feet_on_ground=feet_on_ground,
            ram_active_enemies=active,
        )

    offset = (perception.mario_box[0] - ram_box[0], perception.mario_box[1] - ram_box[1])
    return PerceptionAudit(
        mario_detected_from_frame=True,
        mario_box_from_frame=perception.mario_box,
        mario_box_from_ram=ram_box,
        mario_inside_ram_box=_overlaps(perception.mario_box, ram_box),
        offset=offset,
        feet_on_ground=feet_on_ground,
        ram_active_enemies=active,
    )
