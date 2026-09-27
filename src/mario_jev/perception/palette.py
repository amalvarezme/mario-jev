"""The Super Mario Bros. overworld palette, measured from real frames.

These values were not copied from a reference table or guessed. They were sampled
from frames produced by the emulator in this repository, and
``tests/test_perception.py`` re-verifies them against live frames so a palette
change cannot pass unnoticed.

Because the NES renders through a fixed palette for an overworld level, these
constants are stable for world 1-1. A different tileset (castle, underground,
underwater) would use different entries, which is why the detector also derives
its background colour from the frame itself rather than trusting a constant for
that particular job.
"""

from __future__ import annotations

from typing import Final

import numpy as np

RGB = tuple[int, int, int]

#: The sky. Also the most frequent colour in a typical overworld frame, which is
#: what makes self-calibration possible.
SKY: Final[RGB] = (104, 136, 252)

#: Ground blocks and bricks.
GROUND: Final[RGB] = (228, 92, 16)

#: The light tan highlight inside ground blocks and used-block faces.
GROUND_HIGHLIGHT: Final[RGB] = (240, 208, 176)

#: Pipes, and Koopa Troopa shells/skin.
PIPE_GREEN: Final[RGB] = (0, 168, 0)

#: Light green, used for the pipe highlight and some bricks' trim.
LIGHT_GREEN: Final[RGB] = (184, 248, 24)

#: Mario's hat and shirt. His single most distinctive colour in this palette,
#: which is why the frame-based Mario detector keys on it.
MARIO_RED: Final[RGB] = (248, 56, 0)

#: Clouds and the flag pole.
WHITE: Final[RGB] = (252, 252, 252)

#: Goomba body and other brown sprites.
BROWN: Final[RGB] = (172, 124, 0)

#: Coins and the "?" block face.
COIN_YELLOW: Final[RGB] = (252, 160, 68)

BLACK: Final[RGB] = (0, 0, 0)

#: Every colour the gameplay-classification tables care about.
KNOWN_COLOURS: Final[dict[str, RGB]] = {
    "sky": SKY,
    "ground": GROUND,
    "ground_highlight": GROUND_HIGHLIGHT,
    "pipe_green": PIPE_GREEN,
    "light_green": LIGHT_GREEN,
    "mario_red": MARIO_RED,
    "white": WHITE,
    "brown": BROWN,
    "coin_yellow": COIN_YELLOW,
    "black": BLACK,
}

#: The fixed status bar occupies the top two tile rows and never contains
#: gameplay. Excluding it is a convention, not an approximation: the HUD is
#: chrome drawn over the playfield, and its red coin icon would otherwise be
#: detected as a second Mario.
HUD_ROWS: Final[int] = 32

#: One tile in pixels.
TILE: Final[int] = 16

#: The measured sprite box of the player, expressed as offsets from the RAM
#: readings. Measured on live frames: with ``0x03AD = 112`` and ``0x03B8 = 176``
#: the red cluster sat at rows 193..206, columns 114..125, which falls inside
#: this box. The game draws the player 16 px below its stored Y, which is why
#: the vertical offset is +16 rather than 0.
PLAYER_BOX_DX: Final[int] = 0
PLAYER_BOX_DY: Final[int] = 16
PLAYER_BOX_SIZE: Final[int] = 16


def as_rgb(color: RGB) -> np.ndarray:
    """Turn a palette entry into a comparable ``uint8`` array."""
    return np.array(color, dtype=np.uint8)


def mask_for(frame: np.ndarray, color: RGB, tolerance: int = 0) -> np.ndarray:
    """A boolean mask of the pixels matching one palette colour."""
    target = as_rgb(color).astype(np.int16)
    delta = np.abs(frame.astype(np.int16) - target)
    return np.all(delta <= tolerance, axis=2)


def dominant_colour(frame: np.ndarray) -> RGB:
    """The most frequent colour, derived from the frame itself.

    Used for the background instead of the :data:`SKY` constant so that a level
    with a different sky does not silently break terrain detection.
    """
    flat = np.ascontiguousarray(frame).reshape(-1, 3)
    packed = (
        flat[:, 0].astype(np.uint32) << 16
        | flat[:, 1].astype(np.uint32) << 8
        | flat[:, 2].astype(np.uint32)
    )
    values, counts = np.unique(packed, return_counts=True)
    winner = int(values[counts.argmax()])
    return ((winner >> 16) & 0xFF, (winner >> 8) & 0xFF, winner & 0xFF)
