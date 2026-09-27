"""The Super Mario Bros. RAM map, and a typed view over it.

Addresses come from the *Super Mario Bros./RAM map* at Data Crystal. They are
the ground truth that the frame-based perception layer is cross-checked against:
vision is what we *claim* is on screen, RAM is what the emulator *knows*.

The package's own derived readings were checked against these addresses, and one
initially plausible mapping turned out to be **wrong**:

* ``info['x_pos']`` is ``0x006D * 0x100 + 0x0086`` — a 16-bit absolute position,
  *not* ``0x03AD``. Reading the source (``smb_env.py``, ``_x_position``) is what
  settles it. ``x_pos`` and ``0x03AD`` both read ``40`` at the 1-1 start, which
  is exactly the coincidence that makes this worth pinning in a test.
* ``info['y_pixel']`` is exactly ``0x03B8``. ``0x00CE`` also read ``176`` at the
  same moment, again by coincidence rather than identity.
* ``info['y_pos']`` is a *different* quantity from ``y_pixel``; do not conflate them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

#: Number of enemy slots the game keeps in RAM.
ENEMY_SLOTS = 5

#: Named single-byte addresses.
RAM_ADDRESSES: dict[str, int] = {
    "player_x_in_level": 0x006D,
    "player_x_on_screen": 0x0086,
    "player_x_offset": 0x03AD,
    "player_y": 0x00CE,
    "player_y_on_screen": 0x03B8,
    "screen": 0x071A,
    "screen_scroll": 0x071B,
    "screen_edge_x": 0x071C,
}

#: Parallel per-enemy-slot address blocks, in slot order.
ENEMY_TYPE_ADDRESSES: tuple[int, ...] = tuple(range(0x0016, 0x0016 + ENEMY_SLOTS))
ENEMY_X_LEVEL_ADDRESSES: tuple[int, ...] = tuple(range(0x006E, 0x006E + ENEMY_SLOTS))
ENEMY_X_SCREEN_ADDRESSES: tuple[int, ...] = tuple(range(0x0087, 0x0087 + ENEMY_SLOTS))
ENEMY_X_OFFSET_ADDRESSES: tuple[int, ...] = tuple(range(0x03AE, 0x03AE + ENEMY_SLOTS))
ENEMY_Y_ADDRESSES: tuple[int, ...] = tuple(range(0x03B9, 0x03B9 + ENEMY_SLOTS))


@dataclass(frozen=True, slots=True)
class EnemySlot:
    """One enemy slot: its type byte and its position readings."""

    slot: int
    type_id: int
    x_level: int
    x_screen: int
    x_offset: int
    y: int

    @property
    def is_active(self) -> bool:
        """A zero type byte means the slot is empty."""
        return self.type_id != 0


class RamView:
    """A read-only view over one RAM snapshot.

    The snapshot is copied on construction, so a view stays valid across
    subsequent emulator steps. That matters because the rollout planner steps
    the emulator many times while still needing the state it started from.
    """

    __slots__ = ("_bytes",)

    def __init__(self, ram: np.ndarray) -> None:
        array = np.asarray(ram)
        if array.ndim != 1:
            raise ValueError(f"RAM must be one-dimensional, got shape {array.shape}")
        self._bytes = array.astype(np.uint8, copy=True)

    def __len__(self) -> int:
        return int(self._bytes.shape[0])

    @property
    def raw(self) -> np.ndarray:
        """The raw copied bytes. Mutating this does not affect the emulator."""
        return self._bytes

    def byte(self, address: int) -> int:
        """Read one address as a Python int."""
        if not 0 <= address < self._bytes.shape[0]:
            raise IndexError(f"RAM address {address:#06x} is outside the 2 KiB page")
        return int(self._bytes[address])

    # -- named player readings ------------------------------------------------

    @property
    def player_x_in_level(self) -> int:
        """Absolute horizontal position within the level."""
        return self.byte(RAM_ADDRESSES["player_x_in_level"])

    @property
    def player_x_on_screen(self) -> int:
        return self.byte(RAM_ADDRESSES["player_x_on_screen"])

    @property
    def player_x_offset(self) -> int:
        return self.byte(RAM_ADDRESSES["player_x_offset"])

    @property
    def player_y(self) -> int:
        """Vertical position as the game's pixel reading."""
        return self.byte(RAM_ADDRESSES["player_y"])

    @property
    def player_y_on_screen(self) -> int:
        return self.byte(RAM_ADDRESSES["player_y_on_screen"])

    @property
    def screen(self) -> int:
        return self.byte(RAM_ADDRESSES["screen"])

    @property
    def screen_scroll(self) -> int:
        return self.byte(RAM_ADDRESSES["screen_scroll"])

    @property
    def screen_edge_x(self) -> int:
        return self.byte(RAM_ADDRESSES["screen_edge_x"])

    # -- enemies --------------------------------------------------------------

    def enemy(self, slot: int) -> EnemySlot:
        """Read one enemy slot by index."""
        if not 0 <= slot < ENEMY_SLOTS:
            raise IndexError(f"enemy slot must be within [0, {ENEMY_SLOTS}), got {slot}")
        return EnemySlot(
            slot=slot,
            type_id=self.byte(ENEMY_TYPE_ADDRESSES[slot]),
            x_level=self.byte(ENEMY_X_LEVEL_ADDRESSES[slot]),
            x_screen=self.byte(ENEMY_X_SCREEN_ADDRESSES[slot]),
            x_offset=self.byte(ENEMY_X_OFFSET_ADDRESSES[slot]),
            y=self.byte(ENEMY_Y_ADDRESSES[slot]),
        )

    def enemies(self) -> tuple[EnemySlot, ...]:
        """Every enemy slot, active or not."""
        return tuple(self.enemy(slot) for slot in range(ENEMY_SLOTS))

    def active_enemies(self) -> tuple[EnemySlot, ...]:
        """Only the slots the game currently considers occupied."""
        return tuple(slot for slot in self.enemies() if slot.is_active)

    def __iter__(self) -> Iterator[EnemySlot]:
        return iter(self.enemies())
