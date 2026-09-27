"""The joypad: button bits and the discrete action sets the loop chooses from.

The bit values are copied from ``nes_py.wrappers.joypad_space.JoypadSpace._button_map``
rather than invented, so an action built here is byte-identical to the one the
package would build from the same button names.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Button name -> joypad bit, taken from the installed ``JoypadSpace`` wrapper.
BUTTON_BITS: dict[str, int] = {
    "right": 0b10000000,
    "left": 0b01000000,
    "down": 0b00100000,
    "up": 0b00010000,
    "start": 0b00001000,
    "select": 0b00000100,
    "B": 0b00000010,
    "A": 0b00000001,
    "NOOP": 0b00000000,
}

#: Ordered action set used for navigation. Mirrors ``SIMPLE_MOVEMENT`` in
#: ``gym_super_mario_bros.actions`` — seven actions, the smallest set that can
#: still walk, run, jump, and retreat.
SIMPLE_MOVEMENT: tuple[tuple[str, ...], ...] = (
    ("NOOP",),
    ("right",),
    ("right", "A"),
    ("right", "B"),
    ("right", "A", "B"),
    ("A",),
    ("left",),
)

#: Forward-only subset, useful when retreat is known to be useless.
RIGHT_ONLY: tuple[tuple[str, ...], ...] = (
    ("NOOP",),
    ("right",),
    ("right", "A"),
    ("right", "B"),
    ("right", "A", "B"),
)

ACTION_SETS: dict[str, tuple[tuple[str, ...], ...]] = {
    "simple": SIMPLE_MOVEMENT,
    "right_only": RIGHT_ONLY,
}

DEFAULT_ACTION_SET = "simple"


def bits_for(buttons: tuple[str, ...] | list[str]) -> int:
    """Combine button names into one joypad byte."""
    combined = 0
    for button in buttons:
        try:
            combined |= BUTTON_BITS[button]
        except KeyError as exc:  # pragma: no cover - guarded by the tables above
            raise KeyError(f"unknown joypad button {button!r}") from exc
    return combined


@dataclass(frozen=True, slots=True)
class Action:
    """One addressable action in the agent's control set."""

    index: int
    buttons: tuple[str, ...]
    bits: int

    @property
    def name(self) -> str:
        """A stable, human-readable identifier, e.g. ``right+A``."""
        return "+".join(self.buttons)

    @property
    def moves_right(self) -> bool:
        return "right" in self.buttons

    @property
    def moves_left(self) -> bool:
        return "left" in self.buttons

    @property
    def jumps(self) -> bool:
        """``A`` is the jump button in Super Mario Bros."""
        return "A" in self.buttons

    @property
    def runs(self) -> bool:
        """``B`` is the run/fire button."""
        return "B" in self.buttons

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "bits": self.bits,
            "buttons": list(self.buttons),
        }


def build_actions(action_set: str = DEFAULT_ACTION_SET) -> tuple[Action, ...]:
    """Expand a named action set into indexable :class:`Action` objects."""
    try:
        table = ACTION_SETS[action_set]
    except KeyError as exc:
        known = ", ".join(sorted(ACTION_SETS))
        raise KeyError(f"unknown action set {action_set!r}; known sets: {known}") from exc
    return tuple(
        Action(index=index, buttons=tuple(buttons), bits=bits_for(buttons))
        for index, buttons in enumerate(table)
    )
