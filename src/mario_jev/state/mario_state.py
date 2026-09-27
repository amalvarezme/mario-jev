"""Typed state, and its canonical JSON projection for Jev.

Everything here is **relative**: the distance to the next enemy, the pit ahead,
the frames left before the jump window closes. Absolute world coordinates are
carried only where the projection needs them (``player.x``, ``progress.x_max``
and the candidates' ``final_x``), because those three are directly comparable to
each other and Jev has to be able to compare them.

Two coordinate spaces are in play, and conflating them is a silent bug:

* **in-level** — ``info['x_pos']``, the 16-bit absolute position. This is what
  progress and candidates move along.
* **on-screen** — the RAM ``0x03AD`` player column and the ``0x03AE`` enemy
  block. These are the columns the frame is drawn in, so they are the only two
  readings that may be subtracted from each other when asking "how far away is
  that Goomba". ``0x0087`` is *not* usable for this; see
  :mod:`mario_jev.emulator.ram`.

The projection's key set is a contract, not a convenience: WU4 sends it to Jev
and WU5 journals it. ``to_jev_state()`` therefore emits exactly the documented
keys and nothing else — the extra typed fields (``screen_x``, ``feet_row``,
``frames_to_jump_window``, ``vertical_clearance``, ``speed_px_per_tick``) are for
our own code, and adding them to the JSON would change the contract.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from .outcome import TerminalOutcome

# --- the info keys this package depends on ------------------------------------
# Read from the installed `gym_super_mario_bros.smb_env.SuperMarioBrosEnv`, not
# assumed: `_get_info` merges `_normalized_info` and `_reward_info` into its own
# readings. `tests/test_state.py` pins every one of these against the live
# package, and against the RAM address each is derived from.

#: ``0x006D * 0x100 + 0x0086``: the 16-bit absolute in-level position.
#: Not ``0x03AD``, which is the on-screen column.
INFO_X_POS = "x_pos"
#: The furthest ``x_pos`` reached this episode.
INFO_X_POS_MAX = "x_pos_max"
#: ``0x03B8``. Equal to the RAM byte, and a *different* quantity from ``y_pos``.
INFO_Y_PIXEL = "y_pixel"
#: ``0x075A``. The raw life counter; becomes ``0xFF`` once the game is over.
INFO_LIFE = "life"
#: ``0x07F8..0x07FA``: the HUD timer, in seconds.
INFO_TIME = "time"
#: The world and stage numbers, both 1-based.
INFO_WORLD = "world"
INFO_STAGE = "stage"
#: ``_flag_get`` (``_is_world_over or _is_stage_over``): the level was cleared.
INFO_CLEAR = "clear"
#: The same value as ``clear``, exposed under the older name.
INFO_FLAG_GET = "flag_get"
#: ``_is_dying or _is_dead``.
INFO_DEATH = "death"
#: ``player_state == 0x0B`` or ``y_viewport > 1`` (the dying animation, or a fall).
INFO_IS_DYING = "is_dying"
#: ``player_state == 0x06``.
INFO_IS_DEAD = "is_dead"
#: The raw player state byte ``0x000E`` (``0x0B`` is dying).
INFO_PLAYER_STATE = "player_state"

#: Any of these being true means this frame is a death.
DEATH_INFO_KEYS: tuple[str, ...] = (INFO_DEATH, INFO_IS_DYING, INFO_IS_DEAD)
#: Any of these being true means the level was cleared.
GOAL_INFO_KEYS: tuple[str, ...] = (INFO_CLEAR, INFO_FLAG_GET)

#: The player state byte value for "in the dying animation".
PLAYER_STATE_DYING = 0x0B


def is_death(info: Mapping[str, object]) -> bool:
    """Whether an ``info`` mapping reports a death. See :data:`DEATH_INFO_KEYS`."""
    return any(bool(info.get(key, False)) for key in DEATH_INFO_KEYS)


def is_goal(info: Mapping[str, object]) -> bool:
    """Whether an ``info`` mapping reports a cleared level. See :data:`GOAL_INFO_KEYS`."""
    return any(bool(info.get(key, False)) for key in GOAL_INFO_KEYS)


@dataclass(frozen=True, slots=True)
class PlayerFeatures:
    """Where Mario is and what he is doing."""

    #: Absolute in-level position (``info['x_pos']``). Same space as ``final_x``.
    x: int
    #: The game's vertical pixel reading (``info['y_pixel']``, RAM ``0x03B8``).
    y: int
    #: On-screen sprite column (RAM ``0x03AD``), the same space as enemy ``drawn_x``.
    screen_x: int
    #: Bottom row of the sprite box, ``y + PLAYER_BOX_DY + 15``.
    feet_row: int
    #: Whether the floor directly under Mario is holding him up.
    on_ground: bool
    #: ``"right"`` or ``"left"``; measured from the ``x_pos`` delta, so it is
    #: unknown (and inherited) while Mario is standing still.
    facing: str

    def __post_init__(self) -> None:
        if self.facing not in {"right", "left"}:
            raise ValueError(f"facing must be 'right' or 'left', got {self.facing!r}")


@dataclass(frozen=True, slots=True)
class NearestEnemy:
    """The closest active enemy that is not behind Mario.

    :attr:`distance_px` is the empty space between the two sprite boxes, so it is
    ``0`` when they are touching or overlapping and never negative. ``None``
    means no active enemy was ahead of Mario at all.
    """

    distance_px: int | None
    #: Whether that enemy moved towards Mario since the previous observation.
    #: ``False`` also means "unknown" — the first observation of an episode has no
    #: previous sample to compare against, and guessing would be worse than
    #: reporting the absence of evidence.
    approaching: bool
    #: The RAM slot the reading came from, for the journal. Not projected.
    slot: int | None = None


@dataclass(frozen=True, slots=True)
class TerrainAhead:
    """What the frame says is coming up, measured from Mario's front edge."""

    #: Width of the first pit within the lookahead, in pixels. 0 means "no pit".
    gap_columns: int
    #: Pixels from Mario's front edge to the first pit column, or ``None``.
    gap_distance_px: int | None
    #: How far the highest floor within the lookahead sits above the ground line.
    #: Positive means a step up / wall.
    wall_height: int
    #: Frames until Mario's front edge reaches the first hole column at the
    #: currently measured horizontal speed. ``None`` when there is no hole ahead
    #: *or* the speed is zero — with no speed there is no arrival time.
    frames_to_jump_window: int | None
    #: Free space between Mario's feet and the floor directly under him: 0 while
    #: standing on it, positive while airborne. Overhead headroom is deliberately
    #: not claimed: the palette that draws pipes and stairs is the same one that
    #: draws the floor, so the frame cannot be asked where a ceiling is.
    vertical_clearance: int
    #: How far ahead (pixels) the three readings above were taken.
    lookahead_px: int


@dataclass(frozen=True, slots=True)
class Progress:
    """How far along the episode is."""

    #: ``"<world>-<stage>"``, e.g. ``"1-1"``.
    world: str
    #: Furthest absolute position reached this episode (``info['x_pos_max']``).
    x_max: int
    #: Raw life counter (``info['life']``, RAM ``0x075A``); ``255`` once the
    #: game is over, because that is what the RAM byte holds.
    lives: int
    #: Seconds left on the HUD timer (``info['time']``).
    time_left: int


@dataclass(frozen=True, slots=True)
class CandidateAssessment:
    """One rolled-out action, reduced to what Jev is shown."""

    action: str
    dx: int
    outcome: TerminalOutcome
    risk: float
    final_x: int

    def __post_init__(self) -> None:
        # Accept the plain string too, but store the enum: the vocabulary lives in
        # one place and an invented outcome name is a bug, not a value.
        object.__setattr__(self, "outcome", TerminalOutcome(self.outcome))
        if not 0.0 <= self.risk <= 1.0:
            raise ValueError(f"risk must be within [0, 1], got {self.risk!r}")


@dataclass(frozen=True, slots=True)
class MarioState:
    """The typed state: relative features, plus the candidate table once it exists."""

    tick: int
    player: PlayerFeatures
    nearest_enemy: NearestEnemy
    terrain_ahead: TerrainAhead
    progress: Progress
    #: Measured horizontal speed in in-level pixels per frame. 0 while unknown.
    speed_px_per_tick: float = 0.0
    #: Filled in once the rollout planner has run; empty until then.
    candidates: tuple[CandidateAssessment, ...] = ()

    def with_candidates(
        self, candidates: tuple[CandidateAssessment, ...] | list[CandidateAssessment]
    ) -> "MarioState":
        """A copy carrying the rollout table produced for this tick."""
        return replace(self, candidates=tuple(candidates))

    def candidate(self, action: str) -> CandidateAssessment:
        """The assessment for one action name."""
        for assessment in self.candidates:
            if assessment.action == action:
                return assessment
        raise KeyError(f"no candidate for action {action!r}")

    def to_jev_state(self) -> dict[str, object]:
        """The exact JSON-serializable projection the plan fixes."""
        return to_jev_state(self)


def to_jev_state(state: MarioState) -> dict[str, object]:
    """Project a :class:`MarioState` onto the contracted JSON shape.

    Every value is cast to a plain Python type. That is not decoration: the
    readings pass through numpy scalars on their way here, and ``json.dumps``
    refuses numpy integers. The key set is exactly the plan's — no more, no less.
    """
    player = state.player
    enemy = state.nearest_enemy
    terrain = state.terrain_ahead
    progress = state.progress

    return {
        "player": {
            "x": int(player.x),
            "y": int(player.y),
            "on_ground": bool(player.on_ground),
            "facing": str(player.facing),
        },
        "nearest_enemy": {
            "distance_px": None if enemy.distance_px is None else int(enemy.distance_px),
            "approaching": bool(enemy.approaching),
        },
        "terrain_ahead": {
            "gap_columns": int(terrain.gap_columns),
            "gap_distance_px": (
                None if terrain.gap_distance_px is None else int(terrain.gap_distance_px)
            ),
            "wall_height": int(terrain.wall_height),
        },
        "progress": {
            "world": str(progress.world),
            "x_max": int(progress.x_max),
            "lives": int(progress.lives),
            "time_left": int(progress.time_left),
        },
        "candidates": {
            assessment.action: {
                "dx": int(assessment.dx),
                "outcome": assessment.outcome.value,
                "risk": float(assessment.risk),
                "final_x": int(assessment.final_x),
            }
            for assessment in state.candidates
        },
    }


__all__ = [
    "DEATH_INFO_KEYS",
    "GOAL_INFO_KEYS",
    "INFO_CLEAR",
    "INFO_DEATH",
    "INFO_FLAG_GET",
    "INFO_IS_DEAD",
    "INFO_IS_DYING",
    "INFO_LIFE",
    "INFO_PLAYER_STATE",
    "INFO_STAGE",
    "INFO_TIME",
    "INFO_WORLD",
    "INFO_X_POS",
    "INFO_X_POS_MAX",
    "INFO_Y_PIXEL",
    "CandidateAssessment",
    "MarioState",
    "NearestEnemy",
    "PLAYER_STATE_DYING",
    "PlayerFeatures",
    "Progress",
    "TerrainAhead",
    "is_death",
    "is_goal",
    "to_jev_state",
]
