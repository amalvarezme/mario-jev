"""The terminal-outcome vocabulary, and the precedence it is resolved with.

The five strings are fixed by the plan's JSON contract, so they are defined here,
in the module that owns that contract, rather than in the trajectory package that
produces them. The planner imports them; nothing here imports the planner.

The precedence is part of the contract too, and it is worth writing down why it
is ordered this way: a trajectory that reached the flag is a success even if the
same frame also reports a death bit, and a trajectory that ended in a pit is a
different failure from one that ended on a Goomba, because only the first one
means the planner failed to see a hole.
"""

from __future__ import annotations

from enum import StrEnum


class TerminalOutcome(StrEnum):
    """How a rolled-out candidate action ended."""

    #: Mario reached the flag pole, or the world/stage ended.
    REACHES_GOAL = "reaches_goal"
    #: Mario died while his hitbox was over a hole column.
    FALLS_IN_GAP = "falls_in_gap"
    #: Mario died anywhere else — in practice, on an enemy.
    HITS_ENEMY = "hits_enemy"
    #: The rollout survived the whole horizon without advancing.
    NO_PROGRESS = "no_progress"
    #: The rollout survived the whole horizon and advanced.
    LANDS_SAFE = "lands_safe"


#: Resolution order, most decisive first. ``resolve_outcome`` walks it.
OUTCOME_PRECEDENCE: tuple[TerminalOutcome, ...] = (
    TerminalOutcome.REACHES_GOAL,
    TerminalOutcome.FALLS_IN_GAP,
    TerminalOutcome.HITS_ENEMY,
    TerminalOutcome.NO_PROGRESS,
    TerminalOutcome.LANDS_SAFE,
)

#: Outcomes that cost Mario a life.
HAZARD_OUTCOMES: frozenset[TerminalOutcome] = frozenset(
    {TerminalOutcome.FALLS_IN_GAP, TerminalOutcome.HITS_ENEMY}
)

#: The one outcome that ends the episode as a win.
GOAL_OUTCOME: TerminalOutcome = TerminalOutcome.REACHES_GOAL

#: Outcomes reachable only by surviving the horizon.
SURVIVAL_OUTCOMES: frozenset[TerminalOutcome] = frozenset(
    {TerminalOutcome.NO_PROGRESS, TerminalOutcome.LANDS_SAFE}
)


__all__ = [
    "GOAL_OUTCOME",
    "HAZARD_OUTCOMES",
    "OUTCOME_PRECEDENCE",
    "SURVIVAL_OUTCOMES",
    "TerminalOutcome",
]
