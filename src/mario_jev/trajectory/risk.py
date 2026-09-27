"""One risk number per candidate trajectory: a float in ``[0, 1]``, higher is worse.

The formula is deliberately a small, named, pure function rather than an inline
expression, because a risk number nobody can check is not reviewable. Its tests
assert **orderings**, not magic constants, and this is the ordering they assert:

    falls_in_gap > hits_enemy > no_progress > lands_safe(close to an enemy)
                                                 > lands_safe(far from an enemy)
                                                 > reaches_goal

Two judgement calls are visible here and both are arguable, so they are written
down rather than hidden:

* **Falling into a pit is the worst hazard**, above walking into an enemy. Both
  cost a life, but a pit death means the terrain reading failed, while an enemy
  contact may be information the state simply did not have yet.
* **Doing nothing is worse than any survival**, because a candidate that survives
  the horizon without advancing burns the decision cadence for nothing. That is
  what makes ``NOOP`` lose to every moving action on flat ground.

The enemy-distance term applies only to surviving trajectories: a trajectory that
already died does not care how close the enemy was.
"""

from __future__ import annotations

from mario_jev.state import TerminalOutcome

#: A cleared level is the only outcome with no cost at all, and it is the strict
#: minimum of the range: every other outcome, including the safest survival,
#: costs something.
GOAL_RISK: float = 0.0

#: Death while over a gap column.
FALLS_IN_GAP_RISK: float = 0.9
#: Death anywhere else — in practice, on an enemy.
HITS_ENEMY_RISK: float = 0.8

#: Survived the horizon without advancing.
NO_PROGRESS_RISK: float = 0.6

#: Survived the horizon and advanced, with nothing near.
LANDS_SAFE_BASE_RISK: float = 0.2
#: The extra cost of ending the trajectory right on top of an enemy.
LANDS_SAFE_DANGER_RISK: float = 0.3

#: How close (in pixels, box to box) counts as dangerous. Two tiles: the distance
#: a Goomba covers in about 40 frames, which is longer than a typical horizon.
SAFE_MARGIN_PX: int = 32


def risk_cost(
    outcome: TerminalOutcome,
    *,
    final_enemy_distance: int | None = None,
) -> float:
    """The risk of one rolled-out trajectory.

    ``final_enemy_distance`` is the empty space, in pixels, between Mario's sprite
    box and the nearest enemy at the *end* of the rollout (``None`` when no enemy
    was near). It only moves the answer for surviving trajectories.
    """
    resolved = TerminalOutcome(outcome)
    if resolved is TerminalOutcome.REACHES_GOAL:
        return GOAL_RISK
    if resolved is TerminalOutcome.FALLS_IN_GAP:
        return FALLS_IN_GAP_RISK
    if resolved is TerminalOutcome.HITS_ENEMY:
        return HITS_ENEMY_RISK
    if resolved is TerminalOutcome.NO_PROGRESS:
        return NO_PROGRESS_RISK

    danger = 0.0
    if final_enemy_distance is not None:
        remaining = SAFE_MARGIN_PX - max(0, int(final_enemy_distance))
        danger = min(1.0, max(0.0, remaining / SAFE_MARGIN_PX))
    return round(LANDS_SAFE_BASE_RISK + LANDS_SAFE_DANGER_RISK * danger, 4)


__all__ = [
    "FALLS_IN_GAP_RISK",
    "GOAL_RISK",
    "HITS_ENEMY_RISK",
    "LANDS_SAFE_BASE_RISK",
    "LANDS_SAFE_DANGER_RISK",
    "NO_PROGRESS_RISK",
    "SAFE_MARGIN_PX",
    "risk_cost",
]
