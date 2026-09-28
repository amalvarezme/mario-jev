"""The four questions one decision tick sends to Jev.

The question set is deliberately tiny and fixed: every tick asks the same four
things, so the answers are comparable across ticks and the journal can be read
as a time series. The ids are a contract with the response parser, so they live
in one tuple and nothing else invents a fifth.

Each question carries *instructions built from the state*, not a static sentence:
a question that does not show the model Mario's position, the rollout numbers, and
the pit measurements would be asking it to judge blind, and a blind choice is
indistinguishable from a guess.

The criteria for ``best_action`` are keyed by the candidate action names in the
order the rollouts produced them, so a returned ``choice`` is always an action
some rollout actually measured and the policy can always execute it.
"""

from __future__ import annotations

from mario_jev.state import CandidateAssessment, MarioState

#: The exact question ids every tick asks, in the order they are asked.
TICK_QUESTION_IDS = ("best_action", "risk", "lethal_ahead", "gap_severity")

#: The four-point risk scale, most benign first.
RISK_LEVELS: tuple[str, ...] = ("safe", "minor risk", "serious risk", "near-certain death")

#: The four-point gap-severity scale, most benign first.
GAP_LEVELS: tuple[str, ...] = ("no gap", "small gap", "wide gap", "bottomless pit")

#: ``lethal_ahead`` is a yes/no question, and both options need wording.
LETHAL_CRITERIA: dict[str, str] = {
    "true": "death is likely",
    "false": "no death expected",
}

#: One-line human descriptions for the action names the emulator build can emit.
#: An unknown name falls back to the name itself, which stays truthful.
ACTION_DESCRIPTIONS: dict[str, str] = {
    "NOOP": "stand still",
    "right": "walk right",
    "right+A": "jump to the right, A held",
    "right+B": "run right, no jump",
    "right+A+B": "run right, jump held while running",
    "A": "jump in place",
    "left": "back off to the left",
}

__all__ = [
    "TICK_QUESTION_IDS",
    "build_questions",
]


def _action_verb(action: str) -> str:
    """A one-line description of an action name."""
    return ACTION_DESCRIPTIONS.get(action, action)


def _candidate_line(assessment: CandidateAssessment) -> str:
    """One candidate's action, described with the numbers its rollout measured."""
    return (
        f"{_action_verb(assessment.action)} "
        f"(dx={assessment.dx:+d}, final_x={assessment.final_x}, "
        f"outcome={assessment.outcome.value}, risk={assessment.risk:.2f})"
    )


def _candidate_summary(state: MarioState) -> str:
    """Every candidate rollout on one semicolon-separated line."""
    return "; ".join(_candidate_line(assessment) for assessment in state.candidates)


def _enemy_phrase(state: MarioState) -> str:
    enemy = state.nearest_enemy
    if enemy.distance_px is None:
        return "no active enemy ahead"
    motion = "closing in" if enemy.approaching else "not closing in"
    return f"nearest enemy {enemy.distance_px}px ahead, {motion}"


def _terrain_phrase(state: MarioState) -> str:
    terrain = state.terrain_ahead
    if terrain.gap_distance_px is None:
        gap = "no pit within the lookahead"
    else:
        gap = f"a {terrain.gap_columns}px pit {terrain.gap_distance_px}px ahead"
    wall = f"{terrain.wall_height}px of wall ahead" if terrain.wall_height else "no wall ahead"
    return f"{gap}, {wall}"


def _posture(state: MarioState) -> str:
    return "on the ground" if state.player.on_ground else "airborne"


def _best_action_instructions(state: MarioState) -> str:
    player = state.player
    progress = state.progress
    return (
        f"Tick {state.tick}. Mario is at x={player.x} (furthest this episode "
        f"x_max={progress.x_max}), {_posture(state)}, facing {player.facing}, "
        f"moving at {state.speed_px_per_tick:.1f}px/frame. "
        f"{_enemy_phrase(state)}; {_terrain_phrase(state)}. "
        f"Each candidate was rolled out through the emulator: {_candidate_summary(state)}. "
        "Choose the single action that reaches the furthest final_x without dying."
    )


def _risk_instructions(state: MarioState) -> str:
    risks = ", ".join(
        f"{assessment.action}={assessment.risk:.2f}" for assessment in state.candidates
    )
    return (
        f"Tick {state.tick}. Rate how dangerous Mario's situation is right now. "
        f"{_enemy_phrase(state)}; {_terrain_phrase(state)}; "
        f"Mario is {_posture(state)} at x={state.player.x} with "
        f"{state.progress.lives} lives left. "
        f"The rollouts estimated these per-action risks: {risks}. "
        "Score the danger from 0 (safe) to 3 (near-certain death)."
    )


def _lethal_instructions(state: MarioState) -> str:
    outcomes = ", ".join(
        f"{assessment.action} -> {assessment.outcome.value}" for assessment in state.candidates
    )
    return (
        f"Tick {state.tick}. Would Mario die in the next few frames if nothing "
        f"changes? {_enemy_phrase(state)}; {_terrain_phrase(state)}; "
        f"Mario is {_posture(state)} at x={state.player.x}. "
        f"The rollout outcomes were: {outcomes}. "
        "Answer true only if death is likely."
    )


def _gap_instructions(state: MarioState) -> str:
    terrain = state.terrain_ahead
    return (
        f"Tick {state.tick}. Judge the pit ahead. gap_columns={terrain.gap_columns}px, "
        f"gap_distance_px={terrain.gap_distance_px}, "
        f"wall_height={terrain.wall_height}px, "
        f"Mario's speed={state.speed_px_per_tick:.1f}px/frame at x={state.player.x}. "
        "Score the severity from 0 (no gap) to 3 (bottomless pit)."
    )


def build_questions(state: MarioState) -> dict[str, dict[str, object]]:
    """The four questions for one tick, each with criteria and state-built text.

    Raises:
        ValueError: when the tick carries no rollouts. ``best_action`` would have
            no candidate to offer, and a choice question with no choices is not a
            question at all.
    """
    if not state.candidates:
        raise ValueError("cannot build questions for a tick with no candidate rollouts")

    return {
        "best_action": {
            "type": "choice",
            "instructions": _best_action_instructions(state),
            "criteria": {
                assessment.action: _candidate_line(assessment)
                for assessment in state.candidates
            },
        },
        "risk": {
            "type": "score",
            "instructions": _risk_instructions(state),
            "criteria": list(RISK_LEVELS),
        },
        "lethal_ahead": {
            "type": "noul",
            "instructions": _lethal_instructions(state),
            "criteria": dict(LETHAL_CRITERIA),
        },
        "gap_severity": {
            "type": "score",
            "instructions": _gap_instructions(state),
            "criteria": list(GAP_LEVELS),
        },
    }
