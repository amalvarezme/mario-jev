"""The decision rule: cadence, confidence gate, guardrails, and typed fallbacks.

This module is **pure**. :func:`select_action` reads a rollout table, an optional
parsed Jev response and the settings, and returns a :class:`TickDecision`; it
touches no emulator, no network and no clock, so every branch is reachable in a
test without one. The loop that supplies those values, and the one place a Jev
call can happen, live in :mod:`mario_jev.policy.loop`.

The rule is deterministic and exhaustive, evaluated in exactly this order:

1. no candidates              -> ``NOOP`` / ``fallback_no_candidates``
2. not a cadence tick         -> the last decided action / ``hold``
3. no credential              -> ``table.best`` / ``fallback_no_key``
4. a transient degradation    -> ``table.best`` / ``fallback_degraded``
5. *(no branch)*              -> auth/request/protocol errors abort the run
6. confidence below threshold -> ``table.best`` / ``fallback_below_confidence``
7. the choice is lethal while
   the best candidate is not   -> ``table.best`` / ``guardrail_lethal``
8. otherwise                  -> the chosen action / ``jev``

Step 5 is deliberately an absence, not code: ``JevAuthError``, ``JevRequestError``
and ``JevProtocolError`` are never caught here and must never be caught by the loop
either. Only a *transient* failure (``JevUnavailable``: timeouts, 5xx, or 429/529
once retries are exhausted) is degraded into step 4. A bad key, a malformed request
or a drifted contract must abort the run rather than masquerade as an endless
fallback.

Advisor boundary
----------------

The optional PlayJev advisor is **never** sent to Jev. ``JevClient.ask`` takes a
:class:`~mario_jev.state.MarioState` and projects it itself with the frozen
five-key ``to_jev_state()``; the advisor block belongs only to the *recorded* tick
projection, which this policy layer forms in :func:`journal_state`. That keeps the
committed, separately verified client untouched and the model's input contract
frozen, and it is why ``MarioState.to_jev_state()`` gains nothing here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from mario_jev.state import HAZARD_OUTCOMES

if TYPE_CHECKING:
    from mario_jev.advisor import AdvisorAdvice
    from mario_jev.config import Settings
    from mario_jev.jev import JevResponse
    from mario_jev.state import MarioState
    from mario_jev.trajectory import CandidateRollout, RolloutTable

__all__ = [
    "DecisionCadence",
    "DecisionSource",
    "TickDecision",
    "advisor_view",
    "journal_state",
    "select_action",
]


class DecisionSource(StrEnum):
    """Where a tick's executed action came from.

    The closed set is the contract: the journal counts ticks by these values, so a
    new fallback has to be added here rather than invented as a free string.
    """

    #: Jev answered and its choice was executed.
    JEV = "jev"
    #: No new decision was requested on this frame; the last decided action ran.
    HOLD = "hold"
    #: Jev chose a lethal candidate while a non-lethal one was on the table.
    GUARDRAIL_LETHAL = "guardrail_lethal"
    #: No credential is configured; the local table answer ran, with no call made.
    FALLBACK_NO_KEY = "fallback_no_key"
    #: The call failed transiently (``JevUnavailable``); the table answer ran.
    FALLBACK_DEGRADED = "fallback_degraded"
    #: Jev answered, but below the confidence threshold; the table answer ran.
    FALLBACK_BELOW_CONFIDENCE = "fallback_below_confidence"
    #: There was nothing to choose from; the emulator was held still.
    FALLBACK_NO_CANDIDATES = "fallback_no_candidates"


@dataclass(frozen=True, slots=True)
class DecisionCadence:
    """How often Jev is consulted: once every ``cadence_frames`` executed game frames.

    That is the loop's own ``frame_index``, not ``emulator.tick``: the planner's
    rollout sweep advances ``emulator.tick`` by the whole rollout cost, so keying the
    cadence off it would fire on an arbitrary subset of ticks.
    """

    cadence_frames: int

    def __post_init__(self) -> None:
        # A cadence of zero would make ``due`` divide by zero rather than mean
        # "every frame", so it is rejected here instead of crashing mid-run.
        if self.cadence_frames < 1:
            raise ValueError(f"cadence_frames must be >= 1, got {self.cadence_frames}")

    def due(self, frame_index: int) -> bool:
        """Whether the frame at ``frame_index`` is a decision tick."""
        return frame_index % self.cadence_frames == 0


@dataclass(frozen=True, slots=True)
class TickDecision:
    """The one action this tick will execute, and why.

    ``confidence`` is ``None`` when no *choice* answer was gated (the decision was
    taken before the confidence gate); otherwise it is the gated confidence.
    ``risk`` is the ``risk`` score answer, populated only when the choice was
    executed. ``jev_called`` is true whenever a Jev reply was consulted or a call
    was attempted and degraded, which is exactly what ``RunSummary.decisions``
    counts.
    """

    action: str
    source: DecisionSource
    confidence: float | None
    risk: float | None
    reason: str
    jev_called: bool


def _candidate(table: RolloutTable, name: str) -> CandidateRollout:
    """The rollout for ``name``, or a loud ``ValueError`` if it is not offered.

    The client rejects a choice outside the criteria, which are the candidate
    names in rollout order, so a name the table does not know cannot happen. It is
    a bug, not a state to degrade into, so it raises rather than fall back.
    """
    try:
        return table.by_name(name)
    except KeyError as exc:
        raise ValueError(
            f"Jev chose {name!r}, which is not one of the candidates the tick "
            "offered; the client rejects a choice outside the criteria, so this "
            "is a bug rather than a state to fall back from"
        ) from exc


def select_action(
    table: RolloutTable,
    jev: JevResponse | None,
    settings: Settings,
    *,
    cadence_due: bool = True,
    last_action: str = "NOOP",
    degraded_reason: str | None = None,
) -> TickDecision:
    """Choose this tick's action from the table and an optional Jev answer.

    Pure and exhaustive: the eight steps in the module docstring are evaluated in
    order and every one of them returns, so nothing is consulted after a decision
    is taken. ``jev`` is only dereferenced from step 6 onward, which the loop
    reaches only after a successful call, so a degraded or absent answer never
    needs a :class:`~mario_jev.jev.JevResponse`.
    """
    # 1. Nothing to choose from: hold the emulator still, whatever the cadence or
    #    the credential say. Asking Jev about an empty candidate set is impossible.
    if not table.candidates:
        return TickDecision(
            action="NOOP",
            source=DecisionSource.FALLBACK_NO_CANDIDATES,
            confidence=None,
            risk=None,
            reason="no trajectory candidates to choose from",
            jev_called=False,
        )

    # 2. Not a decision tick: reuse the last decided action verbatim. The rollouts
    #    still ran, but no new decision was requested on this frame.
    if not cadence_due:
        return TickDecision(
            action=last_action,
            source=DecisionSource.HOLD,
            confidence=None,
            risk=None,
            reason="cadence not due on this frame",
            jev_called=False,
        )

    # 3. No credential configured: the measured table answer, with no call made.
    if not settings.jev_available:
        return TickDecision(
            action=table.best.action.name,
            source=DecisionSource.FALLBACK_NO_KEY,
            confidence=None,
            risk=None,
            reason="no Jev credential is configured",
            jev_called=False,
        )

    # 4. A transient failure was caught by the loop: degrade, but record the cause
    #    rather than the symptom, and let it count as a decision attempt.
    if degraded_reason is not None:
        return TickDecision(
            action=table.best.action.name,
            source=DecisionSource.FALLBACK_DEGRADED,
            confidence=None,
            risk=None,
            reason=degraded_reason,
            jev_called=True,
        )

    # 5. There is deliberately no branch here for JevAuthError, JevRequestError or
    #    JevProtocolError: those abort the run, so a well-behaved caller never
    #    reaches this function with one.

    choice = jev.choice("best_action")  # type: ignore[union-attr]
    confidence = float(choice.confidence)
    chosen_name = choice.choice

    # 6. The answer is not trusted below the threshold. The confidence is recorded
    #    even though the answer was rejected: it is what the gate read.
    if confidence < settings.confidence_threshold:
        return TickDecision(
            action=table.best.action.name,
            source=DecisionSource.FALLBACK_BELOW_CONFIDENCE,
            confidence=confidence,
            risk=None,
            reason=(
                f"confidence {confidence:.3f} is below the "
                f"{settings.confidence_threshold:.3f} threshold"
            ),
            jev_called=True,
        )

    chosen = _candidate(table, chosen_name)
    best = table.best

    # 7. The model chose a candidate that dies while a non-lethal one was on the
    #    table: the guardrail overrides the choice, but keeps the gated confidence.
    if chosen.outcome in HAZARD_OUTCOMES and best.outcome not in HAZARD_OUTCOMES:
        return TickDecision(
            action=best.action.name,
            source=DecisionSource.GUARDRAIL_LETHAL,
            confidence=confidence,
            risk=None,
            reason=(
                f"Jev chose {chosen_name!r}, which is lethal while "
                f"{best.name!r} is not"
            ),
            jev_called=True,
        )

    # 8. The model's answer stands. The risk score is carried through when the
    #    response has one; a missing risk answer is not an error.
    risk: float | None = None
    if "risk" in jev.answers:  # type: ignore[union-attr]
        risk = float(jev.score("risk").score)  # type: ignore[union-attr]
    return TickDecision(
        action=chosen_name,
        source=DecisionSource.JEV,
        confidence=confidence,
        risk=risk,
        reason=f"Jev chose {chosen_name!r}",
        jev_called=True,
    )


def advisor_view(advice: AdvisorAdvice | None) -> dict[str, object] | None:
    """The journal projection of one advisor answer, or ``None`` when absent.

    Plain data, in the advisor's own vocabulary. This block never travels to Jev.
    """
    if advice is None:
        return None
    return {
        "model_id": advice.model_id,
        "backend": advice.backend,
        "top": {
            "name": advice.top.name,
            "probability": float(advice.top.probability),
        },
        "moves": [
            {"name": move.name, "probability": float(move.probability)}
            for move in advice.moves
        ],
        "latency_ms": float(advice.latency_ms),
    }


def journal_state(
    state: MarioState,
    advice: AdvisorAdvice | None = None,
) -> dict[str, object]:
    """The recorded tick projection: the frozen Jev state plus the advisor block.

    ``MarioState.to_jev_state()`` gains nothing — the extra ``advisor`` key is added
    here, by the policy, and is **not** sent to Jev. The model receives exactly the
    five frozen keys.
    """
    return {**state.to_jev_state(), "advisor": advisor_view(advice)}
