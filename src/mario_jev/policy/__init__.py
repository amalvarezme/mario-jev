"""The decision layer: cadence, confidence gate, guardrails, and the tick loop.

``mario_jev.policy`` ties the measured rollout table, the Jev client and the
optional PlayJev advisor together into one deterministic tick:

* :class:`DecisionCadence` decides *when* the model is consulted;
* :func:`select_action` decides *what* runs, in a fixed, exhaustive order;
* :class:`ExecutionLoop` runs the tick and records every branch it took;
* :class:`RunSummary` counts the run by :class:`DecisionSource` rather than
  claiming a rate.

Boundaries this package keeps
-----------------------------

* **The model's input is frozen.** The payload sent to Jev is exactly
  ``MarioState.to_jev_state()`` and nothing else; ``to_jev_state()`` gains no keys
  here. ``JevClient.ask`` takes a :class:`~mario_jev.state.MarioState` and projects
  it itself, and that client is a committed, separately verified slice.
* **The advisor is recorded, never sent.** The optional advisor block appears only
  in the *recorded* tick projection, which this layer forms via
  :func:`journal_state`; it is never part of the model's input.
* **Transient failures degrade; the rest abort.** Only ``JevUnavailable`` is caught
  and recorded as a degradation. ``JevAuthError``, ``JevRequestError`` and
  ``JevProtocolError`` propagate and stop the run.
* **A name the table does not know is a bug.** It raises ``ValueError`` rather than
  silently falling back to a stale action.
"""

from .decision import (
    DecisionCadence,
    DecisionSource,
    TickDecision,
    advisor_view,
    journal_state,
    select_action,
)
from .loop import (
    OVERSHOOT_FACTOR,
    ExecutionLoop,
    RunSummary,
    TickRecord,
)

__all__ = [
    "OVERSHOOT_FACTOR",
    "DecisionCadence",
    "DecisionSource",
    "ExecutionLoop",
    "RunSummary",
    "TickDecision",
    "TickRecord",
    "advisor_view",
    "journal_state",
    "select_action",
]
