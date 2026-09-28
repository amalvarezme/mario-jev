"""The decision layer: cadence, confidence gate, guardrails, and the tick loop.

``mario_jev.policy`` ties the measured rollout table, the Jev client and the
optional PlayJev advisor together into one deterministic tick:

* :class:`DecisionCadence` decides *when* the model is consulted;
* :func:`select_action` decides *what* runs, in a fixed, exhaustive order;
* :class:`JevDispatcher` keeps the call **off** the tick: one worker, at most one
  call in flight, and a take that never blocks;
* :class:`ExecutionLoop` runs the tick and records every branch it took;
* :class:`RunSummary` counts the run by :class:`DecisionSource` rather than
  claiming a rate, and names why the run stopped with one member of
  :data:`TERMINATIONS`.

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
  ``JevProtocolError`` propagate and stop the run. With the call off the tick both
  are attributed to the tick that *observes* them.
* **The tick never waits.** A Jev answer arrives one or more ticks after it was
  asked for, so "fired a call" and "applied an answer" are counted separately:
  ``RunSummary.decisions`` keeps counting fired calls, ``answers_applied`` counts
  applied ones, and an answer older than the staleness bound is discarded rather
  than acted on.
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
from .dispatch import JevDispatcher, Runner, Taken
from .loop import (
    JEV_STALE_DISCARDED,
    OVERSHOOT_FACTOR,
    STALE_LIMIT_CADENCES,
    TERMINATIONS,
    ExecutionLoop,
    RunSummary,
    TickRecord,
    classify_termination,
)

__all__ = [
    "JEV_STALE_DISCARDED",
    "OVERSHOOT_FACTOR",
    "STALE_LIMIT_CADENCES",
    "TERMINATIONS",
    "DecisionCadence",
    "DecisionSource",
    "ExecutionLoop",
    "JevDispatcher",
    "RunSummary",
    "Runner",
    "Taken",
    "TickDecision",
    "TickRecord",
    "advisor_view",
    "classify_termination",
    "journal_state",
    "select_action",
]
