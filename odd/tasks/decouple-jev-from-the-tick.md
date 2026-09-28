# Feature — decouple the Jev call from the tick (evaluation finding #1)

Status: authorized by the user (chose "desacoplar la llamada" over accepting ~1 Hz or batching).
Owner: parent session. Implemented task by task; every task closes with one work-unit commit, and
each commit needs the user's explicit go-ahead.

## The defect, measured

`ExecutionLoop.step()` executes **exactly one game frame per tick** and calls `JevClient.ask(...)`
**in series inside that tick** (`policy/loop.py:488`). The emotion cadence
(`decision_cadence_frames`) chooses on which ticks it asks; it does not change the fact that the tick
**waits**.

Consequences measured in the live evaluation:

* unpaced run: **900 frames in 184 s = 4.9 game frames per second**, i.e. the game advances ~12×
  slower than real time;
* one tick spent **21 271.52 ms inside Jev** out of a 21 375.84 ms tick, so a single 20 s timeout
  **froze Mario for 20 seconds**;
* the paced run was worse: **120 frames in 157 s = 0.76 fps**.

So the cost of this defect is not the decision rate — it is that the *world stops* while the model
thinks. A 416 ms median round trip cannot fit inside a 16.7 ms frame, and no amount of client tuning
changes that; the call has to leave the critical path.

## The pinned design

### 1. A non-blocking dispatcher, not a changed client

`src/mario_jev/policy/dispatch.py` (new) owns one worker and at most one call:

```python
class JevDispatcher:
    def __init__(self, client, *, runner: Runner | None = None): ...
    def request(self, questions, state, *, tick: int) -> bool: ...   # False if one is already in flight
    def take(self) -> Taken | None: ...                              # never blocks; None if nothing finished
    def close(self) -> None: ...                                     # idempotent, joins the worker
```

* **`src/mario_jev/jev/**` is not touched at all.** `JevClient` stays synchronous, single-threaded
  and already verified; the dispatcher runs `client.ask(...)` in a worker. This keeps the review
  slice of the committed `jev/` work intact.
* **At most one call in flight.** A cadence tick with a call already in flight does **not** fire
  another one, and the tick records that it did not. Cancellation is not attempted: the standard
  library cannot cancel a running call, and pretending otherwise would be a lie in the evidence.
* **`runner` is a seam**, in the established style of this codebase (`transport`, `sleeper`, and the
  pacer's injected clock): the default is a one-worker `ThreadPoolExecutor`; tests inject a manual
  runner they drive by hand, so every sequence is deterministic and no test consults the real clock.
* **The worker never touches the emulator.** It receives an immutable `MarioState` and the questions
  the main thread already built.

### 2. The loop never waits

* On a cadence tick with a credential, candidates and nothing in flight: `request(...)`, and the tick
  records that it asked (`jev_called` stays "a call was fired").
* At the decision point: `take()`. A result at most `staleness_limit_ticks` old (default
  `3 * decision_cadence_frames`, pinned as a constant with its reason) is used; an older one is
  **discarded** and the tick holds its last action — the same thing a tick with no decision at all
  already did — with the discard recorded. It is deliberately **not** "the rollout takes over": a tick
  with a credential and no answer holds and says `DecisionSource.HOLD`, rather than claiming a decision
  it did not make. The rollout's own answer stays available as the fallback for the branches that
  genuinely have no credential.
* `run()` closes the dispatcher on **every** exit path, including the abort path where a
  `JevAuthError`/`JevRequestError` propagates, so no worker outlives the run.

### 3. The staleness is recorded, because otherwise the evidence lies

Once the call is not awaited, "Jev decided this tick" stops being one fact and becomes three:

| fact | where |
|---|---|
| this tick **fired** a call | `jev_called`, `RunSummary.decisions` |
| this tick **applied** an answer, and how old it was | `asked_at_tick`, `answer_age_ticks`, `RunSummary.answers_applied` |
| what the **call itself** cost, and when | `jev_call_ms` |

Pinned record fields: `asked_at_tick: int \| None`, `answer_age_ticks: int \| None`,
`jev_call_ms: float \| None`, plus a new member of the closed `decision.source` set for an answer
that came from an earlier tick.

**`latencies_ms` keeps its meaning**: the time *this tick* spent in each phase, so the phases it holds
— everything except `total` itself, which is a member of the same mapping — sum to at most `total`. The
call's own duration lives in
`jev_call_ms` instead, precisely so that invariant survives — with the call moved off the tick, putting
21 s of Jev latency into a tick whose `total` no longer contains it would break the arithmetic the
evaluation relied on.

**`decision_rate_hz` keeps counting fired calls** and its docstring says so; `answers_applied` is the
new counter for the other meaning. A silent redefinition of an existing field would invalidate every
number already in `docs/EVALUATION.md`.

### 4. Tests that must exist

1. A call that takes 20 s (manual runner) **does not stop the game**: frames keep advancing and each
   tick's `total` stays small. This is the deterministic proof of the whole feature.
2. At most one in flight: a second cadence tick during a call does not fire, and records it.
3. An answer older than the bound is discarded and the rollout governs.
4. A fresh answer is applied with `answer_age_ticks == 0`; a one-tick-old answer with the real age.
5. A failed call is reported on the tick that **observes** it, as `fallback_degraded`, with its
   measured duration.
6. `run()` leaves no live worker, on the normal path and on the abort path.
7. `answers_applied` and `decisions` count different things and are documented as such.
8. The phases in `latencies_ms` — everything except `total` itself — sum to at most `total`, on every
record of a run.
9. Journal round trip for the new keys, and the schema-enumeration test updated rather than left stale.

## Non-goals

* **No batching or speculative futures** (the option the user did not choose).
* **No change to `src/mario_jev/jev/**`**, to the emulator or to the pacer.
* **No rewrite of the report's existing sections** — only a new section for #1, with measured numbers.

## Tasks

| # | Task | Surfaces | Evidence |
|---|---|---|---|
| T1 | The non-blocking decision round: dispatcher + loop integration + tests | `src/mario_jev/policy/dispatch.py` (new), `policy/loop.py`, `policy/__init__.py`, `tests/test_dispatch.py` (new), `tests/test_policy.py` | tests 1-4, 6, 7, 8; suite totals |
| T2 | Record and show the staleness | `journal/records.py`, `journal/__init__.py`, `dashboard/build.py`, `cli.py`, `tests/test_journal.py`, `tests/test_cli.py` | test 5, 9; a real journal line read back |
| T3 | Measured live run | `artifacts/**` (gitignored) | frames/s before vs after; what a real timeout now costs |
| T4 | Report and plan | `docs/EVALUATION.md`, `odd/tasks/mario-jev-real-time-agent.md` | finding #1 struck through with its measured numbers |

T1 and T2 reopen the decision round that was just verified, so the round's verification must cover
them again. T3 is the only task that needs the credential (the Keychain bridge in `scripts/jev-env.sh`).

## Open product question, recorded rather than decided

With the frames moving and Jev answering at ~2 Hz, **most frames are governed by the rollout, not by
Jev**. The evaluation's own data says the two agreed on 112/113 calls, so this is defensible — but it
changes what the agent *is*: from "Jev decides every tick, slowly" to "Jev steers, the rollout
drives". The report must say that plainly; the user should decide whether that is the deliverable.
