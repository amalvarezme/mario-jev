# Feature — a reproducible run, so an A/B comparison measures the build and not the machine

Status: proposed. Not started. Each task is one work-unit commit.

## Why

`docs/EVALUATION.md` cannot support an A/B claim, and says so twice: *"The same seed does not
reproduce the same run"*, and *"one run per build, and the load differed (4.16 against 5.73), so the
23 % rate gain is a floor rather than a clean measurement"*. Every comparison in the report carries
that excuse because the trajectory is not a function of the inputs.

The exploration established the exact causes, and there are **two**, not one:

1. **The horizon is not a pure function of state and configuration.** `_calibrate`
   (`trajectory/planner.py:698`) and `_sweep` (`:755`) time themselves with `perf_counter`, and
   `sweep_ms` decides both the correction pass (`:564`) and both learning branches
   (`_learn_from_overrun` `:621`, `_relax_towards_measurement` `:648`). The remembered cost model
   (`:442`) carries every earlier reading of that planner instance into every later tick.
2. **Which tick takes an answer is decided by thread timing.** `JevDispatcher.take` is
   non-blocking (`policy/dispatch.py:158`), the answer is applied on the tick that observes it, and
   `in_flight` gates whether a call fires at all (`policy/loop.py:606-616`). The same answers,
   delivered at different wall-clock moments, produce different actions — and one of them is
   discarded as `jev_stale` (`loop.py:626-634`).

Fixing only (1) would therefore not make a run reproducible. A harness that fixes both is the
prerequisite for every later measurement: the rollout bottleneck, the staleness policy, an advisor
comparison.

## Pinned design

### 1. A deterministic cost source (task T1)

`Settings` gains `rollout_cost_source` (`"measured"` default, `"fixed"` opt-in) plus
`rollout_fixed_per_frame_ms` and `rollout_fixed_base_ms`. When the source is `fixed`, the planner
takes those constants for `choose_horizon` and **consults no clock on the horizon path at all**:
no calibration measurement, no overrun learning, no relaxation toward a measurement. The measured
`elapsed_ms`/`sweep_ms` are still recorded per tick — they are real facts about the tick — but they
no longer feed the decision, and the journal says which source decided so.

The observable property to pin: **the decision is clock-free.** With the clock made absurd — a
`perf_counter` returning a constant, then an advancing absurd value — the horizon, the number of
passes, the candidate rows and the selected action are unchanged when the source is `fixed`, and
visibly move when it is `measured`. That is a falsifiable test, not a claim.

The recorded `elapsed_ms`/`sweep_ms` do **not** survive a raising clock, and must not: they are
measurements of a tick that really happened, and the invariant is about what *decides*. So a raising
clock is tested where it can move a measured horizon (the calibration), never against the whole
table — "the whole table is unchanged with a raising clock" is unsatisfiable for any implementation
that still records real times, and pinning it would have forced the measurements out of the journal.

Default stays `measured`, so no existing run or test changes behaviour.

### 2. A recorded-answer feed (task T2)

A journal already carries, per tick, the questions asked and the response received (`questions`,
`response`), the tick that asked, and the answer's age. A run can be driven from it with **no
network and no thread**: the feed answers the k-th fired call with the k-th recorded response, after
that response's recorded delay in ticks, through the same four members the loop already uses on the
dispatcher (`in_flight`, `request`, `take`, `close`). The tick that takes the answer becomes a
function of the transcript, not of the machine.

The feed is keyed by **call order**, not by tick, so a build that fires on a slightly different tick
still receives the transcript's answers in order; a build that fires *more* calls than the transcript
holds must say so rather than invent one.

A fed run's numbers are not its own. Its `jev_call_ms` is the recorded one, so the journal must mark
the run as fed in a way no reader can miss, and no field may present a transcript value as a live
measurement.

This is deliberately **not** action replay. Replaying recorded *actions* proves the emulator physics
and nothing about the deciding build — the two builds would be indistinguishable by construction.
Replaying recorded *answers* keeps each build's planner, policy, guardrail and fallback live, which
is exactly what an A/B comparison is for. What it cannot fix is the answers themselves: whether the
Jev service is deterministic for an identical request is unknown and needs a network measurement,
which this work does not claim either way.

The loop and the journal must distinguish a fed run from a live one, so no journal can be read as
evidence of a live call it never made.

### 3. The harness and its proof (task T3)

`mario_jev run --answers <journal>` (name to be confirmed at implementation) wires T1 + T2, and a
test proves the property end to end: **two runs with the same seed, the same fixed cost source and
the same transcript produce identical action sequences**, compared field by field with the timing
fields excluded. A second test pins the negative: with a measured cost source fed two different
clock readings, the horizon differs — the property is not vacuous.

The report gains one entry saying what a reproducible run is for, what it proves, and what it still
does not (the Jev service's own determinism; every millisecond figure, which stays a sample).

## Tasks

| # | Task | Surfaces | Evidence |
|---|---|---|---|
| T1 | Deterministic cost source: settings, planner, journal field, unit tests | `src/mario_jev/config.py`, `src/mario_jev/trajectory/planner.py`, `src/mario_jev/journal/records.py`, `tests/test_config.py`, `tests/test_trajectory.py`, `tests/test_journal.py` | an absurd `perf_counter` (constant, then raising) leaves the horizon and the roller table identical under `fixed` and moves them under `measured`; `measured` remains the default and every existing test passes untouched |
| T2 | The recorded-answer feed: a transcript reader and a call source the loop can use in place of the dispatcher, behind no flag yet | `src/mario_jev/policy/feed.py` (new), `src/mario_jev/policy/loop.py`, `src/mario_jev/journal/records.py`, `tests/test_feed.py` (new) | a fed run opens no socket and enters no client: the k-th fired call receives the k-th recorded response after its recorded delay; a tick's journal line marks the value as coming from a transcript and never presents it as a live measurement; the loop is otherwise unchanged |
| T3 | The CLI flag, the reproducibility proof and the report entry | `src/mario_jev/cli.py`, `src/mario_jev/config.py`, `tests/test_cli.py`, `tests/test_policy.py` (or a new `tests/test_reproducibility.py`), `docs/EVALUATION.md` | two fed runs, same seed, same cost source ⇒ identical action sequences modulo timing; and the negative that the comparison can fail |

## Risks

* **A transcript is a fixture, and a stale one is a trap.** If the feed is keyed by tick and a
  later build asks on different ticks, the comparison is no longer apples to apples. The feed must
  record when it could not answer, and the harness must say so rather than silently shifting the
  schedule.
* **A deterministic mode can be mistaken for a better mode.** It is an instrument. `measured`
  remains the default and the live behaviour is unchanged; the journal marks a fed run so no
  evidence can be misread as a live call.
* **Thread elimination changes the failure surface.** A run with no worker thread cannot exhibit a
  dispatcher race, so it must not be used to claim the live race is fixed.

## Non-goals

* No claim about the Jev service's determinism — unmeasured, and it needs a network call.
* No claim that the rollout bottleneck is fixed, or that the 7.5 Hz target is reachable; that is a
  separate objective, and it becomes measurable once this lands.
* No change to `measured` behaviour, to the emulator, to perception, or to the Jev client.
