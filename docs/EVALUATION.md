# Evaluation — world 1-1, live

What actually happened when the agent ran Super Mario Bros. 1-1 against the real Jev API on this
machine. Every number below is measured; where something is inferred rather than measured, it says
so. The raw journals are local evidence under `artifacts/`, which `.gitignore` excludes from the
repository, so the numbers and the commands that produced them are written here instead.

Date: 2026-09-28. Machine: macOS arm64, Python 3.14.6. Model: `jev-latest` (resolved by the API to
`jev-1.13.0`), endpoint `https://api.typesafe.ai/v1/systemone`. Seed `0`, stage `1-1`.

## How to reproduce

```sh
# environment readiness, with the credential bridged from the macOS Keychain
PYTHONPATH=src scripts/jev-env.sh .venv/bin/python -m mario_jev doctor

# run 1 — 900 frames, unpaced
PYTHONPATH=src scripts/jev-env.sh .venv/bin/python -m mario_jev run \
    --fast --max-ticks 900 --stage 1-1 --journal artifacts/eval-1-1.jsonl

# run 2 — 120 frames, paced at the configured frame rate
PYTHONPATH=src scripts/jev-env.sh .venv/bin/python -m mario_jev run \
    --real-time --max-ticks 120 --stage 1-1 --journal artifacts/paced-1-1.jsonl

# read a journal back, or render it
PYTHONPATH=src .venv/bin/python -m mario_jev replay artifacts/eval-1-1.jsonl
PYTHONPATH=src .venv/bin/python -m mario_jev dashboard artifacts/eval-1-1.jsonl --out /tmp/report.html
```

`doctor` on this machine reports Python 3.14.6; `numpy`, `PIL`, `requests`, `gym_super_mario_bros`
and `nes_py` present; the ROM present; `JEV_API_KEY` **configured** (printed as a boolean, never the
value); and `torch`/`transformers` **not installed**, reported as the advisor being unavailable
rather than as a failure — which is the intended state here.

## Results

| | run 1 (unpaced) | run 2 (paced, before the fix) |
|---|---|---|
| game frames (decision ticks) | 900 | 120 |
| wall clock | 184.0 s | 157.2 s |
| game time advanced | 15.0 s | 2.0 s |
| **slower than real time** | **12.3×** | **78.6×** |
| Jev calls | 113 | 15 |
| **decisions / second** | **0.614 Hz** | **0.095 Hz** |
| designed decision rate | 7.5 Hz | 7.5 Hz |
| decision sources | jev 100, below-confidence 12, degraded 1, hold 787 | jev 9, below-confidence 6, hold 105 |
| fallback rate over calls | 11.5 % | 40.0 % |
| Jev latency p50 / p90 / max | 416 / 562 / **21 272 ms** | 435 / — / 859 ms |
| rollout sweep p50 | 72 ms | **1 205 ms** |
| horizon (frames) | 14 → 8 | 6 (the floor) |
| progress (`x_max`) | 40 → **724** | 40 → 350 |
| lives | 2 throughout | 2 throughout |
| in-game seconds burned | 45 | 6 |

The paced column is the state **before** the pacing defect was fixed; §2 records the fix and the same
run re-measured afterwards (157 s → 18 s, 0.095 Hz → 0.850 Hz, sweep 1 205 ms → 70 ms). The unpaced
column is unaffected by that fix.

## What worked

**The vision layer is exact against RAM.** Over all 900 frames of run 1, replayed action-by-action
on a fresh seeded emulator (the replay is faithful: zero `x_pos` disagreements with the recorded
states), `PerceptionAudit` reported Mario detected from the frame in **900/900** ticks, **0**
disagreements with the RAM-derived box, and **0** disagreements about whether he was on the ground.
The measured offset between the frame-derived box and the RAM box was 3.02 px mean (5 px max)
horizontally and 1.58 px mean (8 px max) vertically — sub-tile, as the detection geometry predicts.

**The Jev contract works end to end, live.** 113 real calls, each one POST carrying four questions
in parallel, each answer parsed into typed values. The API resolved `jev-latest` to `jev-1.13.0`.
`noul` answers came back without a confidence field, fractional `score` values arrived intact, and
`gap_severity` and `risk` returned legends matching the criteria that were sent — every documented
quirk of this API was exercised by real traffic rather than by a fixture.

**The failure taxonomy behaved as designed.** One call in run 1 met a genuine server-side failure —
HTTP **520** — and the policy degraded to the deterministic fallback, recorded `fallback_degraded`,
and kept running. No `JevAuthError`, `JevRequestError` or `JevProtocolError` occurred, so the
abort path was not triggered live; it remains covered by tests against a real client and a scripted
transport.

**The journal is real evidence.** The file round-trips, every line is independently parseable, and
the numbers in this document were computed from it rather than from console output. The dashboard
renders it (frames, detections, per-tick decision scores, confidence and latency series) with no
server and no extra dependency.

## What degraded, and where it broke

### 1. The cadence does not hold, and it is not close

The design is a decision every 8 game frames — 7.5 Hz at 60 fps. Measured: **0.614 Hz** unpaced and
**0.095 Hz** paced, against a designed 7.5 Hz. In unpaced terms the run covered 900 game frames in
184 s, so the agent runs about **12× slower than real time**; with pacing on, **79× slower**.

The arithmetic is not subtle: a Jev round trip has a **416 ms median** on this machine, a rollout
sweep costs **72 ms**, and a game frame is **16.7 ms**. One decision therefore costs about 30 game
frames of wall clock. The architecture's own degradation ladder is what covers this, and it is why
787 of 900 frames were `hold` rather than decisions — the cadence fired when it could, not when it
was scheduled to.

### 2. `real_time=True` makes the agent dramatically worse, not more real

This was not expected and is the most actionable finding. Enabling pacing raised the rollout sweep
from a **72 ms** median to a **1 205 ms** median — a 17× increase — because the pacing applies to
every emulator step, and the rollout sweep steps the same emulator for every candidate. The
consequences compound:

* the horizon collapsed to its floor of **6 frames** (0.1 s of game time) in every one of the 120
  paced ticks, against 8–14 unpaced;
* with less foresight, the share of calls falling below the confidence threshold rose from
  **10.6 %** to **40 %**;
* the loop ran at **1.31 s per frame**.

Separately, the pacer does not achieve its own configured rate even for bare stepping: 120 direct
steps with `real_time=True` took 3.60 s (**30 ms per step ≈ 33 fps**), against a configured 60 fps.
That defect is **untouched** by the fix below; re-measured afterwards it is 28.12 ms per step, i.e.
**1.69× the configured 16.67 ms period**.

**Fixed and re-measured the same day.** `FramePacer` gained a nestable `suspended()` context manager
(suppressing sleeps, resetting the schedule on the outermost exit, never touching `enabled`), and
`TrajectoryPlanner.plan()` now wraps its whole sweep — calibration, first sweep and correction sweep
— in it. The reasoning is that a rollout is a *simulated* future: nothing about it should consume the
real timeline, and the game is simply paused while the agent thinks.

| measured | before | after |
|---|---|---|
| one `plan()`, paced, fixed 8-frame horizon, tick 0 | 1 821 ms | **58 ms** |
| the same `plan()`, unpaced (control) | 93 ms | 58 ms |
| `time.sleep` calls during that sweep | **67 for 68 steps** | **0** |
| paced run, 120 frames — wall clock | 157 s | **18 s** |
| paced run — decision rate | 0.095 Hz | **0.850 Hz** |
| paced run — sweep p50 | 1 205 ms | **70 ms** |
| paced run — horizon | 6 (the floor) | 7–9 |
| paced run — below-confidence fallback | 40 % (6/15 calls) | **27 %** (4/15) |
| paced run — slower than real time | 79× | **9×** |

The mechanism is measured, not inferred: the sleep count was one per simulated step before the fix
and zero after. Reverting only the planner's context manager reproduces the defect exactly
(2 156 ms and 67 sleeps for 68 steps), which is how the regression test that pins it was checked.

The absolute millisecond figures are samples, not constants, as with every timing number in this
report: an independent verification measured the same pre-fix `plan()` at **3 774 ms** rather than
1 821 ms, because it ran while this evaluation was executing on the same machine. The **sleep count**
is the load-independent signal — 67 versus 0 for the same 68 steps — which is why that, and not the
elapsed time, is what the regression test pins.

What this does **not** fix, and the report still stands on: the cadence is a separate problem (§1 — a
416 ms round trip cannot fit a 16.7 ms frame, with or without pacing), the pacer's own shortfall is
untouched, and the stall in §3 happened in an **unpaced** run whose horizon was already 8. The
conclusion narrows rather than disappears: the paced run is now usable, but the decision rate is
still 11 % of the design and `real_time` still cannot make this agent real-time.

### 3. The agent stalls for two thirds of the run, and Jev is not the reason

`x_max` rose from 40 to **724 by frame 271** and then never moved again: the remaining **600 frames**
(45 in-game seconds) were spent standing at `x = 722`, on the ground, facing left, against a
**48 px (3-tile) wall** with no gap ahead.

The rollout is the reason. At every one of those 600 ticks, **all seven candidates** reported
`dx ∈ {-1, 0}`, `outcome = no_progress`, and an identical `risk = 0.60` over a horizon of 8 frames.
When every candidate is identical, the rollout carries **no signal at all** — neither the safest
candidate nor Jev can distinguish a wall-clearing jump from standing still, and the tie-break lands
on `NOOP`.

Jev is not the problem: it agreed with the rollout's own lowest-risk candidate on **112 of 113**
calls. The agent is faithfully executing a rollout that has stopped being able to see. Two
contributors are visible in the data and neither is proven alone:

* **The horizon ratchet.** The horizon decayed from 14 to 8 over the run (and to its floor of 6 under
  pacing). At 8 frames — 0.13 s of game time — a run-up plus a jump may genuinely not clear a 3-tile
  wall, so the rollouts are honest and the horizon is simply too short to represent the escape.
* **No stall handling.** The lethal guardrail only overrides a *hazardous* choice; standing still is
  not hazardous, and the confidence gate passed Jev's `NOOP` at 0.74–0.79. Nothing in the policy can
  tell "waiting is correct" from "the rollout has gone blind".

**The ratchet is the root cause, and it was measured.** Re-planning from the frozen stalled savestate
at several fixed horizons settles which of the two contributors is doing the damage:

| horizon | candidate `dx` | outcomes | risks | can the agent see the escape? |
|---|---|---|---|---|
| 8 | -1, 0, 0, 0, 0, 0, 0 | all `no_progress` | 0.6 only | **no** |
| 12 | -4, 0, …, 0 | all `no_progress` | 0.6 only | no |
| 16 | -7, 0, …, 0 | all `no_progress` | 0.6 only | no |
| **24** | -17, 0, 0, 0, 0, **2**, **2** | 5 `no_progress`, **2 `lands_safe`** | **0.2** and 0.6 | **yes** |
| 32 | -30, 0, …, **12**, **13** | 5 / 2 | 0.2 and 0.6 | clearly |
| 48 | -58, 0, 0, 0, 0, **40**, **55** | 5 / 2 | 0.2 and 0.6 | unmistakably |

The running jump that clears the wall needs about **24 frames** to show up as progress, and the
horizon was 8. The agent was not choosing badly; it was **blind by budget**, with the budget set by a
cost model that had ratcheted down over 270 ticks of overruns.

**Fixed the same day — measured end to end.** `TrajectoryPlanner.plan()` gained an optional
`min_horizon` floor (clamped into `[rollout_min_frames, 96]`), `RolloutTable.escalated` records
whether the horizon actually exceeded what the budget chose, and `ExecutionLoop` counts consecutive
frames without `x_max` advancing and asks for a 24-frame horizon once that count reaches 30. Scaling
the horizon was chosen over a heuristic fallback ("retry the last action that moved us") because
re-planning produces *real information* — two candidates become visibly safe — where the heuristic is
a guess.

| measured, 900 frames, same stage and seed | before | after |
|---|---|---|
| `x_max` reached | 40 → **724** | 40 → **1363** |
| still advancing at the end | no — dead since frame 271 | **yes — maximum on the last frame** |
| ticks that chose `NOOP` | 700 of 900 | **472** of 900 |
| escalations | 0 | 101, the first at frame 292 |
| horizon | 8 on 735 ticks | 6 on 696, **24 on 105**, 10 on 83 |
| wall clock / decision rate | 184 s / 0.614 Hz | 154 s / 0.735 Hz |

The agent now clears the wall and **nearly doubles its progress**, and it is still moving when the
frames run out.

**The ratchet itself is now fixed too — and the fix was measured the same way.** What the journals
recorded is the defect: `per_frame_ms` began at **0.69** ms and climbed to **2.92**, where it stayed
for **796 of the run's 900 ticks**, while the cost the sweeps actually demonstrated
(`elapsed_ms / steps_taken`, p50) was **1.30** — the memory settled at **2.24× the truth** and never
came back. It never came back because nothing lowered it: `remeasured_per_frame_ms` existed and was
called **only on the overrun path**, so a sweep that fitted the budget never re-measured anything.
The second live run shows the same shape at 2.09 against 1.24, a factor of 1.69, pinned for 734 ticks.

`COST_RECOVERY = 0.97` and `_relax_towards_measurement(...)` make a fitting sweep pull the model one
step back toward what it just measured. The value is **chosen**, not measured — the problem is what
was measured; the recovery takes `ceil(log(1.3048/2.9191)/log(0.97)) = 27` fitting ticks, which is
the same order as the loop's own 30-frame stall threshold, so a transient spike now heals itself in
about the time it takes to declare a stall. A measurement *above* the remembered cost leaves the
model untouched: only an overrun is a refutation.

| measured, same stage and seed | before (900 ticks) | with the ratchet relaxed (805 ticks) |
|---|---|---|
| `per_frame_ms` | 0.69 → **2.92**, p50 2.92 | 1.51 → **0.86**, p50 1.38 |
| real cost implied by the journal (p50) | 1.30 | 1.11 |
| **horizon** | **6** on 696 ticks | **8 to 24**, most often 10–14, with 46 at the escalation floor |
| recorded escalations | 101 | **46** |
| `x_max` reached | 1363 | **1438** |

The three things the change was for are all visible: the memory now tracks the machine instead of a
worst moment, the horizon moves back toward what the machine is demonstrating — its real cost
(p50 1.11 ms/frame) pays for **15** frames where the inflated memory paid for 6 — and **the escalation
drops
from the mechanism to the exception** — 101 uses to 46. It is still needed; it was reduced, not
replaced.

Note the run length: this one stopped at **tick 805**, not 900, and §5 is about why the journal
cannot tell you that.

### 4. A 21-second stall left no trace in the journal

One tick in run 1 took **21 272 ms** of wall clock. The journal's own `degradations` list for that
tick is **empty**, because the tick ultimately succeeded: the recorded response reports the
successful attempt at **577 ms**, and the remaining ~20.7 s is unaccounted for in the file. That
figure is consistent with a first attempt hitting the client's **20 s timeout**, one backoff, and a
successful retry — but the journal cannot prove it, because the run header does not record
`jev_timeout_s` or `jev_max_attempts` at all. A reader of this journal can see that 21 seconds
vanished and cannot see why.

Two gaps in the evidence, both fixable:

* a retry that consumed seconds is an event worth recording as a degradation, not merely a latency;
* the run header should carry the timeout and attempt budget, since they define what "slow" means.

### 5. The journal cannot say why a run stopped

Added by the ratchet run, and it is the sharpest evidence gap in the file. That run stopped after
**805** tick records against a 900-tick budget, and the journal cannot state why. Two code facts, both
confirmed against `policy/loop.py`:

* **The record's `state` is the state observed *before* the action.** `step()` builds it early, then
executes, then assembles the `TickRecord` around the pre-step state. So the final step has a record,
but that record shows the world before it: `lives` reads `[2]` in all 805 lines, including the last.
* **The termination itself is known and then discarded.** `StepResult.terminated`/`truncated` sets an
internal `_ended` flag and is never written anywhere: not on the record, not in a degradation, not in
`RunSummary`, whose seven fields are exactly `ticks`, `frames`, `decisions`, `sources`, `degradations`,
`elapsed_s` and `decision_rate_hz`. (The journal's own summary line carries those seven plus `kind`,
and nothing else.) A reader sees `ticks: 805 < 900` and cannot distinguish a death from
reaching the flag from a run someone stopped by hand.

**What the file does suggest, and why that is only a suggestion.** The last recorded tick has all
seven candidates at `hits_enemy` and 360 in-game seconds still on the timer, which rules out a
timeout. It does **not** settle that Mario died: there is no goal coordinate in this repository to
rule the flag pole in or out, and no recorded signal either way. The honest reading is that the run
ended early, probably by death, and that the journal does not say.

The two runs tabulated earlier never died, which is why this went unnoticed until the agent started
progressing far enough to meet an enemy. So the file records the agent's best run yet and cannot say
how it ended. The fix is small and unbuilt: a termination reason on the summary, plus the post-step
outcome on the final record.

## What the journal recorded honestly, and what it cannot say

The advisor was **disabled** for both runs (`{"available": false, "reason": "disabled",
"measured": false}` on all 1 020 ticks) because `torch` is not installed on this machine. The journal
distinguishes that from an advisor that was enabled and failed (which would read
`"measured": true` with its own reason), so the absence here is not ambiguous.

No tick called Jev and reported nothing: `jev_questions` is present exactly on the 113 ticks that
made a call and `null` on the rest, so a reader cannot mistake "no decision was requested" for "the
model had nothing to say". No-call ticks render in the dashboard as **a gap in the series**, not as a
zero, and no frame artifact is drawn for a tick whose PNG is absent.

## What this evaluation does NOT establish

* **The PlayJev advisor is entirely unexercised.** The 2.2 GB weights are deliberately not
  downloaded and `torch` is absent, so every claim about its distribution is untested here. Its
  contribution to the agent is unknown, not zero.
* **No death occurred *in the two runs tabulated in the results section*.** Lives stayed at 2 for all
  900 frames and neither episode terminated, so the death path — `falls_in_gap`, `hits_enemy`, and the
  lethal guardrail that acts on them — was not exercised in those runs. **The later ratchet run ended
  early, probably by death** (see §5), so the path was very likely exercised live exactly once — and
  the journal cannot say so, because the termination is recorded nowhere. `falls_in_gap` and
  `reaches_goal` still have synthetic unit coverage only; no run reached the flag.
* **The horizon is a sample, never a constant.** Per-frame rollout cost ranged 0.645–2.349 ms within
  a single process in earlier measurements, and the cost model ratchets one way, so the horizons
  above (14 → 8, and 6 under pacing) are what happened on this machine under this load.
* **One machine, one seed, one stage.** Nothing here generalises to another machine's latency, and
  1-1's first wall is not a benchmark for the level.
* **The live cadence claim is a measurement, not a design property.** 0.614 Hz is what this machine
  and this API latency produced; the design target of 7.5 Hz remains unreached and, at a 416 ms
  round trip, unreachable without decoupling the call from the tick.

## Findings worth acting on

In the order I would act on them, each with the evidence above:

1. **Decouple Jev from the tick** (or accept ~1 Hz and say so). A synchronous call of 416 ms median
   inside a 16.7 ms frame budget cannot hold any cadence above about 2 Hz. Nothing else on this list
   matters if the decision rate is 8 % of the design. **Still open.**
2. ~~**Do not pace the shared emulator.**~~ **Fixed and re-measured** — see §2. `FramePacer.suspended()`
   plus one `with` in `plan()`: the sweep sleeps zero times, the paced run went from 157 s to 18 s, the
   horizon recovered from its floor of 6, and the fallback rate fell from 40 % to 27 %.
3. **Record the retry, and the timeout.** A 20 s timeout that ends in success currently leaves an
   empty `degradations` list and an unexplained 21 s latency. **Still open.**
4. ~~**Give the agent a way out of a blind rollout.**~~ **Fixed and measured end to end** — see §3. A
   measured stall (30 frames without `x_max` advancing) escalates the horizon to 24, and the agent now
   reaches `x_max` **1363** instead of 724, still moving at the end of the run. The escalation fires
   101 times and is recorded on every one of those ticks.
5. ~~**Relax the cost model.**~~ **Fixed and measured end to end** — see §3. A fitting sweep now
   relaxes the remembered cost one `COST_RECOVERY` step toward what it measured, with only an overrun
   able to raise it. `per_frame_ms` tracks the machine instead of pinning at its worst value, the
   horizon goes from 6 to 8–24, the escalation falls from 101 uses to 46, and `x_max` reaches
   **1438**.
6. **Record why a run stopped.** A run that ended after 805 of its 900 ticks records no reason:
   the termination signal is discarded, the final record's `state` is the pre-step world, and
   `RunSummary` has no termination field. The journal therefore cannot distinguish a death from
   reaching the flag from a manual stop, and it cannot even settle *that* a death occurred. **Open,
   and new** — see §5.
7. **Fix the pacer's own rate.** 28.12 ms per step against a configured 16.67 ms is **1.69×** the
   configured period. **Still open.**
