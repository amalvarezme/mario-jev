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

### Watching a run while it happens

A run can be watched instead of waited for. The viewer serves the same report, regenerated on every
request, so a browser refresh is the whole mechanism:

```sh
PYTHONPATH=src .venv/bin/python -m mario_jev serve artifacts/run.jsonl   # http://127.0.0.1:8000/
```

The play and the viewer are also one command, for the case where the run does not exist yet:

```sh
PYTHONPATH=src scripts/jev-env.sh .venv/bin/python -m mario_jev demo --max-ticks 300 --no-open
```

`demo` plays stage 1-1, writes `artifacts/demo.jsonl` with the per-tick frame and trajectory PNGs, and
serves the report for it while the run is still going, so a refresh shows the page growing; it opens the
browser at the bound address, which is also what `--port 0` prints. Ctrl-C stops the viewer and
**abandons** the run, and says exactly that: the journal keeps every tick written so far and has **no
summary trailer of its own**, so the readers report the run as **in progress** — which is the honest
reading of a run nobody finished, and not a claim that it completed. Two edge states get their own
sentence rather than that one: a trailer already sitting at the journal path was not written by this run,
so the demo says only that the trailer is there and may not be its own — this run never opened that file
(for instance because it is not writable) — and stopping the viewer does not finish a run. A run that
**fails** prints its error and the viewer keeps serving the ticks up to the failure.

`mario_jev live` is the same idea with the game visible:

```sh
PYTHONPATH=src scripts/jev-env.sh .venv/bin/python -m mario_jev live --max-ticks 900
```

Two windows open. The game plays in its own window at 60 fps, and the browser opens the decision
board at `/live`; `--no-open` prints that address instead of opening a window, and `--no-window`
plays headless and leaves the board as the only view.

The board is two columns. On the left, the decision panel: the buttons the agent pressed, the Jev
scores with the chosen answer marked, the rollout candidates, and the decision with its `reason` —
re-fetched about once a second, which is a snapshot at the decision tick rate, so a fast run means
some refreshes skip ticks. On the right, the game: every real-timeline frame the run produces is
published to a one-slot sink and polled from `/live/frame.png` at about 70 ms, roughly 14 Hz. That is
not 60 fps and the page says so where the game is. The polled frame is **not** evidence — it is
never written to disk and never enters the journal; the recorded per-tick PNGs, perception overlay
and all, stay in the full report at `/`.

The game window holds 60 fps only while the
pacing keeps up, and the run's own timing is the record of what the machine really did. Like `demo`,
`live` writes no summary trailer when it is interrupted, so the readers report that run as **in
progress**, which is the honest reading and not a claim that it completed.

A journal **without** its summary trailer — a run still in progress — is readable through this path
(`read_journal(..., allow_partial=True)`) and every reader then says the run is **in progress** rather
than `not recorded`, which is the label for a *finished* run that lost a key. The strictness that
matters is untouched: the header is still required and every line present is still validated, so the
only thing tolerated is that the trailer has not been written yet. A line caught mid-write is answered
`503` with the reader's own message and a sentence saying the file may be mid-write, never a partial
report presented as a whole one.

Three limits, measured and stated rather than discovered later: the page **embeds its images**, so it is
15.26 MB for the 442-tick run above (0.11 MB for a 3-tick one) and grows with the run — fine to refresh
by hand, unusable as a polling loop; the server is **single-threaded**, so one tab is the target and a
second request waits; and it binds `127.0.0.1` with **no authentication**, so what keeps the evidence
private is the loopback default and nothing else — publishing it takes an explicit `--bind`, which the
startup line then prints.

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

**§6 changed part of this and none of the conclusion.** Taking the call off the tick means a slow
round trip no longer stops the game, and it removed the ticks that used to absorb a whole call — but it
did **not** raise how often the loop decides *per frame*: §6 measures the same 25 calls over the same 200
frames in both builds. A faster loop gets through those same ticks sooner, so the Hz figure rises with
the frame rate and by the same 23 % — which says nothing about reaching 7.5 Hz of decisions.

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
That figure is **not a pacer defect and is not fixed here, because it is not fixable in this code**:
re-measured afterwards it is 28.12 ms per step, i.e. **1.69× the configured 16.67 ms period**, and a
bare `time.sleep(1/60)` loop on the same machine measures **101 ms** per step against a control of
`time.sleep(0.030)` at **116 ms** — measured while a `pi` process sat at 99.7 % CPU with a load average
above 3. Those two controls are properties of that loaded machine, not of the pacer: on an idle machine
they will be faster, and today's load no longer shows that process. What they establish is that the
pacer cannot guarantee its configured rate from inside Python, and it does not hide that: a shortfall
shows up in `total_lag_s`. (Precisely: `last_lag_s` is observed at the head of each wait, so a
`time.sleep` overshoot is charged to the **following** wait as `overshoot - period` — the sleep was
entitled to one period — and `wait()` now returns the sleep it **measured** rather than the one it asked
for. Both are documented in the class and pinned by tests.) What *was* a real defect sat on the
behind-schedule path and is fixed — see finding 7.

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

### 4. A 21-second stall left no trace — now it does

One tick in run 1 (tick 8) took **21 376 ms** of wall clock, of which **21 272 ms** was spent inside the
Jev call — the tick's own `latencies_ms` splits it as `jev: 21 271.52`, `total: 21 375.84`, with 98 ms
of rollout alongside. The journal's own `degradations` list for that
tick is **empty**, because the tick ultimately succeeded: the recorded response reports the
successful attempt at **577 ms**, and the remaining ~20.7 s is unaccounted for in the file. The figure
is consistent with a first attempt hitting the client's **20 s timeout**, one backoff and a successful
retry — but neither *this* file nor the code that wrote it could say so, because the header recorded
neither the timeout nor the attempt budget, and nothing counted attempts.

**Fixed and verified after the run.** `Settings.redacted` now carries `jev_timeout_s`,
`jev_max_attempts` and `rollout_min_frames`, so a header says what "slow" meant; `JevResponse` carries
`attempts` (the transport calls the returned result actually cost), recorded by the client from its own
retry loop; and the loop appends `jev_retried` to a tick whose successful response took more than one
attempt. Measured on a scripted timeout-then-success run:

```
header:              jev_timeout_s=20.0  jev_max_attempts=3  rollout_min_frames=6
tick 0 response:     attempts=2  latency_ms=577.0
degradations:        {'jev_retried': 1}
```

A reader of such a file can now reconstruct the 20.7 s: the header says the timeout was 20 s, the tick
says the call took two attempts, and the successful one is the 577 ms in the same block.

**What this does not do is explain the old file.** `artifacts/eval-1-1.jsonl` was written before the
change, so its header still has no timeout and its ticks still carry no attempt count. The evidence for
that stall remains a discrepancy in a 21-second latency, and no rewrite can add a measurement that was
never taken. That is the difference between fixing a defect and fixing a record.

### 5. The journal could not say why a run stopped — now it does

Added by the ratchet run, and it is the sharpest evidence gap in the file. That run stopped after
**805** tick records against a 900-tick budget, and **that file cannot state why** — nothing written
before this evaluation can, because the facts were never recorded. Two code facts, both confirmed
against `policy/loop.py` as it stood when the run was made:

* **The record's `state` is the state observed *before* the action.** `step()` builds it early, then
executes, then assembles the `TickRecord` around the pre-step state. So the final step has a record,
but that record shows the world before it: `lives` reads `[2]` in all 805 lines, including the last.
* **The termination itself was known and then discarded.** `StepResult.terminated`/`truncated` set an
internal `_ended` flag and were written nowhere: not on the record, not in a degradation, not in
`RunSummary`, whose seven fields are exactly `ticks`, `frames`, `decisions`, `sources`, `degradations`,
`elapsed_s` and `decision_rate_hz`. (The journal's own summary line carried those seven plus `kind`,
and nothing else.) A reader sees `ticks: 805 < 900` and cannot distinguish a death from
reaching the flag from a run someone stopped by hand.

**Fixed: a run now records why it stopped.** `TickRecord` carries the post-step `terminated` and
`truncated` facts; `RunSummary.termination` is one of a closed five-member set (`max_ticks`, `goal`,
`death`, `truncated`, `episode_ended`), refused at construction if it is anything else, and the
classification reads the episode signals through the state layer's own `is_goal`/`is_death` predicates
rather than re-deriving them. The journal writes all three, `render_text` prints the reason beside the
counts, and the dashboard leads its summary with it.

The mechanism is verified against a **real death**, not a fake: driving the emulator with `right` into
the first Goomba terminates at **frame 160** with `is_death(info)` true — reproducibly, and a test
drives the real emulator through the real loop with a scripted decision source until it dies, then
asserts that `termination` reads `"death"` and that the journal writes it, with `terminated: true` on
the final record and on no other.

An earlier observation of the same thing, made by luck rather than by construction, is worth recording
because it says more about this agent than about the classifier: **one** unscripted run of the same loop
with no credential ended at **tick 122** by death and wrote `"termination": "death"`, and **four runs
since — the identical code, seed and stage — all survived to the 400-tick cap.** The death path is real
and the recording works; the agent reaching that death is a coin toss, because its horizon comes from a
wall-clock measurement (see "What this evaluation does NOT establish"). That first run alone is not
evidence a reader can reproduce, which is why the test above exists.

**The old file still cannot say**: it was written before the field existed, and the readers report an
absent termination as `not recorded` rather than guessing.

**What the old file does suggest, and why that is only a suggestion.** The last recorded tick has all
seven candidates at `hits_enemy` and 360 in-game seconds still on the timer, which rules out a
timeout. It does **not** settle that Mario died: there is no goal coordinate in this repository to
rule the flag pole in or out, and no recorded signal either way. The honest reading is that the run
ended early, probably by death, and that the journal does not say.

The two runs tabulated earlier never died, which is why this went unnoticed until the agent started
progressing far enough to meet an enemy. So the file records the agent's best run yet and cannot say
how it ended. The fix is the block above — a termination reason on the summary, plus the post-step
outcome on the final record — and it applied to no file that already existed.

### 6. The world stopped while Jev thought — the call is off the tick now

This is the finding that mattered most, and it was a **redesign**, not a tweak. `step()` executes
exactly one game frame per tick and called the API **in series inside that tick**, so the game clock
was hostage to a 416 ms round trip: the unpaced run advanced **900 frames in 184 s (4.9 fps)**, and the
tick that hit a 20 s timeout spent **21 271.52 ms inside Jev** — Mario froze for twenty seconds.

**What changed.** A dispatcher owns one worker and at most one call in flight: the tick fires a call,
takes whatever has finished, and executes its frame either way. It never waits. An answer older than
three cadences is discarded, and the tick holds — what a non-cadence tick always did. The client itself
is untouched; it is run in a worker instead of rewritten. An answer applied from an earlier tick is
recorded as `jev_stale` with its age, so "Jev decided this" can never silently mean "Jev decided this
four ticks ago".

**Measured, same machine, same stage and seed, same 200 frames, same cadence (8), same day:**

```sh
PYTHONPATH=src scripts/jev-env.sh .venv/bin/python -m mario_jev run --fast \
    --max-ticks 200 --stage 1-1 --journal artifacts/t1-after.jsonl     # this build
# and the same command at the previous commit, in a worktree of it, for artifacts/t1-before.jsonl
```

| | before (call inside the tick) | after (call off the tick) |
|---|---|---|
| frames / wall clock | 200 / **35.63 s** | 200 / **28.87 s** |
| game frames per second | **5.61** | **6.93** |
| tick `total`, p90 (nearest-rank) | **475.8 ms** | **238.1 ms** |
| tick `total`, **maximum** | **1 203.4 ms** | **243.6 ms** |
| ticks whose own cost ≥ 1 s | **1** | **0** |
| ticks whose own cost ≥ 400 ms | **26** | **0** |
| calls fired | 25 | 25 |
| sum of the calls' cost | 10 628 ms | 10 832 ms |
| median call | 422.6 ms | 428.9 ms |

The 25 calls still cost the same **~10.7 s**. What changed is that **no tick contains it any more**:
the worst tick drops from 1.2 s to 244 ms, and every tick that carried a call now costs a couple of
hundred milliseconds instead. Precisely, and this is worth stating because the first reading of these
two rows invites a wrong attribution: in the old build **25 of the 26 ticks above 400 ms carried a
call**, and the **twenty-sixth did not** — the 1 203.4 ms worst tick of that run was a **rollout
overrun** (`rollout: 1 199.7 ms`, Jev phase ≈0), not a Jev stall. The worst *call-carrying* tick was
680.5 ms. So the call was not the only thing that could stop the world, and after the change neither
is the world's cost: the new build's maximum tick, 243.6 ms, is the sweep.

**What this deliberately did not change.** The decision cadence is the cadence, not the latency. Both
runs fired the same 25 calls and held the same 175 frames:

| `decision.source` | before | after |
|---|---|---|
| `hold` | 175 | 175 |
| `jev` | 19 | 0 |
| `jev_stale` | 0 | 18 |
| `fallback_below_confidence` | 6 | 6 |
| `guardrail_lethal` | 0 | 1 |

So the agent does not decide **more often per frame** than before, and it is not closer to the 7.5 Hz
design target: **Jev's own choice was executed on 19 of 200 frames before and 18 after** (the 19th is
the one the lethal guardrail overrode), because the cadence asks every eighth frame and the loop holds
in between. What the redesign removed is the *freeze*, not the cadence. In the paragraph §1's units the
rate did rise, from **0.702 to 0.866 Hz**, and that rise is exactly the same 23 % as the frame rate: the
loop asks at the same density, and a faster loop gets through more of those ticks per second. The
answers arrived **3–5 ticks** after being asked, none was discarded as stale, and all 25 were applied.

**Honest limits of this comparison.** One run per build, and the load differed (the load average read
by `uptime` just before each run was 4.16 for the new build and 5.73 for the old — read at the time and
recorded here, **not** written into either journal, so it is context rather than evidence), so the 23 %
rate gain is a floor rather than a clean measurement — the *structural* figures above are the ones to
trust, because where a call's cost lands is not a matter of load. The frames per second is still ~7, i.e.
the game advances **8.7× slower than real time**, now bounded by the rollout sweep (~120 ms per tick)
rather than by the API. **The next bottleneck is the rollout, not Jev**, and this section claims nothing
about saturating the emulator's 60 Hz.

### 7. The first live death with the fixes in: a pit the agent could see

An **observation**, not a defect claim, and it is the run that exercised the fixes end to end. Same
command as run 1 (`--fast`, stage 1-1, seed 0), extended to the same 900-tick budget, journalled to
`artifacts/t1-900.jsonl`:

```
ticks 442   termination "death"   elapsed 65.2 s   6.78 frames/s
x_max 40 -> 1151 (best at tick 418)   calls fired 55   answers applied 55   discarded 0
tick total p50 119.7 ms   maximum 281.0 ms   ticks >= 1 s: 0
```

The freeze is gone on a real run: **the worst tick of 442 cost 281 ms**, where the same build's
predecessor had a tick at 21 271 ms and the 200-tick comparison had one at 1 203 ms. Every one of the 55
answers was applied within the staleness bound (ages 2–5, one at 9), and the death is recorded as
`"termination": "death"` on the summary and `terminated: true` on the final tick — which is §5's fix,
seen on a real death rather than a scripted one.

**How it died, measured.** Mario ran off a ledge at `x_max` 1151 and fell. Its own state carried the gap
beforehand — `gap_distance_px` went **60 → 0** over the twenty ticks before the fall completed — and the
five fresh decisions taken *during* the fall were all `NOOP`:

| tick | what Jev said | confidence | executed |
|---|---|---|---|
| 404 | **`right+A`** | 0.49 | **NOOP** (below the 0.60 threshold) |
| 412 | `NOOP` | 0.51 | NOOP |
| 419 | `NOOP` | 0.42 | NOOP |
| 429 | `NOOP` | 0.41 | NOOP |
| 441 | `NOOP` | 0.24 | NOOP |

At tick 404 the model proposed **moving** and the confidence gate discarded it, so the table's own answer
ran; the table's best, for a body already in the air over a void, was to do nothing. The gate and the
rollout then agreed on `NOOP` for the remaining four decisions.

**What this does not establish, and it matters.** It does **not** establish that `right+A` would have
cleared the gap — air control in Super Mario Bros. is limited and no run here tried it — nor that a
different confidence threshold would have saved the episode. Both are experiments, not readings. What is
measured is the sequence: a visible gap, one discarded proposal to move, four agreements to stand still,
and a death. The stall escalation added for §3 never fired in this run (it needs 30 frames of frozen
`x_max`; the stall lasted 23), which is worth noting only because a fall gives no recovery to escalate
towards — it is an observation about this death, not a case for a different threshold.

## What the journal recorded honestly, and what it cannot say

The advisor was **disabled** for both runs (`{"available": false, "reason": "disabled",
"measured": false}` on all 1 020 ticks) because `torch` is not installed on this machine. The journal
distinguishes that from an advisor that was enabled and failed (which would read
`"measured": true` with its own reason), so the absence here is not ambiguous.

No tick called Jev and reported nothing: `jev_questions` is present exactly on the 113 ticks that
made a call and `null` on the rest, so a reader cannot mistake "no decision was requested" for "the
model had nothing to say". No-call ticks render in the dashboard as **a gap in the series**, not as a
zero, and no frame artifact is drawn for a tick whose PNG is absent.

**A credential in a URL cannot reach a log.** A URL key is not the only secret an endpoint can carry:
`https://user:password@host/v1` is a legitimate endpoint to accept, and writing it verbatim into the
header would have put a password in a journal, in the dashboard, in the doctor report and in the
client's `repr`. Every one of those now emits the endpoint through a single `redact_url_userinfo`, so
whole userinfo — username included — is dropped, on the way in and again on the way out when a reader
displays a recorded one. The honest limits, both stated in code: redaction is a projection, so the raw
endpoint is still an attribute of the client object (a debugger sees it, a log does not), and journals
written before this change are not migrated — the stored bytes keep what they captured, while the
dashboard redacts what it prints.

## What this evaluation does NOT establish

* **The PlayJev advisor is entirely unexercised.** The 2.2 GB weights are deliberately not
  downloaded and `torch` is absent, so every claim about its distribution is untested here. Its
  contribution to the agent is unknown, not zero.
* **No death occurred *in the two runs tabulated in the results section*.** Lives stayed at 2 for all
  900 frames and neither episode terminated, so the death path — `falls_in_gap`, `hits_enemy`, and the
  lethal guardrail that acts on them — was not exercised in those runs. **The later ratchet run ended
  early, probably by death** (see §5), so the path was very likely exercised live exactly once — and
  that file cannot say so, because it was written before the termination was recorded anywhere. (A run
  made now does say, and the classification is verified against a real death in §5; the old
  journal is a record that cannot be repaired.) `falls_in_gap` and `reaches_goal` still have synthetic
  unit coverage only; no run reached the flag.
* **The same seed does not reproduce the same run.** With the seed fixed at 0 and the stage fixed at
  1-1, one headless loop run against the real emulator ended at **tick 122** by death, and four
  identical runs since (including three consecutive ones made while this report was being verified)
  survived to the **400-tick** cap. The emulator is not the source: given the seed it is deterministic.
  The cost model is, because the rollout horizon is chosen from a wall-clock measurement of per-frame
  cost — under load the agent plans a different depth and therefore takes different actions. The
  procedure replicates; the trajectory does not. Any number in this report that depended on the horizon
  should be read the same way.
* **The horizon is a sample, never a constant.** Per-frame rollout cost ranged 0.645–2.349 ms within
  a single process in earlier measurements, and the cost model ratchets one way, so the horizons
  above (14 → 8, and 6 under pacing) are what happened on this machine under this load.
* **No run has saturated the emulator's frame rate.** With the call off the tick the game still advances
  at ~7 fps, bounded now by the rollout sweep (~120 ms per tick); the 60 Hz of the machine's configured
  rate is untouched, and nothing here claims it. The decision cadence is still the cadence: the loop
  holds between answers rather than deciding more often.
* **The one before-and-after comparison is a single run per build, and the load differed** (4.16 against
  5.73). Its rate figures are a floor; its structural figures — where a call's cost lands — are not
  load-sensitive in the same way.
* **One machine, one seed, one stage.** Nothing here generalises to another machine's latency or
  another machine's rollout cost, and 1-1's first wall is not a benchmark for the level. Note too that
  even a fixed seed does not fix the trajectory (see above): what replicates is the procedure.
* **The live cadence claim is a measurement, not a design property.** 0.614 Hz is what this machine
  and this API latency produced; the design target of 7.5 Hz remains unreached and, at a 416 ms
  round trip, unreachable without decoupling the call from the tick.

## A reproducible run: what the fed harness proves, and what it does not

Every comparison in this report carries the same excuse, and it is stated twice above: *"the same
seed does not reproduce the same run"*. Two causes sat behind that, and both had to go before any A/B
claim could be made.

1. **The horizon was sized from a wall-clock measurement.** `TrajectoryPlanner` calibrates the
   machine with `perf_counter` and sizes later horizons from what it measured; the same measurement
   drives the overrun correction and the learning branches. Under load the agent plans a different
   depth, so it takes different actions.
2. **Which tick applies an answer was decided by thread timing.** `JevDispatcher.take` never blocks,
   and the answer is applied on the tick that observes it. Two runs fed the same answers at different
   wall-clock moments execute different actions -- and one of them can even discard an answer as
   `jev_stale`.

### What it is for

`mario_jev run --answers <journal>` replays a recorded journal's Jev answers instead of calling the
API: the feed serves the k-th recorded answer to the k-th fired call after that answer's recorded
delay in ticks, through the same four members the loop already used on the dispatcher, with **no
socket and no worker thread**. Combined with `rollout_cost_source=fixed` (a horizon sized from two
constants, which consults no clock on the horizon path at all), the trajectory becomes a function of
its inputs.

This is deliberately **not** action replay. Replaying recorded *actions* would prove the emulator
physics and nothing about the deciding build -- the two builds would be indistinguishable by
construction. Replaying recorded *answers* keeps each build's planner, policy, guardrail and fallback
live, which is exactly what an A/B comparison needs.

```sh
# record a transcript once, live (network and credential required)
PYTHONPATH=src scripts/jev-env.sh .venv/bin/python -m mario_jev run --fast \
    --max-ticks 200 --stage 1-1 --journal artifacts/transcript-1-1.jsonl

# replay it for one build, offline: no credential, no request
MARIO_ROLLOUT_COST_SOURCE=fixed MARIO_ROLLOUT_FIXED_BASE_MS=0.5 \
MARIO_ROLLOUT_FIXED_PER_FRAME_MS=1.1 \
PYTHONPATH=src .venv/bin/python -m mario_jev run --fast \
    --max-ticks 200 --stage 1-1 --journal artifacts/fed-a.jsonl \
    --answers artifacts/transcript-1-1.jsonl

# and once more for the build you are comparing it against
PYTHONPATH=src .venv/bin/python -m mario_jev replay artifacts/fed-a.jsonl
```

### What it proves

**Two runs, same seed, same fixed cost source, same transcript, produce identical action sequences.**
`tests/test_reproducibility.py` drives that end to end through `main([...])` with the real emulator and
compares the journals **field by field**, excluding exactly the genuine measurements of the replaying
machine and naming them: each tick's `latencies_ms`, the sweep's own `elapsed_ms`/`sweep_ms`, the cost
model those produced (`base_ms`, `per_frame_ms`, `calibration_frames`), and the summary's `elapsed_s`
and `decision_rate_hz`. Everything left -- the horizons, every candidate row, the applied answers with
their ages, the decisions -- must match, and does. The same test checks that the transcript was really
replayed rather than both runs holding in fallback, and that every tick of a fed run carries the
`transcript_answers` stamp.

**And the proof can fail.** The negative control in the same file takes the default `measured` source,
feeds the same transcript, and injects two different clock readings -- the planner's own `time` module
replaced by an advancing fake, never a slow machine and never a sleep. The horizon moves (48 frames
against 6 in the recorded run), the executed actions move with it (`right+B` against `right`), and the
same field-by-field comparison is unequal. The equality above is therefore a property of the fixed
cost source, not an artefact of a comparison that cannot fail.

### What it still does not

* **The Jev service's own determinism is unmeasured.** The transcript is a recording of what the API
  answered once; nothing here says it would answer the same way again. That needs a network
  measurement, and this work makes no claim either way. A transcript is a fixture, and a stale one is
  a trap.
* **Every millisecond figure stays a load-dependent sample.** The fixed cost source removes the clock
  from the *decision*; it does not turn a measurement into a constant. A fed journal still records the
  replaying machine's real `elapsed_ms`/`sweep_ms`/`latencies_ms`, and they are as load-dependent as
  every other timing number in this report.
* **A fed run cannot exhibit the dispatcher race, so it must never be used to claim that race is
  fixed.** There is no worker and no future on the fed path: the race is *absent*, not *repaired*. The
  live path is still `JevDispatcher` with its one worker, and its correctness is witnessed by the live
  runs above and by its own tests -- never by a fed run.
* **A fed run is not automatically a reproducible one.** The flag alone replays the answers; it leaves
  the horizon on the measured source unless it is told otherwise, so the CLI prints a warning for
  `--answers` without `rollout_cost_source=fixed`. A reader who believed the flag alone bought
  reproducibility would compare two builds and measure the machine again.
* **The transcript's own counts are printed, not assumed aligned.** A transcript holds the answers the
  recording *applied*, not the calls it *fired*: a call that failed, or whose answer was discarded as
  stale, left a `jev_call_ms` but no response and therefore no slot in the transcript. The k-th
  recorded answer is then not the k-th original call, and a replay that fires **fewer** calls than the
  transcript holds never reaches the unused tail, so **no mismatch and no exhaustion signal fires**.
  That silence is why the CLI prints the transcript's fired/applied counts next to the run's after
  every fed run, and says so on the line when the two fired counts differ. `answers_exhausted` and
  `transcript_mismatch` remain the recorded signals for the two cases that *do* fire (the replay asked
  for more calls than the transcript holds, or the record answers a different question).

## Findings worth acting on

In the order I would act on them, each with the evidence above:

1. ~~**Decouple Jev from the tick**~~ **Done, and measured** — see §6. A dispatcher runs the call in a
   worker, the tick never waits, and an answer applied late is recorded as `jev_stale` with its age.
   Same machine, stage, seed, cadence and 200 frames: the worst tick's own cost fell from **1 203 ms to
   244 ms**, the ticks that carried a call from **25 to 0**, and the game went from **5.61 to 6.93 frames
   per second** while the 25 calls cost the same ~10.7 s. It is deliberately **not** a cadence change:
   both builds fired 25 calls and held 175 frames; the same density of decisions, and a faster loop.
   The next bottleneck is the rollout (~120 ms per tick), not the API.
2. ~~**Do not pace the shared emulator.**~~ **Fixed and re-measured** — see §2. `FramePacer.suspended()`
   plus one `with` in `plan()`: the sweep sleeps zero times, the paced run went from 157 s to 18 s, the
   horizon recovered from its floor of 6, and the fallback rate fell from 40 % to 27 %.
3. ~~**Record the retry, and the timeout.**~~ **Fixed and verified** — see §4. The header carries
   `jev_timeout_s`, `jev_max_attempts` and `rollout_min_frames`; a response carries `attempts`; a tick
   whose call took more than one attempt records `jev_retried`, counted in the summary. A scripted
   timeout-then-success run writes `attempts=2` beside `latency_ms=577.0` with a 20 s timeout in its
   header — the 20.7 s gap is now reconstructible. The pre-existing journal still cannot explain its
   own stall, and cannot be made to.
4. ~~**Give the agent a way out of a blind rollout.**~~ **Fixed and measured end to end** — see §3. A
   measured stall (30 frames without `x_max` advancing) escalates the horizon to 24, and the agent now
   reaches `x_max` **1363** instead of 724, still moving at the end of the run. The escalation fires
   101 times and is recorded on every one of those ticks.
5. ~~**Relax the cost model.**~~ **Fixed and measured end to end** — see §3. A fitting sweep now
   relaxes the remembered cost one `COST_RECOVERY` step toward what it measured, with only an overrun
   able to raise it. `per_frame_ms` tracks the machine instead of pinning at its worst value, the
   horizon goes from 6 to 8–24, the escalation falls from 101 uses to 46, and `x_max` reaches
   **1438**.
6. ~~**Record why a run stopped.**~~ **Fixed and verified against a real death** — see §5.
   `RunSummary.termination` names the reason from a closed set, the final record carries `terminated`,
   and the journal writes both. Older files report `not recorded`, which is the honest answer for them.
7. ~~**Fix the pacer's own rate.**~~ **The figure was a misdiagnosis; the arithmetic defect behind it
   is fixed.** 28.12 ms per step was never the pacer: on this machine a bare `time.sleep(1/60)` loop
   achieves **101 ms** per step and a control of `time.sleep(0.030)` takes **116 ms**, with a `pi`
   process at 99.7 % CPU. The pacer cannot beat the platform's timer, and it reported the shortfall
   honestly in `last_lag_s`/`total_lag_s`. What *was* wrong is real: the behind-schedule path
   re-anchored the deadline to *now*, forgiving the whole deficit and permanently adding each slow
   step's work to the period. It now repays the deficit by skipping sleeps — never game frames — and
   writes it off only past a documented bound. Verified on a **virtual clock**: the average interval
   between steps is **1.000000 periods** after the fix and **1.136364** with the old path restored, and
   every pacing test now drives that clock instead of the real one, so what the pacer does to its
   schedule is asserted exactly rather than measured against the machine's load.
