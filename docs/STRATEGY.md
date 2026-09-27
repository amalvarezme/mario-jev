# Strategy: playing Super Mario Bros. in real time with Jev as the decision layer

This document is the strategy, not the changelog. It states what the system does,
why each piece exists, what is measured, and where each piece is weak.

---

## 1. The constraint that shapes the whole design

**Jev does not accept images.** Its published contract is explicit: `state` may be
a string, a JSON object, or an array of text — the OpenAPI schema types that third
case simply as `array` — and "Image, audio, and video inputs are not supported yet".

The request was to use each frame's image to characterize the state. That is
still exactly what happens — but the frame reaches Jev **indirectly**. A
deterministic perception layer converts the frame into structured state, and Jev
decides over that. Two consequences follow, and both are load-bearing:

1. Every feature Jev reasons about is a named number that a human can audit. When
   Jev picks an action, the reason is inspectable, not latent.
2. The perception layer's own error rate becomes a first-class measurement. If
   vision mislocates Mario, Jev is fed a false state and will make a confidently
   wrong decision. That is why every detection is cross-checked against RAM.

An optional local multimodal model (PlayJev-0.8B) also reads the frame directly
and contributes its move distribution as *one more feature*. It is an advisor,
never the authority, for the reasons in §10.

## 2. Architecture and phase contract

```
                    ┌──────────────────────────────────────────────┐
                    │  emulator (nes-py / gym-super-mario-bros)     │
                    │  frame (240,256,3) uint8 + RAM 2 KiB          │
                    └───────────────┬──────────────────────────────┘
                                    │
     every frame ───────────────────┤
                                    ▼
                    ┌──────────────────────────────────────────────┐
                    │  perception — deterministic, local, no net    │
                    │  Mario bbox · enemies · coins · tiles · gaps  │
                    │  each detection cross-checked against RAM     │
                    └───────────────┬──────────────────────────────┘
                                    ▼
                    ┌──────────────────────────────────────────────┐
                    │  state — typed MarioState                     │
                    │  relative features, plus to_jev_state() JSON  │
                    └───────────────┬──────────────────────────────┘
                                    │
     every K frames ───────────────┤
                                    ▼
                    ┌──────────────────────────────────────────────┐
                    │  trajectory — real savestate rollouts         │
                    │  N candidate actions × horizon, under a        │
                    │  wall-clock frame budget                       │
                    └───────────────┬──────────────────────────────┘
                                    ▼
                    ┌──────────────────────────────────────────────┐
                    │  Jev — ONE request, parallel questions        │
                    │  best_action (choice) · risk (score)          │
                    │  lethal_ahead (noul) · gap_severity (score)   │
                    │  → choice + probabilities + confidence        │
                    └───────────────┬──────────────────────────────┘
                                    ▼
                    ┌──────────────────────────────────────────────┐
                    │  policy — application code owns execution     │
                    │  confidence gate · safety override · fallback │
                    │  cadence · degradation recording              │
                    └───────────────┬──────────────────────────────┘
                                    ▼
                    emulator.step(action) ──▶ journal + dashboard
```

The boundary between Jev and the policy is deliberate and matches Jev's own
guidance: **Jev judges, application code executes.** Jev never touches the
emulator, never chooses its own authority, and cannot bypass the confidence gate.

## 3. The Jev contract this system relies on

`POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer <JEV_API_KEY>`,
`Content-Type: application/json`. Every request carries all three of `model`,
`questions`, and `state`; `state` is a string, an object, or an array, and images,
audio, and video are not accepted (§1).

`questions` is a JSON **object keyed by ids the application chooses** — not an
array — and each value is a question discriminated by `type`, of the shape
`{type, instructions, criteria}`. The free-text field is named `instructions`
(a string, object, or array) and is optional. Three question types, **all
evaluated in parallel against the same state**, so one HTTP request resolves every
judgment needed for a tick:

| Type | Use | `criteria` | Returns |
|---|---|---|---|
| `choice` | pick one action from the candidate set (≤255 options) | required: an object mapping each choice name to its description | `choice`, `probabilities` (keyed by choice name, sum 1), `confidence` |
| `score` | rate on an ordered rubric | required: an **ordered array** of levels (the spec requires at least one; the published prose says 2–10) | probability-weighted `score`, `legend`, `probabilities`, `confidence` |
| `noul` | yes/no judgment | optional: `{true, false}` | `noul` (0..1 = P(yes)) only — **a `noul` answer has no `confidence`** |

A `score` is averaged over the level probabilities, so it **may be fractional**
(0.48 and 1.30 were both observed), and its `legend` maps the level index as a
**string** (`"0"`…`"3"`) to the description that was sent; `probabilities` uses
the same string keys. Because a `noul` answer carries no confidence, it cannot be
gated by the same threshold as `choice` and `score` without asking an additional
question. The response is `{model, answers, usage:{input_tokens, output_tokens}}`:
`model` echoes the resolved version (`jev-latest` currently resolves to
`jev-1.13.0`), and `usage.output_tokens` is currently free of charge.

Errors: **401 or 403** for an invalid or missing key (hard failure, no retry — the
published prose says 401, but the live API returns 403 with
`error_type: authentication_error`), 422 validation (hard failure; the body names
the offending field in `detail[].loc`), 429 rate limit and 529 overloaded (retry
with exponential backoff). The API is fronted by Cloudflare, which rejects a
request that lacks a realistic `User-Agent` with its own 403 (`error code: 1010`)
before the request ever reaches Jev, so any client **must** send one.

The credential is held in `Settings.jev_api_key` with `repr=False` and
`compare=False`, so it cannot reach a log, a traceback, a journal line, or a
dataclass diff. The rest of the program sees only `Settings.jev_available`.

## 4. Frame-by-frame characterization

The frame is 240×256, three channels, `uint8` — measured, not assumed.

Perception runs on **every** frame because it is local and cheap. It produces,
per frame:

- Mario's bounding box from NES-palette segmentation, with a confidence,
- enemy positions and their relative distance and approach speed,
- the nearby tile geometry: solid ground, pipes, gaps, stairs, `?` blocks,
- collectibles and power-ups,
- the agreement between every detection and the RAM ground truth.

That last item is the honest quality measure. Vision is what we claim is on
screen; RAM is what the emulator knows. The disagreement rate is logged and
reported, never hidden.

**A corrected assumption worth recording.** `x_pos` is `0x006D * 0x100 + 0x0086`
— a 16-bit absolute level position — and it is *not* `0x03AD`. The two agree only
while `0x006D` is still zero, which is exactly why the wrong mapping looked correct
when only the level start was inspected. Measured: at frame 136 of `1-1`,
`x_pos = 256` while `0x03AD = 112`. Likewise `y_pixel` is exactly `0x03B8`;
`0x00CE` read the same value at the start by coincidence.

## 5. Possible trajectories, from real savestates

The original plan was a reduced physics model. That was replaced, because
`nes_py` exposes real savestates (`dump_state()` / `load_state(snapshot)`) and
they were measured at roughly **350 fps** for a load-state-and-step loop. An exact
rollout through the actual game beats an approximation of it, for less code.

Per decision tick, for each candidate action:

1. `dump_state()`,
2. step the candidate forward for the horizon,
3. observe the outcome and record the trajectory,
4. `load_state()` and repeat for the next candidate.

Each candidate yields a predicted path plus a terminal outcome from a fixed
vocabulary: `lands_safe`, `hits_enemy`, `falls_in_gap`, `reaches_goal`,
`no_progress`.

**The horizon is not fixed.** 7 candidates × 30 frames costs about 0.6 s at the
measured 350 fps, which is far above a 7.5 Hz tick. So the planner is given a
wall-clock **frame budget** and shortens the horizon to fit it, reducing toward
`rollout_min_frames` rather than overrunning the tick.

## 6. What Jev is asked each tick

One request, four parallel questions over the same state, in a single `questions`
object keyed by exactly these ids:

| Question id | Type | Asks |
|---|---|---|
| `best_action` | `choice` | which candidate action to take, each option's `criteria` description carrying that candidate's rolled-out outcome and cost |
| `risk` | `score` | 0–10 risk of losing Mario within the horizon |
| `lethal_ahead` | `noul` | is the current trajectory lethal if unchanged |
| `gap_severity` | `score` | how hard the terrain ahead is on an ordered rubric |

The returned `choice` is the action, `probabilities` is the decision score across
every candidate, and `confidence` is what the policy gates on.

## 7. Real time

60 decisions per second over HTTP is physically impossible, so the loop does not
try. It decouples perception from decision:

```
emulator advances one frame at 60 Hz wall-clock pace
   ├── perception + state          every frame      (local, no network)
   └── rollouts + Jev              every K frames   (default K = 8 ≈ 7.5 Hz)
```

Degradation ladder, in order:

1. **Inside budget, confidence ≥ threshold** → execute Jev's `choice`.
2. **Confidence < threshold** → run the deterministic safe fallback and record
   the low-confidence event with its full probability distribution.
3. **Over budget, or 429/529 after retries, or no credential** → hold the last
   known-safe trajectory and record the degradation.
4. **401, 403 or 422** → hard failure; a wrong or invalid key is not something to
   route around silently. The published prose documents only 401, but the live API
   answers an invalid key with 403 (§3).

Measured headroom: the emulator steps at **261 fps** unpaced, so 60 fps has about
4× headroom before pacing pressure appears. Whether the *Jev* round trip fits
inside a K-frame budget is measured per tick and reported; it is never assumed.

## 8. Risk model

"Minimise Mario's loss, arrive safely" is expressed as a cost ordering, not as a
reward to maximise:

- **Lethal**: falling into a gap, or contact with an enemy while small. Highest cost.
- **Reversible loss**: taking damage while powered up, losing a life. High cost.
- **Stall**: no progress, or being pushed back into the screen edge. Penalised,
  because stalling is how a safe-looking policy ends up never reaching the goal.
- **Progress**: forward movement toward the flag. Reward, but strictly subordinate
  to the two categories above.

Jev never sees this ordering as a single blended number. The `score` and `noul`
questions expose the lethal and severity judgments separately, so the recorded
evidence shows *which* judgment drove a decision instead of one opaque total.

## 9. Evidence

Every decision tick records: frame index, the perceived state, the trajectory
table, the exact question payload sent to Jev, Jev's raw answer with its
probabilities and confidence, the executed action, and measured latency. Alongside
the JSONL trace, each tick can emit an annotated frame PNG with the detection
overlay and the candidate trajectories drawn on it.

The dashboard shows, live: the frame with detections, the candidate trajectories,
the Jev decision-score table, and the confidence and latency evolution over time.

## 10. Honest limits

- **Jev's probability is a signal, not a guarantee.** Its own documentation says
  thresholds should be set by risk. That is why the confidence gate and the
  deterministic fallback exist rather than trusting `choice` unconditionally.
- **The perception layer is the weakest link.** It is hand-written segmentation
  over a 40-year-old palette. Enemy sprites that overlap tile graphics will
  sometimes be missed, and a missed Goomba is exactly the failure that kills
  Mario. The vision-vs-RAM disagreement rate is reported so this is visible
  instead of mysterious.
- **PlayJev is advisor-only, on evidence.** It sees the frame directly, which is
  attractive, but on *Infinite Mario* it scores **0.32 against its own teacher**
  (1757 vs 4229) — second worst of the ten games it was trained on. It is a
  feature, and it is optional: if torch or the 2.2 GB of weights are unavailable,
  the loop keeps running without it.
- **Savestates cannot be persisted.** `NativeStateSnapshot` has no `__bytes__`, no
  `__reduce__`, and no buffer protocol, and `load_state` rejects anything that is
  not that exact type. Reproducing a run means replaying its recorded actions, not
  restoring a file.
- **The rollout horizon is a budget, not a guarantee.** Under load the horizon
  shrinks, so a decision taken under pressure is made with less foresight than the
  same decision taken on an idle machine.
- **No claim is made that this reaches the goal.** What is claimed is that the
  pipeline is real, measurable, and honest about its failures.

## 11. Legal note on the ROM

The ROM used is the one bundled inside `gym-super-mario-bros`
(`gym_super_mario_bros/_roms/super-mario-bros.nes`). It is Nintendo's copyrighted
work, redistributed by that package's author. This project does not download,
copy, or redistribute it, and `*.nes` is git-ignored. Reading and writing it is a
decision for whoever runs the code, not something this project can grant.
