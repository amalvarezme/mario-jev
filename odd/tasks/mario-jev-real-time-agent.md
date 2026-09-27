# Tasks: mario-jev-real-time-agent

Real-time Super Mario Bros. (NES) agent whose decisions come from the Jev System One model
(TypeSafe AI), with a deterministic vision layer, forward trajectory enumeration, and an
optional local multimodal advisor (PlayJev-0.8B).

Branch: `feat/mario-jev-real-time-agent` (off `main`).
Repo: `/Users/andresalvarez/Documents/mario-jev` (greenfield, git initialized this session).

---

## 1. Decisions taken with the user BEFORE any code

| # | Decision | Chosen | Rejected |
|---|----------|--------|----------|
| D1 | Perception / decision split | Deterministic vision layer produces typed JSON state **and** PlayJev-0.8B runs as an optional second-opinion advisor feeding a distribution feature into that state | — |
| D2 | Emulation engine | Python: `gym-super-mario-bros 9.1.0` + `nes-py 9.0.1` | `jsnes 2.1.0` (Node); `stable-retro 1.0.1` |
| D3 | Assets | User has a legal ROM and a Jev API key | — |

**D3 was overtaken by a discovery.** `gym-super-mario-bros 9.1.0` ships the ROM inside the package
(`gym_super_mario_bros/_roms/super-mario-bros.nes`, 40 976 bytes, iNES magic `NES\x1a`, mapper 0,
2 PRG / 1 CHR banks) and `SuperMarioBrosEnv` loads it through `smb1_rom_path()` with no external
path. No other `*.nes` exists on this machine (searched `$HOME` to depth 10 plus `~/Documents`,
`~/Downloads`, `~/Desktop`, `~/ROMs` and the RetroArch paths), so the bundled ROM is the one
actually used. That ROM is Nintendo's copyrighted work redistributed by the package author, not
by this project: our code never downloads, copies, or redistributes it. That is a fact the user
should weigh, and it is not something this plan can grant. The only remaining asset blocker is
`JEV_API_KEY`, absent from the environment and from every shell rc file checked.

### 1.1 Harness fallback — RESOLVED

WU1 and WU2 were executed **inline**, with delegation recorded as unavailable: the Pi session had
been started when this directory was **not** a git repository, so it held no clone identity.
Every launch path failed — the `gentle-ai-worker` writer, a read-only `gentle-ai-explore` probe,
the explicit `repository_root` route, and `session_worktree_register` — all with *"could not
register launched worktree: Select an existing worktree in the same Git clone as this session"*.
The read-only probe failing identically proved this was not writer-worktree isolation, and the
parent therefore executed inline as the routing ladder's documented fallback.

**Resolved.** Pi was restarted inside the now-git directory. A live `gentle-ai-explore` read-only
probe re-ran and returned its result (task `mukbrs7d-1-hkhe`), so the delegated runtime exists
after all. The Mandatory Delegation Triggers (4-file rule, multi-file write rule, verification
rule, long-session rule) are enforceable from WU3 onward, and the remaining work units must be
routed through them instead of defaulting to inline execution.

---

## 2. Verified facts this plan rests on (not assumed)

1. **Jev does not accept images.** Official contract: `state` is `text`, a JSON `object`, or an
   `array` of texts — "Image, audio, and video inputs are not supported yet". The frame must be
   converted to structured state by our own code. This is why D1 needs a perception layer at all.
2. **Jev endpoint — corrected 2026-09-27 against the live API and its published OpenAPI spec.**
   The endpoint is `POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer <key>`,
   `Content-Type: application/json`. `GET https://api.typesafe.ai/v1/models` lists the aliases the
   key may use (`jev-latest`, `jev-preview`). **The endpoint recorded here before was
   `https://thejevai.com/v1/systemone`: a different product, which rejects this key with
   `401 Invalid API key`.** Evidence: the spec at `https://api.typesafe.ai/openapi.json` (title
   `TypeSafe`, version `0.2.0`) and a live `200` for the real tick payload.
   Contract, taken from the spec rather than from prose:
   - The request requires all three of `model`, `questions` and `state` (`model` is required).
   - `state` is a `string`, `object`, or `array`. Images, audio and video are not supported.
   - `questions` is an **object keyed by ids the application chooses**, `minProperties: 1`; each
     value is a `Question` discriminated on `type`. Each is `{type, instructions, criteria}`, and
     **`instructions` is optional**, accepting a string, object, or array.
   - `choice`: `criteria` required, an object mapping choice name -> description. Answer ->
     `choice`, `confidence`, `probabilities` keyed by choice name.
   - `score`: `criteria` required, an **ordered array**, spec `minItems: 1`. Answer -> `score`
     (probability-weighted, **may be fractional**), `confidence`, `legend` (level index as a
     string -> the description that was sent), `probabilities` keyed by the level string.
   - `noul`: `criteria` optional, `{true, false}`. Answer -> `noul` (0..1 = P(yes)) and
     **no `confidence` field at all**, so a noul judgment cannot pass the same confidence gate as
     choice/score without a second question.
   - Response: `{model, answers, usage:{input_tokens, output_tokens}}`. `model` echoes the
     resolved version (`jev-latest` -> `jev-1.13.0`). `usage.output_tokens` is currently free.
   - Errors: `422` carries `{"detail":[{loc, msg, type, input, ctx}]}` and names the offending
     field. Unauthenticated access returns **`403`** with `error_type: authentication_error`,
     while the published prose says `401`; treat **both 401 and 403 as hard authentication
     failures**, and `429`/`529` as backoff.
   - **Cloudflare sits in front of the API**: a request without a realistic `User-Agent` is
     rejected with `403` and body `error code: 1010`, before it ever reaches Jev. The client must
     always send one.
   Key must stay server-side, supplied to the process as `JEV_API_KEY`.
3. **PlayJev-0.8B** (`OmniJev/PlayJev-0.8B`, Apache 2.0, base `Qwen3_5ForConditionalGeneration`):
   sees the frame + the move list and returns a probability per move. `model.safetensors` is
   **2214.6 MB**. Honest caveat driving its advisor-only role: **0.32 vs teacher on Infinite Mario**
   (1757 vs 4229), second worst of its ten games. Its signal is a feature, never the authority.
4. **Dependency feasibility on this machine** (macOS arm64, Python 3.14.6, Node 22.23.1):
   - `nes-py 9.0.1` publishes `cp314-cp314-macosx_11_0_arm64` -> no compilation.
   - `torch 2.14.0` publishes `cp314` macOS arm64 wheels; `transformers 5.17.0` exists.
5. **SMB1 RAM map** (Data Crystal, *Super Mario Bros./RAM map*), the ground truth our vision
   layer is cross-checked against:
   `0x006D` player horizontal position in level · `0x0086` player x on screen ·
   `0x03AD` player x offset within screen · `0x00CE` player y · `0x03B8` player y within screen ·
   `0x006E–0x0072` enemy x in level · `0x0087–0x008B` enemy x on screen ·
   `0x03AE–0x03B2` enemy x offset · `0x03B9–0x03BD` enemy y · `0x071A` current screen ·
   `0x071B` screen scroll · `0x071C` screen-edge x.
   `gym-super-mario-bros` exposes `env.ram` and an `info` dict. Exact `info` keys must be read
   from the installed package, not assumed; task 1.4 verifies them.
6. **Emulator facts measured in this session** against the installed packages, not read from
   documentation:
   - Frame: `env.screen` / `env.observation()` -> `numpy (240, 256, 3)` `uint8`
     (`SCREEN_HEIGHT=240`, `SCREEN_WIDTH=256`, `SCREEN_SHAPE_24_BIT=(240, 256, 3)`). `nes_py.nes_env`
     also exposes `OBSERVATION_MODE_RGB_ARRAY_CONTIGUOUS` and `OBSERVATION_MODE_GRAYSCALE`, the
     latter able to write into a reusable `output` array.
   - RAM: `env.ram` -> `numpy (2048,)` `uint8`; `env.ram_values(specs, output)` reads configured
     values into a reusable `uint32` array.
   - `info` has **42 keys**, not the 26 `_get_info` builds: `_normalized_info()` and
     `_reward_info()` merge in `clear`, `death`, `game`, `game_family`, `lives`, `progress`,
     `progress_max`, `reward_components`, `reward_total_clipped`, `reward_total_unclipped`,
     `rom_mode`, `single_stage`, `target_stage`, `target_world`, `task_id`, `timeout`,
     `world_label`.
   - Action sets: `SIMPLE_MOVEMENT` (7) `[NOOP, right, right+A, right+B, right+A+B, A, left]`;
     `RIGHT_ONLY` (5); `COMPLEX_MOVEMENT` (12).
   - **Savestates work**: `env.dump_state()` returns an opaque `NativeStateSnapshot` and
     `env.load_state(snapshot)` restores it (a rollout left `x_pos` exactly where it started).
     Measured throughput: **261 fps** straight stepping (unpaced, so 60 fps has ~4x headroom) and
     **350 fps** for `load_state` + step rollout loops.
   - RAM cross-check, and a corrected assumption: `info['x_pos']` is `0x006D * 0x100 + 0x0086`
     (a 16-bit absolute position), **not** `0x03AD` as first assumed — both read 40 at the 1-1
     start, which is exactly why the wrong mapping looked right. `info['y_pixel']` is exactly
     `0x03B8`; `0x00CE` also read 176 at that moment, again by coincidence rather than identity.
     `info['y_pos']`=79 is a *different* quantity from `y_pixel` and must not be conflated with it.
   - A savestate is **not serialisable to disk**: `NativeStateSnapshot` exposes no `__bytes__`, no
     `__reduce__`, and no buffer protocol, and `load_state` rejects anything that is not that exact
     type. A file-backed savestate is therefore impossible, so the adapter deliberately has no
     `save_state`; one was written and then deleted rather than left as dead code.
   - `load_state` restores the emulator but leaves the gym wrapper's cached `info` stale, so the
     adapter refreshes it from live RAM (`_get_info()` reads RAM at call time). Without that step
     the rollout planner would compare every candidate against the wrong starting state.

   Consequence for the cadence budget: the sweep cost is **not reproducible on this machine**.
   Measured `per_frame_ms` ranged over `0.65 .. 2.35 ms` *within the same process*, so at the
   default `rollout_budget_ms = 120` the planner chose horizons of `10, 20, 15, 18, 15` across
   five consecutive runs. Two separate hazards follow, and both were measured rather than
   reasoned about:
   * `sweep_ms` mostly lands at or just under the budget, but does overshoot (one run measured
     122 ms against a 120 ms budget).
   * `elapsed_ms` — what the caller actually pays — is **1.0x to 3.5x** the budget, because it also
     contains the calibration pass and a possible second correction sweep (`passes` was 2 in three
     of five runs). The 7.5 Hz tick has 133 ms, so a 419 ms `plan()` cannot hold the cadence; the
     degradation ladder in `docs/STRATEGY.md` §7 is what covers that case today.
   The horizon must therefore be chosen from *measured* cost per machine and per load — never
   assumed, and never reported as a single number. `RolloutTable` carries `horizon`, `sweep_ms`,
   `elapsed_ms`, `base_ms`, `per_frame_ms` and `passes` so the journal can record all of it.
   **Open item for WU4**: the caller must size its tick knowing that one call can cost up to ~3.5x
   the budget.
7. **Stepping past the end of an episode raises.** `nes_py` raises
   `ValueError: cannot step in a done environment! call 'reset'` on the first step after the game
   ends, and a candidate dying mid-rollout is exactly the signal the planner exists to produce.
   The rollout loop must break on `StepResult.terminated` / `.truncated`. `load_state(snapshot)`
   **does** clear the wrapper's done flag, so the sweep continues to the next candidate with no
   `reset()`. Both behaviours are pinned by tests in WU3.

## 3. Real-time strategy (the core design constraint)

60 decisions/second over HTTP is physically impossible. The strategy is a **decoupled decision
cadence**, not a per-frame call:

```
emulator advances every frame at wall-clock pace (60 Hz budget)
        │
        ├── perception + trajectory enumeration run every frame  (cheap, local, no network)
        │
        └── Jev is consulted every K frames (default K=8 ≈ 7.5 Hz)
                 │
                 ├── if the answer arrives inside the budget  -> execute the chosen action
                 ├── if confidence < threshold               -> deterministic safe fallback
                 └── if it exceeds the budget / 429 / 529    -> hold the last safe trajectory
                                                                and record the degradation
```

Every tick records latency, so the real decision rate is **measured, not claimed**.

## 4. Work units

Each work unit is one commit on the feature branch, tests alongside the behaviour, English
artifacts, Conventional Commit message.

### WU1 — Foundation, emulator boundary, strategy document — **DONE**
- [x] 1.1 `.gitignore` extended (`.venv/`, `__pycache__/`, `*.py[cod]`, `.pytest_cache/`,
      `.ruff_cache/`, `artifacts/`, `*.nes`) while keeping `.atl/`. Branch
      `feat/mario-jev-real-time-agent` created off `main`.
- [x] 1.2 `pyproject.toml` with pinned deps and pytest config (including the `rom`, `live_jev`,
      `playjev`, `slow` markers) + `.venv` created with `uv`. Installed cleanly with no source
      build: `gym-super-mario-bros 9.1.0`, `nes-py 9.0.1`, `numpy 2.5.3`, `pillow 12.3.0`,
      `requests 2.34.2`, `gymnasium 1.3.0`, `pytest 9.1.1`.
- [x] 1.3 `docs/STRATEGY.md` written: 11 sections covering the image constraint, the phase
      contract, the Jev contract, frame characterization, savestate rollouts, the per-tick
      question set, the real-time cadence and degradation ladder, the risk model, evidence, the
      evaluation protocol, and per-component honest limits.
- [x] 1.4 `src/mario_jev/emulator/` — `ram.py` (RAM map + `RamView`), `actions.py` (joypad bits
      taken from `JoypadSpace._button_map`, `SIMPLE_MOVEMENT`/`RIGHT_ONLY`, `Action`), `env.py`
      (`MarioEmulator`, `StepResult`, `FramePacer`, stage parsing). **Done as specified:** the real
      `info` keys (42) and the real `env.ram` shape `(2048,)` were verified and pinned in tests,
      as were the frame shape (240, 256, 3) and the corrected `x_pos` / `y_pixel` invariants.
      `rom_path` was deliberately **not** implemented: `SuperMarioBrosEnv` has no external-ROM
      parameter, so such an option could only ever be dead configuration.
- [x] 1.5 `tests/` harness written: 49 tests, all passing, **0 skipped** — the emulator tests run
      for real because the ROM is present. No ROM-gated skip was needed and none was faked.

Two defects found and fixed inside this work unit, both recorded in §2.6: the cached `info` was
stale after `load_state` (which would have made every rollout compare against the wrong starting
state), and a file-backed `save_state` was written and then deleted because
`NativeStateSnapshot` cannot be serialised.

### WU2 — Perception: frame -> terrain and player — **DONE**
- [x] 2.1 `src/mario_jev/perception/` — deterministic detection from the NES frame. Scope was
      **narrowed by measurement**, and the narrowing is the substance of this unit:
      * `palette.py` holds the SMB1 overworld palette sampled from real frames, not guessed.
      * `detector.py` detects the **player** (red-pixel bounding box + confidence) and the
        **terrain** (floor surface row per column, pits, raised surfaces).
      * **Enemies are deliberately NOT frame-detected.** At 1-1 the Goomba's cap uses palette
        entries `(228,92,16)` and `(252,160,68)` — the same entries as ground blocks, bricks and
        coins — so colour cannot separate an enemy from level furniture. The first attempt keyed
        on brown `(172,124,0)` and returned the wrong answer: in sampled frames those 46 brown
        pixels at columns 114..125 are **Mario's own shoes and buttons**, inside his own box,
        while the visible Goomba at column ~139 was not isolated at all. That detector was
        removed rather than shipped behind a caveat. Enemies come from RAM, which is exact.
      * Coins, power-ups, pipes, stairs and `?` blocks were dropped from the checklist for the
        same reason: the palette entries that draw them are shared with the floor and bricks,
        so a colour-only claim about them would not survive contact with a real frame.
- [x] 2.2 Cross-check against RAM is implemented as `PerceptionAudit` and covers Mario's box,
      the offset, an independent `feet_on_ground` geometric check, and the active enemy count.
      **Measured agreement: 30/30 and 32/32 sampled ticks (100%)** across two episodes, from
      `test_the_frame_and_ram_agree_about_mario_in_a_real_episode`.
- [x] 2.3 `src/mario_jev/state/` moved to WU3, together with the trajectory planner, because the
      typed state's relative features are derived from the rollouts.
- [x] 2.4 Tests written: synthetic frames for the cases the game cannot be asked to produce on
      demand (a pit, a raised platform, no player, red hidden in the HUD, too few red pixels,
      a frame/RAM disagreement) plus live emulator frames for everything else.

Three corrections found by measurement in this unit, each now pinned by a test:
1. **The black seam.** A naive "contiguous solid from the bottom edge" scan breaks on the first
   seam between floor blocks and reported the floor as a single row at 239 with 192 phantom
   holes. Counting black as solid turns that into the true profile: floor at 208, zero holes.
2. **The player's box.** `[x, x+16) x [y+16, y+32)` — with `0x03AD=112`, `0x03B8=176` the sprite
   is at rows 192..207, so the feet rest on row 207, exactly one above the floor top at 208.
3. **Mario's red is animated, not constant.** Over 260 frames it ranges 37..55 px (median 45,
   three poses: 45 standing, 37 and 55 walking). A constant of 45 asserted as equality would
   have been a test of one pose; the range is now the pinned fact and 45 the confidence median.

Also measured and recorded, not acted on: RAM `0x0087` ("enemy x on screen") read **215** for a
Goomba whose drawn sprite sat near column 139, while `0x03AE` (the offset block) read 129. The
offset block is the one that corresponds to the drawn sprite; nothing in this project positions
with `0x0087`.

### WU2.5 — Jev contract correction and credential bridge — **DONE** (`78e9798`)
Forced by WU4's dependency: the endpoint and the question shape recorded in §2.2 were wrong, so
the client would have been written against an API that rejects this key.
- [x] 2.5.1 `src/mario_jev/config.py` — `DEFAULT_ENDPOINT` corrected to
      `https://api.typesafe.ai/v1/systemone`. `DEFAULT_MODEL` (`jev-latest`) was already right.
- [x] 2.5.2 `README.md` and `docs/STRATEGY.md` §1/§3/§6/§7 — the endpoint, the
      `questions`-is-a-map shape, `instructions`, the criteria shapes, the `noul`-has-no-
      confidence caveat, the 401/403/422/429/529 ladder, and the mandatory `User-Agent`.
- [x] 2.5.3 `scripts/jev-env.sh` — the credential bridge the user chose: read the key from the
      macOS Keychain, export it as `JEV_API_KEY`, then `exec "$@"`. Verified against the real
      Keychain entry. Sourcing restores the caller's shell options, so strict mode cannot leak.
- [x] 2.5.4 `tests/test_credentials.py` — 8 tests, none of which touches the real credential.
- [x] 2.5.5 `tests/test_config.py` — the corrected host is pinned literally, with a negative
      assertion for the old host.

Test count after this unit: **80 passing, 0 skipped** (baseline 71).

### WU3 — Typed state and possible trajectories
- [ ] 3.0 `src/mario_jev/state/` — typed `MarioState` with **relative** features (distance to
      nearest enemy from RAM, gap width ahead from the frame, frames-to-jump-window, vertical
      clearance, progress, lives, timer) and a canonical `to_jev_state()` JSON projection.
- [ ] 3.1 `src/mario_jev/trajectory/` — rollout planner built on **real emulator savestates**
      (`dump_state` / `load_state`) rather than a hand-written physics approximation. This replaces
      the original reduced-physics plan because the savestate route is exact and measured to be
      affordable. The planner takes an explicit **frame budget** and shortens its horizon to fit
      it instead of assuming a fixed one.
- [ ] 3.2 Candidate control set: the discrete NES action combinations from `SIMPLE_MOVEMENT`
      (`NOOP`, `right`, `right+A`, `right+B`, `right+A+B`, `A`, `left`) with their rolled-out
      trajectories and terminal outcome (`lands_safe`, `hits_enemy`, `falls_in_gap`,
      `reaches_goal`, `no_progress`).
- [ ] 3.3 Per-trajectory risk cost and the trajectory table that becomes the `choice` criteria
      descriptions handed to Jev.
- [ ] 3.4 Tests: synthetic scenes with a known correct answer (a pit that must be jumped, a
      Goomba that must not be walked into).

#### WU3 contract — decided here, not open to the writer

These are the interfaces WU4 will consume, so they are fixed in the plan rather than invented
while coding. The writer chooses the *implementations*; these *shapes* are the review contract.

**`to_jev_state()` must produce exactly this JSON-serializable projection** (`json.dumps` must
accept it, and this is the same shape that Jev already answered coherently in a live 200 call):

```
{"player":        {"x", "y", "on_ground", "facing"},
 "nearest_enemy": {"distance_px", "approaching"},
 "terrain_ahead": {"gap_columns", "gap_distance_px", "wall_height"},
 "progress":      {"world", "x_max", "lives", "time_left"},
 "candidates":    {"<action name>": {"dx", "outcome", "risk", "final_x"}}}
```

`MarioState` carries **relative** features only: distance to the nearest active enemy ahead (from
the RAM `0x03AE` block, never `0x0087`), gap width and gap distance ahead from the frame,
`frames_to_jump_window`, vertical clearance, progress, lives and time left. `frames_to_jump_window`
is the sharpest of these and is defined as: frames until Mario's front edge reaches the first hole
column at the currently measured horizontal speed (computed from the `x_pos` delta between two
successive ticks), `None` when there is no hole ahead or the speed is zero. It gets its own test.

**Terminal outcome** is one of exactly five values from the plan, resolved in this precedence:
`reaches_goal` > `falls_in_gap` > `hits_enemy` > `no_progress` > `lands_safe`. Death splits on
whether Mario's x was over a hole column at the moment of death. `no_progress` means survived the
horizon without advancing. The exact `info` keys for death and for reaching the goal must be read
from the installed package and pinned in a test — assumed key names are how this project got the
`x_pos` mapping wrong.

**Risk cost** is a float in [0, 1], higher is worse, and the exact formula is the writer's — but
it must be a named pure function whose tests assert these **orderings**: any hazard outcome
(`falls_in_gap`, `hits_enemy`) > `no_progress` > the same surviving trajectory ending closer to an
enemy, and `reaches_goal` is the minimum. A risk number that cannot be checked against an ordering
is not reviewable.

**Planner invariants**, each one a test:

1. It MUST break the rollout on `StepResult.terminated` / `.truncated`. A naive
   `for _ in range(H)` loop raises `ValueError: cannot step in a done environment!` the first time
   a candidate dies — measured.
2. It MUST `load_state(snapshot)` before every candidate, and MUST NOT cache `info` across that
   call (the adapter refreshes `info`, the caller must not keep an old copy). A candidate that
   terminated must not prevent the next one from running: `load_state` clears the done flag —
   measured.
3. It MUST leave the emulator exactly where it started, and MUST NOT advance `tick` (the pinned
   invariant is `tick == start + 10` after a 10-step rollout).
4. Horizon selection MUST come from measured cost inside the given `rollout_budget_ms`, clamped to
   `[rollout_min_frames, rollout_max_frames]`, and the chosen horizon and the measured elapsed ms
   MUST be reported in the returned table so WU5's journal can record them.
5. Every measured test MUST run with `real_time=False`; `real_time=True` is the `MarioEmulator`
   default and paces to 60 fps, which would silently make every timing assertion meaningless.

### WU4 — Jev decision layer, PlayJev advisor, execution policy
- [ ] 4.1 `src/mario_jev/jev/` — real HTTP client for `POST /v1/systemone`: one request carrying
      the tick's parallel questions (`best_action` choice, `risk` score, `lethal_ahead` noul,
      `gap_severity` score), typed response validation against the documented shape,
      exponential backoff on 429/529, and hard failure on 401/422.
- [ ] 4.2 Deterministic fallback policy used when the key is absent, the call fails, or
      confidence is below threshold — so the loop always has a safe action.
- [ ] 4.3 `src/mario_jev/advisor/` — optional PlayJev-0.8B advisor, lazily loaded, that
      contributes its move distribution as a state feature and **degrades to absent** when torch
      or the weights are unavailable. Never on the critical path.
- [ ] 4.4 `src/mario_jev/policy/` — cadence (K frames), confidence threshold, safety guardrails,
      degradation recording, and the execution loop.
- [ ] 4.5 Tests: mocked Jev transport (deterministic), live test gated on `JEV_API_KEY`,
      policy tests for threshold/fallback/cadence behaviour, advisor test that skips honestly.

### WU5 — Evidence journal, visualization, entrypoints
- [ ] 5.1 `src/mario_jev/journal/` — JSONL trace with, per decision tick: frame index, perceived
      state, the trajectory table, the Jev question payload, the raw Jev answer with
      probabilities/confidence, the executed action, and the measured latency.
- [ ] 5.2 Per-tick artifacts: annotated frame PNG (detection overlay), trajectory overlay plot.
- [ ] 5.3 `src/mario_jev/dashboard/` — live view showing the frame with detections, the candidate
      trajectories, the Jev decision-score table, and the confidence/latency evolution over time.
- [ ] 5.4 `src/mario_jev/cli.py` — `run`, `replay`, `dashboard`, `doctor` (environment and
      dependency readiness report).
- [ ] 5.5 Tests for journal schema, replay determinism, and the doctor report.

### WU6 — Live evaluation (gated on the user's ROM and key)
- [ ] 6.1 Run world 1-1 live; record progress (`x` reached), deaths, `flag_get`, measured
      decisions/second, Jev latency distribution, fallback rate, and vision-vs-RAM error rate.
- [ ] 6.2 Honest report: what worked, what degraded, what the vision layer got wrong, and where
      the cadence broke down.

## 5. Review workload forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~2800–3600 total (roughly half tests) across 6 work units, plus ~150 for WU2.5 |
| 400-line budget risk | **High if delivered as one change** |
| Chained PRs recommended | **Yes** — one PR per work unit (WU1, WU2, WU2.5, WU3…WU6) |
| Delivery strategy | ask-on-risk; the work-unit commits are the review slices |
| Decision needed before apply | No |

## 6. Allowed edit surfaces per work unit

- WU1: `pyproject.toml`, `.gitignore`, `README.md`, `docs/**`, `src/mario_jev/__init__.py`,
  `src/mario_jev/config.py`, `src/mario_jev/emulator/**`, `tests/conftest.py`,
  `tests/test_config.py`, `tests/test_emulator.py`
- WU2: `src/mario_jev/perception/**`, `tests/test_perception.py`
- WU2.5: `src/mario_jev/config.py`, `README.md`, `docs/STRATEGY.md`, `scripts/jev-env.sh`,
  `tests/test_credentials.py`, `tests/test_config.py`
- WU3: `src/mario_jev/state/**`, `src/mario_jev/trajectory/**`, `tests/test_state.py`,
  `tests/test_trajectory.py`
- WU4: `src/mario_jev/jev/**`, `src/mario_jev/advisor/**`, `src/mario_jev/policy/**`,
  `tests/test_jev_client.py`, `tests/test_policy.py`, `tests/test_advisor.py`
- WU5: `src/mario_jev/journal/**`, `src/mario_jev/dashboard/**`, `src/mario_jev/cli.py`,
  `tests/test_journal.py`, `tests/test_cli.py`
- WU6: `artifacts/**`, `docs/EVALUATION.md`

None of these is `.`, the repo root, or an absolute path.

## 7. Blockers and open items

- **ROM: resolved.** The ROM ships inside `gym-super-mario-bros`; see the D3 note in §1. Live
  emulation and the vision-vs-RAM error rate are therefore runnable without the user supplying a
  path. The copyright caveat in §1 stands and is the user's call.
- **`JEV_API_KEY` not exported, and the endpoint was wrong — both addressed in WU2.5.** The key
  exists in the macOS Keychain (service `jev-api-key`) and is **verified valid** against
  `https://api.typesafe.ai/v1/systemone`. `scripts/jev-env.sh` bridges Keychain -> environment
  rather than putting Keychain code in the library, so the package stays portable and
  `Settings.load()` keeps reading the environment only. The key must never be pasted into a
  prompt, a file, or a commit.
- PlayJev's 2.2 GB weight download is deferred until the core loop is green; the advisor is
  optional by design precisely so this cannot block the critical path.
