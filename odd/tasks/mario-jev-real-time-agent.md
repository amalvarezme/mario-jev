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

### WU3 — Typed state and possible trajectories — **DONE**
(state slice `b5ae236`; the trajectory slice is the commit that adds
`src/mario_jev/trajectory/`. Suite 100 -> 133.)
- [x] 3.0 `src/mario_jev/state/` — typed `MarioState` with **relative** features (distance to
      nearest enemy from RAM, gap width ahead from the frame, frames-to-jump-window, vertical
      clearance, progress, lives, timer) and a canonical `to_jev_state()` JSON projection.
- [x] 3.1 `src/mario_jev/trajectory/` — rollout planner built on **real emulator savestates**
      (`dump_state` / `load_state`) rather than a hand-written physics approximation. This replaces
      the original reduced-physics plan because the savestate route is exact and measured to be
      affordable. The planner takes an explicit **frame budget** and shortens its horizon to fit
      it instead of assuming a fixed one.
- [x] 3.2 Candidate control set: the discrete NES action combinations from `SIMPLE_MOVEMENT`
      (`NOOP`, `right`, `right+A`, `right+B`, `right+A+B`, `A`, `left`) with their rolled-out
      trajectories and terminal outcome (`lands_safe`, `hits_enemy`, `falls_in_gap`,
      `reaches_goal`, `no_progress`).
- [x] 3.3 Per-trajectory risk cost and the trajectory table that becomes the `choice` criteria
      descriptions handed to Jev.
- [x] 3.4 Tests: synthetic scenes with a known correct answer (a pit that must be jumped, a
      Goomba that must not be walked into).

**As landed.** The 7-candidate sweep over real savestates breaks on
`terminated`/`truncated`, reloads the snapshot before every candidate, and re-reads `info` after
each load. The horizon comes from measured cost inside the budget, and `RolloutTable` reports
`horizon`, `sweep_ms`, `elapsed_ms`, `base_ms`, `per_frame_ms`, `calibration_frames`, `passes` and
`cost_source` (`calibrated` | `reused`) so WU5's journal can record the truth.

Two honest limits, both measured, neither hidden:

1. **The cost clock is unstable.** `per_frame_ms` ranged `0.64 .. 2.35 ms` inside a *single*
   process, so the same budget produced horizons from 9 to 27. The warm start removes the
   calibration pass from every call after the first — steady-state overhead fell from
   ~52-176 ms to ~0.1 ms — but it does not stabilise the clock.
2. **The cost model ratchets one way.** An overrun raises the remembered per-frame cost and an
   underrun never lowers it, so a transient stall permanently shrinks the horizon for the life of
   that planner instance: observed decaying `14 -> 13 -> 9` and staying there. Conservative, and a
   known cost. Relaxing it slowly is a candidate for a later unit.

**Requirement this creates for WU4:** the cadence loop MUST hold ONE `TrajectoryPlanner` instance
across ticks. A caller that builds a fresh planner per tick gets no warm start at all and pays a
calibration pass every time.

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
- [x] 4.1 `src/mario_jev/jev/` — real HTTP client for `POST /v1/systemone`: one request carrying
      the tick's parallel questions (`best_action` choice, `risk` score, `lethal_ahead` noul,
      `gap_severity` score), typed response validation against the documented shape,
      exponential backoff on 429/529, and hard failure on 401/422.
      **Landed `f537f9d`** after one corrective round: 192 tests, 0 xfails, and the live test
      passing against the real API with the tightened parser. Two independent verifications; the
      second could not falsify the fix. See "WU4 slice A — independent verification verdict" and
      "corrective round" below.
- [ ] 4.2 Deterministic fallback policy used when the key is absent, the call fails, or
      confidence is below threshold — so the loop always has a safe action.
- [x] 4.3 `src/mario_jev/advisor/` — optional PlayJev-0.8B advisor, lazily loaded, that
      contributes its move distribution as a state feature and **degrades to absent** when torch
      or the weights are unavailable. Never on the critical path.
      **Implemented and verified; four writer rounds, five independent verifications. Landed
      `4070381`.** See "WU4 slice B — verification record" below for what the rounds found and what
      stayed open.
- [ ] 4.4 `src/mario_jev/policy/` — cadence (K frames), confidence threshold, safety guardrails,
      degradation recording, and the execution loop.
- [ ] 4.5 Tests: mocked Jev transport (deterministic), live test gated on `JEV_API_KEY`,
      policy tests for threshold/fallback/cadence behaviour, advisor test that skips honestly.

#### WU4 contract — decided here, not open to the writer

WU4 lands in **three independently verified slices**, one commit each, because the three pieces
have different failure modes and a single 1500-line change would not be reviewable:
**A** `jev/` (the network boundary), **B** `advisor/` (an optional input that must never break
the loop), **C** `policy/` (the decision that ties them together).

**A — `src/mario_jev/jev/`**

Error taxonomy, all subclassing `JevError`. No exception message may contain the key.

| Exception | Trigger | Retried |
|---|---|---|
| `JevAuthError` | 401 or 403 | never |
| `JevRequestError` | 422, and other non-retryable 4xx | never |
| `JevProtocolError` | 2xx whose body violates the documented shape | never |
| `JevUnavailable` | retries exhausted on 429/529/5xx/timeout/transport error | yes, then raised |

The transport is a seam, so every branch is testable with no network:

```python
@dataclass(frozen=True, slots=True)
class HttpResult:
    status: int
    body: str
    elapsed_ms: float
    error: str | None = None   # transport failure: timeout, DNS, refused
```

`transport(request: JevRequest) -> HttpResult`, with `JevRequest(url, headers, json_body,
            timeout_s)`. The default transport uses `requests`; `sleeper` is injectable (default
`time.sleep`) so backoff tests never sleep for real.

Retry only when `HttpResult.error is not None` or status is in `{429, 500, 502, 503, 504, 529}`
(exposed as `RETRYABLE_STATUSES`). Backoff `min(BACKOFF_BASE_S * 2**attempt, BACKOFF_CAP_S)`, up to
`max_attempts`. **401/403/422 are never retried.**

Headers: `Authorization: Bearer <key>`, `Content-Type: application/json`, and a realistic
`User-Agent` from the module constant `USER_AGENT` — without one Cloudflare answers 403 with
`error code: 1010` before the request reaches Jev.

*Key-containment invariant, tested:* the key string never appears in `repr()` of any request,
response or exception, nor in any `.as_dict()`. `Authorization` renders as `"Bearer <redacted>"`.

**As landed, the invariant was made structural rather than conventional.** Verification found one
path the contract did not cover: `dataclasses.asdict(request)` copies a field value verbatim, so a
plain-string `Authorization` header would have been written straight into any journal that used
`asdict` instead of the sanctioned `as_dict()`. The credential is therefore wrapped in a
`RedactedHeader` value, and `JevRequest.__post_init__` **refuses** a plain string. Consequences:
`repr`, `str`, f-strings, `%s` and `%.r` all render `Bearer <redacted>`; concatenation and `join`
reject it, because it is not a `str`; `asdict` copies the wrapper, so `json.dumps` of that copy
raises `TypeError` — loud, not silent; and `reveal()` is the single accessor, called only by
`default_transport`. Equality against the raw string still holds, so a test can assert that the
outgoing header carries the expected value without that value becoming printable. The wrapper is
exported from `mario_jev.jev` so the policy and the journal can name it.

#### WU4 slice A — independent verification verdict (2026-09-28)

The first verifier run **failed with a runtime error, not a verdict**, and the parent session died
before reading that failure. The verification was therefore re-run from scratch against the same
slice, and it returned a real verdict. Recorded here because the plan would otherwise read as if
slice A had already been verified when it had not.

**Reproduced independently by the verifier**: 171 passed / 1 skipped; **0 xfails and no skips other
than the one sanctioned `live_jev`**; the live test is a **genuine 0.63 s network round trip**
against `api.typesafe.ai` and asserts real properties of the response, not merely that no exception
was raised. All of claims A1–A5 (one request, four ids and types, criteria order, `ValueError` on
no candidates, all three headers, instructions built from the state), B6–B9 (exact backoff
`[0.5, 1.0, 2.0, 4.0]`, `max_attempts` calls, `max_attempts - 1` sleeps, 401/403/422 never retried,
`None` key with zero transport calls), C11–C14 (`1.3` survives as a float through the full client
path, `NoulAnswer` has no confidence field, all eight protocol rejections, booleans rejected as
numbers) held under an independent probe of 54 checks, as did 69 credential-containment views.

**Three findings, all confirmed, all now being fixed**:

| # | Severity | Finding |
|---|---|---|
| F1 | important | `pickle.dumps(request)` emits the raw key: `__slots__` state is pickled verbatim. The docstring's "must never be rendered or serialised" was false for `pickle`. |
| F2 | important | `JevClient` kept the plain key in `__dict__`, so `repr(vars(client))` and `pickle.dumps(client)` leaked it. |
| F3 | **real defect** | A response missing `probabilities` (and `legend`) parsed successfully into `{}` with no signal. Evidence-integrity class, not safety class: `confidence` *is* required, so the gate cannot drift — but the journal contract promises the distribution, and `_float_map`/`_str_map` silently masked its absence. |

F4 (minor): `RETRYABLE_STATUSES` was not pinned — the verifier removed `500/502/504` in memory and
the whole test file still passed, because the test iterated the constant it was supposed to check.
F5 (nit): `errors.py` said `JevUnavailable` means "retries exhausted", but a non-retryable 5xx such
as 501 raises it after one call and zero sleeps.

**Assessed and rejected as a defect**: a non-retryable 5xx mapping to `JevUnavailable` is the
documented policy (degrade with a recorded cause naming `HTTP 501`), not a mislabelling; odd statuses
418/408 → `JevRequestError` and 301/302 → `JevProtocolError` are both defensible.

**Left unverified, stated as such**: the fixture's "kept verbatim" provenance (cannot be diffed
offline), whether real `requests` exceptions can embed header values (structurally they do not), the
Cloudflare-1010 rationale for the `User-Agent` (indirect support only: the live call succeeded), and
the policy-side degrade-vs-abort behaviour (WU4 slice C is not written yet — only the contract text
exists).

#### WU4 slice A — corrective round and second verdict (2026-09-28)

Because `assess` returns `risk: unassessable` (the files are untracked, so no native risk can be
produced) the harness treats the candidate as high risk and requires a separate independent
verifier. Both the writer round and its verification are recorded here.

**The fix** (delegated writer, exactly the five allowed files, nothing else): `RedactedHeader`
gains `__reduce__` refusing pickling and `__copy__`/`__deepcopy__` keeping copying safe and
redacted; `JevClient` stores the credential only inside `RedactedHeader`; a missing **or empty**
`probabilities`/`legend` is now a `JevProtocolError` through `_required_map`; `RETRYABLE_STATUSES`
is pinned to its literal value with literal `500/502/504` coverage; the `JevUnavailable` docstring
now matches behaviour (a non-retryable 5xx raises after one call and zero sleeps).

**Parent verification** (own probe, not the writer's count): 12/12 checks, including a
*every-substring-≥5-characters* scan of the sentinel. **192 passed, 1 skipped**, 0 xfails.

**Second independent verdict: clean — the fix could not be falsified.** Pickling is closed across
protocols 0–5, `cloudpickle`, and a `shelve`/dbm write whose 24 576 on-disk bytes contained zero
fragments; every rendered view, 13 failure-path tracebacks, and the mocked-wire header are clean;
the required-distribution rule rejects all ten missing/empty cases and **accepts** extra keys, a
`legend` with 2 or 6 levels against 4 criteria, fractional `score=1.3`, and a confidence-free
`noul` — i.e. no key-set or rubric-count validation was introduced, as ordered. The `F4` pin was
proven by **kill runs**: removing each of `429/500/502/504/529` fails the suite on both the
behavioural and the literal-pin axis, so the original "removal stays green" failure mode is dead.
The decisive `F3` check passed against the real API: **the tightened parser accepts the live
response** (0.47 s round trip).

**Residual design boundary, recorded openly (all equivalent in privilege to calling `reveal()`):**
`client._authorization._value` holds the raw value one level below `__dict__` — unavoidable for any
wrapper, and reachable only by an in-process actor that could equally call `reveal()`; registering a
`copyreg` reducer bypasses `__reduce__` (same privilege; unregistering restores the refusal); and
`header.__getstate__()` exposes slot state when called directly (not on any pickle path). In the
tests, the fragment-scanning helper is used by the two pickle tests; the parent's own probe scanned
fragments across every containment view, which is why the gap is recorded rather than fixed.

Questions: `build_questions(state)` returns exactly four ids — `best_action` (choice), `risk`
(score), `lethal_ahead` (noul), `gap_severity` (score) — each `{type, instructions, criteria}`.
`best_action.criteria` keys are **exactly** the state's candidate action names in planner order, so
a returned choice is always executable; no candidates means `ValueError`.

Responses are typed, never dicts passed around: `Usage(input_tokens, output_tokens)`,
`ChoiceAnswer(type, choice, confidence, probabilities)`, `ScoreAnswer(type, score, confidence,
legend, probabilities)`, `NoulAnswer(type, noul)`, `JevResponse(model, answers, usage,
latency_ms)`. **`NoulAnswer` has no confidence**, which is why `lethal_ahead` can never share the
choice/score gate. Validation rejects a missing id, an answer whose `type` differs from what was
asked, a `choice` outside the criteria sent, a non-numeric `noul`, and a `score` that is not a
number — **`score` may legitimately be fractional** (0.48 and 1.30 were both observed) and must not
be coerced to `int`. Every violation raises `JevProtocolError`.

```python
JevClient(*, endpoint, model, api_key, timeout_s, max_attempts, transport=None, sleeper=None,
          user_agent=USER_AGENT)
JevClient.from_settings(settings, *, transport=None, sleeper=None)
.ask(questions, state) -> JevResponse
.decide(state: MarioState) -> JevResponse     # build_questions + ask, one HTTP request
.redacted -> dict                             # safe diagnostics, no key material
.close() and context-manager support
```

**B — `src/mario_jev/advisor/`**

```python
AdvisorMove(name, probability)
AdvisorAdvice(model_id, backend, moves, top, latency_ms)
AdvisorAbsence(model_id, reason)   # "disabled"|"torch_missing"|"transformers_missing"|
                                   # "weights_missing"|"load_failed"|"no_matching_moves"
LoadedAdvisor(backend, predict)    # what a successful load yields
AdvisorLoadError(reason)           # raised by the default loader; reason from the same set
PlayJevAdvisor.from_settings(settings, *, loader=None)
.available -> bool
.absence -> AdvisorAbsence | None
.advise(frame, action_names) -> AdvisorAdvice | None
```

Availability is decided **without importing torch** (`importlib.util.find_spec`), so asking the
question cannot cost seconds or fail. `advise()` **never raises**: any failure degrades to a
recorded `AdvisorAbsence` and returns `None`. The policy touches it only when
`settings.advisor_enabled`, so it is never on the critical path. The `loader` seam lets both the
present and the absent path be tested **with no weights and no 2.2 GB download**, and the absent
path is additionally tested for real because torch genuinely is not installed here.

**Pinned here rather than left to the writer** — the original contract omitted three decisions, and
each one would otherwise have been invented silently:

1. **The load seam.** `loader: Callable[[str], LoadedAdvisor]`, default is the real one. A successful
   load yields `LoadedAdvisor(backend: str, predict: Callable[[object], Mapping[str, float]])`, where
   `predict` returns probabilities over the *model's own* move vocabulary — it is not given the
   action names, so alignment is testable on its own. The default loader raises
   `AdvisorLoadError(reason)` with a reason drawn from the closed set, which is what makes the
   absent path structural instead of conventional.
2. **Alignment and renormalisation.** `advise(frame, action_names)` reports probabilities **over
   `action_names` only**: moves whose name is not offered are dropped, and the surviving
   probabilities are renormalised to sum to 1. `top` is the single highest-probability
   `AdvisorMove`, ties broken by name so the result is deterministic. `moves` is every surviving
   move in descending probability order.
3. **A degenerate distribution has an honest name.** If the model offers no move that matches any
   offered name, `advise` returns `None` and records `"no_matching_moves"` — a sixth reason, added
   because mapping that case onto `"load_failed"` would be a lie in the journal. The same applies
   to a non-finite or non-positive probability mass: that is a backend failure, recorded as
   `"load_failed"`.

`AdvisorAbsence` is `None` while the advisor is usable, and records the first failure that made it
unusable. The advisor contributes its distribution as a **state feature**; it is never the
authority, and the policy must be able to run with it absent.

#### WU4 slice B — verification record (2026-09-28)

Four writer rounds and five independent verifications. Recorded because the *distribution* of what
was found is the useful part, and because the earlier "this is the final round" claims in this
session were wrong twice.

**What the verifications found, by kind.** Five **behaviour** findings: the frame gate coerced
dimensions with `int(size)` so `shape=("240","256","3")` passed as a real frame; `float(True)`
returned a 1.0 "probability"; `AdvisorAbsence` did not enforce its own closed reason set;
`AdvisorLoadError.reason` was writable after construction; and two **guard-asymmetry escapes** in
`from_settings`/`advise` — the first being the blocking one. Eight **testimony** findings: a
docstring claiming every asset disagreement "is absorbed and never reaches the loop" (false — only
disagreements that *raise* are absorbed, and a runnable-but-wrong reading produces valid-looking
advice), the same present-tense claim about a consumer left in the sibling file after being fixed in
the other, a bare "never raises" that `BaseException` falsifies, a design intent written as wired
behaviour, a test name promising a mechanism its body never observed, a parametrised case that
could not discriminate its own fix, a duplicate test, and a title phrase.

**The blocking finding, and why it matters more than its size.** `_reported_reason` guarded the
*read* of the reason but returned the **caller's object**, which was then re-tested outside the
guard — a `str` subclass passing membership once and raising on the second test escaped both
`from_settings` and `advise`, because a raise in a handler *body* cannot be caught by a sibling
`except`. **The parent's own verification of that round had passed while the defect was present.**
The fix is an invariant rather than a guard: nothing leaving the reason helpers may be the caller's
object — it is always one of the module's own plain-`str` constants, enforced once in
`_canonical_reason` and verified **by identity** (`is REASON_*`), not by equality with a literal.

**Two lessons that changed how this session works.** (1) *Behaviour findings are caught by tests;
testimony findings cannot be* — no test reads a docstring, so all eight would have shipped as
confident lies, and the only instrument that catches them is an independent reader. (2) *A kill-test
is worthless unless the mutation faithfully reproduces the pre-fix code path.* The parent produced
three non-faithful or non-discriminating mutations in this slice — the membership left inside the
`try` so its own `except` absorbed the second collision; a hostile object the gate rejected before
the route under test; a `return` searched at 8 spaces of indentation when it sits at 12, so the
mutation silently did not apply and the "kill" passed green. All three looked like verdicts. The
final verifier was therefore instructed to write its own mutation **and assert it had applied**
before running anything, which it did.

**Accepted residuals, recorded rather than fixed** (all classified by the verifier as boundary or
nit, none affecting the loop):

* **Boundary — private state.** `_absence`, `_reason` and `_canonical_reason`'s inputs are reachable
  by in-process code, the same rule already accepted for slice A's `_authorization._value`. A
  directly-constructed `AdvisorAbsence(model_id, hostile_value)` is the caller's own input, like
  `AdvisorLoadError(...)` refusing at construction. No public or rendered path reaches either.
* **Boundary — `settings` reads.** `from_settings` reads `settings.advisor_enabled`/`advisor_model_id`
  outside every `try`, so a hostile `Settings` escapes. Caller-owned configuration, outside the load
  seam that the fail-closed claim covers.
* **Nit — the suite asserts reason equality and `type is str`, not `is`-identity with the module
  constants.** Identity was proven by a probe in the verifying round, not pinned by a test.
* **Nit — of the two seam-guard tests, only one is mutation-detecting**; the `BaseException`
  propagation test passes under the mutation too, which is inherent to what it asserts.
* **Nit — `isinstance(loaded, LoadedAdvisor)` implicitly reads `loaded.__class__` and can raise.**
  Correctly covered by the guard, but the read is implicit rather than documented.
* **Unverified, and disclosed in the code itself** — the real Hugging Face load and inference path
  (`default_loader` → `_load_snapshot` → `_make_predict` → `_move_vocabulary`/`_label_token_ids`).
  The 2.2 GB weights are deliberately absent, no test fakes it, and the module says so. The
  load-time failure path *is* real: a missing local snapshot raises `weights_missing` **before any
  import**, verified on this machine where `torch` is genuinely absent.

**C — `src/mario_jev/policy/`**

**WU3's projection is frozen.** `to_jev_state()` keeps exactly its five keys and gains nothing. The
**policy**, not the state layer, appends the optional advisor block:
`{**state.to_jev_state(), "advisor": {...}}`.

**Three things the original contract left ambiguous and that are now pinned**, because each would
otherwise have been invented while coding:

1. **`select_action`'s real signature.** The contract's `select_action(table, decision, settings)`
   did not say what `decision` was or how the between-cadence case ("the last decided action")
   reaches a *pure* function. Pinned:
   `select_action(table, jev, settings, *, cadence_due=True, last_action="NOOP", degraded_reason=None)`
   where `jev` is a `JevResponse` or `None`. Pure: no emulator, no network, no clock, every branch
   testable without one.
2. **What actually goes to Jev when the advisor is on.** `JevClient.ask(questions, state)` takes a
   `MarioState` and projects it itself, and that client is a committed, separately verified slice.
   Pinned: the payload sent is exactly `state.to_jev_state()` with its frozen five keys, and the
   advisor block is an input to the **recorded** tick state (journal and dashboard), which is what
   "a state feature" means here. Reopening the verified client to accept a state override is out of
   scope for this unit, and the module docstring must say so rather than leave it looking like an
   oversight.
3. **What counts as "a decision".** `RunSummary.decisions` counts ticks where `jev_called` is True,
   and `decision_rate_hz = decisions / elapsed_s`. Counting cadence ticks instead would overstate
   the rate on a run that never consulted Jev.

**`frame_index` is the loop's own counter, and the plan had this wrong.** The first version pinned
`frame_index = emulator.tick`, which is not a game-frame count: `TrajectoryPlanner.plan(emulator)`
steps the same emulator object while enumerating candidates, so `emulator.tick` jumps by the whole
rollout cost on every tick. Measured on this machine at `real_time=False`: `30 -> 217`,
`217 -> 392`, `392 -> 567`, `567 -> 910` (`steps_taken` 187, 175, 175, 343). With `cadence_frames=8`
the cadence therefore fired on an arbitrary subset of ticks — 1 of those 4, by accident — so the
configured "every 8 game frames, 7.5 Hz" was not the cadence that ran. The loop now owns a counter
that advances by exactly one per executed game frame, and the raw `emulator.tick` is recorded
separately as `emulator_tick` so the planner's inflation stays **visible** in the journal instead of
silently corrupting the cadence.

**The journal records the questions the policy built, not a reconstructed request.** `TickRecord`
carries `jev_questions` — the four questions exactly as `build_questions(state)` produced them — and
not a fabricated `{url, headers, json_body, timeout_s}` mirror. The committed client exposes no
request handle, so such a mirror could only drift from what is actually sent, and a redacted
`Authorization` placeholder in it would invite a reader to believe a header was captured. The state
that accompanied the questions is already on the record.

```python
OVERSHOOT_FACTOR = 3.0
DecisionSource(StrEnum): "jev" | "hold" | "guardrail_lethal" | "fallback_no_key" |
                         "fallback_degraded" | "fallback_below_confidence" |
                         "fallback_no_candidates"
TickDecision(action, source, confidence, risk, reason, jev_called)
DecisionCadence(cadence_frames)                 # .due(frame_index) -> frame_index % K == 0
select_action(table, jev, settings, *, cadence_due=True, last_action="NOOP", degraded_reason=None)
TickRecord(tick, frame_index, emulator_tick, state, rollout, jev_questions, jev_response,
           decision, advisor, latencies_ms, degradations)
RunSummary(ticks, frames, decisions, sources, degradations, elapsed_s, decision_rate_hz)
ExecutionLoop(settings, *, emulator=None, planner=None, client=None, builder=None, advisor=None)
  .step() -> TickRecord
  .run(*, max_ticks=None, observer=None) -> RunSummary
```

**A name the client accepted but the table does not know raises `ValueError`** — it cannot happen
(the client rejects a choice outside the criteria, which are the candidate names in order), so it is
a bug and must be loud rather than degraded into a silent fallback.

Selection is deterministic and exhaustive, evaluated in this order:

1. no candidates → `NOOP`, `fallback_no_candidates`.
2. not a cadence tick → the last decided action, `hold`.
3. no credential (`settings.jev_available` false) → `table.best`, `fallback_no_key`.
4. `JevUnavailable` — timeouts, 5xx, and 429/529 once retries are exhausted — →
   `table.best`, `fallback_degraded`, recording the cause.
5. `JevAuthError`, `JevRequestError` or `JevProtocolError` → **propagate and abort
   the run.** These are not transient: a bad key, a malformed payload or a drifted
   contract must not be masked by an endless fallback that would make the run look
   Jev-driven while it never once consulted Jev.
6. confidence below `settings.confidence_threshold` → `table.best`,
   `fallback_below_confidence`.
7. the chosen candidate `died` and `best` did not → `best`, `guardrail_lethal`.
8. otherwise the chosen candidate, `jev`.

`fallback` is always `table.best` — lowest risk, ties broken by furthest `dx` — which WU3 already
fixes, so the fallback needs no network and no new policy. It is the **current** tick's measured
answer, never a memorised action: the rollouts are local and run every frame, so a fresher and
strictly better answer than "repeat last time" always exists. `hold` therefore means only "no new
decision was requested on this frame", which is exactly the between-cadence case.

Timing is **recorded, never repaired**: the loop holds **one** `TrajectoryPlanner` instance across
ticks (the WU3 requirement) and records `timing_overrun` when
`table.elapsed_ms > settings.rollout_budget_ms * OVERSHOOT_FACTOR` (3.0). WU4 does not attempt to
fix the overrun; the limit stays visible in the journal.

**Test obligations (4.5) — 0 skips in the default suite:** every branch above with a fake transport
and injected sleeper; the key-containment invariant; retry counts and the 401/403/422 no-retry rule;
each protocol violation; advisor absence for real plus the injected-loader present path; and one
`live_jev`-marked test against the real API, which the **parent runs during verification** so the
evidence is real rather than deferred.

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
