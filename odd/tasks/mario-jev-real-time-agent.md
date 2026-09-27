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

### 1.1 Harness fallback (recorded, not silent)

Subagent delegation is **unavailable in this session**. Every launch path was tried: the
`gentle-ai-worker` writer, then a read-only `gentle-ai-explore` probe, then the explicit
`repository_root` route. All three failed with *"could not register launched worktree: Select an
existing worktree in the same Git clone as this session"*, and `session_worktree_register`
returned the same message. Cause: the Pi session was started when this directory was **not** a git
repository, so the session holds no clone identity, and creating one afterwards does not
retroactively give it one. The read-only probe failing identically proves this is not
writer-worktree isolation.

Consequence: the Mandatory Delegation Triggers still fire (4-file rule, multi-file write rule,
verification rule, long-session rule), but the delegated runtime does not exist here. The parent
therefore executes inline and records that as an explicit fallback, which is what the routing
ladder prescribes when the package role is unusable. Restarting Pi inside this now-git directory
should restore delegation for later work units.

---

## 2. Verified facts this plan rests on (not assumed)

1. **Jev does not accept images.** Official contract: `state` is `text`, a JSON `object`, or an
   `array` of texts — "Image, audio, and video inputs are not supported yet". The frame must be
   converted to structured state by our own code. This is why D1 needs a perception layer at all.
2. **Jev endpoint**: `POST https://thejevai.com/v1/systemone`, `Authorization: Bearer <key>`.
   Three question types, **all evaluated in parallel against one state**, so one HTTP request
   resolves every judgment for a tick:
   - `choice` (≤255 options) -> `choice`, `probabilities` (sum 1), `confidence`
   - `score` (2–10 ordered levels) -> probability-weighted `score`, `legend`, `probabilities`, `confidence`
   - `noul` -> `noul` (0..1 = P(yes)); optional `criteria.true` / `criteria.false`
   Errors: 401 invalid key, 422 validation, 429 rate limit, 529 overloaded. Retry with backoff on 429/529.
   Key must stay server-side, read from `JEV_API_KEY`.
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

   Consequence for the cadence budget: at 350 fps a 7-candidate x 30-frame rollout costs ~0.6 s,
   far above a 7.5 Hz tick. The planner must therefore take a **frame budget** and shorten its
   horizon to fit, rather than assuming a fixed horizon.

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

### WU2 — Perception and frame-by-frame state characterization
- [ ] 2.1 `src/mario_jev/perception/` — deterministic detection from the NES frame: NES palette
      segmentation for Mario, enemies, coins, power-ups; tile/geometry reading for ground,
      pipes, gaps, stairs, `?` blocks; bounding boxes + per-detection confidence.
- [ ] 2.2 Cross-check every detection against the §2.5 RAM ground truth; the disagreement rate is
      logged, because that number is the honest quality measure of the vision layer.
- [ ] 2.3 `src/mario_jev/state/` — typed `MarioState` with **relative** features (distance to
      nearest enemy, gap width ahead, frames-to-jump-window, vertical clearance, progress,
      lives, timer) and a canonical `to_jev_state()` JSON projection.
- [ ] 2.4 Tests on synthetic frames + recorded real frames once a ROM is available.

### WU3 — Possible trajectories (forward enumeration)
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
| Estimated changed lines | ~2800–3600 total (roughly half tests) across 6 work units |
| 400-line budget risk | **High if delivered as one change** |
| Chained PRs recommended | **Yes** — one PR per work unit (WU1…WU6) |
| Delivery strategy | ask-on-risk; the work-unit commits are the review slices |
| Decision needed before apply | No |

## 6. Allowed edit surfaces per work unit

- WU1: `pyproject.toml`, `.gitignore`, `README.md`, `docs/**`, `src/mario_jev/__init__.py`,
  `src/mario_jev/config.py`, `src/mario_jev/emulator/**`, `tests/conftest.py`,
  `tests/test_config.py`, `tests/test_emulator.py`
- WU2: `src/mario_jev/perception/**`, `src/mario_jev/state/**`, `tests/test_perception.py`,
  `tests/test_state.py`
- WU3: `src/mario_jev/trajectory/**`, `tests/test_trajectory.py`
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
- **`JEV_API_KEY` not exported.** Every live Jev decision is gated on it. The key must never be
  pasted into a prompt, a file, or a commit.
- PlayJev's 2.2 GB weight download is deferred until the core loop is green; the advisor is
  optional by design precisely so this cannot block the critical path.
