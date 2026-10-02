# mario-jev

A real-time Super Mario Bros. (NES) agent whose decisions come from **Jev**, the
System One typed-decision model from TypeSafe AI.

The agent perceives each frame, turns it into typed state, rolls candidate actions
out through the real emulator with savestates, and asks Jev one question set per
tick. Jev returns a chosen action with probabilities and confidence; application
code owns execution, the confidence gate, and the safety fallback.

Read **[`docs/STRATEGY.md`](docs/STRATEGY.md)** for the full strategy: the phase
contract, the Jev request/response contract, the real-time cadence and its
degradation ladder, the risk model, the evaluation protocol, and the honest limits
of every component.

## Prerequisites

- Python 3.14 (the pinned `nes-py==9.0.1` wheel is `cp314`-`macosx_11_0_arm64`)
- A Jev API key from TypeSafe AI. The product site is <https://typesafe.ai/>;
  the API reference is <https://api.typesafe.ai/docs>

The Super Mario Bros. ROM ships inside `gym-super-mario-bros`. This project does
not download, copy, or redistribute it; see the legal note at the end of the
strategy document.

## Install

```sh
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `JEV_API_KEY` | *(unset)* | Jev credential. **Never write it to a file or commit it.** |
| `JEV_ENDPOINT` | `https://api.typesafe.ai/v1/systemone` | Jev evaluation endpoint |
| `JEV_MODEL` | `jev-latest` | Jev model id |
| `MARIO_STAGE` | `1-1` | world-stage to play |
| `MARIO_DECISION_CADENCE` | `8` | emulator frames between two Jev consultations |
| `MARIO_CONFIDENCE_THRESHOLD` | `0.6` | below this, the deterministic fallback runs |
| `MARIO_ROLLOUT_BUDGET_MS` | `120` | wall-clock budget for the rollout sweep |
| `MARIO_ROLLOUT_COST_SOURCE` | `measured` | what decides the horizon: the machine's own timing, or the constants below |
| `MARIO_ROLLOUT_FIXED_BASE_MS` | *(unset)* | sweep's fixed base cost, required when the source is `fixed` |
| `MARIO_ROLLOUT_FIXED_PER_FRAME_MS` | *(unset)* | fixed per-frame rollout cost, required when the source is `fixed` |
| `MARIO_REAL_TIME` | `true` | pace the emulator at real time |
| `MARIO_TARGET_FPS` | `60` | pacing target |
| `MARIO_MAX_TICKS` | *(uncapped)* | stop after this many decision ticks |
| `MARIO_ADVISOR` | `false` | enable the optional PlayJev advisor |
| `MARIO_ARTIFACTS_DIR` | `artifacts` | where evidence is written |

This table covers the variables worth knowing first; `src/mario_jev/config.py` is
the complete list, and `mario_jev doctor` reports what this machine resolves.

## Credential

The key is a secret and is never written to a file or a commit. On macOS it lives
in the Keychain, and `scripts/jev-env.sh` is the bridge that hands it to a
command:

- Keychain entry: service `jev-api-key`, account `$USER`.
- The bridge exports the value as `JEV_API_KEY`.
- Usage: `scripts/jev-env.sh <command>`.

```sh
scripts/jev-env.sh .venv/bin/mario-jev doctor
```

Sourcing it (`source scripts/jev-env.sh`) exports `JEV_API_KEY` into the current
shell instead of running a command. If `JEV_API_KEY` is already set, the bridge
uses that value and never touches the Keychain, so CI and gateway-injected keys
keep working.

## Run

```sh
export JEV_API_KEY="sk_jev_..."
.venv/bin/mario-jev doctor    # readiness report for the environment
.venv/bin/mario-jev run       # play, recording evidence
.venv/bin/mario-jev replay    # rebuild the view from a recorded trace
.venv/bin/mario-jev dashboard # live view
.venv/bin/mario-jev live      # play in a window and watch the decision board
```

`mario_jev live` plays the stage in its own window at 60 fps and opens the browser at the live
decision board (`/live`), which shows what the agent is doing and why. Pass `--no-open` to print the
address instead of opening a window.

## Tests

```sh
.venv/bin/python -m pytest
```

Markers: `live_jev` (real network call), `playjev` (optional advisor stack),
`slow` (takes more than a couple of seconds).
