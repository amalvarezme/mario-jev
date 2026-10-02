# Feature — the game in a window, and the Jev scores beside it (`mario_jev live`)

Status: authorized by the user, who chose the native 60 fps game window plus a live scores page over
an all-browser page and over a panel drawn inside the game window. One work-unit commit per task, each
needing the user's go-ahead.

## Why

`demo` plays the run and serves the journal's evidence. The page grows as the journal grows and says so
itself: *this is the run's evidence, not the game*. What it cannot show is the game. The loop steps a
headless emulator, and the journal keeps **one frame per decision tick**, so the page can never move at
60 fps no matter how often it refreshes.

The user asked to watch the game iterating under the agent's control with the Jev decision panel
alongside. `mario_jev live` is that: the real game in a window at 60 fps, and a page that refreshes
itself showing the buttons the agent pressed, the candidates with their probabilities, the chosen
action, and why the policy chose it.

## The three facts this design is built on (measured on this machine)

1. **nes-py refuses to render off the main thread.** `ImageViewer.__init__` raises
   `RuntimeError('rendering from python threads is not supported')` when the current thread is not the
   main one (`.venv/lib/python3.14/site-packages/nes_py/_image_viewer.py:24-28`). Measured both ways:
   120 frames from a daemon thread raise exactly that error; 90 frames on the main thread open the
   window and render in 0.50 s (`pyglet 1.5.21`, Python 3.14.6).
   `live` therefore **inverts `demo`'s arrangement**: the run owns the main thread and the viewer runs
   in a daemon thread. `demo` is not touched.
2. **Closing the game window by hand does not break the run.** Measured: after
   `viewer._window.close()`, `is_open` stays `True`, `render()` keeps working and `close()` is clean.
   No guard is added for it: the window can disappear and the run goes on until Ctrl-C or
   `--max-ticks`. Stated rather than engineered around.
3. **The rollout sweep drives the same emulator the gameplay does.** The planner receives the
   loop's own emulator and rolls candidates forward on it (`trajectory/planner.py:534-675`,
   `:772-836`), so a render attached to `step()` unconditionally would draw hundreds of simulated
   futures per decision tick at unbounded speed instead of the game. The sweep already marks
   itself: it runs inside `with emulator.pacer.suspended():` (`:562`), and that context is
   documented as *"its steps are simulated futures and must not consume the real timeline... the
   game is effectively paused while the agent thinks"* (`emulator/env.py:158-172`). That is the
   line the window follows.

## Pinned design

### 1. The emulator draws the real timeline, and only the real timeline

`MarioEmulator` already accepts `render_mode` (`src/mario_jev/emulator/env.py:281`) and hands it to the
inner env (`:292`). It keeps the value and calls the inner `render()` from `reset()` and from `step()`
**only** when the mode is `"human"` — the one mode nes-py opens a window for — **and the pacer is not
suspended**. `FramePacer` gains the read-only `is_suspended` its own `_suspend_depth` already knows
(`emulator/env.py:125`, `:158`), so the rule reads in one line and means what the pacer's docstring
already says: a suspended step is a simulated future, not the real timeline.

The loop still calls `emulator.step()` once per emulator frame and is not touched, and the render sits
inside the step, before the pacing wait — so the frame is on screen during the wait, which is what keeps
the window smooth. The alternative, an explicit `emulator.render()` call from `policy/loop.py`, was
considered and rejected for two reasons: it puts a display concern in the loop, and it moves the render
after the pacing wait, where the presented frame would lag the one being computed.

### 2. The render mode is a display concern, not evidence

`_build_emulator(settings, *, render_mode=None)` (`src/mario_jev/cli.py:86`) and
`_play(..., render_mode=None)` carry the mode from the command to the emulator. `run` and `demo` pass
nothing and are unchanged on every path their tests cover. The mode is deliberately **not** added to
`Settings` and **not** written to the journal header: the header is evidence about the run, and a
window changes nothing the run measured.

### 3. The live page (`GET /live`)

A new `build_live_html(journal, *, artifacts_dir=None)` in `src/mario_jev/dashboard/live.py`, answering
one question: what is the agent doing right now, and why? One page holding:

- the stage, the tick, the frame index, and whether the run is in progress, finished or stopped — the
  same words the report already uses, read off the trailer and never invented;
- **the newest tick's frame**, at full size, referenced through the run's own `frame_png` path with the
  existing artifact rule;
- **the buttons**: the action the tick applied, turned into its button tokens by the same action-name
  parser the action set uses, not by splitting the name in the template. A name outside the set says so
  instead of guessing;
- **the Jev scores**: every question the tick fired, with the choice answers' `probabilities`, the
  chosen one marked, and its `confidence`; score answers with their `score` and `legend`; the recorded
  note where the answer was `noul`. A tick that fired no call says so, and absent data reads
  `not recorded`;
- **the rollout candidates**: `dx`, `outcome`, `risk` and `final_x` per candidate name, with the chosen
  one marked;
- **the decision**: action, source, confidence, risk and `reason`.
It refreshes itself with a `<meta http-equiv="refresh">` and links to the full report at `/`. No
JavaScript and no streaming: every refresh is a snapshot, and the page says in its own words that this
is the decision tick rate, not 60 fps.

### 4. `serve` learns one route and a second way to start

`Viewer` gains an optional live builder and `ViewerHandler` answers `/live` with it when it is
configured, and `404` when it is not. `serve(..., live=False)` keeps every current caller and every
current response byte-identical (`src/mario_jev/serve.py`).

`live` needs the viewer **off** the main thread, so `serve.py` also gains a threaded entry point that
reuses the same construction, URL, startup lines and `on_ready` callback as `serve()`: bind, print,
then loop over `handle_request()` with a short server timeout while checking a stop event, and
`server_close()` on the way out. The blocking `serve()` path is untouched: its tests are the contract.

### 5. `mario_jev live`

```sh
mario_jev live [--stage 1-1] [--max-ticks 900] [--journal artifacts/live.jsonl]
               [--artifacts-dir DIR] [--seed N] [--port 8000] [--bind 127.0.0.1] [--no-open]
```

The same flags as `demo`, so nothing new has to be learned, with two deliberate differences:
`--max-ticks` defaults to **900** instead of 300, because a viewer means to sit and watch and 300 ticks
is about 45 seconds; and the journal defaults to `artifacts/live.jsonl`.

It plays with `render_mode="human"` on the main thread, serves `/live` from a daemon thread, and opens
the browser once at `/live` behind the existing seam; `--no-open` skips the browser and prints the
address instead. The game window is the point of the command and is not optional.

Ctrl-C stops the run and the viewer and prints **the same honest sentence** `demo` prints
(`_demo_exit_sentence`, `src/mario_jev/cli.py:722`): the journal keeps every tick so far and has no
trailer, so reading it reports the run as in progress. A run that fails prints the error on stderr and
the viewer keeps serving, exactly as `demo` does.

### 6. Non-goals

* No change to `policy/`, `jev/`, `trajectory/`, `journal/`, or the loop.
* No new dependency: `pyglet` arrives with `nes-py`; `threading` and `webbrowser` are standard library.
* No panel drawn by hand inside the game window: the scores live in the page, the game lives in the
  window.
* `demo` and `serve` keep their behaviour. `live` is a new command, not a flag that changes them.

## Tasks

| # | Task | Surfaces | Evidence |
|---|---|---|---|
| G1 | The emulator draws in `human` mode, and the render-mode seam | `src/mario_jev/emulator/env.py`, `src/mario_jev/cli.py`, `tests/test_emulator.py`, `tests/test_cli.py` | a fake inner env proves `render()` is called from `reset()` and from every real-timeline step in `human` mode, is **not** called for a step inside `pacer.suspended()`, and is never called in the default mode; `run` and `demo` still build a headless emulator, with their existing tests as the contract |
| G2 | The live page | `src/mario_jev/dashboard/live.py` (new), `src/mario_jev/dashboard/__init__.py`, `tests/test_live_view.py` (new) | for a fixture tick: the buttons come from the action name through the parser, the chosen candidate is marked in both the probabilities and the rollout table, a trailer-less journal reads as in progress, and a tick with no response says so instead of inventing scores |
| G3 | The `/live` route and the threaded viewer | `src/mario_jev/serve.py`, `tests/test_serve.py` | one real request to `/live` over an ephemeral port returns the live page; `live=False` answers 404 there and is byte-identical on `/`; the threaded entry point serves a request, stops on request, and closes its socket |
| G4 | The `live` command, its flags and the docs | `src/mario_jev/cli.py`, `tests/test_cli.py`, `README.md`, `docs/EVALUATION.md` | with injected seams: the run is built with `human` render mode, the viewer starts with the live route before the run, the browser opens once at `/live`, `--no-open` skips it, and Ctrl-C and a failed run produce `demo`'s own sentences. Then one real run on this machine, reported as what was seen and not as what should have happened |

## Risks

* The render rule leans on the sweep suspending the pacer. That coupling is the pacer's own documented
  meaning, is asserted by a test in this feature, and is the reason the loop is untouched; if a future
  rollout ever steps the emulator without suspending the clock, the window would show it, and that test
  is where the change would be caught.
* The run holds the main thread and the viewer is a daemon thread: if the viewer thread dies, the run
  goes on and refreshing stops. Its failure is printed when it happens rather than swallowed.
* `pyglet` is the one dependency that can fail on this machine's Python. It is measured working here,
  and a machine where it fails gets the emulator's own traceback instead of a silent headless run.
* Two windows must be placed side by side by the user. Nothing positions them, and the command says
  which two they are.
* The page's refresh rate is independent of the tick rate, so a fast run means refreshes that skip
  ticks. The page's own numbers make that visible instead of hiding it.
* The frames land in the configured artifacts directory, where an earlier run's frames may already
  live. `live` does not move or rename them; pass `--artifacts-dir` to keep runs apart.
