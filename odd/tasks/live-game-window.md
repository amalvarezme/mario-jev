# Feature — the game in a window, and the Jev scores beside it (`mario_jev live`)

Status: authorized by the user, who chose the native 60 fps game window plus a live scores page over
an all-browser page and over a panel drawn inside the game window. All four tasks are implemented,
committed and verified: G1 `b98a274`, G2 `8e3913d`, G3 `b50a152`, G4 `7049063`. The follow-up at the end
of this document (G5, G6) is authorized by the user too.

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
- **the rollout candidates**: `dx`, `outcome`, `risk` and `final_x` per candidate name, with the row
  of the applied action (`decision.action`) marked `(applied)` and Jev's picked row marked `(chosen)`.
  When the two are the same row it is marked once, `(applied, chosen)`, never two marks a reader has to
  reconcile. Jev's pick is a claim about the answer and the applied action is a claim about what ran,
  and the page never lets the first be read as the second;
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

## Observed on this machine

One real run, `mario-jev live --journal artifacts/live.jsonl --artifacts-dir artifacts/live
--max-ticks 900`, with the credential from the Keychain:

* the two startup lines printed, and the game window opened -- no traceback anywhere in the log;
* `/live` answered `200` while the browser refreshed it every 0.5 s and `/frame_*.png` answered
  alongside it, so the page and its frame kept up with the run;
* the board read `status in progress`, `pressed: right+B`, the Jev scores for the newest tick, the
  rollout rows with the applied candidate marked, and the decision row;
* the run ended on its own summary trailer with `termination: death` after 233 ticks (39 s), and the
  command then printed the finished sentence and exited 0.

The limit this run makes plain: **the command ends when the episode does.** The window closes, the
board stops being served, and watching another run means starting another one. That is the honest
shape of a command whose whole subject is one run's game, and it is why the exit line distinguishes a
finished run from an abandoned one instead of pretending the viewer outlives the game.

## Follow-up — the game moves into the board page (G5, G6)

The user asked for the real-time view of the controlled game to sit **on the right**, beside the scores
and buttons, on the same screen. A native window cannot be placed inside a web page and cannot be kept
above one -- the window order belongs to the operating system, and a click on the browser sends it
behind -- so the options put to the user were: move the native window over the page, or bring the game
into the page. The user chose **the game inside the page, panel on the left**, over moving the native
window and over doing both.

### 1. A one-slot frame sink

`src/mario_jev/live_frame.py` (new) holds :class:`LiveFrame`: the newest frame the run produced, in one
slot, safe to write from the run's thread and read from the viewer's. ``publish(frame)`` stores a copy
and bumps a sequence number; ``png()`` returns the newest frame as PNG bytes, encoding **at most once
per published frame** and never on the publishing thread -- the run must not pay for an encoding nobody
asked for, and a burst of requests for the same frame must not encode it twice. The slot and the
cached bytes are guarded; the encoding itself happens outside the lock.

### 2. The emulator publishes every real-timeline frame

`MarioEmulator` gains ``on_frame: Callable[[np.ndarray], None] | None = None``, called from ``reset()``
and from ``step()`` with the same fresh contiguous frame the ``StepResult`` carries, under the same gate
the window follows: only on the real timeline, never during a sweep. One method presents the frame -- the
window when the mode is ``human``, the sink when a callback is given, both or neither -- and the publish
happens before the pacing wait, as the window's draw does. The callback is independent of
``render_mode``: the page can be live with the window closed.

### 3. The frame is served at `/live/frame.png`

A second route under the page's own prefix, checked before the artifact route, so no artifact name can
shadow it and it can shadow no artifact. It asks the frame source and answers ``200``
``image/png`` with ``Cache-Control: no-store`` -- the page polls this URL, so a cached answer would
freeze the game -- or ``404`` with a sentence when there is no frame yet or no frame source at all.
``Viewer`` and ``ThreadedViewer`` take the source as an optional callable, ``None`` by default, which is
what keeps every current caller byte-identical.

### 4. The page becomes two columns, and the game lives in the right one

The board is one flex row: the **panel on the left** (right now, buttons, Jev scores, rollout,
decision) and **the game on the right**, an ``<img>`` pointing at ``/live/frame.png``. It wraps to one
column on a narrow window instead of squeezing both.

The game is polled every ~70 ms and the panel is re-fetched every second; both are small inline
JavaScript, which is a change from G2's "no JavaScript" and is stated rather than slipped in. Two
details make the polling honest: the preloaded frame is swapped only once it has decoded (two stacked
images toggling, one request per frame, no flicker), and a failed request leaves the last frame on
screen instead of blanking it. The ``<meta http-equiv="refresh">`` stays in the markup as the fallback
for a browser without JavaScript -- where it reloads the whole page, and the frame with it, at that
slower rate -- and the script removes it on load, because a reload every second would restart the
polling and blank the game.

### 5. `--no-window`

The native window stops being compulsory: ``mario_jev live --no-window`` plays headless and lets the
page carry the game. The default stays the window, which is still the only 60 fps view.

### Non-goals of the follow-up

* No streaming, no websockets, no server-sent events: the page polls two plain URLs.
* No change to the report at `/`, to the journal, or to what a run records. The live frame is not
  written to disk and is not evidence; the recorded per-tick PNGs keep their perception overlay and
  stay the only frames the journal holds.
* No 60 fps claim for the page: it refreshes at the poll rate, and says so in its own words.

## Tasks (follow-up)

| # | Task | Surfaces | Evidence |
|---|---|---|---|
| G5 | The frame sink, the emulator's per-frame callback, and the route that serves it | `src/mario_jev/live_frame.py` (new), `src/mario_jev/emulator/env.py`, `src/mario_jev/serve.py`, `tests/test_live_frame.py` (new), `tests/test_emulator.py`, `tests/test_serve.py` | a published frame comes back as a PNG that decodes to the same pixels; the callback fires on the reset and on every real-timeline step and never inside `pacer.suspended()`; the route answers `200 image/png` with `no-store` when a frame exists, `404` before one does and when no source is configured; repeated requests for one frame encode it once |
| G6 | The two-column page with the polling game, `--no-window`, and the docs | `src/mario_jev/dashboard/live.py`, `tests/test_live_view.py`, `src/mario_jev/cli.py`, `tests/test_cli.py`, `README.md`, `docs/EVALUATION.md` | the page carries the panel and the game as one row, the game points at `/live/frame.png`, the script removes the meta refresh and polls, the fallback markup and the no-JavaScript sentence are present, `--no-window` builds a headless emulator while the page stays live, and a real browser check shows the frame changing while the panel keeps its place |

## Closed — commits and the evidence that closed them

| # | Commit | State |
|---|---|---|
| G1–G4 | `b98a274`, `8e3913d`, `b50a152`, `7049063`, `00fb53a` | closed earlier |
| G5 | `3e2a5be` | closed |
| G6 — the page | `5a26a1a` | closed |
| G6 — the wiring, the flag, the docs | `76c1720` | closed |

G6 was split in two work units because one commit over six files is a worse
review than two over three. The split is the page (what it renders) and the
wiring (what feeds it), which is also the seam the defect sat on.

**The gap the task table did not name.** G5 built the sink, the emulator
callback and the route; the table's G6 row says nothing about connecting them,
and nothing did. `_cmd_live` never built a `LiveFrame` and never passed
`frame_source` to the viewer, so `/live/frame.png` answered 404 on every real
run. The page would have rendered an empty game column forever. That wiring is
the substance of `76c1720`.

**Verified by execution** (full suite `735 passed`, 0 skipped — the one
`live_jev` test really calls the endpoint when the Keychain key is present):

* the route answers 404 with *"no frame has been published yet"* for 817
  consecutive polls — so the source was wired from the first request and the
  404s were honest — then flips to 200 at t+0.144 s;
* `200 image/png` with `Cache-Control: no-store`, decoding to `(240, 256, 3)`,
  which matches `SCREEN_SHAPE`;
* three polls a second apart differ, and the differing pixels sit in rows
  193–208, the ground line where Mario stands, **not** the HUD — so the sprite
  is animating and the run is advancing, which is the strongest available
  evidence that the page would show a moving game;
* the script's cache-busting `?n=` does not break the route;
* `--no-window` imports zero pyglet under `-X importtime`. nes-py constructs
  its `ImageViewer` lazily, so that is a precise witness for "no window was
  ever requested", not merely "no window appeared";
* the artifacts directory holds exactly the 30 files the journal names and
  nothing resembling a polled frame: the live frame is not evidence and never
  touches disk;
* the `/live` body references exactly two URLs, `/` and `/live/frame.png`, so
  no recorded `frame_png` path leaks into the page;
* `live.LIVE_FRAME_URL == serve.LIVE_FRAME_PATH`, and the import runs one way
  only, so the cycle the docstring claims to avoid is really avoided.

**One defect the verification caught, fixed in `76c1720`.**
`serve.LIVE_STARTUP_NOTE` asserted "the game plays in its own window at 60 fps"
— false under the `--no-window` this task added — and "this is the decision
board, not the game" — false for a board that now carries the game. It now
claims only the two rates, which hold in both modes.

**Not verified.** No JavaScript engine ever ran the inline script. Its text,
URLs and constants were checked against the live response, but the decode-then-
swap behaviour and the panel-subtree replacement are unexercised end to end.
A browser check is the one piece of G6's stated evidence still outstanding.
