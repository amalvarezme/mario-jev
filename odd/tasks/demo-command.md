# Feature — one command that plays world 1-1, writes the evidence and serves the page

Status: authorized by the user. One work-unit commit, needing the user's go-ahead.

## Why

Watching a run takes three steps today: start a run, start `serve` on the journal it is writing, open the
URL. `mario_jev demo` does all three: it plays stage 1-1 automatically, writes the journal (with the
per-tick frame and trajectory PNGs, which is the "simulation" the page shows), and serves the report so
a refresh shows the run growing.

## Pinned design

### 1. The run becomes a reusable function

`_cmd_run` currently does settings → writer → emulator/planner/client/advisor → `loop.run(observer)` →
`write_summary` → `render_text`. Extract the middle into

```python
def _play(settings, journal_path, *, artifacts_dir, max_ticks=None, observer=None) -> RunSummary
```

so `run` and `demo` use the same path and `run`'s behaviour is **unchanged on every path that its tests
cover** (its existing tests are the contract). One path does change, and it is stated here rather than
left to be discovered: an `OSError` raised by `writer.close()` — the writer's `finally` — used to escape
`main` as a traceback, because `close()` sat outside `_cmd_run`'s handler while `write_summary` sat
inside it. With the extraction, `close()` runs in `_play`'s `finally` and its `OSError` reaches
`_cmd_run`'s `except (JevError, OSError)`, so the same failure is reported as `run failed: OSError: ...`
with exit code 1. The exit code is the same either way; the improvement is a diagnosis instead of a
traceback, and it is pinned by a test rather than asserted. `_play` writes the summary trailer itself, as
`_cmd_run` does today.

### 2. `serve` learns where it is listening

`serve(...)` gains `on_ready: Callable[[str], None] | None = None`, called with the bound URL once the
socket is bound and **before** it starts serving, and **inside** the `try` that owns
`finally: server_close()` so a callback that raises cannot leak the socket. Default `None` keeps every
current caller unchanged. It exists so the demo can open the browser at the real address even when
`--port 0` lets the OS choose one. `serve` prints its own two startup lines unconditionally and prints
nothing on the callback's behalf; the callback may print (the demo's does, when the browser does not
open), and a callback that raises is the caller's business, not a serve failure.

### 3. `mario_jev demo`

```sh
mario_jev demo [--stage 1-1] [--max-ticks 300] [--journal artifacts/demo.jsonl]
               [--artifacts-dir DIR] [--seed N] [--port 8000] [--bind 127.0.0.1] [--no-open]
```

* Defaults chosen for a demo: **stage 1-1**, 300 ticks, journal `artifacts/demo.jsonl`, loopback, port
  8000. Every one of them overridable; `--stage` and `--seed` go through the same overrides `run` uses.
* The run goes in a **daemon thread**; the server owns the main thread. Nothing in the loop needs the
  main thread (the emulator is headless and the Jev call already runs in the dispatcher's worker), so
  this is a composition, not a concurrency change.
* `on_ready` opens the browser with `webbrowser.open(url)`, behind a seam so a test can assert it and
  `--no-open` can skip it. If the open returns false, that is not an error: the URL is already printed
  and the demo says so rather than pretending a window appeared.
* **Ctrl-C stops serving and abandons the run**, and says exactly that: the journal keeps every tick
  written so far and has **no summary trailer**, so reading it later reports the run as *in progress*.
  That is the honest reading of a run nobody finished, and the demo must not imply it completed. Waiting
  for the run would be worse — it can take minutes.
* A run that **fails** (a rejected credential, a protocol error) prints the error on stderr and the
  server keeps serving: the journal still holds the ticks up to the failure, and the page shows them.
  It does not fabricate a summary and it does not claim the run is still going anywhere but through the
  same "no trailer" sentence the viewer already uses.

### 4. Non-goals

* No window, no animated replay, no video (the page embeds the frames; the viewer's limits are stated in
  the report).
* No new dependencies: `threading` and `webbrowser` are standard library.
* No change to `policy/`, `jev/`, `emulator/`, `trajectory/`, `journal/`, `dashboard/`.

## Tasks

| # | Task | Surfaces | Evidence |
|---|---|---|---|
| D1 | `_play` extraction + `on_ready` + the `demo` subcommand + tests | `src/mario_jev/cli.py`, `src/mario_jev/serve.py`, `tests/test_cli.py`, `tests/test_serve.py` | `run` unchanged on its covered paths — plus the `close()` path above, which changed deliberately and is pinned by a test; a demo with injected seams proves order (run started, URL opened once, `--no-open` skips it), the flags reach the run, and Ctrl-C/abort leave the journal trailer-less on purpose. The exit sentence is **derived, never guessed**: this run's own return after it wrote its trailer means finished; a trailer found while this run has *not* returned is reported as present with no author claimed, because it may be this run's (a `close()` that raised after the write) or an earlier run's at the same path; a readable trailer-less journal means *in progress*; and a missing or unreadable one has its own sentence rather than a claim about ticks it may not hold |
| D2 | A one-line entry in the report's "How to reproduce" | `docs/EVALUATION.md` | the command, and what the page will say for a run stopped by hand |

## Risks

* A daemon run thread abandoned at Ctrl-C: the process exits while the emulator object is still in use.
  That is a teardown, and the honest cost is stated on exit rather than hidden.
* Two threads sharing `Settings` and the journal path: the run writes, the server reads. That is exactly
  what `serve` already does with a separate process, and the reader's partial mode is what makes it
  safe.
