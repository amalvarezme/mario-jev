# Feature — a local viewer for a run, live (`mario_jev serve`)

Status: authorized by the user (chose the live viewer over a finished-runs-only one). One work-unit
commit per task, each needing the user's go-ahead.

## Why

The agent's evidence already exists as a journal plus per-tick PNGs, and `dashboard <journal> --out`
renders it to a file. Watching a run today means waiting for it to finish and then opening a file. A
small local server fixes that: `mario_jev serve artifacts/run.jsonl` serves the same report,
regenerated **per request**, so a browser refresh shows the run as it grows.

## The blocker, measured

A journal **without** its summary trailer is unreadable: `read_journal` raises
`JournalFormatError("missing summary trailer; the journal must end with one")`
(`src/mario_jev/journal/reader.py:120`). A run in progress has no trailer, so today it cannot be
rendered at all. Two pieces are therefore needed before a server can show anything live.

## Pinned design

### 1. A partial read that stays strict about lines

`read_journal(path, *, allow_partial: bool = False)`. In partial mode:

* the run **header is still required** — a journal with no header is not a run in progress, it is a
  broken file;
* **every line that is present must still be valid**. A malformed line, a second header, a tick after
  the trailer: all still raise exactly as before. The project's rule is that a bad line is never
  skipped, and this mode does not touch it;
* only the **absence of the trailer** is tolerated, and then `Journal.summary is None`.

`Journal.summary` becomes `Mapping[str, object] | None` and every reader must handle `None`. The
docstring states the rule and the distinction: a *missing* summary is a run in progress; a summary that
is present but lacks a key is the `not recorded` case, which is a different fact about a finished run.

### 2. "In progress" is its own label

`render_text` and `build_html` must say the run is **in progress** when `summary is None` — never
`not recorded`. `not recorded` is the honest reading for absent data in a *finished* run; printing it
over a live one would describe a completed run that lost part of itself. The summary block says so
plainly, in the same place it currently prints the counts.

### 3. The server

`src/mario_jev/serve.py` plus a CLI subcommand:

```sh
mario_jev serve <journal> [--port N] [--bind ADDR] [--artifacts-dir DIR] [--once]
```

* **Binds `127.0.0.1` by default.** This is the user's own evidence; putting it on the network is an
  explicit act, so a non-loopback `--bind` is allowed only when given explicitly and the startup line
  says which address it took.
* **`GET /` → the report, regenerated on every request** from `read_journal(path, allow_partial=True)`
  and `build_html(...)`, so refreshing is the whole "live" mechanism. No streaming, no polling
  machinery, no SSE.
* **A torn read is reported, never invented.** If the read raises `JournalFormatError` (a line caught
  mid-write is the expected case during a live run), the handler retries **once** after a short,
  documented delay; if it still fails it answers **503** whose body says the journal could not be read
  at this moment and may be mid-write, with the reader's own message beneath it. It never renders a
  partial report as if it were complete, and it never skips the bad line to make one.
* **Artifacts are served strictly inside the artifacts directory.** The request path is resolved and
  must stay inside it; `..`, an absolute path, or a symlink escaping the directory is a 404. No
  directory listing. `Content-Type` by extension (`png`, `html`, `txt`, `json`), `405` for the methods
  this handler dispatches — everything but `GET`/`HEAD`. Two deliberate exceptions, found by
  verification and now stated rather than implied: a verb with **no** handler never reaches this code
  and gets the standard library's own `501`, and a null byte in the path makes `Path.resolve` raise, so
  it is caught (`ValueError` as well as `OSError`) and answered 404 like every other path with no
  artifact behind it.
* **`--once`** serves exactly one request and exits, so the end-to-end path is testable without a
  lingering process.
* On startup it prints the URL and **one honest line**: this serves the journal's evidence, not the game
  itself — the emulator advances one game frame per tick, so a live run grows at roughly the tick rate,
  not at 60 fps.
* `KeyboardInterrupt` exits 0 with no traceback.

### 4. Non-goals

* No animated replay, no video, no window (the `pyglet` window was a separate option and is not this).
* No new dependencies: `http.server`, `socketserver` and `urllib.parse` from the standard library.
* No change to `policy/`, `jev/`, `emulator/`, `trajectory/`, or the loop.

## Tasks

| # | Task | Surfaces | Evidence |
|---|---|---|---|
| V1 | The partial read and the "in progress" rendering | `src/mario_jev/journal/reader.py`, `journal/__init__.py`, `dashboard/build.py`, `tests/test_journal.py`, `tests/test_cli.py` | a live-style fixture (header + ticks, no trailer) reads and renders as in progress in both readers; a malformed line still raises in partial mode |
| V2 | The server and the CLI subcommand | `src/mario_jev/serve.py` (new), `cli.py`, `tests/test_serve.py` (new) | one real request over an ephemeral port; path-escape attempts 404; a torn journal gives 503 and not a fabricated report; `--once` exits |
| V3 | Verification and the report | `docs/EVALUATION.md`, plan | an independent verification round over V1+V2, then a short section on how to watch a run |

## Risks

* `http.server` is single-threaded and blocking by design here: one browser tab is the target, and a
  second request waits. Stated rather than engineered around — a threaded server would add concurrency
  to a viewer for no benefit.
* The live page is a *snapshot* per refresh. Nothing in it pretends to push updates.
