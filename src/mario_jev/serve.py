"""A local viewer for one run: the report and the live board, regenerated per request.

``mario_jev serve <journal>`` starts a **single-threaded** HTTP server whose job
is to answer ``GET /`` with ``build_html(read_journal(path,
allow_partial=True))`` and, when it is configured for it, ``GET /live`` with
``build_live_html(...)``. There is no streaming, no server-sent events, no polling
loop and no worker thread: a request reads the journal as it is at that moment,
builds one page, sends it, and the next request starts over. One request is
handled at a time, on purpose -- a second viewer tab waits for the first.

The two report routes are one read path: ``/live`` uses the same partial read,
the same single retry after the same delay, and the same ``503`` with the reader's
own words. It is a second page, deliberately not a second way to read a journal.
A viewer built without the live builder -- the default -- does not answer
``/live`` at all: that path is then the same ``404`` as every unknown path.

``mario_jev live`` needs the viewer off the main thread, because the game window
owns it. :class:`ThreadedViewer` is that: the same bind, URL, startup lines and
``on_ready`` promise as :func:`serve`, serving from a daemon thread that polls its
stop event and closes its socket on the way out. ``serve`` itself keeps blocking
the calling thread by design, and its tests are the contract for that path.

A third route, ``/live/frame.png``, answers the newest frame the run published,
when the viewer was built with a frame source. It is checked before the artifact
route, so a file that happens to sit at the same relative path cannot shadow the
live game; the source is optional and ``None`` by default, which is what keeps
every viewer built before this route existed answering exactly as it did. The
route never lets the source's failure reach the handler: a source that raises is
answered ``500`` with the failure named, because the alternative -- an exception
out of a request handler -- is a dropped connection with nothing on it.

A journal mid-write has a short window in which it is not readable, because a
line can be half-written when the reader reaches it. The viewer does not paper
over that: it retries the read once after a short documented delay and, if the
journal still cannot be read, answers ``503`` with a sentence saying the file may
be mid-write and the reader's own message beneath it -- the line and the reason.
It never renders a half-read journal as if it were complete and it never skips a
line it could not parse -- showing part of a run as the whole run is the one
thing a viewer built for evidence must not do.

Request paths other than ``/``, ``/live`` and ``/live/frame.png`` are resolved
against the run's artifacts directory and served only from inside it; anything
that is not inside it is answered ``404``, including a path the filesystem
refuses to look at -- a null
byte makes ``resolve`` raise rather than fail to match, and no request may make
this handler raise. What is *not* necessarily a 404 is a request the standard
library rejects before dispatching here at all: a verb with no handler, or a
malformed request line, gets the library's own ``501``/``400`` and never a file.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from socketserver import TCPServer
from typing import Callable, TextIO
from urllib.parse import unquote, urlsplit

from mario_jev.dashboard import build_html, build_live_html
from mario_jev.journal import Journal, JournalFormatError, read_journal

__all__ = [
    "DEFAULT_BIND",
    "DEFAULT_PORT",
    "LIVE_FRAME_PATH",
    "LIVE_PATH",
    "RETRY_DELAY_S",
    "THREADED_JOIN_TIMEOUT_S",
    "THREADED_POLL_S",
    "ThreadedViewer",
    "Viewer",
    "ViewerServer",
    "run",
    "serve",
]

#: Where the viewer listens unless another address is asked for, and why: the page
#: shows one machine's run to the person at that machine, and binding a wider
#: interface would publish the evidence to the network by accident.
DEFAULT_BIND = "127.0.0.1"
#: The port used when none is given.
DEFAULT_PORT = 8000
#: How long the viewer waits before its single retry of an unreadable journal.
#: Short enough not to be felt, long enough to cover the write of one line.
RETRY_DELAY_S = 0.05
#: The path that serves the live decision board rather than the report or an
#: artifact. It is a fixed module-level path, not a per-viewer one, because the
#: `live` command links to it and the route has to name the same thing everywhere.
LIVE_PATH = "/live"
#: The path that serves the newest frame the run published, for the board page's
#: game image. It sits *under* the board's own prefix on purpose: one page, one
#: prefix, and a path the router checks before the artifact route so no artifact
#: name can shadow the live game.
LIVE_FRAME_PATH = "/live/frame.png"
#: The one path that serves the report rather than an artifact.
REPORT_PATH = "/"
#: How long the threaded entry point's accept loop waits for a connection before
#: looking at its stop event again. Short enough that a stop is felt in well under
#: a human blink, long enough not to spin the CPU while serving nobody.
THREADED_POLL_S = 0.05
#: How long ``ThreadedViewer.stop`` waits for its accept loop to notice the stop
#: before giving up on the join. Generous compared with ``THREADED_POLL_S`` so a
#: working thread always finishes well inside it, and bounded so a thread that
#: somehow does not cannot hang the process trying to shut down.
THREADED_JOIN_TIMEOUT_S = 2.0
#: The startup lines. Two pages, two second lines: `serve` answers the run's
#: evidence, `live` answers the decision board beside the game window, and each
#: says which one it is rather than one sentence pretending to fit both.
REPORT_STARTUP_NOTE = (
    "this is the run's evidence, not the game: the loop advances one frame per "
    "tick, so the page grows at the tick rate, not at 60 fps"
)
LIVE_STARTUP_NOTE = (
    "this is the decision board, not the game: the game plays in its own window "
    "at 60 fps, and this page refreshes at the decision tick rate, not at 60 fps"
)
#: Content types named for the artifact kinds this project writes. Anything else
#: is served as bytes: the extension is the only claim the viewer can check.
CONTENT_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".html": "text/html; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".json": "application/json",
}
#: What an artifact of an unnamed kind is served as.
FALLBACK_CONTENT_TYPE = "application/octet-stream"


def _artifacts_root(journal_path: Path, artifacts_dir: object | None) -> Path | None:
    """Where artifacts are served from: the argument, else the recorded directory.

    The recorded directory is read from the journal header with the same partial
    read the page uses, so a run still in progress has one. A journal that cannot
    be read at all falls back to the journal's own directory rather than refusing
    to start: the viewer exists to watch a run that has not finished, so an
    unreadable journal now is a state it must be able to outlive.
    """
    if artifacts_dir is not None:
        return Path(artifacts_dir)
    try:
        header = read_journal(journal_path, allow_partial=True).run
    except (JournalFormatError, OSError):
        return journal_path.parent
    recorded = header.get("artifacts_dir")
    if recorded is None or str(recorded) == "":
        return journal_path.parent
    return Path(str(recorded))


@dataclass(frozen=True)
class Viewer:
    """Everything one viewer needs to answer a request.

    ``read`` and ``report`` are the two seams: the real ones are the journal
    reader and :func:`~mario_jev.dashboard.build_html`, and a test may replace
    either to exercise the handler without a journal on disk. ``live`` is the
    third, and the only one that changes a route: the decision board's builder,
    or ``None`` when this viewer does not answer ``/live`` -- a viewer without it
    treats that path exactly like every other path with no artifact behind it.
    ``frame_source`` is the fourth and is a different kind of option again: not a
    page builder but the thing ``/live/frame.png`` asks for the newest frame. It
    defaults to ``None``, and a viewer without one answers that route with a 404
    saying so.
    """

    journal_path: Path
    artifacts_dir: Path | None = None
    read: Callable[..., Journal] = read_journal
    report: Callable[..., str] = build_html
    retry_delay_s: float = RETRY_DELAY_S
    live: Callable[..., str] | None = None
    frame_source: Callable[[], bytes | None] | None = None


class ViewerHandler(BaseHTTPRequestHandler):
    """One request, one freshly built page, one connection."""

    server_version = "mario-jev-viewer"
    sys_version = ""
    #: HTTP/1.0 closes the connection after the response. That is what makes
    #: ``--once`` exactly one request, and it is honest about what this server is:
    #: one request at a time, with nothing kept alive between them.
    protocol_version = "HTTP/1.0"

    # -- routing --

    def do_GET(self) -> None:
        self._respond(with_body=True)

    def do_HEAD(self) -> None:
        self._respond(with_body=False)

    def _method_not_allowed(self) -> None:
        """Anything but GET and HEAD: said out loud, and nothing else happens."""
        self._send(
            405,
            "text/plain; charset=utf-8",
            b"method not allowed: this viewer serves GET and HEAD\n",
            allow="GET, HEAD",
        )

    do_POST = _method_not_allowed
    do_PUT = _method_not_allowed
    do_DELETE = _method_not_allowed
    do_PATCH = _method_not_allowed
    do_OPTIONS = _method_not_allowed
    do_TRACE = _method_not_allowed

    def _respond(self, *, with_body: bool) -> None:
        target = unquote(urlsplit(self.path).path)
        if target == REPORT_PATH:
            self._report(with_body=with_body)
        elif target == LIVE_PATH and self.viewer.live is not None:
            self._live_page(with_body=with_body)
        elif target == LIVE_FRAME_PATH:
            # Before the artifact route, deliberately: ``live/frame.png`` is also a
            # path the artifact resolver would happily look up inside the run's
            # directory, and a file that happened to sit there must not be able to
            # shadow the live game -- nor the game to shadow an artifact, because
            # this is the one reserved name under the page's own prefix.
            self._live_frame(with_body=with_body)
        else:
            self._artifact(target, with_body=with_body)

    # -- the newest frame: the game inside the page --

    def _live_frame(self, *, with_body: bool) -> None:
        """Answer the newest published frame, or say which 404 this is.

        The source is asked on every request rather than cached here: the sink
        behind it is the one place a frame lives, and it is the one that knows
        how to answer without encoding the same frame twice. A second cache in
        the handler would be a second answer to "which frame is newest".

        ``no-store`` because the page polls this URL: a cached answer would keep
        showing the same frame while the browser reported a hit, and the game on
        the page would freeze with nothing failing.

        The two 404s are different sentences because they are different facts.
        "No source configured" is a viewer that will never serve a frame, and "no
        frame published yet" is a run that has not started stepping. Only the
        second one fixes itself by reloading, and a person reading the page is
        owed the difference.
        """
        source = self.viewer.frame_source
        if source is None:
            self._send(
                404,
                "text/plain; charset=utf-8",
                b"not found: this viewer has no live frame source\n",
                with_body=with_body,
            )
            return
        try:
            data = source()
        except Exception as exc:
            # A request must never make this handler raise: an exception here is a
            # dropped connection with no status line and a traceback on the server,
            # which tells the person at the page nothing. The source's own words are
            # included for the same reason the torn-journal 503 includes the
            # reader's: a bare 500 leaves them with a frozen game and no cause.
            self._send(
                500,
                "text/plain; charset=utf-8",
                (
                    "the live frame could not be produced; the frame source "
                    f"failed: {type(exc).__name__}: {exc}\n"
                ).encode("utf-8"),
                with_body=with_body,
            )
            return
        if data is None:
            self._send(
                404,
                "text/plain; charset=utf-8",
                b"not found: no frame has been published yet\n",
                with_body=with_body,
            )
            return
        self._send(
            200,
            "image/png",
            data,
            with_body=with_body,
            cache_control="no-store",
        )

    # -- the report and the live page --

    def _report(self, *, with_body: bool) -> None:
        self._read_page(
            lambda journal: self.viewer.report(
                journal, artifacts_dir=self.viewer.artifacts_dir
            ),
            with_body=with_body,
        )

    def _live_page(self, *, with_body: bool) -> None:
        """The decision board, through the report's own read-retry-503 path.

        Deliberately the *same* path: the live page is the same journal read the
        same partial way, retried the same single time after the same delay, and
        answered with the same 503 and the same reader's words when it still
        cannot be read. A second copy would be a second chance for the two pages
        to disagree about a torn journal, and there is only one journal.
        """
        assert self.viewer.live is not None  # routed only when configured
        self._read_page(
            lambda journal: self.viewer.live(  # type: ignore[misc]
                journal, artifacts_dir=self.viewer.artifacts_dir
            ),
            with_body=with_body,
        )

    def _read_page(
        self, build: Callable[[Journal], str], *, with_body: bool
    ) -> None:
        """Read the journal, build the page, and answer 200 or the shared 503.

        The retry and the 503 live here once so ``/`` and ``/live`` cannot drift:
        the error path is the contract, and duplicating it would let one page keep
        a fix the other lost.
        """
        try:
            page = self._render(build)
        except (JournalFormatError, OSError) as exc:
            # The reader's own words, not a summary of them: the line number and
            # the reason are the only useful thing a viewer can say about a
            # journal it could not read. They are preceded by the one thing the
            # reader cannot know -- that this is a *snapshot*, and the file may be
            # mid-write rather than broken, which is what a refresh fixes.
            self._send(
                503,
                "text/plain; charset=utf-8",
                (
                    "the journal could not be read at this moment: it may be "
                    "mid-write, in which case reloading will help. The reader said:\n"
                    f"{exc}\n"
                ).encode("utf-8"),
                with_body=with_body,
            )
            return
        self._send(
            200, "text/html; charset=utf-8", page.encode("utf-8"), with_body=with_body
        )

    def _render(self, build: Callable[[Journal], str]) -> str:
        """The page for the journal as it is now, with exactly one retry.

        The retry covers the window in which the writer has flushed part of a
        line: reading again after a short pause usually finds it complete. It is
        one retry and not a wait-for-readiness loop -- a viewer that blocks until
        the file becomes readable would hide a journal that is genuinely broken.
        """
        try:
            return self._page(build)
        except (JournalFormatError, OSError):
            time.sleep(self.viewer.retry_delay_s)
            return self._page(build)

    def _page(self, build: Callable[[Journal], str]) -> str:
        journal = self.viewer.read(self.viewer.journal_path, allow_partial=True)
        return build(journal)

    # -- artifacts --

    def _artifact(self, target: str, *, with_body: bool) -> None:
        resolved = self._resolve(target)
        if resolved is None:
            self._not_found(with_body=with_body)
            return
        try:
            data = resolved.read_bytes()
        except OSError:
            self._not_found(with_body=with_body)
            return
        self._send(200, self._content_type(resolved), data, with_body=with_body)

    def _resolve(self, target: str) -> Path | None:
        """The artifact ``target`` names, or ``None`` if it is not one.

        ``None`` covers every way a request can fail to name a file inside the
        artifacts directory: no directory is configured, the path is absolute,
        it escapes with ``..``, a symlink inside the directory points out of it,
        the file does not exist, or it is a directory. All of them become the
        same ``404``: outside the artifacts directory nothing exists as far as
        this server is concerned, and saying which of those it was would leak the
        shape of the filesystem to whoever asked.
        """
        root = self.viewer.artifacts_dir
        if root is None:
            return None
        relative = target[1:] if target.startswith("/") else target
        if not relative or relative.startswith("/") or Path(relative).is_absolute():
            return None
        try:
            base = Path(root).resolve()
            candidate = (base / relative).resolve()
            # ``ValueError`` as well as ``OSError``: a path carrying a null byte
            # makes ``resolve`` raise rather than fail to match, and a request must
            # never be able to make this handler raise. It is answered 404 like every
            # other path that has no artifact behind it.
            if not candidate.is_relative_to(base) or not candidate.is_file():
                return None
        except (OSError, ValueError):
            return None
        return candidate

    @staticmethod
    def _content_type(path: Path) -> str:
        return CONTENT_TYPES.get(path.suffix.lower(), FALLBACK_CONTENT_TYPE)

    # -- responses --

    def _not_found(self, *, with_body: bool) -> None:
        self._send(
            404, "text/plain; charset=utf-8", b"not found\n", with_body=with_body
        )

    def _send(
        self,
        status: int,
        content_type: str,
        body: bytes,
        *,
        with_body: bool = True,
        allow: str | None = None,
        cache_control: str | None = None,
    ) -> None:
        """One response, with the extra headers the caller asked for.

        ``allow`` and ``cache_control`` are ``None`` by default, and ``None`` adds
        nothing: every answer that predates them is byte-identical to what it was.
        """
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if allow is not None:
            self.send_header("Allow", allow)
        if cache_control is not None:
            self.send_header("Cache-Control", cache_control)
        self.end_headers()
        if with_body:
            self.wfile.write(body)

    @property
    def viewer(self) -> Viewer:
        return self.server.viewer  # type: ignore[attr-defined]


class ViewerServer(TCPServer):
    """A one-request-at-a-time server for one :class:`Viewer`."""

    # A viewer is started, stopped and started again while a run goes on; a
    # ``TIME_WAIT`` socket from the previous run must not make that fail.
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], viewer: Viewer) -> None:
        self.viewer = viewer
        super().__init__(address, ViewerHandler)


def _viewer_for(
    journal_path: Path,
    *,
    artifacts_dir: object | None,
    retry_delay_s: float,
    live: bool,
    frame_source: Callable[[], bytes | None] | None = None,
) -> Viewer:
    """The one viewer both entry points build, so they cannot disagree.

    ``live`` decides whether ``Viewer.live`` is the decision board's builder or
    ``None``. ``None`` is the default everywhere it is not asked for, which is
    what keeps ``/live`` a plain 404 on every viewer built before this route
    existed. ``frame_source`` is passed through untouched and defaults to
    ``None`` for the same reason: a viewer nobody gave a source to must answer
    ``/live/frame.png`` with its own 404, not with an invented frame.
    """
    return Viewer(
        journal_path=journal_path,
        artifacts_dir=_artifacts_root(journal_path, artifacts_dir),
        retry_delay_s=retry_delay_s,
        live=build_live_html if live else None,
        frame_source=frame_source,
    )


def _bind_server(
    journal_path: Path,
    *,
    artifacts_dir: object | None,
    bind: str,
    port: int,
    retry_delay_s: float,
    live: bool,
    url_path: str,
    frame_source: Callable[[], bytes | None] | None = None,
) -> tuple[ViewerServer, str]:
    """Bind the socket, compute the URL the OS actually chose, and return both.

    Shared by :func:`serve` and :class:`ThreadedViewer` so the two entry points
    reuse one binding and one URL computation; a second copy is a second place
    for ``port=0`` to be handled differently. The URL always names the page this
    viewer actually answers -- ``/`` for the report, ``/live`` for the board -- and
    never a path the router would 404.
    """
    viewer = _viewer_for(
        journal_path,
        artifacts_dir=artifacts_dir,
        retry_delay_s=retry_delay_s,
        live=live,
        frame_source=frame_source,
    )
    server = ViewerServer((bind, port), viewer)
    host, actual_port = server.server_address[0], server.server_address[1]
    return server, f"http://{host}:{actual_port}{url_path}"


def _startup_lines(path: Path, url: str, *, live: bool) -> tuple[str, str]:
    """The two lines either entry point prints, shared so they cannot drift.

    The first names the journal and the bound URL. The second says what the page
    is, and the two paths say different things because they are different pages:
    ``serve`` shows the run's evidence, ``live`` shows the decision board beside
    the game window.
    """
    note = LIVE_STARTUP_NOTE if live else REPORT_STARTUP_NOTE
    return f"serving {path} at {url}", note


def run(server: ViewerServer, *, once: bool = False) -> int:
    """Serve until interrupted, or exactly one request when ``once`` is set.

    Returns the exit code. ``KeyboardInterrupt`` is the Ctrl-C the user meant and
    is not an error: it returns ``0`` with no traceback.
    """
    try:
        server.handle_request()
        if once:
            return 0
        while True:
            server.handle_request()
    except KeyboardInterrupt:
        return 0


def serve(
    journal_path: object,
    *,
    artifacts_dir: object | None = None,
    bind: str = DEFAULT_BIND,
    port: int = DEFAULT_PORT,
    once: bool = False,
    retry_delay_s: float = RETRY_DELAY_S,
    out: TextIO | None = None,
    on_ready: Callable[[str], None] | None = None,
    live: bool = False,
    frame_source: Callable[[], bytes | None] | None = None,
) -> int:
    """Start the viewer for ``journal_path`` and serve until interrupted.

    ``on_ready``, when given, is called with the bound URL once the socket is
    bound and **before** the first request is served. It exists so a caller can
    act on the address the OS actually chose (``port=0``). ``serve`` prints its
    own startup lines unconditionally and decides nothing on the callback's
    behalf; passing ``None`` keeps every current caller unchanged.

    ``live`` is ``False`` by default and changes not one byte for every current
    caller: with it off, no live builder is configured and ``/live`` is the same
    404 as any unknown path. With it on, this viewer is the decision board -- the
    route the ``live`` command serves from its background thread.

    ``frame_source`` is the newest frame for ``/live/frame.png``, and is ``None``
    by default for the same reason: no source means that route answers its own
    404 instead of an image, so a caller that never passed one sees the same
    responses as before the route existed.
    """
    stream = sys.stdout if out is None else out
    path = Path(journal_path)
    server, url = _bind_server(
        path,
        artifacts_dir=artifacts_dir,
        bind=bind,
        port=port,
        retry_delay_s=retry_delay_s,
        live=live,
        url_path=LIVE_PATH if live else REPORT_PATH,
        frame_source=frame_source,
    )
    try:
        # Both prints and the callback sit inside the ``try``: a callback that
        # raises, or a stream that fails (a closed pipe, say), must not skip
        # ``server_close`` and leak the socket the viewer just bound. A failing
        # print is the caller's environment failing, not swallowed here.
        first, note = _startup_lines(path, url, live=live)
        print(first, file=stream)
        print(note, file=stream)
        if on_ready is not None:
            on_ready(url)
        return run(server, once=once)
    finally:
        server.server_close()


class ThreadedViewer:
    """The viewer on a daemon thread, for a command that owns the main one.

    ``live`` plays the game on the main thread, because nes-py refuses to render
    anywhere else, so the page beside the window has to be served from a
    background thread or not at all. This is that thread and the small surface
    around it: it binds the socket and computes the URL exactly as :func:`serve`
    does (through the same helpers), prints the same first line and the live
    page's own second line, calls ``on_ready`` before the first request can be
    served, and then loops over :meth:`~socketserver.BaseServer.handle_request`
    with a short server timeout until :meth:`stop` asks it to quit.

    Reusing :func:`serve` itself is impossible here and that is not an oversight:
    its ``run`` blocks the calling thread by design, and blocking the calling
    thread is the one thing this path cannot do. What is shared is everything
    that must not differ -- the bind, the URL, the startup lines and
    ``on_ready`` -- while the loop is the one piece a background thread needs.

    ``stop`` closes the socket and joins the thread with a bounded wait. The wait
    is bounded because the accept loop polls its stop event every
    ``THREADED_POLL_S``; a socket close from another thread can also surface as
    ``OSError`` inside the loop, which is treated as the stop it is rather than
    as a failure to print.
    """

    def __init__(
        self,
        journal_path: object,
        *,
        artifacts_dir: object | None = None,
        bind: str = DEFAULT_BIND,
        port: int = DEFAULT_PORT,
        retry_delay_s: float = RETRY_DELAY_S,
        out: TextIO | None = None,
        on_ready: Callable[[str], None] | None = None,
        live: bool = True,
        frame_source: Callable[[], bytes | None] | None = None,
    ) -> None:
        self._path = Path(journal_path)
        self._stream = sys.stdout if out is None else out
        self._on_ready = on_ready
        self._live = live
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        url_path = LIVE_PATH if live else REPORT_PATH
        self._server, self.url = _bind_server(
            self._path,
            artifacts_dir=artifacts_dir,
            bind=bind,
            port=port,
            retry_delay_s=retry_delay_s,
            live=live,
            url_path=url_path,
            frame_source=frame_source,
        )

    @property
    def address(self) -> tuple[str, int]:
        """The host and port the socket is bound to, as the OS chose them."""
        return self._server.server_address

    def start(self) -> str:
        """Print, call ``on_ready``, then serve on a daemon thread. Returns the URL.

        The prints and the callback are wrapped so a raising callback still closes
        the socket this instance bound: exactly the promise :func:`serve` makes on
        the blocking path, and for the same reason -- a callback is the caller's
        business, not a licence to leak a port.
        """
        try:
            first, note = _startup_lines(self._path, self.url, live=self._live)
            print(first, file=self._stream)
            print(note, file=self._stream)
            if self._on_ready is not None:
                self._on_ready(self.url)
        except BaseException:
            self._server.server_close()
            raise
        self._server.timeout = THREADED_POLL_S
        self._thread = threading.Thread(
            target=self._serve, name="mario-jev-viewer", daemon=True
        )
        self._thread.start()
        return self.url

    def _serve(self) -> None:
        """Accept one request at a time until ``stop`` sets the event.

        ``handle_request`` returns after the server timeout when nobody connects,
        so the event is seen promptly without a busy loop. ``OSError`` is the
        socket closing under a blocked ``accept`` during ``stop``; that is the
        stop, not an error, and is not printed.
        """
        while not self._stop.is_set():
            try:
                self._server.handle_request()
            except OSError:
                return

    def stop(self) -> None:
        """Close the socket and join the thread with a bounded wait.

        Closing first is what unblocks a thread waiting in ``accept``; the join is
        bounded so a thread that somehow does not notice cannot hang the caller
        that is trying to shut the process down.
        """
        self._stop.set()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=THREADED_JOIN_TIMEOUT_S)

    def __enter__(self) -> ThreadedViewer:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
