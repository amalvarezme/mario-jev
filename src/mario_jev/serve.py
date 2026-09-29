"""A local viewer for one run, live: the report, regenerated per request.

``mario_jev serve <journal>`` starts a **single-threaded** HTTP server whose only
job is to answer ``GET /`` with ``build_html(read_journal(path,
allow_partial=True))``. There is no streaming, no server-sent events, no polling
loop and no worker thread: a request reads the journal as it is at that moment,
builds one page, sends it, and the next request starts over. One request is
handled at a time, on purpose -- a second viewer tab waits for the first.

A journal mid-write has a short window in which it is not readable, because a
line can be half-written when the reader reaches it. The viewer does not paper
over that: it retries the read once after a short documented delay and, if the
journal still cannot be read, answers ``503`` with a sentence saying the file may
be mid-write and the reader's own message beneath it -- the line and the reason.
It never renders a half-read journal as if it were complete and it never skips a
line it could not parse -- showing part of a run as the whole run is the one
thing a viewer built for evidence must not do.

Request paths other than ``/`` are resolved against the run's artifacts
directory and served only from inside it; anything that is not inside it is
answered ``404``, including a path the filesystem refuses to look at -- a null
byte makes ``resolve`` raise rather than fail to match, and no request may make
this handler raise. What is *not* necessarily a 404 is a request the standard
library rejects before dispatching here at all: a verb with no handler, or a
malformed request line, gets the library's own ``501``/``400`` and never a file.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from socketserver import TCPServer
from typing import Callable, TextIO
from urllib.parse import unquote, urlsplit

from mario_jev.dashboard import build_html
from mario_jev.journal import Journal, JournalFormatError, read_journal

__all__ = [
    "DEFAULT_BIND",
    "DEFAULT_PORT",
    "RETRY_DELAY_S",
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
#: The one path that serves the report rather than an artifact.
REPORT_PATH = "/"
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
    either to exercise the handler without a journal on disk.
    """

    journal_path: Path
    artifacts_dir: Path | None = None
    read: Callable[..., Journal] = read_journal
    report: Callable[..., str] = build_html
    retry_delay_s: float = RETRY_DELAY_S


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
        else:
            self._artifact(target, with_body=with_body)

    # -- the report --

    def _report(self, *, with_body: bool) -> None:
        try:
            page = self._render()
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

    def _render(self) -> str:
        """The page for the journal as it is now, with exactly one retry.

        The retry covers the window in which the writer has flushed part of a
        line: reading again after a short pause usually finds it complete. It is
        one retry and not a wait-for-readiness loop -- a viewer that blocks until
        the file becomes readable would hide a journal that is genuinely broken.
        """
        try:
            return self._page()
        except (JournalFormatError, OSError):
            time.sleep(self.viewer.retry_delay_s)
            return self._page()

    def _page(self) -> str:
        journal = self.viewer.read(self.viewer.journal_path, allow_partial=True)
        return self.viewer.report(journal, artifacts_dir=self.viewer.artifacts_dir)

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
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if allow is not None:
            self.send_header("Allow", allow)
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
) -> int:
    """Start the viewer for ``journal_path`` and serve until interrupted."""
    stream = sys.stdout if out is None else out
    path = Path(journal_path)
    viewer = Viewer(
        journal_path=path,
        artifacts_dir=_artifacts_root(path, artifacts_dir),
        retry_delay_s=retry_delay_s,
    )
    server = ViewerServer((bind, port), viewer)
    host, actual_port = server.server_address[0], server.server_address[1]
    print(f"serving {path} at http://{host}:{actual_port}/", file=stream)
    print(
        "this is the run's evidence, not the game: the loop advances one frame per "
        "tick, so the page grows at the tick rate, not at 60 fps",
        file=stream,
    )
    try:
        return run(server, once=once)
    finally:
        server.server_close()
