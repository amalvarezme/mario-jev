"""The live viewer, driven over a real socket -- with no thread and no sleep.

Every test here sends a raw request on a connection opened *before* the server
starts serving, then lets the server accept it. The kernel holds the connection
in the listen backlog, so the accept is immediate and the test needs no worker
thread, no ``join`` and no waiting: a failure is a failure, never a flake.

The honesty rules pinned here are the ones a viewer of a running experiment can
break: it must say "in progress" for a journal whose trailer has not been written
yet (never "not recorded", which would describe lost evidence), it must serve
nothing from outside the artifacts directory, and it must report a journal it
cannot read as ``503`` with a line saying the file may be mid-write and the
reader's own message beneath it, rather than a page that
looks complete.
"""

from __future__ import annotations

import json
import socket
from io import StringIO
from pathlib import Path

import pytest

from mario_jev import serve as serve_module
from mario_jev.dashboard import build_live_html
from mario_jev.journal import Journal, JournalFormatError
from mario_jev.serve import (
    DEFAULT_BIND,
    LIVE_PATH,
    ThreadedViewer,
    Viewer,
    ViewerServer,
    run,
    serve,
)

HOST = "127.0.0.1"


# -- crafted journals --------------------------------------------------------------


def _header(artifacts_dir: object | None = None) -> str:
    run: dict[str, object] = {
        "kind": "run",
        "started_at": "2026-01-01T00:00:00+00:00",
        "stage": "1-1",
        "seed": 0,
        "settings": {
            "jev_endpoint": "https://example.invalid/v1/systemone",
            "jev_model": "jev-latest",
            "jev_available": False,
            "advisor_enabled": False,
        },
    }
    if artifacts_dir is not None:
        run["artifacts_dir"] = str(artifacts_dir)
    return json.dumps(run)


def _tick(number: int) -> str:
    return json.dumps(
        {
            "kind": "tick",
            "tick": number,
            "frame_index": number,
            "emulator_tick": number * 185,
            "state": {},
            "rollout": {},
            "questions": None,
            "response": None,
            "asked_at_tick": None,
            "answer_age_ticks": None,
            "jev_call_ms": None,
            "decision": {
                "action": "NOOP",
                "source": "fallback_no_key",
                "confidence": None,
                "risk": None,
                "reason": "no_credential",
                "jev_called": False,
            },
            "advisor": {"available": False, "reason": "disabled", "measured": False},
            "latencies_ms": {"rollout": 1.0, "jev": 0.0, "advisor": 0.0, "total": 2.0},
            "degradations": [],
            "terminated": False,
            "truncated": False,
            "frame_png": None,
            "trajectory_png": None,
        }
    )


def _live_journal(path: Path, *, ticks: int = 2, artifacts_dir: object | None = None) -> Path:
    """A journal a running loop would leave behind: header, ticks, no trailer."""
    lines = [_header(artifacts_dir)] + [_tick(number) for number in range(ticks)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# -- request/response plumbing -----------------------------------------------------


def _exchange(server: ViewerServer, request: bytes) -> tuple[int, bytes]:
    """Send one raw request, serve the one request, and return ``(exit code, bytes)``."""
    host, port = server.server_address
    client = socket.create_connection((host, port))
    try:
        client.sendall(request)
        code = run(server, once=True)
        chunks: list[bytes] = []
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        client.close()
        server.server_close()
    return code, b"".join(chunks)


def _request(viewer: Viewer, request: bytes) -> tuple[str, bytes, dict[str, str]]:
    """One request against a fresh server: status line, body, and headers."""
    server = ViewerServer((HOST, 0), viewer)
    code, response = _exchange(server, request)
    assert code == 0
    head, _, body = response.partition(b"\r\n\r\n")
    lines = head.decode("ascii").splitlines()
    headers = {}
    for line in lines[1:]:
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    return lines[0], body, headers


def _get(target: str, *, method: str = "GET") -> bytes:
    return f"{method} {target} HTTP/1.0\r\nHost: localhost\r\n\r\n".encode("ascii")


def _viewer(path: Path, artifacts_dir: object | None = None) -> Viewer:
    return Viewer(
        journal_path=path,
        artifacts_dir=None if artifacts_dir is None else Path(artifacts_dir),
        retry_delay_s=0.0,
    )


def _live_viewer(path: Path, artifacts_dir: object | None = None) -> Viewer:
    """The report viewer plus the decision board, for the ``/live`` route tests."""
    return Viewer(
        journal_path=path,
        artifacts_dir=None if artifacts_dir is None else Path(artifacts_dir),
        retry_delay_s=0.0,
        live=build_live_html,
    )


def _read_all(client: socket.socket) -> bytes:
    """Everything the server sends before it closes the ``HTTP/1.0`` connection."""
    chunks: list[bytes] = []
    while True:
        chunk = client.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


# -- the report --------------------------------------------------------------------


def test_the_root_serves_the_report_for_a_run_still_writing(tmp_path: Path) -> None:
    journal = _live_journal(tmp_path / "live.jsonl", ticks=2)

    status, body, headers = _request(_viewer(journal), _get("/"))

    assert status == "HTTP/1.0 200 OK"
    assert headers["content-type"] == "text/html; charset=utf-8"
    assert headers["content-length"] == str(len(body))
    assert body.startswith(b"<!DOCTYPE html>")
    # The run has no trailer yet, so the page says so rather than inventing counts.
    assert b"in progress" in body
    assert b"Tick 0" in body and b"Tick 1" in body
    # And the summary table is not quietly rendered as a complete run's counts.
    assert b"<th>termination</th>" not in body


def test_the_report_is_regenerated_for_every_request(tmp_path: Path) -> None:
    journal = _live_journal(tmp_path / "live.jsonl", ticks=1)
    server = ViewerServer((HOST, 0), _viewer(journal))
    try:
        assert b"Tick 1" not in _request_on_open_server(server)[1]
        # The loop ticks again between the two requests, exactly as it does live.
        with journal.open("a", encoding="utf-8") as handle:
            handle.write(_tick(1) + "\n")
        assert b"Tick 1" in _request_on_open_server(server)[1]
    finally:
        server.server_close()


def _request_on_open_server(server: ViewerServer, target: str = "/") -> tuple[str, bytes]:
    """One request against a server that keeps serving (not ``--once``)."""
    host, port = server.server_address
    client = socket.create_connection((host, port))
    try:
        client.sendall(_get(target))
        server.handle_request()
        chunks: list[bytes] = []
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        client.close()
    head, _, body = b"".join(chunks).partition(b"\r\n\r\n")
    return head.decode("ascii").splitlines()[0], body


# -- one request, and only the methods that make sense ------------------------------


def test_once_serves_one_request_then_exits_zero(tmp_path: Path) -> None:
    journal = _live_journal(tmp_path / "live.jsonl")
    server = ViewerServer((HOST, 0), _viewer(journal))

    code, response = _exchange(server, _get("/"))

    assert code == 0
    assert response.startswith(b"HTTP/1.0 200 OK")


def test_head_answers_the_headers_and_no_body(tmp_path: Path) -> None:
    journal = _live_journal(tmp_path / "live.jsonl")

    status, body, headers = _request(_viewer(journal), _get("/", method="HEAD"))

    assert status == "HTTP/1.0 200 OK"
    assert body == b""
    assert int(headers["content-length"]) > 0


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
def test_every_other_method_is_405(tmp_path: Path, method: str) -> None:
    journal = _live_journal(tmp_path / "live.jsonl")

    status, body, headers = _request(_viewer(journal), _get("/", method=method))

    assert status == "HTTP/1.0 405 Method Not Allowed"
    assert headers["allow"] == "GET, HEAD"
    assert b"method not allowed" in body


def test_a_keyboard_interrupt_is_an_exit_code_not_a_traceback(tmp_path: Path) -> None:
    journal = _live_journal(tmp_path / "live.jsonl")
    server = ViewerServer((HOST, 0), _viewer(journal))

    def interrupt() -> None:
        raise KeyboardInterrupt

    server.handle_request = interrupt  # type: ignore[method-assign]
    try:
        assert serve_module.run(server) == 0
    finally:
        server.server_close()


# -- artifacts, and only artifacts --------------------------------------------------


@pytest.mark.parametrize(
    ("name", "content", "expected"),
    [
        ("frame_000000.png", b"\x89PNG\r\n\x1a\n", "image/png"),
        ("report.html", b"<!DOCTYPE html>\n", "text/html; charset=utf-8"),
        ("note.txt", b"evidence\n", "text/plain; charset=utf-8"),
        ("run.json", b"{}\n", "application/json"),
        ("blob.bin", b"\x00\x01", "application/octet-stream"),
    ],
)
def test_an_artifact_inside_the_directory_is_served(
    tmp_path: Path, name: str, content: bytes, expected: str
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / name).write_bytes(content)
    journal = _live_journal(tmp_path / "live.jsonl", artifacts_dir=artifacts)

    status, body, headers = _request(_viewer(journal, artifacts), _get(f"/{name}"))

    assert status == "HTTP/1.0 200 OK"
    assert body == content
    assert headers["content-type"] == expected


def test_a_null_byte_path_is_a_404_and_never_a_raised_handler(tmp_path: Path) -> None:
    """A path the filesystem refuses to look at is answered, not thrown out of.

    ``Path.resolve`` raises ``ValueError`` on a null byte rather than failing to
    match, so an ``except OSError`` alone let it escape: the connection was dropped
    with no status line and a traceback on the server, which is not the 404 this
    handler promises for every path with no artifact behind it. Found by an
    independent round, which saw the connection go unanswered.
    """
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "fine.txt").write_text("fine\n", encoding="utf-8")
    journal = _live_journal(tmp_path / "live.jsonl", artifacts_dir=artifacts)

    for target in ("\x00secret.txt", "%00etc/passwd", "..%00/secret.txt"):
        status, body, _ = _request(_viewer(journal, artifacts), _get(f"/{target}"))
        assert status == "HTTP/1.0 404 Not Found", (target, status)
        assert b"SECRET" not in body

    # And the handler is still alive for the next request.
    status, _, _ = _request(_viewer(journal, artifacts), _get("/fine.txt"))
    assert status == "HTTP/1.0 200 OK"


@pytest.mark.parametrize(
    "target",
    [
        "/../secret.txt",
        "/%2e%2e/secret.txt",
        "/a/../../secret.txt",
        "//{absolute}",
        "/{absolute}",
        "/escape.txt",
        "/sub",
    ],
)
def test_a_path_outside_the_artifacts_directory_is_a_404(tmp_path: Path, target: str) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "fine.txt").write_text("fine\n", encoding="utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("SECRET-EVIDENCE-NOBODY-MAY-SEE\n", encoding="utf-8")
    # A symlink that points out of the directory is the same escape by another
    # route, and resolving the real path is what closes it.
    (artifacts / "escape.txt").symlink_to(secret)
    (artifacts / "sub").mkdir()
    journal = _live_journal(tmp_path / "live.jsonl", artifacts_dir=artifacts)

    status, body, _ = _request(
        _viewer(journal, artifacts), _get(target.format(absolute=secret))
    )

    assert status == "HTTP/1.0 404 Not Found"
    assert b"SECRET" not in body


def test_the_report_never_lists_the_artifacts_directory(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "fine.txt").write_text("fine\n", encoding="utf-8")
    journal = _live_journal(tmp_path / "live.jsonl", artifacts_dir=artifacts)

    status, body, _ = _request(_viewer(journal, artifacts), _get("/"))

    assert status == "HTTP/1.0 200 OK"
    assert b"fine.txt" not in body
    # No link to any artifact either: the report embeds what it can and says
    # "not recorded" for the rest, rather than pointing at a directory listing.
    assert b"<a href" not in body


def test_a_missing_artifact_is_a_404_not_a_listing(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "fine.txt").write_text("fine\n", encoding="utf-8")
    journal = _live_journal(tmp_path / "live.jsonl", artifacts_dir=artifacts)

    status, body, _ = _request(_viewer(journal, artifacts), _get("/missing.txt"))
    listing_status, listing_body, _ = _request(
        _viewer(journal, artifacts), _get("/sub/")
    )

    assert status == "HTTP/1.0 404 Not Found"
    assert listing_status == "HTTP/1.0 404 Not Found"
    assert b"fine.txt" not in listing_body
    assert body == b"not found\n"


def test_artifacts_are_not_served_without_a_directory(tmp_path: Path) -> None:
    journal = _live_journal(tmp_path / "live.jsonl")

    status, _, _ = _request(_viewer(journal, None), _get("/frame_000000.png"))

    # ``/`` still works: the report reads the header, it does not need a directory.
    root_status, _, _ = _request(_viewer(journal, None), _get("/"))
    assert root_status == "HTTP/1.0 200 OK"
    assert status == "HTTP/1.0 404 Not Found"


def test_the_artifacts_directory_defaults_to_the_one_the_run_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "note.txt").write_text("recorded\n", encoding="utf-8")
    journal = _live_journal(tmp_path / "live.jsonl", artifacts_dir=artifacts)
    served: dict[str, object] = {}

    def fake_run(server: ViewerServer, *, once: bool = False) -> int:
        # Serving from inside the run hook is what lets this test use the real
        # ``serve`` -- and therefore the real default-directory resolution --
        # without a thread and without guessing the port it bound.
        served["status"], served["body"] = _request_on_open_server(server, "/note.txt")
        return 0

    monkeypatch.setattr(serve_module, "run", fake_run)

    code = serve(journal, port=0, once=True, out=StringIO())

    assert code == 0
    assert served["status"] == "HTTP/1.0 200 OK"
    assert served["body"] == b"recorded\n"


# -- a journal that cannot be read ---------------------------------------------------


def test_a_torn_journal_is_a_503_with_the_readers_own_message(tmp_path: Path) -> None:
    journal = tmp_path / "torn.jsonl"
    # The writer flushed half a line and the reader arrived in the middle of it.
    journal.write_text('{"kind": "run"}\n{"kind": "tick"', encoding="utf-8")

    status, body, headers = _request(_viewer(journal), _get("/"))

    assert status == "HTTP/1.0 503 Service Unavailable"
    assert headers["content-type"] == "text/plain; charset=utf-8"
    # Two things, and both matter. The reader's message, not a paraphrase -- the line
    # number is the useful part -- and ahead of it the one fact the reader cannot
    # know: that this is a snapshot, so the file may be mid-write rather than broken.
    # A bare "journal line 2" leaves a person reloading a healthy live run with no
    # idea that reloading is the answer.
    assert body.startswith(b"the journal could not be read at this moment")
    assert b"journal line 2: not valid JSON" in body
    assert b"<!DOCTYPE html>" not in body


def test_the_read_is_retried_once_and_only_once(tmp_path: Path) -> None:
    journal = _live_journal(tmp_path / "live.jsonl")
    calls: list[int] = []

    def always_torn(path: object, *, allow_partial: bool = False) -> Journal:
        calls.append(1)
        raise JournalFormatError(2, "not valid JSON (Expecting value)")

    viewer = Viewer(journal_path=journal, read=always_torn, retry_delay_s=0.0)

    status, body, _ = _request(viewer, _get("/"))

    assert status == "HTTP/1.0 503 Service Unavailable"
    # The whole body, exactly: the snapshot sentence, then the reader's message.
    # Both halves are asserted here because this test is also the one that pins the
    # body of a failed read as plain text rather than a page.
    assert body == (
        b"the journal could not be read at this moment: it may be mid-write, in "
        b"which case reloading will help. The reader said:\n"
        b"journal line 2: not valid JSON (Expecting value)\n"
    )
    # One read, one retry. A viewer that kept re-reading would hide a journal that
    # is genuinely broken behind an apparently slow one.
    assert len(calls) == 2


def test_the_retry_really_rescues_a_journal_that_was_mid_write(tmp_path: Path) -> None:
    journal = _live_journal(tmp_path / "live.jsonl")
    calls: list[int] = []

    def torn_once(path: object, *, allow_partial: bool = False) -> Journal:
        calls.append(1)
        if len(calls) == 1:
            raise JournalFormatError(2, "not valid JSON (Expecting value)")
        return Journal(run={"kind": "run", "settings": {}}, ticks=(), summary=None)

    viewer = Viewer(journal_path=journal, read=torn_once, retry_delay_s=0.0)

    status, body, _ = _request(viewer, _get("/"))

    assert status == "HTTP/1.0 200 OK"
    assert b"<!DOCTYPE html>" in body
    assert len(calls) == 2


def test_a_page_is_never_built_from_a_journal_that_could_not_be_read(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "empty.jsonl"
    journal.write_text("", encoding="utf-8")

    status, body, _ = _request(_viewer(journal), _get("/"))

    assert status == "HTTP/1.0 503 Service Unavailable"
    assert b"the journal is empty" in body


def test_a_live_page_labels_the_recorded_summary_in_progress(tmp_path: Path) -> None:
    journal = _live_journal(tmp_path / "live.jsonl")

    status, body, _ = _request(_viewer(journal), _get("/"))

    assert status == "HTTP/1.0 200 OK"
    # The recorded block is where the counts would be, and the words there are
    # "in progress". The derived block below it is not part of this claim: it is
    # counted from the ticks really on disk.
    recorded = body.split(b"<h3>Recorded</h3>", 1)[1].split(b"<h3>", 1)[0]
    assert b"in progress" in recorded
    assert b"not recorded" not in recorded


# -- the startup line ----------------------------------------------------------------


def test_serve_says_where_it_listens_and_what_the_page_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = _live_journal(tmp_path / "live.jsonl")
    seen: dict[str, object] = {}

    def fake_run(server: ViewerServer, *, once: bool = False) -> int:
        seen["address"] = server.server_address
        seen["once"] = once
        return 0

    monkeypatch.setattr(serve_module, "run", fake_run)
    out = StringIO()

    code = serve(journal, port=0, once=True, out=out)

    lines = out.getvalue().splitlines()
    assert code == 0
    assert seen["once"] is True
    host, port = seen["address"]  # type: ignore[misc]
    assert lines[0] == f"serving {journal} at http://{host}:{port}/"
    assert host == DEFAULT_BIND
    # The one honest line: the page is evidence, and it grows at the tick rate.
    assert "not the game" in lines[1]
    assert "not at 60 fps" in lines[1]


def test_the_default_bind_is_loopback_only() -> None:
    assert DEFAULT_BIND == "127.0.0.1"


# -- on_ready: the address the OS actually chose, before the first request ------
# ``--port 0`` means the caller cannot know the URL until the socket is bound, so
# ``on_ready`` is the seam that hands it over. It prints nothing itself: the demo
# owns what happens when there is nobody to open it.


def test_serve_hands_on_ready_the_url_it_bound_before_it_serves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = _live_journal(tmp_path / "live.jsonl")
    order: list[object] = []
    bound: dict[str, object] = {}

    def fake_run(server: ViewerServer, *, once: bool = False) -> int:
        order.append("serving")
        bound["address"] = server.server_address
        return 0

    monkeypatch.setattr(serve_module, "run", fake_run)

    ready: list[str] = []
    code = serve(
        journal,
        port=0,
        once=True,
        out=StringIO(),
        on_ready=lambda url: (order.append("ready"), ready.append(url)),
    )

    assert code == 0
    host, port = bound["address"]  # type: ignore[misc]
    assert ready == [f"http://{host}:{port}/"]
    # Bound first, then named, then served: the whole point is that the URL is the
    # real one before anything can ask for it.
    assert order == ["ready", "serving"]


#: A server stand-in with a fixed address, so a test can assert the exact stdout
#: without a real bind choosing a random port. ``run`` is stubbed out, so the
#: stand-in only has to answer ``server_address`` and ``server_close``.
class _FixedAddressServer:
    def __init__(self, address: tuple[str, int], handler: object) -> None:
        self.server_address = ("127.0.0.1", 8000)

    def server_close(self) -> None:
        pass


def test_serve_without_on_ready_serves_exactly_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = _live_journal(tmp_path / "live.jsonl")
    monkeypatch.setattr(serve_module, "ViewerServer", _FixedAddressServer)
    monkeypatch.setattr(serve_module, "run", lambda server, *, once=False: 0)
    out = StringIO()

    code = serve(journal, port=0, once=True, out=out)

    assert code == 0
    # The whole startup output, not just the first line: "exactly as before" is a
    # claim about every byte serve prints.
    assert out.getvalue().splitlines() == [
        f"serving {journal} at http://127.0.0.1:8000/",
        "this is the run's evidence, not the game: the loop advances one frame per "
        "tick, so the page grows at the tick rate, not at 60 fps",
    ]


def test_serve_prints_the_same_lines_with_a_callback_as_without_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = _live_journal(tmp_path / "live.jsonl")
    monkeypatch.setattr(serve_module, "ViewerServer", _FixedAddressServer)
    monkeypatch.setattr(serve_module, "run", lambda server, *, once=False: 0)

    without = StringIO()
    with_callback = StringIO()
    assert serve(journal, port=0, once=True, out=without) == 0
    assert (
        serve(journal, port=0, once=True, out=with_callback, on_ready=lambda url: None)
        == 0
    )

    # serve prints its own lines and nothing on the callback's behalf: the silent
    # callback changes not one byte of them.
    assert without.getvalue() == with_callback.getvalue()


def test_a_raising_on_ready_still_closes_the_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = _live_journal(tmp_path / "live.jsonl")
    servers: list[object] = []

    class _RecordingServer:
        def __init__(self, address: tuple[str, int], handler: object) -> None:
            self.server_address = ("127.0.0.1", 8000)
            self.closed = False
            servers.append(self)

        def server_close(self) -> None:
            self.closed = True

    monkeypatch.setattr(serve_module, "ViewerServer", _RecordingServer)
    monkeypatch.setattr(serve_module, "run", lambda server, *, once=False: 0)

    def explode(url: str) -> None:
        raise RuntimeError("the callback is the caller's business, not serve's")

    with pytest.raises(RuntimeError):
        serve(journal, port=0, once=True, out=StringIO(), on_ready=explode)

    # The callback ran inside the ``try/finally``: the socket it was handed is
    # released even though the callback raised.
    assert servers[0].closed is True  # type: ignore[attr-defined]


# -- the live route: the decision board beside the game ---------------------------
# ``/live`` is a second report route, not a second read path. What these tests pin
# is that it answers the live builder, that a viewer without one treats it as the
# unknown path it is, and that the report route does not change by one byte when
# the board is configured next to it.


def test_the_live_route_serves_the_live_page_over_a_real_socket(tmp_path: Path) -> None:
    journal = _live_journal(tmp_path / "live.jsonl", ticks=2)

    status, body, headers = _request(_live_viewer(journal), _get(LIVE_PATH))

    assert status == "HTTP/1.0 200 OK"
    assert headers["content-type"] == "text/html; charset=utf-8"
    assert headers["content-length"] == str(len(body))
    assert body.startswith(b"<!DOCTYPE html>")
    # The board's own two marks: it refreshes itself at the tick rate and links to
    # the full report, rather than being the report under another name.
    assert b'http-equiv="refresh"' in body
    assert b'<a href="/">' in body
    assert b"not 60 fps" in body


def test_live_is_a_404_when_no_live_builder_is_configured(tmp_path: Path) -> None:
    journal = _live_journal(tmp_path / "live.jsonl", ticks=1)

    status, body, _ = _request(_viewer(journal), _get(LIVE_PATH))

    # The same 404 every unknown path gets: a viewer that was never asked to serve
    # the board must not pretend there is one.
    assert status == "HTTP/1.0 404 Not Found"
    assert body == b"not found\n"


def test_configuring_live_does_not_change_the_report_route(tmp_path: Path) -> None:
    journal = _live_journal(tmp_path / "live.jsonl", ticks=1)

    without = _request(_viewer(journal), _get("/"))
    with_live = _request(_live_viewer(journal), _get("/"))

    # ``/`` is byte-identical whether or not the live builder is configured. The
    # route is one more branch, not a change to the report.
    assert without == with_live


def test_a_torn_journal_on_the_live_route_is_the_same_503_once(tmp_path: Path) -> None:
    journal = _live_journal(tmp_path / "live.jsonl")
    calls: list[int] = []

    def always_torn(path: object, *, allow_partial: bool = False) -> Journal:
        calls.append(1)
        raise JournalFormatError(2, "not valid JSON (Expecting value)")

    viewer = Viewer(
        journal_path=journal, read=always_torn, retry_delay_s=0.0, live=build_live_html
    )

    status, body, _ = _request(viewer, _get(LIVE_PATH))

    assert status == "HTTP/1.0 503 Service Unavailable"
    # The live page goes through the report's own read-retry-503 path, so the body
    # is exactly the one the report route already promises.
    assert body == (
        b"the journal could not be read at this moment: it may be mid-write, in "
        b"which case reloading will help. The reader said:\n"
        b"journal line 2: not valid JSON (Expecting value)\n"
    )
    # One read, one retry: the live route does not grow a second, longer loop.
    assert len(calls) == 2


def test_a_real_torn_journal_on_the_live_route_is_a_503(tmp_path: Path) -> None:
    journal = tmp_path / "torn.jsonl"
    journal.write_text('{"kind": "run"}\n{"kind": "tick"', encoding="utf-8")

    status, body, _ = _request(_live_viewer(journal), _get(LIVE_PATH))

    assert status == "HTTP/1.0 503 Service Unavailable"
    assert b"journal line 2: not valid JSON" in body
    assert b"<!DOCTYPE html>" not in body


def test_serve_with_live_serves_the_board_and_the_default_still_404s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = _live_journal(tmp_path / "live.jsonl", ticks=1)
    served: dict[str, object] = {}

    def fake_run(server: ViewerServer, *, once: bool = False) -> int:
        served["status"], served["body"] = _request_on_open_server(server, LIVE_PATH)
        return 0

    monkeypatch.setattr(serve_module, "run", fake_run)

    assert serve(journal, port=0, once=True, out=StringIO(), live=True) == 0
    assert served["status"] == "HTTP/1.0 200 OK"
    assert bytes(served["body"]).startswith(b"<!DOCTYPE html>")  # type: ignore[arg-type]

    # And with the default ``live=False`` the same route is refused, proving the
    # flag is the live route's switch and not decoration.
    assert serve(journal, port=0, once=True, out=StringIO()) == 0
    assert served["status"] == "HTTP/1.0 404 Not Found"
    assert served["body"] == b"not found\n"


# -- the threaded entry point: serving while the game owns the main thread -------


def test_the_threaded_viewer_serves_live_from_another_thread_and_stops(
    tmp_path: Path,
) -> None:
    journal = _live_journal(tmp_path / "live.jsonl", ticks=2)
    out = StringIO()
    ready: list[str] = []
    viewer = ThreadedViewer(
        journal, port=0, out=out, retry_delay_s=0.0, on_ready=ready.append
    )

    url = viewer.start()
    host, port = viewer.address
    try:
        # The URL on_ready saw is the one the socket really answers on, live path
        # and all: the same bind-and-compute serve uses, reused rather than recopied.
        assert ready == [url]
        assert url == f"http://{host}:{port}{LIVE_PATH}"

        # The request is sent from this thread; the page is built on the daemon.
        client = socket.create_connection((host, port))
        try:
            client.sendall(_get(LIVE_PATH))
            response = _read_all(client)
        finally:
            client.close()
        head, _, body = response.partition(b"\r\n\r\n")
        assert head.startswith(b"HTTP/1.0 200 OK")
        assert body.startswith(b"<!DOCTYPE html>")
    finally:
        viewer.stop()

    lines = out.getvalue().splitlines()
    assert lines[0] == f"serving {journal} at {url}"
    # The honest second line: the game is the window at 60 fps, this page is the
    # board at the decision tick rate. Different from serve's line, on purpose.
    assert "60 fps" in lines[1]
    assert "decision" in lines[1]

    # Stop closed the socket: the port it bound now refuses a further connection.
    with pytest.raises(OSError):
        socket.create_connection((host, port), timeout=0.5).close()


def test_a_raising_on_ready_on_the_threaded_path_closes_the_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = _live_journal(tmp_path / "live.jsonl")
    servers: list[object] = []

    class _RecordingServer:
        def __init__(self, address: tuple[str, int], handler: object) -> None:
            self.server_address = ("127.0.0.1", 8000)
            self.closed = False
            servers.append(self)

        def server_close(self) -> None:
            self.closed = True

    monkeypatch.setattr(serve_module, "ViewerServer", _RecordingServer)

    def explode(url: str) -> None:
        raise RuntimeError("the callback is the caller's business, not the viewer's")

    viewer = ThreadedViewer(journal, port=0, out=StringIO(), on_ready=explode)

    with pytest.raises(RuntimeError):
        viewer.start()

    # A raising callback is the caller's failure, not a licence to leak the port:
    # the socket is released before the exception leaves ``start``.
    assert servers[0].closed is True  # type: ignore[attr-defined]

