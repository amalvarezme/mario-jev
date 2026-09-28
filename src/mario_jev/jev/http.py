"""The transport seam: a request value, a result value, and the one real call.

Everything above this module speaks in :class:`JevRequest` and
:class:`HttpResult`; only :func:`default_transport` knows about ``requests``. That
seam is what lets the tests drive every retry path without touching the network,
and it is also the only place a ``requests`` exception can ever appear.

Why the ``User-Agent`` matters
------------------------------

Cloudflare fronts this API. A request without a realistic ``User-Agent`` is
rejected with **HTTP 403** and the body ``error code: 1010`` *before* it ever
reaches Jev, so a bare client looks like an authentication failure when it is
really a bot filter. The default :data:`USER_AGENT` is deliberately not the
``python-requests`` default for that reason.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping, NoReturn

import requests

#: A realistic, project-identifying User-Agent. See the module docstring: the
#: ``requests`` default is filtered by Cloudflare with 403/1010 before Jev sees it.
USER_AGENT = "mario-jev/0.1 (+https://github.com/; python-requests)"

#: The placeholder an Authorization header renders as in a journal-safe view.
REDACTED_AUTHORIZATION = "Bearer <redacted>"

__all__ = [
    "HttpResult",
    "JevRequest",
    "REDACTED_AUTHORIZATION",
    "RedactedHeader",
    "USER_AGENT",
    "default_transport",
]


class RedactedHeader:
    """A header value that must be sent, but must never be rendered or serialised.

    The credential has to travel in the outgoing ``Authorization`` header, so it
    cannot simply be left out of :class:`JevRequest`. Wrapping it is what makes the
    one inviolable rule of this project — the key never reaches a log, a traceback,
    a journal entry, a file, or a commit — a property of the type rather than a
    matter of discipline:

    * ``repr()``, ``str()`` and every f-string render the redacted placeholder;
    * concatenation and ``join`` reject it, because it is not a ``str``;
    * ``dataclasses.asdict(request)`` copies the wrapper, not the key, so a
      ``json.dumps`` of that copy raises ``TypeError`` — a loud failure instead of
      a silent leak into a journal;
    * ``pickle`` is refused outright: a slot-state serialisation would otherwise
      copy the key out verbatim, so :meth:`__reduce__` raises ``TypeError`` rather
      than emit it;
    * :meth:`reveal` is the only way to obtain the real value, and
      :func:`default_transport` is the only caller.

    Copying (``copy.copy``/``copy.deepcopy``) still works: a copy is just another
    wrapper around the same private value, and it stays redacted.

    Equality with a plain string still holds, so a caller can assert that the header
    carries the value it expects without that value becoming printable.
    """

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        """The real header value. Only the transport should call this."""
        return self._value

    def __reduce__(self) -> NoReturn:
        """Refuse to be pickled: a pickle copy would carry the credential verbatim.

        ``pickle`` serialises slot state directly, so without this it would emit
        the real ``Bearer`` value — the same leak that ``dataclasses.asdict`` +
        ``json.dumps`` already refuses. The message names the type and never the
        value.
        """
        raise TypeError(
            "RedactedHeader refuses to be pickled: it wraps the Jev credential "
            "and must never be serialised"
        )

    def __copy__(self) -> "RedactedHeader":
        """A shallow copy is a fresh wrapper around the same private value."""
        return RedactedHeader(self._value)

    def __deepcopy__(self, memo: dict[int, object]) -> "RedactedHeader":
        """A deep copy keeps the credential wrapped and redacted.

        Copying must not raise — unlike pickling, ``copy`` runs on requests in
        ordinary code — so this returns a new wrapper with the same value instead
        of falling through to the refusing :meth:`__reduce__`.
        """
        clone = RedactedHeader(self._value)
        memo[id(self)] = clone
        return clone

    def __repr__(self) -> str:
        return REDACTED_AUTHORIZATION

    def __str__(self) -> str:
        return REDACTED_AUTHORIZATION

    def __format__(self, format_spec: str) -> str:
        return format(REDACTED_AUTHORIZATION, format_spec)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RedactedHeader):
            return other._value == self._value
        if isinstance(other, str):
            return self._value == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._value)


def _redacted_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """A copy of ``headers`` with any Authorization value replaced by a placeholder."""
    redacted: dict[str, str] = {}
    for name, value in headers.items():
        if name.lower() == "authorization":
            redacted[name] = REDACTED_AUTHORIZATION
        else:
            redacted[name] = value
    return redacted


@dataclass(frozen=True, slots=True, repr=False)
class JevRequest:
    """One HTTP request, as a value.

    ``headers`` carries a :class:`RedactedHeader` for ``Authorization`` — the
    request has to send the credential — so ``repr`` and :meth:`as_dict` render the
    redacted form and are the safe way to journal or log this object. Wrapping the
    value rather than storing a plain string also closes the one path a plain string
    left open: ``dataclasses.asdict`` on this object copies the wrapper, and
    serialising that copy fails instead of writing the key out.
    """

    url: str
    #: Header values. ``Authorization`` holds a :class:`RedactedHeader`, not a
    #: plain string, so no accidental serialiser (``json``, ``pickle``) can copy
    #: the key out of it.
    headers: Mapping[str, str | RedactedHeader]
    json_body: Mapping[str, object]
    timeout_s: float

    def __post_init__(self) -> None:
        """Refuse a plain-string Authorization header.

        The wrapper only protects the credential if nothing can put an unwrapped
        value here, so this is enforced rather than merely documented: a plain
        string is exactly what ``dataclasses.asdict`` copies out verbatim. The
        message names the offending type and never the value.
        """
        for name, value in self.headers.items():
            if name.lower() == "authorization" and not isinstance(value, RedactedHeader):
                raise ValueError(
                    "a JevRequest Authorization header must be a RedactedHeader, "
                    f"not a {type(value).__name__}: a plain string is copied "
                    "verbatim by dataclasses.asdict and would leak the credential"
                )

    def as_dict(self) -> dict[str, object]:
        """A journal-safe view: the Authorization header renders as a placeholder."""
        return {
            "url": self.url,
            "headers": _redacted_headers(self.headers),
            "json_body": dict(self.json_body),
            "timeout_s": self.timeout_s,
        }

    def __repr__(self) -> str:
        view = self.as_dict()
        return (
            "JevRequest("
            f"url={view['url']!r}, "
            f"headers={view['headers']!r}, "
            f"json_body={view['json_body']!r}, "
            f"timeout_s={view['timeout_s']!r})"
        )


@dataclass(frozen=True, slots=True)
class HttpResult:
    """One HTTP outcome, or one transport failure.

    ``error`` is the transport-level discriminator: when it is set the call never
    produced an HTTP response (timeout, DNS failure, refused connection), and
    ``status`` is deliberately ``0`` so it cannot be mistaken for a real status.
    """

    status: int
    body: str
    elapsed_ms: float
    error: str | None = None


def _short_error(exc: requests.RequestException) -> str:
    """A short, header-free description of a transport exception.

    Only the exception's own message is used; the request headers are never
    interpolated, so a credential that lives in a header cannot leak through an
    error string.
    """
    message = str(exc).strip()
    if not message:
        return type(exc).__name__
    described = f"{type(exc).__name__}: {message}"
    # Keep a pathological message from bloating a journal line.
    return described[:300]


def default_transport(request: JevRequest) -> HttpResult:
    """Send one request with ``requests``, and never let its exception escape.

    A short-lived session is used per call: the retry loop lives in the client,
    so a session here would only hold onto a stale connection across a backoff.
    """
    started = time.perf_counter()
    # Reaching for the real credential happens here and nowhere else.
    headers = {
        name: (value.reveal() if isinstance(value, RedactedHeader) else value)
        for name, value in request.headers.items()
    }
    try:
        with requests.Session() as session:
            response = session.post(
                request.url,
                headers=headers,
                json=dict(request.json_body),
                timeout=request.timeout_s,
            )
    except requests.RequestException as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return HttpResult(status=0, body="", elapsed_ms=elapsed_ms, error=_short_error(exc))

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return HttpResult(status=response.status_code, body=response.text, elapsed_ms=elapsed_ms)
