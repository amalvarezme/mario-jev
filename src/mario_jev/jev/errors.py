"""The Jev failure taxonomy.

Four failures are possible when this package talks to Jev, and they are told
apart because the caller treats them differently:

* :class:`JevAuthError` — the credential was rejected (401/403). Retrying the
  same key cannot help, so this never loops.
* :class:`JevRequestError` — the request itself was refused (422 validation, or
  any other non-retryable 4xx). Retrying an invalid body cannot help either.
* :class:`JevProtocolError` — the server answered 2xx but the body does not match
  the documented shape. This is a contract break, not a transient fault.
* :class:`JevUnavailable` — the transport failed, or the server answered
  429/529/5xx. A retryable status is retried and raised once retries are
  exhausted, while a non-retryable 5xx (e.g. 501) is raised immediately after one
  call. This is the only failure the caller may retry at a higher level.

The API key is the one value that must never reach a log or a traceback, so no
message or attribute built here ever interpolates it: those two exceptions carry
a status code and, for 422, the parsed ``detail`` entries instead.
"""

from __future__ import annotations

from typing import Mapping

__all__ = [
    "JevAuthError",
    "JevError",
    "JevProtocolError",
    "JevRequestError",
    "JevUnavailable",
]


class JevError(Exception):
    """Base class for every failure raised by :mod:`mario_jev.jev`."""


def _field_names(detail: tuple[Mapping[str, object], ...]) -> tuple[str, ...]:
    """The offending field names carried by a validation ``detail`` list.

    FastAPI-style validation bodies report ``{"loc": ["body", "questions"], ...}``.
    The last string in ``loc`` is the field the server actually rejected, which is
    the part worth naming; anything else in the entry is noise for a caller.
    """
    names: list[str] = []
    for entry in detail:
        loc = entry.get("loc")
        if isinstance(loc, (list, tuple)):
            for part in reversed(loc):
                if isinstance(part, str):
                    names.append(part)
                    break
        elif isinstance(loc, str):
            names.append(loc)
    return tuple(names)


class JevAuthError(JevError):
    """The credential was rejected, or no credential was configured.

    ``status`` is the HTTP status that triggered it, or ``0`` when the failure
    happened before any request was made (a missing key).
    """

    def __init__(self, message: str, *, status: int) -> None:
        super().__init__(message)
        self.status = status


class JevRequestError(JevError):
    """A non-retryable request failure: 422, or another non-retryable 4xx.

    ``detail`` holds the server's validation entries verbatim, so a caller can
    name the offending field with :attr:`fields` instead of guessing.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int,
        detail: tuple[Mapping[str, object], ...] = (),
    ) -> None:
        super().__init__(message)
        self.status = status
        self.detail = tuple(detail)

    @property
    def fields(self) -> tuple[str, ...]:
        """The field names the server rejected, derived from ``detail``."""
        return _field_names(self.detail)


class JevProtocolError(JevError):
    """A 2xx response whose body violates the documented shape."""


class JevUnavailable(JevError):
    """The transport failed, or the server answered 429/529/5xx.

    A retryable status is raised once retries are exhausted; a non-retryable 5xx
    is raised after a single call.
    """
