"""The Jev HTTP client: one request per tick, with the retry policy that fits it.

The client owns three things the rest of the pipeline must not have to think
about:

* **the request shape** — one POST carrying the model, the projected state, and
  the four questions;
* **the retry policy** — only retryable failures are retried, with capped
  exponential backoff, and the credential failures never are, and the returned
  response carries the number of attempts it actually cost;
* **key containment** — the API key is held only inside the ``Authorization``
  :class:`RedactedHeader` and never in a plain attribute, so it appears in no
  ``repr``, no ``vars()`` view, no ``.redacted`` view, no pickle, no journal
  projection, and no exception, while still reaching the outgoing request.

The transport is injectable so the tests can drive every path without a socket,
and the sleeper is injectable so backoff never actually sleeps during a test run.
"""

from __future__ import annotations

import json
import time
from typing import Callable, Mapping

from mario_jev.config import Settings, redact_url_userinfo
from mario_jev.state import MarioState

from .errors import (
    JevAuthError,
    JevProtocolError,
    JevRequestError,
    JevUnavailable,
)
from .http import USER_AGENT, HttpResult, JevRequest, RedactedHeader, default_transport
from .questions import build_questions
from .responses import JevResponse, parse_response

#: Statuses that are worth a second attempt. Every other 4xx is a bug in the
#: request, not a transient fault, and is never retried.
RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504, 529})

#: Base delay for the exponential backoff, in seconds.
BACKOFF_BASE_S = 0.5

#: Ceiling for the exponential backoff, in seconds. A real-time agent cannot
#: afford an unbounded wait, so the doubling is clipped here.
BACKOFF_CAP_S = 8.0

Transport = Callable[[JevRequest], HttpResult]
Sleeper = Callable[[float], None]

__all__ = [
    "BACKOFF_BASE_S",
    "BACKOFF_CAP_S",
    "JevClient",
    "RETRYABLE_STATUSES",
]


def _backoff_delay(attempt: int) -> float:
    """The capped exponential backoff for a zero-based retry ``attempt``."""
    return min(BACKOFF_BASE_S * 2**attempt, BACKOFF_CAP_S)


def _validation_detail(body: str) -> tuple[Mapping[str, object], ...]:
    """The server's validation entries from a 422 body, best-effort.

    A malformed or absent body yields no entries rather than a parse error: the
    caller already knows the request failed, and losing the field names is worse
    than losing the exception.
    """
    try:
        payload = json.loads(body)
    except (ValueError, TypeError):
        return ()
    if not isinstance(payload, Mapping):
        return ()
    detail = payload.get("detail")
    if isinstance(detail, list):
        return tuple(item for item in detail if isinstance(item, Mapping))
    if isinstance(detail, str):
        return ({"msg": detail},)
    return ()


class JevClient:
    """A Jev client bound to one endpoint, one model, and one credential.

    The credential is held only inside a
    :class:`~mario_jev.jev.http.RedactedHeader` and never as a plain attribute, so
    it is absent from :meth:`redacted`, :func:`repr`, ``vars()`` and every
    serialiser; only the outgoing request reveals it.

    A credential carried by the *endpoint* (``https://user:password@host``) is a
    second, weaker case, and it is handled differently on purpose. It is absent
    from :meth:`redacted` and therefore from :func:`repr`, because those are the
    views that reach a log; it is present in ``vars()``, because redaction here is
    a projection of the object rather than a scrub of it, and no projection can
    make a stored attribute forget itself. Nothing in this package reads it, so the
    leak needs a debugger or a traceback rather than a normal call path — but that
    is still a narrower guarantee than the key's, and stating it beats implying it.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        api_key: str | None,
        timeout_s: float,
        max_attempts: int,
        transport: Transport | None = None,
        sleeper: Sleeper | None = None,
        user_agent: str = USER_AGENT,
    ) -> None:
        self._endpoint = endpoint
        self._model = model
        # The raw key never lives in the instance: the wrapper is both the
        # availability check and the outgoing header, and it refuses to render
        # or serialise. ``repr(vars(client))`` and ``pickle`` therefore cannot
        # copy the credential out.
        self._authorization: RedactedHeader | None = (
            None if api_key is None else RedactedHeader(f"Bearer {api_key}")
        )
        self._timeout_s = timeout_s
        self._max_attempts = max_attempts
        self._transport: Transport = default_transport if transport is None else transport
        self._sleeper: Sleeper = time.sleep if sleeper is None else sleeper
        self._user_agent = user_agent
        self._closed = False

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        transport: Transport | None = None,
        sleeper: Sleeper | None = None,
    ) -> "JevClient":
        """Build a client from :class:`~mario_jev.config.Settings`."""
        return cls(
            endpoint=settings.jev_endpoint,
            model=settings.jev_model,
            api_key=settings.jev_api_key,
            timeout_s=settings.jev_timeout_s,
            max_attempts=settings.jev_max_attempts,
            transport=transport,
            sleeper=sleeper,
        )

    @property
    def redacted(self) -> dict[str, object]:
        """A diagnostics-safe view: endpoint, model, user agent, availability. No key.

        The endpoint goes through :func:`~mario_jev.config.redact_url_userinfo` for
        the same reason the journal header does. A key is not the only credential a
        URL can carry: ``https://alice:hunter2@host/v1`` is a legitimate endpoint to
        accept, and this view reaches a log through :func:`repr` or a traceback, so
        emitting it verbatim would put a password there.
        """
        return {
            "endpoint": redact_url_userinfo(self._endpoint),
            "model": self._model,
            "user_agent": self._user_agent,
            "jev_available": self._authorization is not None,
        }

    def __repr__(self) -> str:
        return f"JevClient({self.redacted!r})"

    def close(self) -> None:
        """Release the client. Idempotent; there is no persistent session to drop."""
        self._closed = True

    def __enter__(self) -> "JevClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def ask(
        self,
        questions: Mapping[str, Mapping[str, object]],
        state: MarioState,
    ) -> JevResponse:
        """Ask one batch of questions about one state, in a single HTTP request.

        Raises:
            JevAuthError: no credential is configured. This layer never sends an
                unauthenticated request, because the policy gates on
                ``settings.jev_available`` before it gets here.
        """
        if self._authorization is None:
            raise JevAuthError("no Jev API key is configured", status=0)
        request = self._build_request(questions, state)
        result, attempts = self._send(request)
        return self._handle(result, questions, attempts)

    def decide(self, state: MarioState) -> JevResponse:
        """Build the four tick questions and ask them about ``state``."""
        return self.ask(build_questions(state), state)

    # -- internals ------------------------------------------------------------------

    def _build_request(
        self,
        questions: Mapping[str, Mapping[str, object]],
        state: MarioState,
    ) -> JevRequest:
        body = {
            "model": self._model,
            "state": state.to_jev_state(),
            "questions": questions,
        }
        authorization = self._authorization
        assert authorization is not None  # ask() refuses a client without a credential
        headers: dict[str, str | RedactedHeader] = {
            # The wrapper, reused as-is: the credential must be sent, but no
            # accidental serialiser may copy it into a journal.
            "Authorization": authorization,
            "Content-Type": "application/json",
            "User-Agent": self._user_agent,
        }
        return JevRequest(
            url=self._endpoint,
            headers=headers,
            json_body=body,
            timeout_s=self._timeout_s,
        )

    def _send(self, request: JevRequest) -> tuple[HttpResult, int]:
        """The result, and how many transport calls it took to get it.

        The count is returned rather than stored because it belongs to one call,
        not to the client, and because it is the only record that a tick paid for
        a failed attempt before the one that worked. A call that exhausts its
        attempts raises instead, and so has no count to carry.
        """
        attempt = 0
        while True:
            result = self._transport(request)
            if result.error is None and result.status not in RETRYABLE_STATUSES:
                return result, attempt + 1
            if attempt + 1 >= self._max_attempts:
                raise JevUnavailable(self._unavailable_message(result, attempt + 1))
            self._sleeper(_backoff_delay(attempt))
            attempt += 1

    @staticmethod
    def _unavailable_message(result: HttpResult, attempts: int) -> str:
        if result.error is not None:
            detail = f"transport error: {result.error}"
        else:
            detail = f"HTTP {result.status}"
        return f"Jev unavailable after {attempts} attempt(s): {detail}"

    def _handle(
        self,
        result: HttpResult,
        questions: Mapping[str, Mapping[str, object]],
        attempts: int,
    ) -> JevResponse:
        status = result.status
        if 200 <= status < 300:
            return self._parse(result, questions, attempts)
        if status in (401, 403):
            raise JevAuthError(f"Jev rejected the credential (HTTP {status})", status=status)
        if status == 422:
            raise JevRequestError(
                "Jev rejected the request as invalid (HTTP 422)",
                status=422,
                detail=_validation_detail(result.body),
            )
        if 400 <= status < 500:
            raise JevRequestError(f"Jev rejected the request (HTTP {status})", status=status)
        if status >= 500:
            raise JevUnavailable(f"Jev returned HTTP {status}")
        raise JevProtocolError(f"unexpected HTTP status {status}")

    def _parse(
        self,
        result: HttpResult,
        questions: Mapping[str, Mapping[str, object]],
        attempts: int,
    ) -> JevResponse:
        try:
            payload = json.loads(result.body)
        except (ValueError, TypeError) as exc:
            raise JevProtocolError(f"Jev returned a non-JSON 2xx body: {exc}") from exc
        requested = {
            question_id: str(question["type"]) for question_id, question in questions.items()
        }
        allowed_choices: dict[str, list[str]] = {}
        for question_id, question in questions.items():
            if question.get("type") != "choice":
                continue
            criteria = question.get("criteria")
            if isinstance(criteria, Mapping):
                allowed_choices[question_id] = list(criteria)
        return parse_response(
            payload,
            requested=requested,
            latency_ms=result.elapsed_ms,
            allowed_choices=allowed_choices,
            attempts=attempts,
        )
