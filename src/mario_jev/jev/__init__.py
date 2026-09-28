"""The Jev decision layer: the four tick questions, the HTTP client, and the answers.

``mario_jev.jev`` is the only package that speaks HTTP. It builds the question
set for a tick, sends it to the Jev System One endpoint in one request, and turns
the reply into typed answers the policy can gate on. The credential lives only in
the outgoing ``Authorization`` header and never reaches a ``repr``, a journal
entry, or an exception.
"""

from .client import (
    BACKOFF_BASE_S,
    BACKOFF_CAP_S,
    RETRYABLE_STATUSES,
    JevClient,
)
from .errors import (
    JevAuthError,
    JevError,
    JevProtocolError,
    JevRequestError,
    JevUnavailable,
)
from .http import (
    USER_AGENT,
    HttpResult,
    JevRequest,
    RedactedHeader,
    default_transport,
)
from .questions import (
    TICK_QUESTION_IDS,
    build_questions,
)
from .responses import (
    ChoiceAnswer,
    JevResponse,
    NoulAnswer,
    ScoreAnswer,
    Usage,
    parse_response,
)

__all__ = [
    "BACKOFF_BASE_S",
    "BACKOFF_CAP_S",
    "ChoiceAnswer",
    "HttpResult",
    "JevAuthError",
    "JevClient",
    "JevError",
    "JevProtocolError",
    "JevRequest",
    "JevRequestError",
    "JevResponse",
    "JevUnavailable",
    "NoulAnswer",
    "RETRYABLE_STATUSES",
    "RedactedHeader",
    "ScoreAnswer",
    "TICK_QUESTION_IDS",
    "USER_AGENT",
    "Usage",
    "build_questions",
    "default_transport",
    "parse_response",
]
