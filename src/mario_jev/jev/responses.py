"""The typed answers, and the strict parser that produces them.

Loose dicts through the rest of the pipeline would defeat the point of the
question contract: the policy gates on ``confidence``, the journal serialises the
answers, and neither should have to remember which keys a given question type
happens to carry. So every answer becomes a dataclass here, and the parser refuses
anything that does not match the documented shape.

Two facts about the live API are load-bearing and easy to get wrong:

* ``score`` is *fractional*. ``gap_severity`` returns ``1.3`` for "between small
  and wide", so the value is never coerced to an ``int``.
* ``noul`` carries **no** ``confidence``. That is exactly why ``lethal_ahead``
  can never share the confidence gate the choice/score answers use, and why
  :class:`NoulAnswer` has no such field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .errors import JevProtocolError

__all__ = [
    "ChoiceAnswer",
    "JevResponse",
    "NoulAnswer",
    "ScoreAnswer",
    "Usage",
    "parse_response",
]


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting for one call."""

    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class ChoiceAnswer:
    """The answer to a ``choice`` question. ``choice`` is an offered candidate."""

    type: str
    choice: str
    confidence: float
    probabilities: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class ScoreAnswer:
    """The answer to a ``score`` question.

    ``score`` may be fractional: ``1.3`` is a valid value between "small gap" and
    "wide gap", and truncating it to ``1`` would silently change the answer.
    """

    type: str
    score: float
    confidence: float
    legend: Mapping[str, str]
    probabilities: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class NoulAnswer:
    """The answer to a ``noul`` question.

    There is deliberately no ``confidence`` field: the live API does not return
    one for ``noul``.
    """

    type: str
    noul: float


@dataclass(frozen=True, slots=True)
class JevResponse:
    """One parsed response, keyed by the question ids that were asked."""

    model: str
    answers: Mapping[str, ChoiceAnswer | ScoreAnswer | NoulAnswer]
    usage: Usage
    latency_ms: float

    def _require(self, question_id: str, expected: type) -> object:
        try:
            answer = self.answers[question_id]
        except KeyError as exc:
            raise KeyError(f"no answer for question {question_id!r}") from exc
        if not isinstance(answer, expected):
            raise JevProtocolError(
                f"answer {question_id!r} is a {type(answer).__name__}, "
                f"not a {expected.__name__}"
            )
        return answer

    def choice(self, question_id: str) -> ChoiceAnswer:
        """The :class:`ChoiceAnswer` for one question id."""
        answer = self._require(question_id, ChoiceAnswer)
        assert isinstance(answer, ChoiceAnswer)
        return answer

    def score(self, question_id: str) -> ScoreAnswer:
        """The :class:`ScoreAnswer` for one question id."""
        answer = self._require(question_id, ScoreAnswer)
        assert isinstance(answer, ScoreAnswer)
        return answer

    def noul(self, question_id: str) -> NoulAnswer:
        """The :class:`NoulAnswer` for one question id."""
        answer = self._require(question_id, NoulAnswer)
        assert isinstance(answer, NoulAnswer)
        return answer

    def as_dict(self) -> dict[str, object]:
        """Journal-ready plain data. Carries no credential material."""
        return {
            "model": self.model,
            "latency_ms": self.latency_ms,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
            },
            "answers": {
                question_id: _answer_as_dict(answer)
                for question_id, answer in self.answers.items()
            },
        }


def _answer_as_dict(answer: ChoiceAnswer | ScoreAnswer | NoulAnswer) -> dict[str, object]:
    if isinstance(answer, ChoiceAnswer):
        return {
            "type": answer.type,
            "choice": answer.choice,
            "confidence": answer.confidence,
            "probabilities": dict(answer.probabilities),
        }
    if isinstance(answer, ScoreAnswer):
        return {
            "type": answer.type,
            "score": answer.score,
            "confidence": answer.confidence,
            "legend": dict(answer.legend),
            "probabilities": dict(answer.probabilities),
        }
    return {"type": answer.type, "noul": answer.noul}


# -- strict parsing -----------------------------------------------------------------


def _number(value: object, *, field: str, question_id: str) -> float:
    # bool is an int subclass, and a JSON true where a number belongs is a shape
    # violation, not a 1.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JevProtocolError(
            f"question {question_id!r}: {field} must be a number, "
            f"got {type(value).__name__}"
        )
    return float(value)


def _string(value: object, *, field: str, question_id: str) -> str:
    if not isinstance(value, str):
        raise JevProtocolError(
            f"question {question_id!r}: {field} must be a string, "
            f"got {type(value).__name__}"
        )
    return value


def _mapping(value: object, *, field: str, question_id: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise JevProtocolError(
            f"question {question_id!r}: {field} must be an object, "
            f"got {type(value).__name__}"
        )
    return value


def _required_map(value: object, *, field: str, question_id: str) -> Mapping[str, object]:
    """A required, non-empty object field.

    A missing distribution must not be masked into ``{}``: the journal contract
    records the answer *with* its probabilities (and, for ``score``, its legend),
    so an absent or empty one is a contract break, not an empty answer.
    """
    if value is None:
        raise JevProtocolError(
            f"question {question_id!r}: {field} is required and must be a "
            "non-empty object, got none"
        )
    mapping = _mapping(value, field=field, question_id=question_id)
    if not mapping:
        raise JevProtocolError(
            f"question {question_id!r}: {field} must be a non-empty object, "
            "got an empty one"
        )
    return mapping


def _float_map(value: object, *, field: str, question_id: str) -> dict[str, float]:
    mapping = _required_map(value, field=field, question_id=question_id)
    return {
        str(key): _number(item, field=f"{field}[{key}]", question_id=question_id)
        for key, item in mapping.items()
    }


def _str_map(value: object, *, field: str, question_id: str) -> dict[str, str]:
    mapping = _required_map(value, field=field, question_id=question_id)
    return {str(key): str(item) for key, item in mapping.items()}


def _parse_usage(value: object) -> Usage:
    if not isinstance(value, Mapping):
        raise JevProtocolError("response body is missing a 'usage' object")
    input_tokens = _number(value.get("input_tokens"), field="input_tokens", question_id="usage")
    output_tokens = _number(
        value.get("output_tokens"), field="output_tokens", question_id="usage"
    )
    return Usage(input_tokens=int(input_tokens), output_tokens=int(output_tokens))


def _parse_choice(
    question_id: str,
    answer: Mapping[str, object],
    allowed: Sequence[str] | None,
) -> ChoiceAnswer:
    choice = _string(answer.get("choice"), field="choice", question_id=question_id)
    if allowed is not None and choice not in allowed:
        raise JevProtocolError(
            f"question {question_id!r}: choice {choice!r} is not one of the "
            "candidates that were offered"
        )
    confidence = _number(answer.get("confidence"), field="confidence", question_id=question_id)
    probabilities = _float_map(
        answer.get("probabilities"), field="probabilities", question_id=question_id
    )
    return ChoiceAnswer(
        type="choice",
        choice=choice,
        confidence=confidence,
        probabilities=probabilities,
    )


def _parse_score(question_id: str, answer: Mapping[str, object]) -> ScoreAnswer:
    # ``score`` stays a float on purpose: 1.3 is a real answer, not a rounding bug.
    score = _number(answer.get("score"), field="score", question_id=question_id)
    confidence = _number(answer.get("confidence"), field="confidence", question_id=question_id)
    legend = _str_map(answer.get("legend"), field="legend", question_id=question_id)
    probabilities = _float_map(
        answer.get("probabilities"), field="probabilities", question_id=question_id
    )
    return ScoreAnswer(
        type="score",
        score=score,
        confidence=confidence,
        legend=legend,
        probabilities=probabilities,
    )


def _parse_noul(question_id: str, answer: Mapping[str, object]) -> NoulAnswer:
    # No confidence is demanded: the live API does not send one for noul.
    noul = _number(answer.get("noul"), field="noul", question_id=question_id)
    if not 0.0 <= noul <= 1.0:
        raise JevProtocolError(
            f"question {question_id!r}: noul must be within [0, 1], got {noul!r}"
        )
    return NoulAnswer(type="noul", noul=noul)


def _parse_answer(
    question_id: str,
    expected_type: str,
    raw: object,
    allowed_choices: Mapping[str, Sequence[str]] | None,
) -> ChoiceAnswer | ScoreAnswer | NoulAnswer:
    answer = _mapping(raw, field="answer", question_id=question_id)
    actual_type = answer.get("type")
    if actual_type != expected_type:
        raise JevProtocolError(
            f"question {question_id!r}: expected type {expected_type!r}, "
            f"got {actual_type!r}"
        )
    if expected_type == "choice":
        allowed = None if allowed_choices is None else allowed_choices.get(question_id)
        return _parse_choice(question_id, answer, allowed)
    if expected_type == "score":
        return _parse_score(question_id, answer)
    if expected_type == "noul":
        return _parse_noul(question_id, answer)
    raise JevProtocolError(
        f"question {question_id!r}: unsupported requested type {expected_type!r}"
    )


def parse_response(
    payload: Mapping[str, object],
    *,
    requested: Mapping[str, str],
    latency_ms: float,
    allowed_choices: Mapping[str, Sequence[str]] | None = None,
) -> JevResponse:
    """Parse a response body against the question ids and types that were sent.

    Args:
        payload: the decoded JSON body.
        requested: question id -> expected answer type, in the order asked.
        latency_ms: the measured round-trip time, carried straight into the result.
        allowed_choices: optional question id -> the candidate names that were
            offered, so a ``choice`` outside them is rejected as a protocol break.

    Raises:
        JevProtocolError: when the body is not an object, is missing an answer for
            a requested id, answers a different type, or carries a malformed value
            (including a missing or empty ``probabilities``/``legend``).
    """
    if not isinstance(payload, Mapping):
        raise JevProtocolError(
            f"response body must be a JSON object, got {type(payload).__name__}"
        )
    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise JevProtocolError("response body is missing a non-empty string 'model'")

    answers_value = payload.get("answers")
    if not isinstance(answers_value, Mapping):
        raise JevProtocolError("response body is missing an 'answers' object")
    missing = [question_id for question_id in requested if question_id not in answers_value]
    if missing:
        raise JevProtocolError(
            "response is missing answers for: " + ", ".join(sorted(missing))
        )

    usage = _parse_usage(payload.get("usage"))
    answers: dict[str, ChoiceAnswer | ScoreAnswer | NoulAnswer] = {}
    for question_id, expected_type in requested.items():
        answers[question_id] = _parse_answer(
            question_id, expected_type, answers_value[question_id], allowed_choices
        )
    return JevResponse(
        model=model,
        answers=answers,
        usage=usage,
        latency_ms=float(latency_ms),
    )
