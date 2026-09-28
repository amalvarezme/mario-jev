"""The Jev client: question construction, the wire request, retries, and containment.

There is no network here. The transport is a small recorder that hands back
queued :class:`HttpResult` values, and the sleeper only records the delays it was
asked to wait, so the retry policy can be asserted exactly and quickly. The one
exception is the single ``live_jev`` test at the end, which is skipped whenever
no credential is configured.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import pickle

import pytest

from mario_jev.config import Settings
from mario_jev.jev import (
    BACKOFF_BASE_S,
    BACKOFF_CAP_S,
    RETRYABLE_STATUSES,
    TICK_QUESTION_IDS,
    USER_AGENT,
    ChoiceAnswer,
    HttpResult,
    JevAuthError,
    JevClient,
    JevProtocolError,
    JevRequest,
    JevRequestError,
    JevResponse,
    JevUnavailable,
    NoulAnswer,
    RedactedHeader,
    ScoreAnswer,
    Usage,
    build_questions,
    parse_response,
)
from mario_jev.state import (
    CandidateAssessment,
    MarioState,
    NearestEnemy,
    PlayerFeatures,
    Progress,
    TerrainAhead,
    TerminalOutcome,
)

#: A synthetic key. The tests never use a real credential.
SENTINEL_KEY = "apikey_SENTINEL_do_not_leak"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
TIMEOUT_S = 20.0

#: The verified live answer, kept verbatim. Two values in it are load-bearing:
#: ``gap_severity.score`` is fractional (1.3) and ``lethal_ahead`` has no confidence.
VERIFIED_RESPONSE: dict[str, object] = {
    "model": "jev-1.13.0",
    "answers": {
        "best_action": {
            "type": "choice",
            "choice": "right+A",
            "confidence": 0.73,
            "probabilities": {
                "noop": 0.0,
                "right+B": 0.0,
                "right": 0.21,
                "right+A": 0.79,
                "left": 0.0,
            },
        },
        "risk": {
            "type": "score",
            "score": 0.48,
            "confidence": 0.52,
            "legend": {
                "0": "safe",
                "1": "minor risk",
                "2": "serious risk",
                "3": "near-certain death",
            },
            "probabilities": {"0": 0.66, "1": 0.22, "2": 0.09, "3": 0.03},
        },
        "lethal_ahead": {"type": "noul", "noul": 0.46},
        "gap_severity": {
            "type": "score",
            "score": 1.3,
            "confidence": 0.59,
            "legend": {
                "0": "no gap",
                "1": "small gap",
                "2": "wide gap",
                "3": "bottomless pit",
            },
            "probabilities": {"0": 0.05, "1": 0.61, "2": 0.32, "3": 0.02},
        },
    },
    "usage": {"input_tokens": 839, "output_tokens": 105},
}

#: The types the four questions request, in the order they are asked.
REQUESTED_TYPES = {
    "best_action": "choice",
    "risk": "score",
    "lethal_ahead": "noul",
    "gap_severity": "score",
}

DEFAULT_CANDIDATES: tuple[CandidateAssessment, ...] = (
    CandidateAssessment(
        action="NOOP", dx=0, outcome=TerminalOutcome.NO_PROGRESS, risk=0.40, final_x=40
    ),
    CandidateAssessment(
        action="right", dx=2, outcome=TerminalOutcome.LANDS_SAFE, risk=0.20, final_x=42
    ),
    CandidateAssessment(
        action="right+A", dx=13, outcome=TerminalOutcome.LANDS_SAFE, risk=0.25, final_x=53
    ),
    CandidateAssessment(
        action="right+B", dx=8, outcome=TerminalOutcome.HITS_ENEMY, risk=0.80, final_x=48
    ),
    CandidateAssessment(
        action="left", dx=-2, outcome=TerminalOutcome.NO_PROGRESS, risk=0.10, final_x=38
    ),
)


# -- builders and fakes -------------------------------------------------------------


def make_state(
    candidates: tuple[CandidateAssessment, ...] = DEFAULT_CANDIDATES,
    *,
    tick: int = 7,
) -> MarioState:
    """A real :class:`MarioState`, built directly rather than through the emulator."""
    return MarioState(
        tick=tick,
        player=PlayerFeatures(
            x=40, y=176, screen_x=40, feet_row=191, on_ground=True, facing="right"
        ),
        nearest_enemy=NearestEnemy(distance_px=64, approaching=True),
        terrain_ahead=TerrainAhead(
            gap_columns=32,
            gap_distance_px=48,
            wall_height=0,
            frames_to_jump_window=24,
            vertical_clearance=0,
            lookahead_px=64,
        ),
        progress=Progress(world="1-1", x_max=40, lives=3, time_left=400),
        speed_px_per_tick=2.0,
        candidates=candidates,
    )


class FakeTransport:
    """A recorder that returns queued results and remembers every request."""

    def __init__(self, *results: HttpResult) -> None:
        self.queued = list(results)
        self.requests: list[JevRequest] = []

    def __call__(self, request: JevRequest) -> HttpResult:
        self.requests.append(request)
        if not self.queued:
            raise AssertionError("the client sent more requests than the test queued")
        return self.queued.pop(0)

    @property
    def calls(self) -> int:
        return len(self.requests)


class RecordingSleeper:
    """Records the backoff delays instead of sleeping."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def ok_response() -> HttpResult:
    return HttpResult(status=200, body=json.dumps(VERIFIED_RESPONSE), elapsed_ms=12.5)


def status_response(status: int, body: str = "{}") -> HttpResult:
    return HttpResult(status=status, body=body, elapsed_ms=1.0)


def timeout_response() -> HttpResult:
    return HttpResult(status=0, body="", elapsed_ms=1.0, error="timeout")


def make_client(
    transport: FakeTransport,
    sleeper: RecordingSleeper | None = None,
    *,
    api_key: str | None = SENTINEL_KEY,
    max_attempts: int = 3,
) -> JevClient:
    return JevClient(
        endpoint=ENDPOINT,
        model=MODEL,
        api_key=api_key,
        timeout_s=TIMEOUT_S,
        max_attempts=max_attempts,
        transport=transport,
        sleeper=RecordingSleeper() if sleeper is None else sleeper,
    )


def payload_copy() -> dict[str, object]:
    """A deep copy of the verified body, safe to mutate in a test."""
    return json.loads(json.dumps(VERIFIED_RESPONSE))


def expected_backoff(count: int) -> list[float]:
    return [min(BACKOFF_BASE_S * 2**attempt, BACKOFF_CAP_S) for attempt in range(count)]


def assert_no_key_material(text: str) -> None:
    """Fail if ``text`` carries the sentinel key, or any usable fragment of it."""
    assert SENTINEL_KEY not in text
    for start in range(len(SENTINEL_KEY)):
        for end in range(start + 5, len(SENTINEL_KEY) + 1):
            fragment = SENTINEL_KEY[start:end]
            assert fragment not in text, fragment


def captured_request() -> JevRequest:
    """The real request a full ``decide`` sends, captured by a fake transport."""
    transport = FakeTransport(ok_response())
    make_client(transport, RecordingSleeper()).decide(make_state())
    return transport.requests[0]


# -- 1. question construction -------------------------------------------------------


def test_build_questions_returns_exactly_the_four_contracted_ids():
    questions = build_questions(make_state())

    assert tuple(questions) == TICK_QUESTION_IDS
    assert [questions[question_id]["type"] for question_id in TICK_QUESTION_IDS] == [
        "choice",
        "score",
        "noul",
        "score",
    ]
    for question in questions.values():
        assert set(question) == {"type", "instructions", "criteria"}
        assert isinstance(question["instructions"], str) and question["instructions"]


def test_best_action_criteria_are_the_candidate_names_in_rollout_order():
    state = make_state()
    criteria = build_questions(state)["best_action"]["criteria"]

    assert list(criteria) == [assessment.action for assessment in state.candidates]
    assert all(isinstance(description, str) for description in criteria.values())


def test_the_scored_questions_carry_the_contracted_scales():
    questions = build_questions(make_state())

    assert questions["risk"]["criteria"] == [
        "safe",
        "minor risk",
        "serious risk",
        "near-certain death",
    ]
    assert questions["gap_severity"]["criteria"] == [
        "no gap",
        "small gap",
        "wide gap",
        "bottomless pit",
    ]
    assert questions["lethal_ahead"]["criteria"] == {
        "true": "death is likely",
        "false": "no death expected",
    }


def test_the_instructions_name_the_numbers_being_judged():
    state = make_state()
    questions = build_questions(state)
    text = " ".join(question["instructions"] for question in questions.values())

    for assessment in state.candidates:
        assert assessment.action in text
    # The rollout numbers must reach the prompt, not just the action names.
    assert "dx=+13" in text
    assert "outcome=hits_enemy" in text
    assert "risk=0.80" in text
    # And the terrain numbers the gap question is about.
    assert "gap_columns=32" in text
    assert "gap_distance_px=48" in text


def test_build_questions_refuses_a_tick_with_no_rollouts():
    with pytest.raises(ValueError):
        build_questions(make_state(candidates=()))


# -- 2. the request on the wire -----------------------------------------------------


def test_the_client_sends_exactly_the_documented_request():
    state = make_state()
    questions = build_questions(state)
    transport = FakeTransport(ok_response())

    response = make_client(transport, RecordingSleeper()).ask(questions, state)

    assert isinstance(response, JevResponse)
    assert transport.calls == 1

    request = transport.requests[0]
    assert request.url == ENDPOINT
    assert request.headers["Authorization"] == f"Bearer {SENTINEL_KEY}"
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["User-Agent"] == USER_AGENT
    assert request.timeout_s == TIMEOUT_S

    body = request.json_body
    assert body["model"] == MODEL
    assert body["state"] == state.to_jev_state()
    assert body["questions"] == questions
    # Assert against the recorded request, not a rebuild.
    assert tuple(body["questions"]) == TICK_QUESTION_IDS
    assert body["state"]["progress"] == {
        "world": "1-1",
        "x_max": 40,
        "lives": 3,
        "time_left": 400,
    }


def test_decide_builds_the_questions_and_sends_one_request():
    state = make_state()
    transport = FakeTransport(ok_response())

    make_client(transport, RecordingSleeper()).decide(state)

    assert transport.calls == 1
    assert transport.requests[0].json_body["questions"] == build_questions(state)


def test_the_client_defaults_to_the_project_user_agent():
    # The default is the point: the requests default is filtered by Cloudflare.
    transport = FakeTransport(ok_response())
    client = make_client(transport, RecordingSleeper())
    client.decide(make_state())
    assert transport.requests[0].headers["User-Agent"] == USER_AGENT
    assert client.redacted["user_agent"] == USER_AGENT


def test_a_custom_user_agent_is_honoured():
    state = make_state()
    transport = FakeTransport(ok_response())
    client = JevClient(
        endpoint=ENDPOINT,
        model=MODEL,
        api_key=SENTINEL_KEY,
        timeout_s=TIMEOUT_S,
        max_attempts=3,
        transport=transport,
        sleeper=RecordingSleeper(),
        user_agent="mario-jev/0.1 (+test)",
    )
    client.decide(state)
    assert transport.requests[0].headers["User-Agent"] == "mario-jev/0.1 (+test)"


# -- 3. parsing the verified response -----------------------------------------------


def test_the_verified_live_response_parses_into_typed_answers():
    response = parse_response(VERIFIED_RESPONSE, requested=REQUESTED_TYPES, latency_ms=42.0)

    assert response.model == "jev-1.13.0"
    assert response.usage == Usage(839, 105)
    assert response.latency_ms == 42.0

    best = response.choice("best_action")
    assert isinstance(best, ChoiceAnswer)
    assert best.choice == "right+A"
    assert best.confidence == 0.73

    risk = response.score("risk")
    assert isinstance(risk, ScoreAnswer)
    assert risk.score == 0.48

    gap = response.score("gap_severity")
    assert gap.score == 1.3
    # The fractional score is not coerced to an int.
    assert gap.score != 1
    assert int(gap.score) == 1

    lethal = response.noul("lethal_ahead")
    assert isinstance(lethal, NoulAnswer)
    assert lethal.noul == 0.46


def test_the_client_parses_the_verified_response_end_to_end():
    transport = FakeTransport(ok_response())
    response = make_client(transport, RecordingSleeper()).decide(make_state())

    assert response.choice("best_action").choice == "right+A"
    assert response.score("gap_severity").score == 1.3
    assert response.noul("lethal_ahead").noul == 0.46
    assert response.latency_ms == 12.5


def test_response_as_dict_is_journal_ready():
    response = parse_response(VERIFIED_RESPONSE, requested=REQUESTED_TYPES, latency_ms=5.0)
    data = response.as_dict()

    assert data["model"] == "jev-1.13.0"
    assert data["latency_ms"] == 5.0
    assert data["usage"] == {"input_tokens": 839, "output_tokens": 105}
    assert set(data["answers"]) == set(REQUESTED_TYPES)
    assert data["answers"]["lethal_ahead"] == {"type": "noul", "noul": 0.46}
    assert "confidence" not in data["answers"]["lethal_ahead"]
    assert data["answers"]["best_action"]["choice"] == "right+A"


def test_typed_accessors_reject_a_missing_or_mismatched_answer():
    response = parse_response(VERIFIED_RESPONSE, requested=REQUESTED_TYPES, latency_ms=1.0)

    with pytest.raises(KeyError):
        response.choice("does_not_exist")
    with pytest.raises(JevProtocolError):
        response.choice("risk")  # risk is a score answer
    with pytest.raises(JevProtocolError):
        response.score("best_action")
    with pytest.raises(JevProtocolError):
        response.noul("gap_severity")


# -- 4. the noul shape --------------------------------------------------------------


def test_noul_answer_has_no_confidence_field_at_all():
    assert not hasattr(NoulAnswer, "confidence")
    assert {field.name for field in dataclasses.fields(NoulAnswer)} == {"type", "noul"}
    assert "confidence" in {field.name for field in dataclasses.fields(ChoiceAnswer)}
    assert "confidence" in {field.name for field in dataclasses.fields(ScoreAnswer)}

    parsed = parse_response(VERIFIED_RESPONSE, requested=REQUESTED_TYPES, latency_ms=1.0)
    assert not hasattr(parsed.noul("lethal_ahead"), "confidence")


# -- 5. protocol violations ---------------------------------------------------------


def test_a_missing_answer_id_is_a_protocol_error():
    payload = payload_copy()
    del payload["answers"]["gap_severity"]  # type: ignore[index]

    with pytest.raises(JevProtocolError):
        parse_response(payload, requested=REQUESTED_TYPES, latency_ms=1.0)


def test_an_answer_of_the_wrong_type_is_a_protocol_error():
    payload = payload_copy()
    payload["answers"]["risk"] = {  # type: ignore[index]
        "type": "choice",
        "choice": "right",
        "confidence": 0.5,
    }

    with pytest.raises(JevProtocolError):
        parse_response(payload, requested=REQUESTED_TYPES, latency_ms=1.0)


def test_a_choice_outside_the_offered_candidates_is_a_protocol_error():
    payload = payload_copy()
    payload["answers"]["best_action"]["choice"] = "up+B"  # type: ignore[index]

    with pytest.raises(JevProtocolError):
        parse_response(
            payload,
            requested=REQUESTED_TYPES,
            latency_ms=1.0,
            allowed_choices={"best_action": ["NOOP", "right", "right+A"]},
        )


def test_a_choice_outside_the_offered_candidates_is_rejected_by_the_client():
    payload = payload_copy()
    payload["answers"]["best_action"]["choice"] = "up+B"  # type: ignore[index]
    transport = FakeTransport(
        HttpResult(status=200, body=json.dumps(payload), elapsed_ms=1.0)
    )

    with pytest.raises(JevProtocolError):
        make_client(transport, RecordingSleeper()).decide(make_state())


def test_a_non_numeric_score_is_a_protocol_error():
    payload = payload_copy()
    payload["answers"]["risk"]["score"] = "high"  # type: ignore[index]

    with pytest.raises(JevProtocolError):
        parse_response(payload, requested=REQUESTED_TYPES, latency_ms=1.0)


def test_a_noul_above_one_is_a_protocol_error():
    payload = payload_copy()
    payload["answers"]["lethal_ahead"]["noul"] = 1.5  # type: ignore[index]

    with pytest.raises(JevProtocolError):
        parse_response(payload, requested=REQUESTED_TYPES, latency_ms=1.0)


def test_confidence_is_required_for_choice_and_score_but_not_noul():
    payload = payload_copy()
    del payload["answers"]["best_action"]["confidence"]  # type: ignore[index]
    with pytest.raises(JevProtocolError):
        parse_response(payload, requested=REQUESTED_TYPES, latency_ms=1.0)

    # noul parses with no confidence at all.
    response = parse_response(VERIFIED_RESPONSE, requested=REQUESTED_TYPES, latency_ms=1.0)
    assert response.noul("lethal_ahead").noul == 0.46


DISTRIBUTION_FIELDS = [
    ("best_action", "probabilities"),
    ("risk", "probabilities"),
    ("risk", "legend"),
    ("gap_severity", "probabilities"),
    ("gap_severity", "legend"),
]


def _body_without_field(question_id: str, field: str) -> str:
    payload = payload_copy()
    del payload["answers"][question_id][field]  # type: ignore[index]
    return json.dumps(payload)


def _body_with_empty_field(question_id: str, field: str) -> str:
    payload = payload_copy()
    payload["answers"][question_id][field] = {}  # type: ignore[index]
    return json.dumps(payload)


@pytest.mark.parametrize("question_id, field", DISTRIBUTION_FIELDS)
def test_a_missing_distribution_is_a_protocol_error_through_the_client(
    question_id: str, field: str
) -> None:
    transport = FakeTransport(
        HttpResult(
            status=200,
            body=_body_without_field(question_id, field),
            elapsed_ms=1.0,
        )
    )

    with pytest.raises(JevProtocolError) as excinfo:
        make_client(transport, RecordingSleeper()).decide(make_state())

    assert question_id in str(excinfo.value)
    assert field in str(excinfo.value)


@pytest.mark.parametrize("question_id, field", DISTRIBUTION_FIELDS)
def test_an_empty_distribution_is_a_protocol_error_through_the_client(
    question_id: str, field: str
) -> None:
    transport = FakeTransport(
        HttpResult(
            status=200,
            body=_body_with_empty_field(question_id, field),
            elapsed_ms=1.0,
        )
    )

    with pytest.raises(JevProtocolError) as excinfo:
        make_client(transport, RecordingSleeper()).decide(make_state())

    assert question_id in str(excinfo.value)
    assert field in str(excinfo.value)


def test_a_non_json_2xx_body_is_a_protocol_error():
    transport = FakeTransport(HttpResult(status=200, body="not json", elapsed_ms=1.0))

    with pytest.raises(JevProtocolError):
        make_client(transport, RecordingSleeper()).decide(make_state())


# -- 6. retry behaviour -------------------------------------------------------------


def test_a_429_followed_by_a_200_succeeds_with_one_delay():
    sleeper = RecordingSleeper()
    transport = FakeTransport(status_response(429), ok_response())

    response = make_client(transport, sleeper).decide(make_state())

    assert isinstance(response, JevResponse)
    assert transport.calls == 2
    assert sleeper.delays == expected_backoff(1)


def test_a_persistent_529_exhausts_the_attempts_and_raises_unavailable():
    max_attempts = 3
    sleeper = RecordingSleeper()
    transport = FakeTransport(*(status_response(529) for _ in range(max_attempts)))

    with pytest.raises(JevUnavailable):
        make_client(transport, sleeper, max_attempts=max_attempts).decide(make_state())

    assert transport.calls == max_attempts
    assert sleeper.delays == expected_backoff(max_attempts - 1)


def test_the_backoff_doubles_from_the_base_and_is_capped():
    max_attempts = 8
    sleeper = RecordingSleeper()
    transport = FakeTransport(*(status_response(503) for _ in range(max_attempts)))

    with pytest.raises(JevUnavailable):
        make_client(transport, sleeper, max_attempts=max_attempts).decide(make_state())

    assert sleeper.delays == expected_backoff(max_attempts - 1)
    # The doubling really does reach the cap inside this many attempts.
    assert sleeper.delays[-1] == BACKOFF_CAP_S


def test_every_declared_retryable_status_is_retried():
    # Pinned literally: iterating the constant alone cannot catch a shrink.
    assert RETRYABLE_STATUSES == frozenset({429, 500, 502, 503, 504, 529})
    for status in sorted(RETRYABLE_STATUSES):
        sleeper = RecordingSleeper()
        transport = FakeTransport(status_response(status), ok_response())
        response = make_client(transport, sleeper).decide(make_state())
        assert isinstance(response, JevResponse), status
        assert transport.calls == 2, status
        assert sleeper.delays == expected_backoff(1), status


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 529])
def test_each_retryable_status_is_retried_literally(status: int) -> None:
    sleeper = RecordingSleeper()
    transport = FakeTransport(status_response(status), ok_response())

    response = make_client(transport, sleeper).decide(make_state())

    assert isinstance(response, JevResponse)
    assert transport.calls == 2
    assert sleeper.delays == expected_backoff(1)


# -- 7. no retry for credential and validation failures -----------------------------


@pytest.mark.parametrize(
    "status, expected",
    [(401, JevAuthError), (403, JevAuthError), (422, JevRequestError)],
)
def test_credential_and_validation_failures_are_never_retried(
    status: int, expected: type[Exception]
) -> None:
    sleeper = RecordingSleeper()
    transport = FakeTransport(status_response(status))

    with pytest.raises(expected) as excinfo:
        make_client(transport, sleeper).decide(make_state())

    assert transport.calls == 1
    assert sleeper.delays == []
    assert excinfo.value.status == status  # type: ignore[attr-defined]


def test_a_422_exposes_the_offending_field_name():
    body = json.dumps(
        {"detail": [{"loc": ["body", "questions"], "msg": "field required", "type": "missing"}]}
    )
    transport = FakeTransport(status_response(422, body))

    with pytest.raises(JevRequestError) as excinfo:
        make_client(transport, RecordingSleeper()).decide(make_state())

    assert excinfo.value.status == 422
    assert "questions" in excinfo.value.fields
    assert transport.calls == 1


def test_another_non_retryable_4xx_maps_to_a_request_error_without_a_retry():
    sleeper = RecordingSleeper()
    transport = FakeTransport(status_response(400, "bad request"))

    with pytest.raises(JevRequestError):
        make_client(transport, sleeper).decide(make_state())

    assert transport.calls == 1
    assert sleeper.delays == []


# -- 8. transport failures ----------------------------------------------------------


def test_a_transport_failure_is_retried_and_then_raises_unavailable():
    max_attempts = 3
    sleeper = RecordingSleeper()
    transport = FakeTransport(*(timeout_response() for _ in range(max_attempts)))

    with pytest.raises(JevUnavailable):
        make_client(transport, sleeper, max_attempts=max_attempts).decide(make_state())

    assert transport.calls == max_attempts
    assert sleeper.delays == expected_backoff(max_attempts - 1)


def test_a_transport_failure_that_recovers_still_returns_a_response():
    sleeper = RecordingSleeper()
    transport = FakeTransport(timeout_response(), ok_response())

    response = make_client(transport, sleeper).decide(make_state())

    assert isinstance(response, JevResponse)
    assert transport.calls == 2
    assert sleeper.delays == expected_backoff(1)


# -- 9. a missing credential --------------------------------------------------------


def test_a_none_api_key_raises_auth_error_without_any_transport_call():
    transport = FakeTransport(ok_response())

    with pytest.raises(JevAuthError):
        make_client(transport, RecordingSleeper(), api_key=None).decide(make_state())

    assert transport.calls == 0


# -- 10. key containment ------------------------------------------------------------


def test_the_api_key_never_leaks_into_any_rendered_view():
    state = make_state()
    transport = FakeTransport(ok_response())
    client = make_client(transport)

    assert SENTINEL_KEY not in repr(client)
    assert SENTINEL_KEY not in repr(client.redacted)
    assert client.redacted["jev_available"] is True

    client.decide(state)
    request = transport.requests[0]
    assert SENTINEL_KEY not in repr(request)
    assert SENTINEL_KEY not in json.dumps(request.as_dict())
    # The wire value is the real one; only the rendered views redact.
    assert request.headers["Authorization"] == f"Bearer {SENTINEL_KEY}"
    assert request.as_dict()["headers"]["Authorization"] == "Bearer <redacted>"

    response = parse_response(VERIFIED_RESPONSE, requested=REQUESTED_TYPES, latency_ms=1.0)
    assert SENTINEL_KEY not in json.dumps(response.as_dict())
    assert SENTINEL_KEY not in repr(response)

    exceptions: list[tuple[HttpResult, type[Exception]]] = [
        (status_response(401), JevAuthError),
        (status_response(403), JevAuthError),
        (status_response(422), JevRequestError),
        (HttpResult(status=200, body="not json", elapsed_ms=1.0), JevProtocolError),
        (timeout_response(), JevUnavailable),
    ]
    for result, expected in exceptions:
        failing = FakeTransport(result, result, result)
        with pytest.raises(expected) as excinfo:
            make_client(failing, RecordingSleeper()).decide(state)
        assert SENTINEL_KEY not in repr(excinfo.value)
        assert SENTINEL_KEY not in str(excinfo.value)


def test_a_redacted_request_view_keeps_the_other_headers():
    request = JevRequest(
        url=ENDPOINT,
        headers={
            "Authorization": RedactedHeader(f"Bearer {SENTINEL_KEY}"),
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        json_body={"model": MODEL},
        timeout_s=TIMEOUT_S,
    )

    view = request.as_dict()

    assert SENTINEL_KEY not in json.dumps(view)
    assert view["headers"]["Content-Type"] == "application/json"
    assert view["headers"]["User-Agent"] == USER_AGENT
    assert view["headers"]["Authorization"] == "Bearer <redacted>"
    assert view["url"] == ENDPOINT


def test_the_credential_reaches_the_transport_and_no_serialiser():
    """The wrapper, not discipline, is what keeps the key out of a journal.

    ``dataclasses.asdict`` is the one serialiser a plain header string cannot
    defend against, because it copies the value verbatim. Wrapping the credential
    means that copy carries the wrapper instead, so serialising it fails loudly
    rather than writing the key into a file. The real value must still be
    obtainable by the transport, which is the only thing that needs it.
    """
    request = JevRequest(
        url=ENDPOINT,
        headers={
            "Authorization": RedactedHeader(f"Bearer {SENTINEL_KEY}"),
            "Content-Type": "application/json",
        },
        json_body={"model": MODEL},
        timeout_s=TIMEOUT_S,
    )

    header = request.headers["Authorization"]
    assert header.reveal() == f"Bearer {SENTINEL_KEY}"
    assert str(header) == "Bearer <redacted>"
    assert f"{header}" == "Bearer <redacted>"
    assert SENTINEL_KEY not in repr(request)
    assert SENTINEL_KEY not in str(dataclasses.asdict(request))
    with pytest.raises(TypeError):
        json.dumps(dataclasses.asdict(request))
    with pytest.raises(TypeError):
        "prefix" + header


def test_a_plain_string_authorization_header_is_refused():
    """A plain string would be copied out verbatim by ``dataclasses.asdict``."""
    with pytest.raises(ValueError, match="RedactedHeader") as excinfo:
        JevRequest(
            url=ENDPOINT,
            headers={"Authorization": f"Bearer {SENTINEL_KEY}"},
            json_body={"model": MODEL},
            timeout_s=TIMEOUT_S,
        )

    assert SENTINEL_KEY not in str(excinfo.value)


def test_pickling_a_request_is_refused_without_naming_the_key():
    """``pickle`` serialises slot state, so the wrapper must refuse it loudly."""
    request = captured_request()

    with pytest.raises(TypeError, match="RedactedHeader") as excinfo:
        pickle.dumps(request)

    assert_no_key_material(str(excinfo.value))


def test_pickling_the_client_is_refused_without_naming_the_key():
    """The wrapper on the client is what a client pickle would otherwise copy out."""
    client = make_client(FakeTransport(ok_response()), RecordingSleeper())

    with pytest.raises(TypeError, match="RedactedHeader") as excinfo:
        pickle.dumps(client)

    assert_no_key_material(str(excinfo.value))


def test_the_client_instance_holds_no_raw_key_material():
    client = make_client(FakeTransport(ok_response()), RecordingSleeper())

    rendered = repr(vars(client)) + str(vars(client)) + json.dumps(client.redacted)
    assert SENTINEL_KEY not in rendered
    assert "Bearer <redacted>" in repr(vars(client))
    for name, value in vars(client).items():
        assert value != SENTINEL_KEY, name
        assert SENTINEL_KEY not in repr(value), name


@pytest.mark.parametrize("clone_with", [copy.copy, copy.deepcopy])
def test_copying_a_request_keeps_it_working_and_redacted(clone_with):
    request = captured_request()

    clone = clone_with(request)

    assert clone is not request
    assert SENTINEL_KEY not in repr(clone)
    assert SENTINEL_KEY not in json.dumps(clone.as_dict())
    assert repr(clone.headers["Authorization"]) == "Bearer <redacted>"
    # The copy still carries the real wire value for the transport.
    assert clone.headers["Authorization"] == f"Bearer {SENTINEL_KEY}"


# -- 11. the one sanctioned live test -----------------------------------------------


@pytest.mark.live_jev
@pytest.mark.skipif(not Settings.load().jev_available, reason="JEV_API_KEY is not set")
def test_a_real_jev_call_returns_a_valid_response():
    state = make_state()
    with JevClient.from_settings(Settings.load()) as client:
        response = client.decide(state)

    assert isinstance(response, JevResponse)
    assert response.model
    assert set(response.answers) == set(TICK_QUESTION_IDS)
    assert isinstance(response.score("risk").score, float)
    assert 0.0 <= response.noul("lethal_ahead").noul <= 1.0
