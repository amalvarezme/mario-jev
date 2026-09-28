"""The decision layer: cadence, confidence gate, guardrails, and the tick loop.

Nothing here touches the network or the clock. The pure rule is exercised on
constructed tables and responses; the loop is exercised with a fake emulator, a
constructed rollout table, fakes for the client/planner/advisor, and — for the one
behaviour that must abort rather than degrade — a *real* ``JevClient`` driving a
scripted transport, so the exception the loop must not catch is the real one the
client raises. One extra test runs a single real tick on the real emulator with
every default collaborator, to pin that the defaults are wired up.
"""

from __future__ import annotations

import json
from dataclasses import fields

import numpy as np
import pytest

from mario_jev.advisor import (
    REASON_TORCH_MISSING,
    AdvisorAbsence,
    AdvisorAdvice,
    AdvisorMove,
)
from mario_jev.config import Settings
from mario_jev.emulator import StepResult, build_actions
from mario_jev.jev import (
    ChoiceAnswer,
    HttpResult,
    JevAuthError,
    JevClient,
    JevProtocolError,
    JevRequestError,
    JevResponse,
    JevUnavailable,
    ScoreAnswer,
    Usage,
    build_questions,
)
from mario_jev.policy import (
    OVERSHOOT_FACTOR,
    DecisionCadence,
    DecisionSource,
    ExecutionLoop,
    TickDecision,
    TickRecord,
    advisor_view,
    journal_state,
    select_action,
)
from mario_jev.perception import detect
from mario_jev.state import (
    HAZARD_OUTCOMES,
    MarioState,
    NearestEnemy,
    PlayerFeatures,
    Progress,
    TerrainAhead,
    TerminalOutcome,
)
from mario_jev.trajectory import CandidateRollout, CostSource, RolloutTable

#: A synthetic credential. A real key is never used, and never printed by a test.
SENTINEL_KEY = "apikey_SENTINEL_policy_do_not_leak"

#: The seven ``simple`` actions, keyed by name, exactly as the emulator builds them.
ACTIONS = {action.name: action for action in build_actions("simple")}

#: The frozen projection key set. The policy must add to it, never edit it.
FROZEN_STATE_KEYS = {"player", "nearest_enemy", "terrain_ahead", "progress", "candidates"}

SCREEN = (240, 256, 3)


# -- pure builders ------------------------------------------------------------------


def candidate(
    name: str,
    *,
    risk: float,
    outcome: TerminalOutcome = TerminalOutcome.LANDS_SAFE,
    dx: int = 0,
) -> CandidateRollout:
    final_x = 40 + dx
    return CandidateRollout(
        action=ACTIONS[name],
        horizon=8,
        frames=8,
        start_x=40,
        final_x=final_x,
        x_max=final_x,
        outcome=outcome,
        risk=risk,
        final_enemy_distance=None,
        terminated=outcome in HAZARD_OUTCOMES,
        truncated=False,
    )


def table_of(*candidates: CandidateRollout, elapsed_ms: float = 0.5) -> RolloutTable:
    return RolloutTable(
        tick=0,
        start_x=40,
        horizon=8,
        budget_ms=120.0,
        elapsed_ms=elapsed_ms,
        sweep_ms=elapsed_ms,
        base_ms=0.5,
        per_frame_ms=1.0,
        calibration_frames=0,
        passes=1,
        cost_source=CostSource.REUSED,
        steps_taken=0,
        candidates=tuple(candidates),
    )


#: Seven survivable candidates. ``right+A+B`` has the lowest risk, so it is ``best``.
def all_safe_table(*, elapsed_ms: float = 0.5) -> RolloutTable:
    return table_of(
        candidate("NOOP", risk=0.40, outcome=TerminalOutcome.NO_PROGRESS),
        candidate("right", risk=0.30),
        candidate("right+A", risk=0.25),
        candidate("right+B", risk=0.20),
        candidate("right+A+B", risk=0.15),
        candidate("A", risk=0.50, outcome=TerminalOutcome.NO_PROGRESS),
        candidate("left", risk=0.60, outcome=TerminalOutcome.NO_PROGRESS),
        elapsed_ms=elapsed_ms,
    )


def response(
    choice: str,
    confidence: float,
    *,
    risk: float | None = None,
    model: str = "jev-test",
    latency_ms: float = 3.0,
) -> JevResponse:
    answers: dict[str, ChoiceAnswer | ScoreAnswer] = {
        "best_action": ChoiceAnswer(
            type="choice",
            choice=choice,
            confidence=confidence,
            probabilities={choice: 1.0},
        )
    }
    if risk is not None:
        answers["risk"] = ScoreAnswer(
            type="score",
            score=risk,
            confidence=confidence,
            legend={"0": "safe", "1": "minor risk"},
            probabilities={"0": 0.5, "1": 0.5},
        )
    return JevResponse(
        model=model,
        answers=answers,
        usage=Usage(input_tokens=10, output_tokens=5),
        latency_ms=latency_ms,
    )


def make_settings(
    *,
    key: str | None = SENTINEL_KEY,
    cadence: int = 1,
    threshold: float = 0.6,
    budget_ms: float = 120.0,
    max_ticks: int | None = None,
    advisor_enabled: bool = False,
) -> Settings:
    return Settings(
        jev_api_key=key,
        decision_cadence_frames=cadence,
        confidence_threshold=threshold,
        rollout_budget_ms=budget_ms,
        max_ticks=max_ticks,
        advisor_enabled=advisor_enabled,
    )


def make_state(tick: int = 0) -> MarioState:
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
    )


def advice(top: str, *, probability: float = 0.9) -> AdvisorAdvice:
    move = AdvisorMove(name=top, probability=probability)
    return AdvisorAdvice(
        model_id="OmniJev/PlayJev-0.8B",
        backend="fake",
        moves=(move,),
        top=move,
        latency_ms=2.0,
    )


def assert_no_key_material(text: str) -> None:
    """Fail if ``text`` carries the sentinel key, or any usable fragment of it."""
    assert SENTINEL_KEY not in text
    for start in range(len(SENTINEL_KEY)):
        for end in range(start + 5, len(SENTINEL_KEY) + 1):
            fragment = SENTINEL_KEY[start:end]
            assert fragment not in text, fragment


def ok_body(choice: str, *, confidence: float = 0.9, risk: float = 0.2) -> str:
    return json.dumps(
        {
            "model": "jev-test",
            "answers": {
                "best_action": {
                    "type": "choice",
                    "choice": choice,
                    "confidence": confidence,
                    "probabilities": {choice: 1.0},
                },
                "risk": {
                    "type": "score",
                    "score": risk,
                    "confidence": confidence,
                    "legend": {"0": "safe", "1": "minor risk"},
                    "probabilities": {"0": 0.6, "1": 0.4},
                },
                "lethal_ahead": {"type": "noul", "noul": 0.1},
                "gap_severity": {
                    "type": "score",
                    "score": 0.0,
                    "confidence": confidence,
                    "legend": {"0": "no gap"},
                    "probabilities": {"0": 1.0},
                },
            },
            "usage": {"input_tokens": 5, "output_tokens": 5},
        }
    )


# -- loop fakes ---------------------------------------------------------------------


class FakeEmulator:
    """A deterministic stand-in for the real emulator: a black frame, one frame per step."""

    def __init__(self, *, terminate_at: int | None = None, truncate_at: int | None = None) -> None:
        self._actions = build_actions("simple")
        self._tick = 0
        self._steps = 0
        self._terminate_at = terminate_at
        self._truncate_at = truncate_at
        self.executed: list[object] = []

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def action_names(self) -> tuple[str, ...]:
        return tuple(action.name for action in self._actions)

    def action(self, index: int):
        return self._actions[index]

    def frame(self) -> np.ndarray:
        return np.zeros(SCREEN, dtype=np.uint8)

    def ram(self) -> object:
        return None

    @property
    def info(self) -> dict[str, object]:
        return {}

    def step(self, action: object) -> StepResult:
        self._tick += 1
        self._steps += 1
        self.executed.append(action)
        return StepResult(
            frame=self.frame(),
            info={},
            reward=0.0,
            terminated=self._terminate_at is not None and self._steps >= self._terminate_at,
            truncated=self._truncate_at is not None and self._steps >= self._truncate_at,
            tick=self._tick,
        )


class FakeBuilder:
    """Builds a fixed state; the loop attaches the rollout's candidates to it."""

    def __init__(self) -> None:
        self.observed_ticks: list[int] = []

    def observe(self, perception: object, ram: object, info: object) -> MarioState:
        tick = int(perception.tick)  # type: ignore[attr-defined]
        self.observed_ticks.append(tick)
        return make_state(tick=tick)


class FakePlanner:
    """Returns one constructed table and never touches the emulator.

    The loop must call the instance it was given, once per tick: that is the WU3
    requirement that a single planner is held across ticks, and it is why the
    injected instance is the only thing here that could answer.
    """

    def __init__(self, table: RolloutTable) -> None:
        self.table = table
        self.plans = 0
        self.seen_ticks: list[int] = []

    def plan(self, emulator: object) -> RolloutTable:
        self.plans += 1
        self.seen_ticks.append(int(emulator.tick))  # type: ignore[attr-defined]
        return self.table


class InflatingPlanner:
    """A planner whose sweep advances the emulator's tick, as the real one does.

    The real :class:`~mario_jev.trajectory.TrajectoryPlanner` steps the same
    emulator while enumerating candidates and never rewinds its counter, so
    ``emulator.tick`` jumps by the whole rollout cost on every loop tick. This fake
    reproduces that inflation deterministically, so a cadence keyed off
    ``emulator.tick`` would fire on an arbitrary subset of ticks.
    """

    def __init__(self, table: RolloutTable, *, frames_per_plan: int) -> None:
        self.table = table
        self.frames_per_plan = frames_per_plan
        self.plans = 0

    def plan(self, emulator: object) -> RolloutTable:
        self.plans += 1
        for _ in range(self.frames_per_plan):
            emulator.step(emulator.action(0))  # type: ignore[attr-defined]
        return self.table


class FakeClient:
    """A queued-response client, so cadence and summary tests never build a real one."""

    def __init__(self, *responses: JevResponse, error: Exception | None = None) -> None:
        self._responses = list(responses)
        self._error = error
        self.calls = 0

    def ask(self, questions: object, state: object) -> JevResponse:
        self.calls += 1
        if self._error is not None:
            raise self._error
        if not self._responses:
            raise AssertionError("the loop called Jev more often than the test queued")
        return self._responses.pop(0)


class FakeAdvisor:
    def __init__(
        self,
        answer: AdvisorAdvice | None,
        *,
        available: bool = True,
        absence: AdvisorAbsence | None = None,
    ) -> None:
        self.answer = answer
        self.available = available
        self.absence = absence
        self.calls = 0

    def advise(self, frame: object, action_names: object) -> AdvisorAdvice | None:
        self.calls += 1
        return self.answer


class ScriptedTransport:
    """Hands back queued results and remembers the requests; never opens a socket."""

    def __init__(self, *results: HttpResult) -> None:
        self.queued = list(results)
        self.requests: list[object] = []

    def __call__(self, request: object) -> HttpResult:
        self.requests.append(request)
        if not self.queued:
            raise AssertionError("the client sent more requests than the test queued")
        return self.queued.pop(0)


class SilentSleeper:
    def __call__(self, delay: float) -> None:
        raise AssertionError("a retry must not sleep in these tests")


def real_client(settings: Settings, transport: ScriptedTransport) -> JevClient:
    return JevClient(
        endpoint=settings.jev_endpoint,
        model=settings.jev_model,
        api_key=settings.jev_api_key,
        timeout_s=settings.jev_timeout_s,
        max_attempts=1,
        transport=transport,
        sleeper=SilentSleeper(),
    )


def loop_with(
    *,
    settings: Settings,
    table: RolloutTable | None = None,
    client: object | None = None,
    emulator: FakeEmulator | None = None,
    planner: object | None = None,
    advisor: object | None = None,
) -> tuple[ExecutionLoop, FakeEmulator, object]:
    emulator = emulator if emulator is not None else FakeEmulator()
    planner = planner if planner is not None else FakePlanner(table or all_safe_table())
    loop = ExecutionLoop(
        settings,
        emulator=emulator,
        planner=planner,
        client=client,  # type: ignore[arg-type]
        builder=FakeBuilder(),
        advisor=advisor,  # type: ignore[arg-type]
    )
    return loop, emulator, planner


# -- 1. the eight branches, in order ------------------------------------------------


def test_branch_1_no_candidates_holds_still() -> None:
    decision = select_action(table_of(), None, make_settings())

    assert decision.source is DecisionSource.FALLBACK_NO_CANDIDATES
    assert decision.action == "NOOP"
    assert decision.confidence is None and decision.risk is None
    assert decision.jev_called is False


def test_branch_2_off_cadence_reuses_the_last_decided_action() -> None:
    decision = select_action(
        all_safe_table(),
        None,
        make_settings(),
        cadence_due=False,
        last_action="right+B",
    )

    assert decision.source is DecisionSource.HOLD
    assert decision.action == "right+B"
    assert decision.confidence is None and decision.risk is None
    assert decision.jev_called is False


def test_branch_3_no_credential_uses_the_measured_best_without_calling() -> None:
    table = all_safe_table()

    decision = select_action(table, None, make_settings(key=None))

    assert decision.source is DecisionSource.FALLBACK_NO_KEY
    assert decision.action == table.best.action.name == "right+A+B"
    assert decision.jev_called is False


def test_branch_4_degraded_reason_is_recorded_verbatim_as_the_cause() -> None:
    table = all_safe_table()
    cause = "Jev unavailable after 1 attempt(s): HTTP 503"

    decision = select_action(table, None, make_settings(), degraded_reason=cause)

    assert decision.source is DecisionSource.FALLBACK_DEGRADED
    assert decision.action == table.best.action.name
    assert decision.reason == cause
    assert decision.jev_called is True


def test_branch_5_a_protocol_drift_is_not_handled_here() -> None:
    # A best_action answered as a score is a protocol break, not a fallback state.
    broken = JevResponse(
        model="jev-test",
        answers={
            "best_action": ScoreAnswer(
                type="score",
                score=0.5,
                confidence=0.9,
                legend={"0": "safe"},
                probabilities={"0": 1.0},
            )
        },
        usage=Usage(1, 1),
        latency_ms=1.0,
    )

    with pytest.raises(JevProtocolError):
        select_action(all_safe_table(), broken, make_settings())


def test_branch_6_low_confidence_falls_back_and_records_what_the_gate_read() -> None:
    table = all_safe_table()

    decision = select_action(
        table, response("right+A+B", 0.42), make_settings(threshold=0.6)
    )

    assert decision.source is DecisionSource.FALLBACK_BELOW_CONFIDENCE
    assert decision.action == table.best.action.name
    assert decision.confidence == pytest.approx(0.42)
    assert decision.risk is None
    assert decision.jev_called is True


def test_branch_7_guardrail_overrides_a_lethal_choice() -> None:
    table = table_of(
        candidate("right+A", risk=0.8, outcome=TerminalOutcome.HITS_ENEMY),
        candidate("right+B", risk=0.2, outcome=TerminalOutcome.LANDS_SAFE),
    )

    decision = select_action(table, response("right+A", 0.9), make_settings())

    assert decision.source is DecisionSource.GUARDRAIL_LETHAL
    assert decision.action == "right+B"
    assert decision.confidence == pytest.approx(0.9)
    assert decision.jev_called is True


def test_branch_8_high_confidence_executes_the_choice_and_carries_the_risk() -> None:
    decision = select_action(
        all_safe_table(), response("right+B", 0.88, risk=0.31), make_settings()
    )

    assert decision.source is DecisionSource.JEV
    assert decision.action == "right+B"
    assert decision.confidence == pytest.approx(0.88)
    assert decision.risk == pytest.approx(0.31)
    assert decision.jev_called is True


def test_branch_8_without_a_risk_answer_records_no_risk() -> None:
    decision = select_action(all_safe_table(), response("right+B", 0.9), make_settings())

    assert decision.source is DecisionSource.JEV
    assert decision.risk is None


def test_a_choice_outside_the_table_raises_rather_than_falling_back() -> None:
    with pytest.raises(ValueError):
        select_action(all_safe_table(), response("up", 0.9), make_settings())


def test_a_cadence_of_zero_is_rejected() -> None:
    with pytest.raises(ValueError):
        DecisionCadence(0)


# -- 2. the crux: what degrades, and what aborts ------------------------------------


def test_a_rejected_credential_aborts_the_run() -> None:
    settings = make_settings()
    transport = ScriptedTransport(HttpResult(status=401, body="{}", elapsed_ms=1.0))
    loop, _, _ = loop_with(
        settings=settings, client=real_client(settings, transport)
    )

    with pytest.raises(JevAuthError):
        loop.run(max_ticks=3)

    assert len(transport.requests) == 1  # the call happened, then the run stopped


def test_a_forbidden_credential_aborts_the_run() -> None:
    settings = make_settings()
    transport = ScriptedTransport(HttpResult(status=403, body="{}", elapsed_ms=1.0))
    loop, _, _ = loop_with(
        settings=settings, client=real_client(settings, transport)
    )

    with pytest.raises(JevAuthError):
        loop.run(max_ticks=3)


def test_a_malformed_request_aborts_the_run() -> None:
    settings = make_settings()
    transport = ScriptedTransport(HttpResult(status=422, body="{}", elapsed_ms=1.0))
    loop, _, _ = loop_with(
        settings=settings, client=real_client(settings, transport)
    )

    with pytest.raises(JevRequestError):
        loop.run(max_ticks=3)


def test_a_drifted_2xx_body_aborts_the_run() -> None:
    # A 200 whose body is missing three of the four answers: the contract drifted.
    drifted = json.dumps(
        {
            "model": "jev-test",
            "answers": {
                "best_action": {
                    "type": "choice",
                    "choice": "right",
                    "confidence": 0.9,
                    "probabilities": {"right": 1.0},
                }
            },
        }
    )
    settings = make_settings()
    transport = ScriptedTransport(HttpResult(status=200, body=drifted, elapsed_ms=1.0))
    loop, _, _ = loop_with(
        settings=settings, client=real_client(settings, transport)
    )

    with pytest.raises(JevProtocolError):
        loop.run(max_ticks=3)


def test_a_transient_503_degrades_instead_of_aborting() -> None:
    settings = make_settings()
    transport = ScriptedTransport(HttpResult(status=503, body="", elapsed_ms=1.0))
    loop, _, _ = loop_with(
        settings=settings, client=real_client(settings, transport)
    )

    records = []
    summary = loop.run(max_ticks=1, observer=records.append)

    assert summary.sources[DecisionSource.FALLBACK_DEGRADED] == 1
    assert summary.decisions == 1
    assert len(transport.requests) == 1
    assert records[0].decision.source is DecisionSource.FALLBACK_DEGRADED
    assert "503" in records[0].decision.reason


def test_a_transport_timeout_degrades_instead_of_aborting() -> None:
    settings = make_settings()
    transport = ScriptedTransport(
        HttpResult(status=0, body="", elapsed_ms=1.0, error="timeout")
    )
    loop, _, _ = loop_with(
        settings=settings, client=real_client(settings, transport)
    )

    summary = loop.run(max_ticks=1)

    assert summary.sources[DecisionSource.FALLBACK_DEGRADED] == 1


# -- 3. cadence ---------------------------------------------------------------------


def test_cadence_one_consults_jev_on_every_tick() -> None:
    client = FakeClient(response("right", 0.9), response("right", 0.9), response("right", 0.9))
    loop, _, _ = loop_with(settings=make_settings(cadence=1), client=client)

    records: list[object] = []
    summary = loop.run(max_ticks=3, observer=records.append)

    assert client.calls == 3
    assert summary.sources[DecisionSource.JEV] == 3
    assert all(record.decision.source is DecisionSource.JEV for record in records)


def test_cadence_eight_consults_only_on_multiples_of_eight() -> None:
    client = FakeClient(response("right", 0.9), response("right", 0.9))
    loop, emulator, _ = loop_with(settings=make_settings(cadence=8), client=client)

    records: list[object] = []
    summary = loop.run(max_ticks=12, observer=records.append)

    assert [record.frame_index for record in records] == list(range(12))
    decided = [
        (record.frame_index, record.decision.action)
        for record in records
        if record.decision.source is DecisionSource.JEV
    ]
    assert decided == [(0, "right"), (8, "right")]
    assert client.calls == 2
    assert summary.sources[DecisionSource.HOLD] == 10
    assert summary.sources[DecisionSource.JEV] == 2
    assert summary.decisions == 2

    # Between decisions the last decided action is executed verbatim.
    assert all(
        record.decision.action == "right"
        for record in records
        if record.decision.source is DecisionSource.HOLD
    )
    assert [action.name for action in emulator.executed] == ["right"] * 12


def test_cadence_counts_executed_frames_even_when_the_planner_inflates_the_tick() -> None:
    # The real planner advances emulator.tick by the whole rollout cost while it
    # enumerates candidates, so a cadence keyed off that counter fires on an
    # arbitrary subset of ticks. Keyed off the loop's own frame counter,
    # cadence_frames=8 must mean every 8 executed game frames.
    client = FakeClient(*(response("right", 0.9) for _ in range(3)))
    planner = InflatingPlanner(all_safe_table(), frames_per_plan=185)
    loop, _, _ = loop_with(
        settings=make_settings(cadence=8), planner=planner, client=client
    )

    records: list[object] = []
    loop.run(max_ticks=20, observer=records.append)

    decided = [
        record.frame_index
        for record in records
        if record.decision.source is DecisionSource.JEV
    ]
    assert decided == [0, 8, 16]
    assert client.calls == 3
    # 185 rollout frames + 1 executed frame per tick: emulator.tick can never have
    # produced this cadence.
    assert [record.emulator_tick for record in records[:3]] == [0, 186, 372]


def test_frame_index_counts_game_frames_while_emulator_tick_records_the_inflation() -> None:
    planner = InflatingPlanner(all_safe_table(), frames_per_plan=185)
    loop, _, _ = loop_with(settings=make_settings(key=None), planner=planner)

    records: list[object] = []
    loop.run(max_ticks=5, observer=records.append)

    frame_indexes = [record.frame_index for record in records]
    emulator_ticks = [record.emulator_tick for record in records]

    assert frame_indexes == [0, 1, 2, 3, 4]
    assert emulator_ticks == [0, 186, 372, 558, 744]
    assert all(later - earlier > 1 for earlier, later in zip(emulator_ticks, emulator_ticks[1:]))


def test_an_inflated_emulator_tick_is_not_a_decision_unless_the_frame_index_is() -> None:
    # frames_per_plan=7 makes emulator.tick land on a multiple of 8 every tick
    # (0, 8, 16, ...) while the loop's frame index is 0, 1, 2, ... A cadence keyed
    # off the raw counter would find "due" on every tick; the frame index says only
    # frame 0 is.
    client = FakeClient(response("right", 0.9))
    planner = InflatingPlanner(all_safe_table(), frames_per_plan=7)
    loop, _, _ = loop_with(
        settings=make_settings(cadence=8), planner=planner, client=client
    )

    first = loop.step()
    second = loop.step()

    assert first.frame_index == 0 and first.emulator_tick == 0
    assert second.frame_index == 1 and second.emulator_tick == 8
    assert second.emulator_tick % 8 == 0  # the raw counter would call this due
    assert second.decision.source is DecisionSource.HOLD
    assert client.calls == 1


# -- 4. the guardrail ---------------------------------------------------------------


def test_guardrail_stands_down_when_the_best_candidate_is_also_lethal() -> None:
    table = table_of(
        candidate("right+A", risk=0.9, outcome=TerminalOutcome.HITS_ENEMY),
        candidate("right+B", risk=0.2, outcome=TerminalOutcome.HITS_ENEMY),
    )

    decision = select_action(table, response("right+A", 0.9), make_settings())

    # Both are lethal, so there is no safe alternative to override towards.
    assert decision.source is DecisionSource.JEV
    assert decision.action == "right+A"


# -- 5. no candidates ---------------------------------------------------------------


def test_no_candidates_holds_the_emulator_still_without_consulting_jev() -> None:
    client = FakeClient()
    loop, emulator, _ = loop_with(
        settings=make_settings(), table=table_of(), client=client
    )

    record = loop.step()

    assert record.decision.source is DecisionSource.FALLBACK_NO_CANDIDATES
    assert record.decision.action == "NOOP"
    assert record.jev_questions is None
    assert client.calls == 0
    assert [action.name for action in emulator.executed] == ["NOOP"]


# -- 6. one planner across ticks ----------------------------------------------------


def test_the_loop_reuses_the_injected_planner_for_every_tick() -> None:
    planner = FakePlanner(all_safe_table())
    client = FakeClient(response("right", 0.9), response("right", 0.9), response("right", 0.9))
    loop, _, _ = loop_with(
        settings=make_settings(), planner=planner, client=client
    )

    loop.run(max_ticks=3)

    # A planner built per tick would not be this object and would not be called here.
    assert planner.plans == 3
    assert planner.seen_ticks == [0, 1, 2]


# -- 7. the timing overrun is recorded, never repaired ------------------------------


@pytest.mark.parametrize(
    ("elapsed_ms", "overrun"),
    [(100.0, False), (359.9, False), (360.0, False), (360.1, True), (1000.0, True)],
)
def test_timing_overrun_is_recorded_exactly_past_three_times_the_budget(
    elapsed_ms: float, overrun: bool
) -> None:
    assert OVERSHOOT_FACTOR == 3.0
    planner = FakePlanner(all_safe_table(elapsed_ms=elapsed_ms))
    loop, _, _ = loop_with(
        settings=make_settings(key=None, budget_ms=120.0), planner=planner
    )

    record = loop.step()

    assert ("timing_overrun" in record.degradations) is overrun


def test_the_summary_counts_timing_overruns() -> None:
    planner = FakePlanner(all_safe_table(elapsed_ms=1000.0))
    loop, _, _ = loop_with(
        settings=make_settings(key=None, budget_ms=120.0), planner=planner
    )

    summary = loop.run(max_ticks=2)

    assert summary.degradations["timing_overrun"] == 2


# -- 8. the advisor never changes the tick ------------------------------------------


def test_an_advisor_that_disagrees_does_not_change_the_tick() -> None:
    answer = advice("left")
    advisor = FakeAdvisor(answer)
    client = FakeClient(response("right", 0.9))
    loop, emulator, _ = loop_with(
        settings=make_settings(advisor_enabled=True), client=client, advisor=advisor
    )

    record = loop.step()

    assert advisor.calls == 1
    assert record.advisor is answer
    assert record.decision.action == "right"
    assert [action.name for action in emulator.executed] == ["right"]
    # The advisor is recorded, and only recorded: it never reaches Jev.
    assert record.journal_state["advisor"]["top"]["name"] == "left"
    assert "advisor" not in json.dumps(record.jev_questions)


def test_an_advisor_with_no_answer_leaves_the_tick_unchanged() -> None:
    advisor = FakeAdvisor(None)
    client = FakeClient(response("right", 0.9))
    loop, emulator, _ = loop_with(
        settings=make_settings(advisor_enabled=True), client=client, advisor=advisor
    )

    record = loop.step()

    assert advisor.calls == 1
    assert record.advisor is None
    assert record.decision.action == "right"


def test_a_disabled_advisor_is_not_consulted() -> None:
    advisor = FakeAdvisor(advice("left"))
    client = FakeClient(response("right", 0.9))
    loop, _, _ = loop_with(
        settings=make_settings(advisor_enabled=False), client=client, advisor=advisor
    )

    record = loop.step()

    assert advisor.calls == 0
    assert record.advisor is None
    assert record.journal_state["advisor"] is None


def test_journal_state_adds_the_advisor_block_while_to_jev_state_stays_frozen() -> None:
    table = all_safe_table()
    state = make_state(tick=1).with_candidates(table.assessments)
    frozen = state.to_jev_state()

    assert set(frozen) == FROZEN_STATE_KEYS

    projected = journal_state(state, advice("left"))

    assert set(projected) == FROZEN_STATE_KEYS | {"advisor"}
    assert projected["advisor"]["top"]["name"] == "left"
    assert journal_state(state, None)["advisor"] is None
    assert advisor_view(None) is None


# -- 9. the recorded request carries no credential and sends the frozen state -------


def test_the_tick_records_no_credential_and_sends_exactly_the_frozen_state() -> None:
    settings = make_settings()
    transport = ScriptedTransport(
        HttpResult(status=200, body=ok_body("right"), elapsed_ms=3.0)
    )
    loop, _, _ = loop_with(
        settings=settings, client=real_client(settings, transport)
    )

    record = loop.step()

    assert record.jev_questions is not None and record.jev_response is not None
    assert_no_key_material(json.dumps(record.jev_questions))
    assert_no_key_material(json.dumps(record.jev_response))
    assert_no_key_material(repr(record.jev_questions))

    assert set(record.state.to_jev_state()) == FROZEN_STATE_KEYS
    sent_state = record.state.to_jev_state()
    assert set(sent_state) == FROZEN_STATE_KEYS
    assert "advisor" not in sent_state

    # ... and the request the client really sent agrees with the recorded view.
    sent = transport.requests[0]
    assert set(sent.json_body["state"]) == FROZEN_STATE_KEYS


def test_jev_questions_are_the_built_questions_and_none_when_no_call_was_made() -> None:
    settings = make_settings(cadence=8)
    client = FakeClient(response("right", 0.9))
    loop, _, _ = loop_with(settings=settings, client=client)

    consulted = loop.step()  # frame 0: the cadence is due
    held = loop.step()  # frame 1: no call was made

    assert consulted.jev_questions is not None
    assert consulted.jev_questions == build_questions(consulted.state)
    assert held.jev_questions is None
    assert client.calls == 1


def test_the_tick_record_carries_only_built_questions_and_no_request_mirror() -> None:
    settings = make_settings()
    transport = ScriptedTransport(
        HttpResult(status=200, body=ok_body("right"), elapsed_ms=3.0)
    )
    loop, _, _ = loop_with(settings=settings, client=real_client(settings, transport))

    record = loop.step()

    assert record.jev_questions is not None
    assert record.jev_response is not None
    # The old record carried a fabricated `Authorization: Bearer <redacted>`
    # placeholder and a `url`/`headers` mirror of a request it never observed.
    assert not hasattr(record, "jev_request")
    text = json.dumps(record.jev_questions) + json.dumps(record.jev_response)
    assert "Authorization" not in text
    assert "Bearer" not in text
    assert "headers" not in text
    assert '"url"' not in text
    assert_no_key_material(text)


# -- 10. RunSummary arithmetic ------------------------------------------------------


def test_the_summary_counts_decisions_by_source_and_measures_the_rate() -> None:
    client = FakeClient(*(response("right", 0.9) for _ in range(3)))
    loop, _, _ = loop_with(settings=make_settings(), client=client)

    summary = loop.run(max_ticks=3)

    assert summary.ticks == 3
    assert summary.frames == 3
    assert summary.decisions == 3
    assert summary.sources[DecisionSource.JEV] == 3
    assert sum(summary.sources.values()) == summary.ticks
    assert summary.elapsed_s > 0.0
    assert summary.decision_rate_hz > 0.0
    assert summary.decision_rate_hz == pytest.approx(summary.decisions / summary.elapsed_s)


def test_a_zero_tick_run_reports_a_zero_rate() -> None:
    loop, _, _ = loop_with(settings=make_settings(), client=FakeClient())

    summary = loop.run(max_ticks=0)

    assert summary.ticks == 0
    assert summary.frames == 0
    assert summary.decisions == 0
    assert summary.decision_rate_hz == 0.0


# -- 11. the episode ends the loop cleanly ------------------------------------------


@pytest.mark.parametrize("ending", ["terminated", "truncated"])
def test_the_loop_stops_on_the_first_ended_step_and_never_steps_again(ending: str) -> None:
    emulator = FakeEmulator(
        terminate_at=3 if ending == "terminated" else None,
        truncate_at=3 if ending == "truncated" else None,
    )
    loop, _, _ = loop_with(
        settings=make_settings(key=None), emulator=emulator
    )

    records: list[object] = []
    summary = loop.run(observer=records.append)

    assert summary.ticks == 3
    assert len(records) == 3
    assert records[-1].decision.source is DecisionSource.FALLBACK_NO_KEY
    with pytest.raises(RuntimeError):
        loop.step()


# -- extra: the default collaborators are wired up ----------------------------------


def test_one_real_tick_runs_with_the_default_collaborators(emulator) -> None:
    settings = make_settings(key=None, cadence=8)

    loop = ExecutionLoop(settings, emulator=emulator)
    record = loop.step()

    assert record.decision.source is DecisionSource.FALLBACK_NO_KEY
    assert record.rollout.candidates
    assert set(record.state.to_jev_state()) == FROZEN_STATE_KEYS
    assert record.jev_questions is None
    assert set(record.latencies_ms) >= {"rollout", "jev", "advisor", "total"}


# -- 12. the record carries its own evidence ----------------------------------------


def tick_record_with_frame(frame: np.ndarray) -> TickRecord:
    """A hand-built record with fixed non-array fields, so equality is the only variable."""
    return TickRecord(
        tick=0,
        frame_index=0,
        emulator_tick=0,
        frame=frame,
        perception=detect(frame, tick=0),
        state=make_state(tick=0),
        rollout=all_safe_table(),
        jev_questions=None,
        jev_response=None,
        decision=TickDecision(
            action="right",
            source=DecisionSource.FALLBACK_NO_KEY,
            confidence=None,
            risk=None,
            reason="no_credential",
            jev_called=False,
        ),
        advisor=None,
        advisor_absence=None,
        latencies_ms={"rollout": 1.0, "jev": 0.0, "advisor": 0.0, "total": 2.0},
        degradations=(),
    )


def test_two_structurally_identical_records_compare_without_numpy_ambiguity() -> None:
    # Two distinct arrays with the same contents. Before compare=False this comparison
    # raised "the truth value of an array with more than one element is ambiguous".
    first = tick_record_with_frame(np.zeros(SCREEN, dtype=np.uint8))
    second = tick_record_with_frame(np.zeros(SCREEN, dtype=np.uint8))

    by_name = {spec.name: spec for spec in fields(TickRecord)}
    for name in ("frame", "perception", "advisor_absence"):
        assert by_name[name].compare is False, name

    assert first is not second
    assert first.frame is not second.frame
    assert first == second


def test_a_real_tick_carries_its_own_frame_and_perception(emulator) -> None:
    loop = ExecutionLoop(make_settings(key=None, cadence=8), emulator=emulator)

    record = loop.step()

    assert isinstance(record.frame, np.ndarray)
    assert record.frame.shape == (240, 256, 3)
    assert record.frame.dtype == np.uint8
    # The perception is exactly what detect() saw in that frame, and its tick is the
    # record's own frame index: the two cannot drift apart.
    assert record.perception == detect(record.frame, tick=record.frame_index)
    assert record.perception.tick == record.frame_index


def test_a_recorded_frame_is_a_copy_and_not_the_live_screen(emulator) -> None:
    """The record must hold pixels that survive later steps.

    ``MarioEmulator.frame()`` returns a fresh copy; ``screen_view()`` is the buffer
    every step mutates in place. A record holding the live buffer would show the
    *last* frame of the run in every tick, and the annotated artifact would be drawn
    from a frame that tick never saw -- silently, with no exception anywhere. This
    asserts the copy directly, against the real emulator, rather than relying on a
    fake that simply lacks the method.
    """
    loop = ExecutionLoop(make_settings(key=None, cadence=8), emulator=emulator)

    record = loop.step()
    captured = record.frame.copy()
    live = emulator.screen_view()

    assert not np.shares_memory(record.frame, live)
    assert record.frame is not live

    # Scribble on the live buffer. If the record aliased it, this would change the
    # recorded pixels; a copy cannot be reached this way.
    live[:] = 255

    assert not np.shares_memory(record.frame, live)
    assert np.array_equal(record.frame, captured)
    assert not np.array_equal(live, captured)


def test_an_enabled_but_absent_advisor_records_its_measured_absence() -> None:
    absence = AdvisorAbsence(model_id="OmniJev/PlayJev-0.8B", reason=REASON_TORCH_MISSING)
    advisor = FakeAdvisor(None, available=False, absence=absence)
    loop, _, _ = loop_with(
        settings=make_settings(key=None, advisor_enabled=True), advisor=advisor
    )

    record = loop.step()

    assert advisor.calls == 0
    assert record.advisor is None
    assert record.advisor_absence is absence


def test_a_disabled_advisor_records_no_measured_absence() -> None:
    advisor = FakeAdvisor(None)
    loop, _, _ = loop_with(
        settings=make_settings(key=None, advisor_enabled=False), advisor=advisor
    )

    record = loop.step()

    assert advisor.calls == 0
    assert record.advisor is None
    assert record.advisor_absence is None
