"""The decision layer: cadence, confidence gate, guardrails, and the tick loop.

Nothing here touches the network or the clock. The pure rule is exercised on
constructed tables and responses; the loop is exercised with a fake emulator, a
constructed rollout table, fakes for the client/planner/advisor, and — for the one
behaviour that must abort rather than degrade — a *real* ``JevClient`` driving a
scripted transport, so the exception the loop must not catch is the real one the
client raises. Two tests reach the real emulator: one runs a single real tick with
every default collaborator to pin that the defaults are wired up, and one runs a
whole scripted run into a real death to pin the ``"death"`` classification against
the real thing rather than against a fake.
"""

from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Callable

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
    TERMINATIONS,
    DecisionCadence,
    DecisionSource,
    ExecutionLoop,
    RunSummary,
    TickDecision,
    TickRecord,
    advisor_view,
    journal_state,
    select_action,
)
from mario_jev.journal import JournalWriter, read_journal
from mario_jev.perception import detect

# From the modules rather than the packages: the stall constants are new, and the
# curated package ``__all__``s do not re-export them.
from mario_jev.policy.loop import JEV_RETRIED, STALL_ESCALATION, STALL_FRAMES
from mario_jev.state import (
    HAZARD_OUTCOMES,
    INFO_DEATH,
    INFO_FLAG_GET,
    MarioState,
    NearestEnemy,
    PlayerFeatures,
    Progress,
    TerrainAhead,
    TerminalOutcome,
)
from mario_jev.trajectory import CandidateRollout, CostSource, RolloutTable
from mario_jev.trajectory.planner import STALL_MIN_HORIZON

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
    attempts: int = 1,
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
        attempts=attempts,
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


def make_state(tick: int = 0, *, x_max: int = 40) -> MarioState:
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
        progress=Progress(world="1-1", x_max=x_max, lives=3, time_left=400),
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

    def __init__(
        self,
        *,
        terminate_at: int | None = None,
        truncate_at: int | None = None,
        end_info: dict[str, object] | None = None,
    ) -> None:
        self._actions = build_actions("simple")
        self._tick = 0
        self._steps = 0
        self._terminate_at = terminate_at
        self._truncate_at = truncate_at
        # What the *ending* step reports in its info. The real emulator sets the
        # death/clear bits on the step that ends the episode, not before it, so
        # this fake does the same: an ordinary step carries an empty info.
        self._end_info = {} if end_info is None else dict(end_info)
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
        terminated = self._terminate_at is not None and self._steps >= self._terminate_at
        truncated = self._truncate_at is not None and self._steps >= self._truncate_at
        return StepResult(
            frame=self.frame(),
            info=dict(self._end_info) if terminated or truncated else {},
            reward=0.0,
            terminated=terminated,
            truncated=truncated,
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


class ProgressBuilder:
    """A builder whose ``x_max`` follows a script, so the loop has a real history.

    The stall is a property of the run over time, not of one frame, so the only
    way to exercise it is to hand the loop a sequence of states that does or does
    not advance. Everything else about the state is what :func:`make_state`
    builds.
    """

    def __init__(self, x_max_at: Callable[[int], int]) -> None:
        self._x_max_at = x_max_at
        self.observed: list[int] = []

    def observe(self, perception: object, ram: object, info: object) -> MarioState:
        tick = int(perception.tick)  # type: ignore[attr-defined]
        x_max = self._x_max_at(tick)
        self.observed.append(x_max)
        return make_state(tick=tick, x_max=x_max)


class StallPlanner:
    """Records what floor each tick asked for, and answers the way the real planner does.

    ``budget_horizon`` stands in for what the measured cost model would have
    chosen on its own; a floor above it raises the horizon and is an escalation,
    a floor below it changes nothing and is not one.
    """

    def __init__(self, *, budget_horizon: int = 8) -> None:
        self.budget_horizon = budget_horizon
        self.asked: list[int | None] = []

    def plan(self, emulator: object, *, min_horizon: int | None = None) -> RolloutTable:
        self.asked.append(min_horizon)
        horizon = self.budget_horizon
        if min_horizon is not None:
            horizon = max(horizon, min_horizon)
        return replace(
            all_safe_table(),
            horizon=horizon,
            escalated=horizon > self.budget_horizon,
        )


class EmptyTablePlanner:
    """A planner that answers with no candidates at all.

    Stands in for a caller-injected planner whose action set is empty -- the only way
    to reach a candidate-less table through the real path is an emulator with no
    actions, which the action sets never produce. The loop must still behave: the
    ticket recounts no escalation and degrades to ``fallback_no_candidates``.
    """

    def __init__(self) -> None:
        self.asked: list[int | None] = []

    def plan(self, emulator: object, *, min_horizon: int | None = None) -> RolloutTable:
        self.asked.append(min_horizon)
        return replace(all_safe_table(), candidates=())


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
    builder: object | None = None,
) -> tuple[ExecutionLoop, FakeEmulator, object]:
    emulator = emulator if emulator is not None else FakeEmulator()
    planner = planner if planner is not None else FakePlanner(table or all_safe_table())
    loop = ExecutionLoop(
        settings,
        emulator=emulator,
        planner=planner,
        client=client,  # type: ignore[arg-type]
        builder=builder if builder is not None else FakeBuilder(),
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


# -- 11b. why the run stopped -------------------------------------------------------
# The live run that motivated these: 805 tick records against a 900-tick budget,
# and nothing in the journal that could tell a death from the flag from a hand on
# the keyboard. The classification below is the missing fact.


@pytest.mark.parametrize(
    ("terminate_at", "truncate_at", "end_info", "expected"),
    [
        # The episode's own signals, read through the state layer's predicates.
        (3, None, {INFO_DEATH: True}, "death"),
        (3, None, {INFO_FLAG_GET: True}, "goal"),
        # Neither signal, so the catch-alls: ``truncated`` names itself, and an
        # episode that simply ended is ``episode_ended`` rather than a guess.
        (None, 3, {}, "truncated"),
        (3, None, {}, "episode_ended"),
        # Precedence, stated as a test and not only as a comment: ``truncated``
        # may accompany a real signal, and the real signal wins.
        (3, 3, {INFO_DEATH: True}, "death"),
        (3, 3, {INFO_FLAG_GET: True}, "goal"),
    ],
)
def test_the_summary_names_why_the_episode_ended(
    terminate_at: int | None,
    truncate_at: int | None,
    end_info: dict[str, object],
    expected: str,
) -> None:
    emulator = FakeEmulator(
        terminate_at=terminate_at, truncate_at=truncate_at, end_info=end_info
    )
    loop, _, _ = loop_with(settings=make_settings(key=None), emulator=emulator)

    summary = loop.run(max_ticks=10)

    assert summary.termination == expected
    assert summary.termination in TERMINATIONS
    # The episode ended early, and the count alone could not have said so.
    assert summary.ticks == 3


def test_reaching_the_tick_cap_is_recorded_as_max_ticks() -> None:
    emulator = FakeEmulator()
    loop, _, _ = loop_with(settings=make_settings(key=None), emulator=emulator)

    records: list[TickRecord] = []
    summary = loop.run(max_ticks=4, observer=records.append)

    assert summary.termination == "max_ticks"
    assert summary.ticks == 4
    # The cap stopped the run, so nothing about the last step ended the episode.
    assert records[-1].terminated is False
    assert records[-1].truncated is False


def test_a_zero_tick_run_stopped_at_its_cap() -> None:
    loop, _, _ = loop_with(settings=make_settings(key=None))

    assert loop.run(max_ticks=0).termination == "max_ticks"


@pytest.mark.parametrize("ending", ["terminated", "truncated"])
def test_only_the_final_record_carries_the_end_of_the_episode(ending: str) -> None:
    """The fact the journal was missing: *which* tick ended the run.

    Every other field on a record describes the world the tick decided from,
    read before the action ran, so the last line of the old journal still showed
    two lives on the tick that lost the third.
    """
    emulator = FakeEmulator(
        terminate_at=3 if ending == "terminated" else None,
        truncate_at=3 if ending == "truncated" else None,
        end_info={INFO_DEATH: True} if ending == "terminated" else {},
    )
    loop, _, _ = loop_with(settings=make_settings(key=None), emulator=emulator)

    records: list[TickRecord] = []
    loop.run(max_ticks=10, observer=records.append)

    assert len(records) == 3
    for record in records[:-1]:
        assert record.terminated is False
        assert record.truncated is False
    final = records[-1]
    assert getattr(final, ending) is True
    assert (final.terminated, final.truncated) == (ending == "terminated", ending == "truncated")


def summary_with(termination: str) -> RunSummary:
    return RunSummary(
        ticks=1,
        frames=1,
        decisions=0,
        sources={},
        degradations={},
        elapsed_s=1.0,
        decision_rate_hz=0.0,
        termination=termination,
    )


@pytest.mark.parametrize("termination", sorted(TERMINATIONS))
def test_every_member_of_the_closed_set_constructs(termination: str) -> None:
    assert summary_with(termination).termination == termination


@pytest.mark.parametrize("invalid", ["", "finished", "MAX_TICKS", "crashed", "None"])
def test_the_summary_refuses_a_termination_outside_the_closed_set(invalid: str) -> None:
    with pytest.raises(ValueError, match="unknown termination"):
        summary_with(invalid)


def test_the_refusal_message_names_every_valid_termination() -> None:
    """Refusing is not enough: the message has to say what is allowed.

    A caller who got this wrong is holding a traceback, and "unknown termination:
    'made_up'" tells them only that they are wrong. Naming the closed set means
    they do not have to open this module to find the answer, and it means the set
    cannot grow without this test noticing.
    """
    with pytest.raises(ValueError) as caught:
        summary_with("made_up")

    message = str(caught.value)
    assert "unknown termination: 'made_up'" in message
    for member in TERMINATIONS:
        assert member in message, member
    # Every member, not merely some: five names, and the list is not a prefix of
    # a longer set that went unmentioned.
    assert len(TERMINATIONS) == 5


def test_a_wrong_but_valid_termination_is_accepted_and_that_is_documented() -> None:
    """The check is set membership, not truth, and the docstring says so.

    ``termination="goal"`` on a run that hit its tick cap is a member of the closed
    set and constructs fine. That is inherent to a plain dataclass whose public
    construction is legitimate -- the journal readers rebuild summaries -- and no
    validator here could tell a wrong-but-valid reason from a right one. Only
    :func:`classify_termination`, which watched the run, can. This test pins the
    acceptance so nobody mistakes the type check for a correctness guarantee, and
    pins that the field's own documentation states the limit.
    """
    wrong = summary_with("goal")

    assert wrong.termination == "goal"
    # Whitespace-normalised: the assertion is about the sentence being there, not
    # about where the docstring happens to wrap.
    documentation = " ".join((RunSummary.__doc__ or "").split())
    assert "set membership, not truth" in documentation
    assert "classify_termination` is the single producer" in documentation


def test_the_closed_set_is_exactly_the_five_documented_reasons() -> None:
    assert TERMINATIONS == frozenset(
        {"max_ticks", "goal", "death", "truncated", "episode_ended"}
    )


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


# -- extra: the death classification, against the real emulator ---------------------
# The fakes above pin the classification's *rule*. They cannot pin that the rule
# fires on a real death, because they decide what ``terminated`` and ``info`` say.
# An unscripted real run is no better as evidence: four such runs survived to their
# 400-tick cap, so a death that happened once was luck and does not reproduce.
#
# What does reproduce is a *scripted* real run. Holding ``right`` from the reset of
# 1-1 walks into the first Goomba on frame 160, every time, so this test is
# deterministic without depending on anything that varies.


class RightOnlyPlanner:
    """One fixed table whose only candidate -- and so whose ``best`` -- is ``right``.

    The point is to remove every source of variation *except* the emulator. It runs
    no sweep, so it never steps the emulator and the loop's frame count is exactly
    the number of game frames; it touches no clock, so the rollout's cost model
    cannot change the answer; and one candidate means the fallback's tie-break
    cannot either. ``min_horizon`` is accepted because a stalled tick passes one,
    and is ignored because there is no horizon here to lengthen.
    """

    def __init__(self) -> None:
        self.plans = 0
        self.asked: list[int | None] = []

    def plan(self, emulator: object, *, min_horizon: int | None = None) -> RolloutTable:
        self.plans += 1
        self.asked.append(min_horizon)
        return table_of(candidate("right", risk=0.1, dx=8))


def test_a_scripted_real_run_dies_and_the_journal_says_so(
    emulator, tmp_path: Path
) -> None:
    """The real emulator, the real ``run()``, and a scripted decision source.

    No credential, so no network: the decision is ``FALLBACK_NO_KEY`` and takes the
    table's best candidate, which is ``right`` on every tick. What is under test is
    everything between the emulator's own ``terminated`` flag and the two places a
    reader looks for it -- ``RunSummary.termination`` and the journal file.

    The cap is generous on purpose. It is a guard against an infinite loop, not a
    prediction: if it were ever reached the termination would read ``max_ticks`` and
    this test would fail loudly rather than quietly testing nothing.
    """
    planner = RightOnlyPlanner()
    settings = make_settings(key=None, cadence=1)
    loop = ExecutionLoop(settings, emulator=emulator, planner=planner)
    path = tmp_path / "death.jsonl"
    writer = JournalWriter(path, settings=settings)

    records: list[TickRecord] = []

    def observe(record: TickRecord) -> None:
        records.append(record)
        writer.observer()(record)

    summary = loop.run(max_ticks=400, observer=observe)
    writer.write_summary(summary)
    writer.close()

    # 1. the classification, from the real episode's own signals.
    assert summary.termination == "death"
    assert summary.ticks == len(records) < 400
    assert all(record.decision.action == "right" for record in records)
    assert planner.plans == summary.ticks

    # 2. the terminating tick is the last one, and only the last one.
    assert records[-1].terminated is True
    assert [index for index, r in enumerate(records) if r.terminated] == [
        len(records) - 1
    ]
    assert not any(record.truncated for record in records)

    # 3. the same run's file agrees, read back from disk.
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    tick_lines = [line for line in lines if line["kind"] == "tick"]
    summary_line = lines[-1]

    assert summary_line["kind"] == "summary"
    assert summary_line["termination"] == "death"
    assert len(tick_lines) == summary.ticks
    assert tick_lines[-1]["terminated"] is True
    assert [line["terminated"] for line in tick_lines[:-1]] == [False] * (
        len(tick_lines) - 1
    )
    # And through the reader the dashboard uses, not only through raw json.
    assert read_journal(path).summary["termination"] == "death"


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


# -- 10. the stall escalation -------------------------------------------------------
#
# Only the loop can see a stall: the planner is handed one savestate and cannot
# know whether the last thirty frames went anywhere. The measured case is the
# recorded 1-1 run, where x_max stopped at 724 and stayed there for 600 frames
# while every candidate at the budget's horizon came back `no_progress` at one
# identical risk. These tests pin the rule that gives the loop a way out of that,
# and the rule that it stops the moment it is no longer needed.


def stalled_loop(
    x_max_at: Callable[[int], int], *, budget_horizon: int = 8
) -> tuple[ExecutionLoop, StallPlanner]:
    planner = StallPlanner(budget_horizon=budget_horizon)
    loop, _, _ = loop_with(
        settings=make_settings(key=None),
        planner=planner,
        builder=ProgressBuilder(x_max_at),
    )
    return loop, planner


def test_a_stall_escalates_the_horizon_and_the_tick_records_that_it_did() -> None:
    loop, planner = stalled_loop(lambda tick: 724)
    records: list[TickRecord] = []

    loop.run(max_ticks=STALL_FRAMES + 2, observer=records.append)

    # Before the stall is measured, nothing changes: the loop asks for no floor,
    # the planner plans on its budget, and the tick records no degradation.
    for record in records[:STALL_FRAMES]:
        assert record.rollout.escalated is False
        assert STALL_ESCALATION not in record.degradations
    assert planner.asked[:STALL_FRAMES] == [None] * STALL_FRAMES

    # From the STALL_FRAMES'th frame without progress, every tick escalates, and
    # says so.
    for record in records[STALL_FRAMES:]:
        assert record.rollout.escalated is True
        assert record.rollout.horizon == STALL_MIN_HORIZON
        assert STALL_ESCALATION in record.degradations
    assert planner.asked[STALL_FRAMES:] == [STALL_MIN_HORIZON] * 2


def test_a_stalled_tick_with_no_candidates_claims_no_escalation() -> None:
    """No candidates is no horizon to lengthen, so nothing may be claimed.

    The loop measures the stall before it plans, so ``plan()`` still receives the
    floor -- it cannot know the table will be empty. What it must not do is record an
    escalation on a tick where no cost could have been paid, when
    ``fallback_no_candidates`` already says what happened.
    """
    planner = EmptyTablePlanner()
    loop, _, _ = loop_with(
        settings=make_settings(key=None),
        planner=planner,
        builder=ProgressBuilder(lambda tick: 724),
    )
    records: list[TickRecord] = []

    loop.run(max_ticks=STALL_FRAMES + 1, observer=records.append)

    stalled = records[-1]
    # The stall was measured and the floor was asked for ...
    assert planner.asked[-1] == STALL_MIN_HORIZON
    assert stalled.rollout.candidates == ()
    # ... but the tick claims no escalation, and points at the reason instead.
    assert STALL_ESCALATION not in stalled.degradations
    assert stalled.decision.source is DecisionSource.FALLBACK_NO_CANDIDATES


def test_escalation_stops_on_the_first_tick_that_moves_again() -> None:
    # Stalled through the escalating tick, then advancing one pixel per frame.
    resumed = STALL_FRAMES + 1
    loop, planner = stalled_loop(
        lambda tick: 724 if tick < resumed else 724 + (tick - resumed) + 1
    )
    records: list[TickRecord] = []

    loop.run(max_ticks=resumed + 2, observer=records.append)

    assert records[STALL_FRAMES].rollout.escalated is True
    assert STALL_ESCALATION in records[STALL_FRAMES].degradations
    # The counter is reset by the progress itself, not decayed: the very next
    # tick plans on its budget again, and so does the one after it.
    for record in records[resumed:]:
        assert record.rollout.escalated is False
        assert record.rollout.horizon == planner.budget_horizon
        assert STALL_ESCALATION not in record.degradations
    assert planner.asked[resumed:] == [None, None]


def test_a_tick_with_no_stall_plans_on_the_budget_alone() -> None:
    loop, planner = stalled_loop(lambda tick: 724 + 4 * tick, budget_horizon=8)
    records: list[TickRecord] = []

    loop.run(max_ticks=STALL_FRAMES + 2, observer=records.append)

    assert planner.asked == [None] * (STALL_FRAMES + 2)
    for record in records:
        assert record.rollout.escalated is False
        assert record.rollout.horizon == 8
        assert STALL_ESCALATION not in record.degradations


# -- 11. the retry the tick paid for -------------------------------------------------
#
# The client retries 429s, 5xxs and timeouts, and a tick that succeeded on its
# second attempt is not a clean tick: it spent a whole failed call's wall clock
# first. The measured case is a tick whose total read 21 272 ms while its
# recorded response read 577 ms, with an empty ``degradations`` list.


def test_a_retried_response_is_recorded_as_a_degradation() -> None:
    client = FakeClient(response("right", 0.9, risk=0.2, attempts=2))
    loop, _, _ = loop_with(settings=make_settings(cadence=1), client=client)

    record = loop.step()

    assert record.decision.jev_called is True
    assert JEV_RETRIED in record.degradations


def test_a_first_try_response_records_no_retry() -> None:
    client = FakeClient(response("right", 0.9, risk=0.2, attempts=1))
    loop, _, _ = loop_with(settings=make_settings(cadence=1), client=client)

    record = loop.step()

    assert record.decision.jev_called is True
    assert JEV_RETRIED not in record.degradations


def test_a_tick_that_made_no_call_records_no_retry() -> None:
    """No call is no attempt count, and certainly not a retry.

    With ``cadence=2`` the second tick is off cadence and never reaches the
    client, so there is no response to read a count from.
    """
    client = FakeClient(response("right", 0.9, risk=0.2, attempts=2))
    loop, _, _ = loop_with(settings=make_settings(cadence=2), client=client)

    called = loop.step()
    uncalled = loop.step()

    assert called.decision.jev_called is True
    assert JEV_RETRIED in called.degradations
    assert uncalled.decision.jev_called is False
    assert uncalled.jev_response is None
    assert JEV_RETRIED not in uncalled.degradations


def test_a_retry_and_a_stall_escalation_co_occur_and_are_both_counted() -> None:
    ticks = STALL_FRAMES + 1
    client = FakeClient(
        *(response("right", 0.9, risk=0.2, attempts=2) for _ in range(ticks))
    )
    loop, _, _ = loop_with(
        settings=make_settings(cadence=1),
        planner=StallPlanner(),
        client=client,
        builder=ProgressBuilder(lambda tick: 724),
    )
    records: list[TickRecord] = []

    summary = loop.run(max_ticks=ticks, observer=records.append)

    stalled = records[-1]
    assert JEV_RETRIED in stalled.degradations
    assert STALL_ESCALATION in stalled.degradations
    # The existing counter aggregates both; neither needs a summary field of its own.
    assert summary.degradations[JEV_RETRIED] == ticks
    assert summary.degradations[STALL_ESCALATION] == 1
