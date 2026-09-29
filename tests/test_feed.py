"""The recorded-answer feed: a run driven from a transcript, with no socket and no thread.

These tests use a fake emulator and a constructed rollout table (the same shape
``tests/test_policy.py`` uses), so a fed run is deterministic and reaches several
ticks without depending on how far Mario happens to walk. The feed itself is
exercised three ways: its own bookkeeping, a real ``ExecutionLoop`` over it, and a
real ``JournalWriter`` so the marking claim is made about the bytes on disk.

Nothing here imports another test file: the doubles are local, which is the
convention this suite already follows.
"""

from __future__ import annotations

import json
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from mario_jev.config import Settings
from mario_jev.emulator import StepResult, build_actions
from mario_jev.jev import (
    ChoiceAnswer,
    HttpResult,
    JevClient,
    JevResponse,
    NoulAnswer,
    ScoreAnswer,
    Usage,
)
from mario_jev.journal import JournalWriter, read_journal
from mario_jev.policy import (
    ExecutionLoop,
    JevDispatcher,
    RecordedAnswer,
    RecordedAnswerFeed,
)
from mario_jev.policy.loop import (
    ANSWERS_EXHAUSTED,
    TRANSCRIPT_ANSWERS,
    TRANSCRIPT_MISMATCH,
)
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
SENTINEL_KEY = "apikey_SENTINEL_feed_do_not_leak"

#: The seven ``simple`` actions, keyed by name, exactly as the emulator builds them.
ACTIONS = {action.name: action for action in build_actions("simple")}

SCREEN = (240, 256, 3)

#: The injected instant. Every deterministic test pins this and never sleeps.
FIXED_NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def fixed_clock() -> datetime:
    return FIXED_NOW


# -- builders -----------------------------------------------------------------------


def make_settings(*, cadence: int = 1) -> Settings:
    return Settings(
        jev_api_key=SENTINEL_KEY,
        decision_cadence_frames=cadence,
        confidence_threshold=0.6,
        rollout_budget_ms=120.0,
        real_time=False,
    )


def make_state(tick: int = 0) -> MarioState:
    return MarioState(
        tick=tick,
        player=PlayerFeatures(
            x=40, y=176, screen_x=40, feet_row=191, on_ground=True, facing="right"
        ),
        nearest_enemy=NearestEnemy(distance_px=64, approaching=True),
        terrain_ahead=TerrainAhead(
            gap_columns=0,
            gap_distance_px=None,
            wall_height=0,
            frames_to_jump_window=None,
            vertical_clearance=0,
            lookahead_px=64,
        ),
        progress=Progress(world="1-1", x_max=40, lives=3, time_left=400),
        speed_px_per_tick=2.0,
    )


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
    risk: float = 0.2,
    latency_ms: float = 3.0,
    attempts: int = 1,
) -> JevResponse:
    """A full four-answer body, the shape a live call records.

    Every question the loop asks is answered. A transcript that answered only
    ``best_action`` would be a record this build's own questions cannot bind to,
    which is exactly the mismatch the feed refuses, so a fixture standing in for a
    real recording must answer all four.
    """
    return JevResponse(
        model="jev-test",
        answers={
            "best_action": ChoiceAnswer(
                type="choice",
                choice=choice,
                confidence=confidence,
                probabilities={choice: 1.0},
            ),
            "risk": ScoreAnswer(
                type="score",
                score=risk,
                confidence=confidence,
                legend={"0": "safe", "1": "minor risk"},
                probabilities={"0": 0.5, "1": 0.5},
            ),
            "lethal_ahead": NoulAnswer(type="noul", noul=0.1),
            "gap_severity": ScoreAnswer(
                type="score",
                score=0.0,
                confidence=confidence,
                legend={"0": "no gap"},
                probabilities={"0": 1.0},
            ),
        },
        usage=Usage(input_tokens=10, output_tokens=5),
        latency_ms=latency_ms,
        attempts=attempts,
    )


def recorded(answer: JevResponse, *, delay_ticks: int, call_ms: float) -> RecordedAnswer:
    """One transcript entry: the recorded body, its age and what the call cost."""
    return RecordedAnswer(
        response=answer.as_dict(), delay_ticks=delay_ticks, call_ms=call_ms
    )


# -- loop doubles -------------------------------------------------------------------


class FakeEmulator:
    """A deterministic stand-in: a black frame and one frame per step."""

    def __init__(self) -> None:
        self._actions = build_actions("simple")
        self._tick = 0

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def action_names(self) -> tuple[str, ...]:
        return tuple(action.name for action in self._actions)

    def action(self, index: int):  # type: ignore[no-untyped-def]
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
        return StepResult(
            frame=self.frame(),
            info={},
            reward=0.0,
            terminated=False,
            truncated=False,
            tick=self._tick,
        )


class FakeBuilder:
    """Builds a fixed state; the loop attaches the rollout's candidates to it."""

    def observe(self, perception: object, ram: object, info: object) -> MarioState:
        return make_state(tick=int(perception.tick))  # type: ignore[attr-defined]


class FakePlanner:
    """Returns one constructed table and never touches the emulator."""

    def __init__(self, table: RolloutTable) -> None:
        self._table = table

    def plan(self, emulator: object, min_horizon: int | None = None) -> RolloutTable:
        return self._table


class InlineRunner:
    """A runner whose call finishes during ``submit``, on the calling thread."""

    def submit(self, fn: Callable[[], object]) -> Future[object]:
        future: Future[object] = Future()
        future.set_result(fn())
        return future

    def shutdown(self, *, wait: bool = True) -> None:
        """Nothing was started, so there is nothing to join."""


def feed_loop(
    feed: RecordedAnswerFeed,
    *,
    cadence: int = 1,
    settings: Settings | None = None,
    emulator: object | None = None,
    client: object | None = None,
) -> ExecutionLoop:
    """A loop whose only Jev source is ``feed``."""
    return ExecutionLoop(
        settings if settings is not None else make_settings(cadence=cadence),
        emulator=emulator if emulator is not None else FakeEmulator(),  # type: ignore[arg-type]
        planner=FakePlanner(all_safe_table()),
        client=client,  # type: ignore[arg-type]
        builder=FakeBuilder(),
        dispatcher=feed,
    )


# -- the feed's own bookkeeping -----------------------------------------------------


def test_a_second_request_is_refused_while_one_is_in_flight() -> None:
    feed = RecordedAnswerFeed(
        [recorded(response("right", 0.9), delay_ticks=2, call_ms=5.0)] * 2
    )
    state = make_state()

    assert feed.request({}, state, tick=0) is True
    assert feed.in_flight is True
    assert feed.request({}, state, tick=0) is False
    assert feed.exhausted is False


def test_an_exhausted_feed_refuses_the_call_and_fabricates_nothing() -> None:
    feed = RecordedAnswerFeed([])
    state = make_state()

    assert feed.request({}, state, tick=0) is False
    assert feed.in_flight is False
    assert feed.exhausted is True
    assert feed.take() is None


def test_the_recorded_delay_is_measured_in_takes_never_a_clock() -> None:
    feed = RecordedAnswerFeed(
        [recorded(response("right", 0.9), delay_ticks=2, call_ms=42.0)]
    )
    state = make_state()

    assert feed.request({}, state, tick=0) is True
    # ``take`` is called once per tick; the answer lands on the second call after
    # the fire, which is the recording's ``answer_age_ticks`` of 2 -- counting the
    # tick the fire happened on as zero.
    assert feed.take() is None
    assert feed.take() is None
    taken = feed.take()
    assert taken is not None
    assert taken.asked_at_tick == 0
    assert taken.call_ms == 42.0
    assert taken.error is None
    assert taken.response is not None


# -- a fed run over the loop --------------------------------------------------------


def test_a_fed_run_never_enters_a_client_or_opens_a_socket() -> None:
    class ExplodingTransport:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, request: object) -> HttpResult:
            self.calls += 1
            raise AssertionError("a fed run must not reach the transport")

    class RecordingClient:
        def __init__(self, inner: JevClient) -> None:
            self._inner = inner
            self.calls = 0

        def ask(self, questions: object, state: object) -> JevResponse:
            self.calls += 1
            return self._inner.ask(questions, state)  # type: ignore[arg-type]

    settings = make_settings(cadence=1)
    transport = ExplodingTransport()
    client = RecordingClient(
        JevClient(
            endpoint=settings.jev_endpoint,
            model=settings.jev_model,
            api_key=settings.jev_api_key,
            timeout_s=settings.jev_timeout_s,
            max_attempts=1,
            transport=transport,
            sleeper=lambda delay: None,  # type: ignore[arg-type,return-value]
        )
    )
    feed = RecordedAnswerFeed(
        [recorded(response("right", 0.9), delay_ticks=0, call_ms=250.0)]
    )

    # By construction the loop holds the feed itself, not a worker-backed
    # dispatcher: there is no thread to race and nothing that could open a socket.
    loop = feed_loop(feed, settings=settings, client=client)

    summary = loop.run(max_ticks=3)

    assert loop._dispatcher is feed  # noqa: SLF001 -- the construction is the claim
    assert not isinstance(loop._dispatcher, JevDispatcher)
    assert client.calls == 0, "a fed run entered the client"
    assert transport.calls == 0, "a fed run opened a socket"
    assert summary.ticks == 3


def test_the_kth_fired_call_gets_the_kth_recorded_answer_after_its_delay() -> None:
    first = response("right", 0.9, latency_ms=111.0)
    second = response("right+A", 0.9, latency_ms=222.0)
    feed = RecordedAnswerFeed(
        [
            recorded(first, delay_ticks=1, call_ms=111.0),
            recorded(second, delay_ticks=2, call_ms=222.0),
        ]
    )
    loop = feed_loop(feed, cadence=1)

    records = [loop.step() for _ in range(5)]

    # Call #1 fires on tick 0 and lands on tick 1; call #2 fires on tick 2 and
    # lands two of its own ticks later, on tick 4.
    assert [record.jev_questions is not None for record in records] == [
        True,
        False,
        True,
        False,
        False,
    ]
    applied = records[1]
    assert applied.asked_at_tick == 0
    assert applied.answer_age_ticks == 1
    assert applied.jev_response == first.as_dict()
    assert applied.jev_call_ms == 111.0
    assert applied.decision.action == "right"

    later = records[4]
    assert later.asked_at_tick == 2
    assert later.answer_age_ticks == 2
    assert later.jev_response == second.as_dict()
    assert later.jev_call_ms == 222.0
    assert later.decision.action == "right+A"


def test_a_build_that_fires_on_different_ticks_still_gets_the_answers_in_order() -> None:
    first = response("right", 0.9)
    second = response("right+A", 0.9)
    feed = RecordedAnswerFeed(
        [
            recorded(first, delay_ticks=1, call_ms=10.0),
            recorded(second, delay_ticks=2, call_ms=20.0),
        ]
    )
    loop = feed_loop(feed, cadence=3)

    records = [loop.step() for _ in range(6)]

    # The feed is keyed by call order, not by tick: this build fires at ticks 0
    # and 3 rather than 0 and 2, and still receives the first then the second
    # recorded answer, each after its own recorded delay.
    fires = [record.tick for record in records if record.jev_questions is not None]
    takes = [record.tick for record in records if record.answer_age_ticks is not None]
    assert fires == [0, 3]
    assert takes == [1, 5]
    applied = [
        record.jev_response["answers"]["best_action"]["choice"]  # type: ignore[index]
        for record in records
        if record.jev_response is not None
    ]
    assert applied == ["right", "right+A"]


def test_running_out_is_recorded_and_never_fabricated() -> None:
    feed = RecordedAnswerFeed(
        [recorded(response("right", 0.9), delay_ticks=0, call_ms=50.0)]
    )
    loop = feed_loop(feed, cadence=1)
    seen = []
    summary = loop.run(max_ticks=3, observer=seen.append)

    first, second, third = seen
    # Tick 0 fires and takes the one recorded answer (a delay of zero).
    assert first.jev_response is not None
    # Tick 1 and tick 2 try to fire again; the transcript is empty, so the feed
    # refuses and the loop records it rather than inventing a response.
    for record in (second, third):
        assert record.jev_questions is None
        assert record.jev_response is None
        assert record.answer_age_ticks is None
        assert record.decision.jev_called is False
        assert ANSWERS_EXHAUSTED in record.degradations
        assert TRANSCRIPT_MISMATCH not in record.degradations
    assert feed.exhausted is True
    assert summary.decisions == 1
    assert summary.answers_applied == 1
    assert summary.degradations[ANSWERS_EXHAUSTED] == 2


def test_an_unused_tail_is_never_asked_for_and_never_reported() -> None:
    feed = RecordedAnswerFeed(
        [recorded(response("right", 0.9), delay_ticks=0, call_ms=5.0)] * 3
    )
    loop = feed_loop(feed, cadence=1)

    summary = loop.run(max_ticks=1)

    assert summary.decisions == 1
    assert feed.exhausted is False
    assert ANSWERS_EXHAUSTED not in summary.degradations
    assert TRANSCRIPT_MISMATCH not in summary.degradations


# -- the question the record answers -------------------------------------------------


def test_a_build_that_renames_a_question_is_refused_and_the_record_is_kept() -> None:
    feed = RecordedAnswerFeed(
        [recorded(response("right", 0.9), delay_ticks=0, call_ms=5.0)]
    )
    state = make_state()
    renamed = {"best_action_v2": {"type": "choice", "criteria": {"right": "walk right"}}}

    assert feed.request(renamed, state, tick=0) is False
    assert feed.in_flight is False
    assert feed.mismatch is True
    assert feed.exhausted is False
    assert feed.take() is None

    # The record was not consumed: a later tick that asks the recorded question
    # still gets the answer, and the response answers exactly the ids it asked.
    asked = {"best_action": {"type": "choice", "criteria": {"right": "walk right"}}}
    assert feed.request(asked, state, tick=1) is True
    taken = feed.take()
    assert taken is not None
    assert taken.response is not None
    assert set(taken.response.answers) == {"best_action"}
    assert taken.response.choice("best_action").choice == "right"


def test_a_recorded_answer_under_a_changed_id_is_refused_and_recorded() -> None:
    body = response("right", 0.9).as_dict()
    answers = dict(body["answers"])  # type: ignore[arg-type]
    answers["best_action_v2"] = answers.pop("best_action")
    body["answers"] = answers
    feed = RecordedAnswerFeed(
        [RecordedAnswer(response=body, delay_ticks=0, call_ms=7.0)]
    )
    loop = feed_loop(feed, cadence=1)

    records = [loop.step() for _ in range(2)]

    for record in records:
        assert record.jev_questions is None
        assert record.jev_response is None
        assert record.decision.jev_called is False
        assert TRANSCRIPT_MISMATCH in record.degradations
        assert ANSWERS_EXHAUSTED not in record.degradations
    assert feed.mismatch is True
    assert feed.exhausted is False
    assert feed.in_flight is False


def test_a_recorded_answer_of_a_changed_type_is_refused_and_recorded() -> None:
    body = response("right", 0.9).as_dict()
    answers = dict(body["answers"])  # type: ignore[arg-type]
    # The recording answered ``best_action`` as a score, not the choice the build
    # asks for. The record parses on its own terms; the build cannot use it.
    answers["best_action"] = {
        "type": "score",
        "score": 1.0,
        "confidence": 0.9,
        "legend": {"0": "safe"},
        "probabilities": {"0": 1.0},
    }
    body["answers"] = answers
    feed = RecordedAnswerFeed(
        [RecordedAnswer(response=body, delay_ticks=0, call_ms=7.0)]
    )
    loop = feed_loop(feed, cadence=1)

    record = loop.step()

    assert record.jev_questions is None
    assert record.jev_response is None
    assert TRANSCRIPT_MISMATCH in record.degradations
    assert ANSWERS_EXHAUSTED not in record.degradations
    assert feed.mismatch is True


def test_a_recorded_choice_the_build_never_offers_is_refused_and_recorded() -> None:
    # The table offers the seven ``simple`` actions; this recording chose one it
    # never offered, so the build's own criteria do not contain it.
    feed = RecordedAnswerFeed(
        [recorded(response("left+A", 0.9), delay_ticks=0, call_ms=7.0)]
    )
    loop = feed_loop(feed, cadence=1)

    record = loop.step()

    assert record.jev_questions is None
    assert record.jev_response is None
    assert TRANSCRIPT_MISMATCH in record.degradations
    assert ANSWERS_EXHAUSTED not in record.degradations
    assert feed.mismatch is True


def test_a_recorded_legend_naming_an_unasked_level_is_refused_and_recorded() -> None:
    feed = RecordedAnswerFeed(
        [recorded(response("right", 0.9), delay_ticks=0, call_ms=5.0)]
    )
    asked = {"risk": {"type": "score", "criteria": ["calm", "uneasy"]}}

    assert feed.request(asked, make_state(), tick=0) is False
    assert feed.in_flight is False
    assert feed.mismatch is True
    assert feed.exhausted is False
    assert feed.take() is None


def test_a_build_that_asks_a_subset_is_answered_and_is_not_a_mismatch() -> None:
    feed = RecordedAnswerFeed(
        [recorded(response("right", 0.9), delay_ticks=0, call_ms=5.0)]
    )
    asked = {
        "best_action": {"type": "choice", "criteria": {"right": "walk right"}},
        "risk": {
            "type": "score",
            "criteria": ["safe", "minor risk", "serious risk", "near-certain death"],
        },
    }

    assert feed.request(asked, make_state(), tick=0) is True
    taken = feed.take()

    assert taken is not None
    assert taken.response is not None
    # Answered for exactly the ids asked. The record's unanswered questions were
    # not asked, so they are not a mismatch and do not appear in the response: an
    # answer the build did not ask for is not this build's evidence.
    assert set(taken.response.answers) == {"best_action", "risk"}
    assert taken.response.choice("best_action").choice == "right"
    assert feed.mismatch is False
    assert feed.exhausted is False


# -- the journal's own claim --------------------------------------------------------


def test_the_journal_marks_a_fed_run_and_its_recorded_latency(tmp_path: Path) -> None:
    settings = make_settings(cadence=1)
    feed = RecordedAnswerFeed(
        [recorded(response("right", 0.9), delay_ticks=0, call_ms=500.0)]
    )
    loop = feed_loop(feed, settings=settings)
    path = tmp_path / "fed.jsonl"
    writer = JournalWriter(path, settings=settings, clock=fixed_clock)
    summary = loop.run(max_ticks=2, observer=writer.observer())
    writer.write_summary(summary)
    writer.close()

    journal = read_journal(path)

    applied = journal.ticks[0]
    # The value is the recording's latency, and the same line says where it came
    # from, so a reader cannot mistake it for a measurement of this machine.
    assert applied["jev_call_ms"] == 500.0
    assert TRANSCRIPT_ANSWERS in applied["degradations"]
    assert TRANSCRIPT_MISMATCH not in applied["degradations"]
    assert applied["response"] is not None
    # The run-level stamp: the summary aggregates the per-tick mark.
    assert journal.summary is not None
    assert journal.summary["degradations"][TRANSCRIPT_ANSWERS] == 2

    # And the written journal is itself a usable transcript: the fed run's own
    # file round-trips through the reader the feed is built from.
    rebuilt = RecordedAnswerFeed.from_ticks(journal.ticks)
    second = feed_loop(rebuilt, settings=settings)
    records = [second.step() for _ in range(5)]
    first_takes = [
        (tick["asked_at_tick"], tick["answer_age_ticks"])
        for tick in journal.ticks
        if tick["answer_age_ticks"] is not None
    ]
    second_takes = [
        (record.asked_at_tick, record.answer_age_ticks)
        for record in records
        if record.answer_age_ticks is not None
    ]
    assert second_takes == first_takes
    assert second_takes == [(0, 0)]


def test_a_live_dispatcher_run_carries_no_transcript_stamp() -> None:
    settings = make_settings(cadence=1)
    transport = ScriptedTransport(ok_body("right"))
    client = real_client(settings, transport)
    loop = ExecutionLoop(
        settings,
        emulator=FakeEmulator(),
        planner=FakePlanner(all_safe_table()),
        client=client,
        builder=FakeBuilder(),
        dispatcher=JevDispatcher(client, runner=InlineRunner()),
    )

    record = loop.step()

    assert record.jev_call_ms is not None, "the live path must still measure its call"
    assert TRANSCRIPT_ANSWERS not in record.degradations
    assert ANSWERS_EXHAUSTED not in record.degradations
    assert TRANSCRIPT_MISMATCH not in record.degradations


def test_a_mismatched_run_and_an_exhausted_run_journal_different_signals(
    tmp_path: Path,
) -> None:
    settings = make_settings(cadence=1)

    body = response("right", 0.9).as_dict()
    answers = dict(body["answers"])  # type: ignore[arg-type]
    answers["best_action_v2"] = answers.pop("best_action")
    body["answers"] = answers
    mismatched_loop = feed_loop(
        RecordedAnswerFeed([RecordedAnswer(response=body, delay_ticks=0, call_ms=9.0)]),
        settings=settings,
    )
    mismatched_path = tmp_path / "mismatched.jsonl"
    writer = JournalWriter(mismatched_path, settings=settings, clock=fixed_clock)
    mismatched_summary = mismatched_loop.run(max_ticks=2, observer=writer.observer())
    writer.write_summary(mismatched_summary)
    writer.close()

    exhausted_loop = feed_loop(
        RecordedAnswerFeed([recorded(response("right", 0.9), delay_ticks=0, call_ms=9.0)]),
        settings=settings,
    )
    exhausted_path = tmp_path / "exhausted.jsonl"
    writer = JournalWriter(exhausted_path, settings=settings, clock=fixed_clock)
    exhausted_summary = exhausted_loop.run(max_ticks=2, observer=writer.observer())
    writer.write_summary(exhausted_summary)
    writer.close()

    mismatch_journal = read_journal(mismatched_path)
    exhausted_journal = read_journal(exhausted_path)
    assert mismatch_journal.summary is not None
    assert exhausted_journal.summary is not None

    # Side by side, from the same reader: two fed runs, two different reasons, and
    # neither reason appears on the other run. "The transcript ran out" can never
    # be read as "the transcript answers someone else".
    mismatch_counts = mismatch_journal.summary["degradations"]
    exhausted_counts = exhausted_journal.summary["degradations"]
    assert mismatch_counts[TRANSCRIPT_MISMATCH] == 2
    assert ANSWERS_EXHAUSTED not in mismatch_counts
    assert exhausted_counts[ANSWERS_EXHAUSTED] == 1
    assert TRANSCRIPT_MISMATCH not in exhausted_counts

    # The fed stamp is on both, so it is not what tells them apart.
    assert TRANSCRIPT_ANSWERS in mismatch_counts
    assert TRANSCRIPT_ANSWERS in exhausted_counts

    # And the reason is on the tick line itself, not only in the summary.
    assert TRANSCRIPT_MISMATCH in mismatch_journal.ticks[0]["degradations"]
    assert ANSWERS_EXHAUSTED in exhausted_journal.ticks[1]["degradations"]


# -- a scripted real client, for the live half of the comparison --------------------


class ScriptedTransport:
    """Hands back queued bodies and remembers the requests; never opens a socket."""

    def __init__(self, *bodies: str) -> None:
        self._bodies = list(bodies)
        self.calls = 0

    def __call__(self, request: object) -> HttpResult:
        self.calls += 1
        return HttpResult(status=200, body=self._bodies.pop(0), elapsed_ms=1.0)


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


def ok_body(choice: str = "right") -> str:
    """A full four-answer body, the only shape the real client accepts."""
    return json.dumps(
        {
            "model": "jev-test",
            "answers": {
                "best_action": {
                    "type": "choice",
                    "choice": choice,
                    "confidence": 0.9,
                    "probabilities": {choice: 1.0},
                },
                "risk": {
                    "type": "score",
                    "score": 0.2,
                    "confidence": 0.9,
                    "legend": {"0": "safe", "1": "minor risk"},
                    "probabilities": {"0": 0.6, "1": 0.4},
                },
                "lethal_ahead": {"type": "noul", "noul": 0.1},
                "gap_severity": {
                    "type": "score",
                    "score": 0.0,
                    "confidence": 0.9,
                    "legend": {"0": "no gap"},
                    "probabilities": {"0": 1.0},
                },
            },
            "usage": {"input_tokens": 5, "output_tokens": 5},
        }
    )
