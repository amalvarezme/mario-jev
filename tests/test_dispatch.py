"""The dispatcher: one worker, at most one call in flight, a take that never blocks.

Nothing here consults the real clock or sleeps for a call to land. Every sequence
is driven through :class:`ManualRunner`, which holds the submitted work until the
test decides it has finished, so \"the call is still running\" and \"the call just
finished\" are states a test *sets* rather than states it waits for.

Two tests do use a real :class:`~concurrent.futures.ThreadPoolExecutor`, because
the default runner is part of the contract and a manual double cannot show that
``close()`` really joins a thread. They use a client that returns immediately and
they assert on ``threading.enumerate()``, which ``shutdown(wait=True)`` makes
deterministic.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from typing import Callable

import pytest

from mario_jev.jev import ChoiceAnswer, JevAuthError, JevResponse, JevUnavailable, Usage
from mario_jev.policy import JevDispatcher, Taken
from mario_jev.state import (
    MarioState,
    NearestEnemy,
    PlayerFeatures,
    Progress,
    TerrainAhead,
)

#: A synthetic credential. A real key is never used, and never printed by a test.
SENTINEL_KEY = "apikey_SENTINEL_dispatch_do_not_leak"

QUESTIONS: dict[str, dict[str, object]] = {
    "best_action": {"type": "choice", "criteria": ["right"], "text": "which?"}
}


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


def response(choice: str = "right", *, confidence: float = 0.9) -> JevResponse:
    return JevResponse(
        model="jev-test",
        answers={
            "best_action": ChoiceAnswer(
                type="choice",
                choice=choice,
                confidence=confidence,
                probabilities={choice: 1.0},
            )
        },
        usage=Usage(input_tokens=10, output_tokens=5),
        latency_ms=3.0,
        attempts=1,
    )


class ManualRunner:
    """A runner the test drives by hand: nothing runs until :meth:`run_next` says so.

    This is the seam that makes every sequence below deterministic. A real worker
    would make \"has it finished yet?\" a race; here it is a question the test has
    already answered.
    """

    def __init__(self) -> None:
        self.pending: list[tuple[Future[object], Callable[[], object]]] = []
        self.submitted = 0
        self.shutdowns = 0

    def submit(self, fn: Callable[[], object]) -> Future[object]:
        self.submitted += 1
        future: Future[object] = Future()
        self.pending.append((future, fn))
        return future

    def run_next(self) -> None:
        """Finish the oldest submitted call, as a real worker eventually would."""
        future, fn = self.pending.pop(0)
        future.set_result(fn())

    def shutdown(self, *, wait: bool = True) -> None:
        self.shutdowns += 1


class FakeClient:
    """A queued-response client that records the arguments each call received."""

    def __init__(self, *responses: JevResponse, error: Exception | None = None) -> None:
        self._responses = list(responses)
        self._error = error
        self.calls: list[tuple[object, object]] = []

    def ask(self, questions: object, state: object) -> JevResponse:
        self.calls.append((questions, state))
        if self._error is not None:
            raise self._error
        if not self._responses:
            raise AssertionError("the dispatcher called the client more often than queued")
        return self._responses.pop(0)


class StepClock:
    """A clock that advances a fixed amount per reading, so durations are exact."""

    def __init__(self, *, step_s: float) -> None:
        self.step_s = step_s
        self.now = 0.0

    def __call__(self) -> float:
        value = self.now
        self.now += self.step_s
        return value


def dispatcher_with(
    client: object, runner: ManualRunner, *, step_s: float = 0.0
) -> JevDispatcher:
    return JevDispatcher(
        client,  # type: ignore[arg-type]
        runner=runner,
        clock=StepClock(step_s=step_s),
    )


# -- 1. at most one call in flight --------------------------------------------------


def test_a_request_while_one_is_in_flight_is_refused() -> None:
    runner = ManualRunner()
    dispatcher = dispatcher_with(FakeClient(response()), runner)

    assert dispatcher.request(QUESTIONS, make_state(), tick=0) is True
    assert dispatcher.in_flight is True
    # The second one is refused, and nothing reaches the runner.
    assert dispatcher.request(QUESTIONS, make_state(), tick=1) is False
    assert runner.submitted == 1


def test_a_finished_but_untaken_call_still_occupies_the_slot() -> None:
    """The worker finishing does not free the dispatcher; taking the result does.

    That is deliberate: the tick that acts on an answer is the tick that frees the
    dispatcher to ask again, so a call can never be fired by a tick that has not
    yet accounted for the previous one.
    """
    runner = ManualRunner()
    dispatcher = dispatcher_with(FakeClient(response(), response()), runner)

    dispatcher.request(QUESTIONS, make_state(), tick=0)
    runner.run_next()  # the call has finished, but nobody has taken it

    assert dispatcher.in_flight is True
    assert dispatcher.request(QUESTIONS, make_state(), tick=1) is False

    assert dispatcher.take() is not None
    assert dispatcher.in_flight is False
    assert dispatcher.request(QUESTIONS, make_state(), tick=2) is True


# -- 2. take never blocks -----------------------------------------------------------


def test_take_returns_none_when_nothing_was_ever_asked() -> None:
    dispatcher = dispatcher_with(FakeClient(), ManualRunner())

    assert dispatcher.take() is None
    assert dispatcher.in_flight is False


def test_take_returns_none_while_the_call_is_still_running_and_does_not_block() -> None:
    runner = ManualRunner()
    dispatcher = dispatcher_with(FakeClient(response()), runner)
    dispatcher.request(QUESTIONS, make_state(), tick=0)

    # The runner is holding the work: a blocking take would hang here forever, and
    # the wall clock around it is the proof that it did not.
    started = time.perf_counter()
    for _ in range(1000):
        assert dispatcher.take() is None
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    assert elapsed_ms < 500.0
    assert dispatcher.in_flight is True
    assert runner.pending  # still unfinished, exactly as the test left it


def test_a_finished_call_comes_back_with_its_tick_and_its_measured_cost() -> None:
    runner = ManualRunner()
    # The worker reads the clock twice per call, so a 10 s step is a 10 s call.
    dispatcher = dispatcher_with(FakeClient(response("right+B")), runner, step_s=10.0)
    dispatcher.request(QUESTIONS, make_state(), tick=7)
    runner.run_next()

    taken = dispatcher.take()

    assert isinstance(taken, Taken)
    assert taken.error is None
    assert taken.response is not None
    assert taken.response.choice("best_action").choice == "right+B"
    assert taken.asked_at_tick == 7
    assert taken.call_ms == pytest.approx(10_000.0)


def test_the_worker_receives_exactly_the_questions_and_state_it_was_given() -> None:
    runner = ManualRunner()
    client = FakeClient(response())
    dispatcher = dispatcher_with(client, runner)
    state = make_state(tick=3)

    dispatcher.request(QUESTIONS, state, tick=3)
    runner.run_next()

    assert client.calls == [(QUESTIONS, state)]


# -- 3. a failure comes back as a value, never out of the worker ---------------------


def test_a_transient_failure_comes_back_as_an_error_rather_than_being_raised() -> None:
    runner = ManualRunner()
    failure = JevUnavailable("Jev unavailable after 2 attempt(s): HTTP 503")
    dispatcher = dispatcher_with(FakeClient(error=failure), runner, step_s=0.25)
    dispatcher.request(QUESTIONS, make_state(), tick=0)

    runner.run_next()  # the worker must not raise here
    taken = dispatcher.take()

    assert taken is not None
    assert taken.response is None
    assert taken.error is failure
    assert taken.call_ms == pytest.approx(250.0)


def test_a_fatal_failure_also_comes_back_as_a_value_for_the_caller_to_judge() -> None:
    """The worker never decides what aborts a run: it reports, the tick judges."""
    runner = ManualRunner()
    fatal = JevAuthError("rejected credential", status=401)
    dispatcher = dispatcher_with(FakeClient(error=fatal), runner)
    dispatcher.request(QUESTIONS, make_state(), tick=0)
    runner.run_next()

    taken = dispatcher.take()

    assert taken is not None and taken.error is fatal
    assert taken.response is None


def test_a_failure_still_frees_the_slot_for_the_next_call() -> None:
    runner = ManualRunner()
    dispatcher = dispatcher_with(
        FakeClient(error=JevUnavailable("Jev unavailable after 1 attempt(s): timeout")),
        runner,
    )
    dispatcher.request(QUESTIONS, make_state(), tick=0)
    runner.run_next()

    assert dispatcher.take() is not None
    assert dispatcher.in_flight is False
    assert dispatcher.request(QUESTIONS, make_state(), tick=1) is True


# -- 4. close ------------------------------------------------------------------------


def test_close_shuts_the_runner_down_and_is_idempotent() -> None:
    runner = ManualRunner()
    dispatcher = dispatcher_with(FakeClient(), runner)

    dispatcher.close()
    dispatcher.close()
    dispatcher.close()

    assert runner.shutdowns == 1


def test_a_closed_dispatcher_refuses_to_fire() -> None:
    runner = ManualRunner()
    dispatcher = dispatcher_with(FakeClient(response()), runner)
    dispatcher.close()

    assert dispatcher.request(QUESTIONS, make_state(), tick=0) is False
    assert runner.submitted == 0


def test_close_joins_the_real_worker_thread() -> None:
    """The default runner is a real thread, and ``close`` must leave none behind."""
    before = set(threading.enumerate())
    dispatcher = JevDispatcher(FakeClient(response()))  # type: ignore[arg-type]
    dispatcher.request(QUESTIONS, make_state(), tick=0)

    dispatcher.close()  # shutdown(wait=True) joins, so this is not a race

    assert set(threading.enumerate()) == before


def test_the_default_runner_really_runs_the_call_off_the_caller_thread() -> None:
    """One end-to-end pass on the real executor, so the default is not only a claim."""
    worker_threads: list[int] = []

    class ThreadRecordingClient:
        def ask(self, questions: object, state: object) -> JevResponse:
            worker_threads.append(threading.get_ident())
            return response()

    dispatcher = JevDispatcher(ThreadRecordingClient())  # type: ignore[arg-type]
    try:
        assert dispatcher.request(QUESTIONS, make_state(), tick=0) is True
        # Polling, never blocking: ``take`` is the only thing being exercised, and
        # it must return without waiting on anything.
        deadline = time.perf_counter() + 5.0
        taken = dispatcher.take()
        while taken is None and time.perf_counter() < deadline:
            taken = dispatcher.take()
    finally:
        dispatcher.close()

    assert taken is not None and taken.error is None
    assert worker_threads and worker_threads[0] != threading.get_ident()
