"""The Jev call, moved off the tick: one worker, at most one call in flight.

The measured defect this module exists to remove: ``ExecutionLoop.step`` executed
exactly one game frame per tick and called ``JevClient.ask`` **in series inside that
tick**. An unpaced run advanced 900 frames in 184 s (4.9 game frames per second),
and one tick spent 21 271.52 ms inside Jev out of a 21 375.84 ms tick -- a single
20 s timeout froze Mario for twenty seconds. A 416 ms median round trip does not
fit inside a 16.7 ms frame, and no client tuning changes that; the call has to
leave the critical path.

What this is, and what it is not
--------------------------------

:class:`JevDispatcher` submits ``client.ask(...)`` to one worker and hands the tick
back control immediately. :meth:`JevDispatcher.take` never blocks: it returns a
:class:`Taken` only when a call has finished, and ``None`` otherwise.

**``mario_jev.jev`` is not touched.** :class:`~mario_jev.jev.JevClient` stays
synchronous and single-threaded, exactly as it was committed and verified; this
module runs it in a worker instead of rewriting it. That is safe *because* at most
one call is in flight: the client is never entered twice concurrently, so its
single-threaded assumptions hold unchanged. The rule is not an optimisation, it is
what makes reusing the committed client legitimate.

**No cancellation is attempted.** The standard library cannot cancel a call that is
already running -- ``Future.cancel()`` refuses once the worker has picked the work
up -- so a discarded answer is a *discarded answer*, not a cancelled request. The
call still runs to completion in the worker and still costs whatever it costs; what
the loop discards is only the right to act on it. Claiming otherwise would be a lie
in the evidence, so this module never claims it.

**The worker never touches the emulator.** It receives the questions the main thread
already built and an immutable :class:`~mario_jev.state.MarioState`, and it reads
nothing else. Everything mutable -- the emulator, the planner, the loop's counters --
stays on the tick thread.

The runner seam
---------------

``runner`` is injected in the established style of this codebase (the client's
``transport`` and ``sleeper``, the pacer's clock). The default is a one-worker
:class:`~concurrent.futures.ThreadPoolExecutor`; tests inject a runner they drive by
hand, so every sequence is deterministic and no test has to consult the real clock
or sleep for a call to land.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, TypeVar

from mario_jev.jev import JevClient, JevResponse
from mario_jev.state import MarioState

__all__ = ["JevDispatcher", "Runner", "Taken"]

_T = TypeVar("_T")


class Runner(Protocol):
    """What the dispatcher needs from whatever runs its one call.

    Deliberately smaller than :class:`~concurrent.futures.Executor`: two methods,
    both satisfied by a stock ``ThreadPoolExecutor`` and by a hand-driven test
    double. ``submit`` takes a no-argument callable because the dispatcher binds
    the questions and the state itself, so nothing about the call's arguments has
    to travel through this protocol.
    """

    def submit(self, fn: Callable[[], _T]) -> Future[_T]:
        """Start ``fn`` and return the future that will carry its result."""

    def shutdown(self, *, wait: bool = True) -> None:
        """Stop accepting work; with ``wait`` true, join what is still running."""


@dataclass(frozen=True, slots=True)
class Taken:
    """One finished call, as the tick that took it sees it.

    Exactly one of ``response`` and ``error`` is set. A worker exception is carried
    here rather than raised inside the worker, so the tick decides what a failure
    means: the loop degrades a :class:`~mario_jev.jev.JevUnavailable` and re-raises
    the fatal ones on the tick that observes them.

    ``asked_at_tick`` is the loop tick that fired the call, which is what lets the
    taking tick measure how old the answer is. ``call_ms`` is what the *call* cost,
    measured around ``client.ask`` in the worker -- it is **not** time the taking
    tick spent, and the loop records it in its own field precisely so that the
    per-tick ``latencies_ms`` arithmetic keeps holding.
    """

    response: JevResponse | None
    error: Exception | None
    asked_at_tick: int
    call_ms: float


class JevDispatcher:
    """One worker, at most one call in flight, and a take that never blocks.

    ``in_flight`` means "a call was submitted and its result has not been taken
    yet", which is the same predicate :meth:`request` refuses on. A finished but
    untaken call still counts: the slot is freed by :meth:`take`, not by the
    worker, so the tick that acts on an answer is always the tick that frees the
    dispatcher to ask again.
    """

    __slots__ = ("_asked_at_tick", "_clock", "_client", "_closed", "_future", "_runner")

    def __init__(
        self,
        client: JevClient,
        *,
        runner: Runner | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._client = client
        # One worker, so the committed single-threaded client is never entered
        # twice at once. ``max_workers=1`` is the invariant written down in code.
        self._runner: Runner = (
            runner if runner is not None else ThreadPoolExecutor(max_workers=1)
        )
        self._clock = clock
        self._future: Future[Taken] | None = None
        self._asked_at_tick = 0
        self._closed = False

    @property
    def in_flight(self) -> bool:
        """Whether a submitted call is still waiting to be taken."""
        return self._future is not None

    def request(
        self,
        questions: Mapping[str, Mapping[str, object]],
        state: MarioState,
        *,
        tick: int,
    ) -> bool:
        """Fire one call, and answer whether it was fired.

        ``False`` means the call was refused, and the two reasons are the same
        kind of fact: one is already in flight, or the dispatcher is closed. The
        caller is told rather than queued -- a queue would hide staleness behind a
        backlog, and the whole point is that the tick knows what it asked for.

        ``questions`` and ``state`` are handed to the worker as they are. The state
        is frozen, and the questions are built fresh per tick and never mutated
        afterwards, so the worker reads only what the tick already finished with.
        """
        if self._closed or self._future is not None:
            return False
        asked_at_tick = tick

        def call() -> Taken:
            started = self._clock()
            try:
                response = self._client.ask(questions, state)
            except Exception as exc:  # noqa: BLE001 -- carried, never swallowed
                # Every failure comes back through ``Taken.error``. The worker has
                # no idea which exceptions abort the run and which degrade it;
                # that is the tick's judgement, and it is made in the loop.
                return Taken(
                    response=None,
                    error=exc,
                    asked_at_tick=asked_at_tick,
                    call_ms=(self._clock() - started) * 1000.0,
                )
            return Taken(
                response=response,
                error=None,
                asked_at_tick=asked_at_tick,
                call_ms=(self._clock() - started) * 1000.0,
            )

        self._asked_at_tick = asked_at_tick
        self._future = self._runner.submit(call)
        return True

    def take(self) -> Taken | None:
        """The finished call, or ``None``; never blocks and never raises.

        ``None`` covers both "nothing was asked" and "the call has not finished",
        and the caller must treat them the same: neither is a reason to wait.
        Taking clears the in-flight slot, so the next :meth:`request` can fire.
        """
        future = self._future
        if future is None or not future.done():
            return None
        self._future = None
        # ``done()`` is already true, so neither call below can block.
        escaped = future.exception()
        if escaped is not None:
            # A safety net, not a path the worker takes: ``call`` catches every
            # ``Exception`` itself, so only a ``BaseException`` escaping it can
            # land here. It still must not be raised out of ``take``, which is
            # documented as total, so it is reported like any other failure. There
            # is no measured duration to report for a call that never returned.
            return Taken(
                response=None,
                error=escaped if isinstance(escaped, Exception) else RuntimeError(str(escaped)),
                asked_at_tick=self._asked_at_tick,
                call_ms=0.0,
            )
        return future.result()

    def close(self) -> None:
        """Stop accepting work and join the worker. Idempotent.

        Joining is the point: a run must not leave a thread alive behind it. A
        call still running is waited for rather than abandoned, because there is
        nothing to abandon it *with* -- see the module docstring on cancellation.
        """
        if self._closed:
            return
        self._closed = True
        self._runner.shutdown(wait=True)
