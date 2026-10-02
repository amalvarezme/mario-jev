"""The command line: play, replay, render the dashboard, serve it, watch it live,
and check readiness.

Seven subcommands, one entry point. :func:`main` takes an explicit ``argv`` so the
tests drive every path in-process and never spawn a subprocess, and it returns an
exit code rather than raising: a missing journal, an unreadable journal or a
broken configuration becomes an actionable message, not a traceback.

* ``run`` plays the stage with the real emulator and the Jev client, records a
  journal, and prints the report for the run it just wrote. It works with no
  credential (degrading to ``fallback_no_key`` on every tick) and with no
  advisor, and it stops cleanly when the episode ends. With ``--answers`` it is
  instead driven from a recorded transcript: the same emulator and planner run
  live, but every Jev answer comes from the transcript, so no socket is opened,
  no worker thread starts, and the run header names the journal its answers came
  from.
* ``replay`` reads a journal and prints the report. It constructs no client, no
  emulator and no planner, and touches no network; replaying the same file twice
  is byte-identical.
* ``dashboard`` writes the static HTML report.
* ``serve`` answers HTTP requests with that same report, regenerated for each
  one, so a run can be watched while it happens.
* ``demo`` is the whole thing in one command: it plays world 1-1, writes the
  run's evidence (the journal, with the per-tick frame and trajectory artifacts),
  and serves the page for it while the run goes.
* ``live`` inverts that arrangement for a viewer: the game owns the main thread
  (nes-py refuses to render a window anywhere else) and the live decision board
  is served from a daemon thread. Its ``--max-ticks`` defaults to 900 and its
  journal to ``artifacts/live.jsonl``.
* ``doctor`` reports the environment and dependency readiness, with an exit code
  that the optional advisor stack can never change.

Every collaborator is built by a small module-level factory so a test can replace
one seam without rebuilding the rest: :func:`_build_emulator`,
:func:`_build_planner`, :func:`_build_client`, :func:`_build_advisor`.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import sys
import threading
import webbrowser
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

from mario_jev.advisor import PlayJevAdvisor
from mario_jev.config import Settings, redact_url_userinfo
from mario_jev.dashboard import render_text, write_html
from mario_jev.emulator import MarioEmulator
from mario_jev.jev import JevClient, JevError
from mario_jev.journal import JournalFormatError, JournalWriter, read_journal
from mario_jev.policy import (
    ExecutionLoop,
    RecordedAnswerFeed,
    RunSummary,
    TickRecord,
    recorded_answers_from_ticks,
)
from mario_jev.serve import DEFAULT_BIND, DEFAULT_PORT, ThreadedViewer
from mario_jev.serve import serve as serve_journal
from mario_jev.trajectory import TrajectoryPlanner

__all__ = ["main"]

#: The default journal path for ``run``.
DEFAULT_JOURNAL = "journal.jsonl"

#: Modules the core needs. A missing one is a hard failure for ``doctor``.
CORE_MODULES: tuple[str, ...] = (
    "numpy",
    "PIL",
    "requests",
    "gym_super_mario_bros",
    "nes_py",
)

#: Modules only the optional PlayJev advisor needs. Their absence never changes
#: the exit code; it is reported as "advisor unavailable".
OPTIONAL_MODULES: tuple[str, ...] = ("torch", "transformers")


# -- collaborator factories (the seams a test may replace) -------------------------


def _build_emulator(settings: Settings, *, render_mode: str | None = None) -> MarioEmulator:
    """A reset real emulator for ``settings``.

    ``render_mode`` is a display choice, so it is handed straight to the
    emulator and nowhere else: it is deliberately not a ``Settings`` field and
    not written to the journal header, because a window changes nothing the run
    measured. ``None`` -- what every current caller passes -- keeps the run
    headless, exactly as before the mode was threaded through.
    """
    emulator = MarioEmulator(
        stage=settings.stage,
        seed=settings.seed,
        real_time=settings.real_time,
        target_fps=settings.target_fps,
        render_mode=render_mode,
    )
    emulator.reset()
    return emulator


def _build_planner(settings: Settings) -> TrajectoryPlanner:
    return TrajectoryPlanner.from_settings(settings)


def _build_client(settings: Settings) -> JevClient:
    return JevClient.from_settings(settings)


def _build_advisor(settings: Settings) -> PlayJevAdvisor | None:
    if not settings.advisor_enabled:
        return None
    return PlayJevAdvisor.from_settings(settings)


# -- doctor ------------------------------------------------------------------------


def _importable(name: str) -> bool:
    """Whether a module can be imported, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _rom_path() -> str | None:
    """The ROM path the emulator itself uses, or ``None`` if it cannot be asked."""
    try:
        from gym_super_mario_bros._roms import smb1_rom_path

        return smb1_rom_path()
    except Exception:
        return None


def _writable(target: Path) -> bool:
    """Whether ``target`` (or its nearest existing parent) can be written to.

    Nothing is created: the check walks up to the nearest existing directory and
    asks the operating system about write permission there.
    """
    probe = target
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    return probe.is_dir() and os.access(probe, os.W_OK)


def collect_doctor(
    *,
    importable: Callable[[str], bool] | None = None,
    rom_path: Callable[[], str | None] | None = None,
    writable: Callable[[Path], bool] | None = None,
    settings: Settings | None = None,
) -> dict[str, object]:
    """Gather the readiness facts. Every probe is injectable for testing."""
    probe = _importable if importable is None else importable
    find_rom = _rom_path if rom_path is None else rom_path
    can_write = _writable if writable is None else writable
    resolved = Settings.load() if settings is None else settings

    core = {name: probe(name) for name in CORE_MODULES}
    optional = {name: probe(name) for name in OPTIONAL_MODULES}
    rom_file = find_rom()
    rom_present = bool(rom_file) and Path(str(rom_file)).is_file()
    artifacts_dir = Path(resolved.artifacts_dir)

    return {
        "python": platform.python_version(),
        "core": core,
        "optional": optional,
        "rom_path": rom_file,
        "rom_present": rom_present,
        "key_configured": resolved.jev_available,
        # Redacted here, not at print time: the report itself is a value a caller
        # may serialise, so the userinfo must never enter it in the first place.
        "endpoint": redact_url_userinfo(resolved.jev_endpoint),
        "model": resolved.jev_model,
        "advisor_model": resolved.advisor_model_id,
        "advisor_enabled": resolved.advisor_enabled,
        "artifacts_dir": str(artifacts_dir),
        "artifacts_writable": can_write(artifacts_dir),
    }


def format_doctor(report: dict[str, object]) -> str:
    """The doctor report as plain text. The credential is a boolean, never a value."""
    lines: list[str] = ["mario-jev doctor", ""]
    lines.append(f"python: {report['python']}")

    core = report["core"]
    assert isinstance(core, dict)
    for name in CORE_MODULES:
        mark = "ok" if core.get(name) else "MISSING (required)"
        lines.append(f"{name}: {mark}")

    optional = report["optional"]
    assert isinstance(optional, dict)
    for name in OPTIONAL_MODULES:
        if optional.get(name):
            lines.append(f"{name}: installed (optional advisor backend)")
        else:
            lines.append(f"{name}: not installed (optional; advisor unavailable)")

    rom_present = bool(report["rom_present"])
    if rom_present:
        lines.append(f"ROM: present at {report['rom_path']}")
    else:
        lines.append(f"ROM: MISSING (looked for {report['rom_path']})")

    configured = "configured" if report["key_configured"] else "not configured"
    lines.append(f"JEV_API_KEY: {configured}")
    lines.append(f"endpoint: {report['endpoint']}")
    lines.append(f"model: {report['model']}")
    advisor_state = "enabled" if report["advisor_enabled"] else "disabled"
    lines.append(f"advisor: {advisor_state} ({report['advisor_model']})")

    writable = "writable" if report["artifacts_writable"] else "NOT WRITABLE"
    lines.append(f"artifacts_dir: {report['artifacts_dir']} ({writable})")
    lines.append("")
    lines.append(
        "advisor stack is optional; its absence is reported, not counted as a failure"
    )
    return "\n".join(lines)


def doctor_exit_code(report: dict[str, object]) -> int:
    """``0`` when the core is ready, ``1`` when something the core needs is missing.

    The optional modules are deliberately absent from this function: ``torch`` and
    ``transformers`` never affect the exit code.
    """
    core = report["core"]
    assert isinstance(core, dict)
    if not all(bool(core.get(name)) for name in CORE_MODULES):
        return 1
    if not report["rom_present"]:
        return 1
    if not report["artifacts_writable"]:
        return 1
    return 0


# -- run ---------------------------------------------------------------------------


def _apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    # ``getattr``: ``run`` and ``demo`` do not carry the same options, and a
    # command that does not offer one must leave the setting alone rather than
    # raise on a missing attribute.
    changes: dict[str, object] = {}
    if getattr(args, "stage", None) is not None:
        changes["stage"] = args.stage
    if getattr(args, "max_ticks", None) is not None:
        changes["max_ticks"] = args.max_ticks
    if getattr(args, "real_time", None) is not None:
        changes["real_time"] = args.real_time
    if getattr(args, "artifacts_dir", None) is not None:
        changes["artifacts_dir"] = Path(args.artifacts_dir)
    if getattr(args, "seed", None) is not None:
        changes["seed"] = args.seed
    if getattr(args, "answers", None) is not None:
        # The flag wins over ``MARIO_ANSWERS_JOURNAL``: ``Settings.load`` already
        # read the environment, and this is the operator asking for this file.
        changes["answers_journal"] = Path(args.answers)
    if not changes:
        return settings
    updated = replace(settings, **changes)
    updated.validate()
    return updated


class _JournalOpenError(OSError):
    """The journal writer could not be built.

    Raised when opening the journal for writing -- or writing the header that
    follows -- fails, before the run proper starts. An ``OSError`` like the
    failure it wraps, but a distinct type so ``run`` can keep its two messages
    apart: "cannot open journal" for this setup failure, and "run failed" for
    everything the run itself raised. The header write can fail *after* the file
    was opened and truncated, so the type marks where the failure happened, not
    a file that was never touched. Before the shared :func:`_play` existed those
    two branches sat next to each other, and the type is what carries the
    distinction across the function boundary.
    """


#: The value a fed run substitutes for ``Settings.jev_api_key`` when the operator
#: configured none. ``ExecutionLoop`` fires a call only when ``settings.jev_available``
#: is true, and a fed run's call is answered by the transcript, so without a value
#: here the run would never fire and the transcript would never be replayed -- the
#: exact silent non-replay ``--answers`` exists to prevent. It is not a credential
#: and never reaches one: a fed run builds no client at all. The journal's header
#: records the operator's own settings, so a no-key fed run still reads
#: ``jev_available: false`` there and names its source under ``answers_journal``.
TRANSCRIPT_CREDENTIAL = "transcript-answers"


class _TranscriptError(OSError):
    """The ``--answers`` journal cannot drive a run.

    An ``OSError`` like the file failures it wraps, so ``demo``'s existing
    ``except (JevError, OSError)`` reports a bad transcript as a failed run rather
    than letting it escape a daemon thread; ``run`` catches it first and gives it a
    message of its own. Nothing here ever falls back to a live call: the whole
    point of the check is that a fed run either replays a transcript or stops.
    """


class _UnenteredClient:
    """The stand-in ``ExecutionLoop`` stores as its client on a fed run.

    ``ExecutionLoop`` keeps a ``client`` attribute but only ever hands it to the
    dispatcher it builds *itself*; with a recorded-answer feed injected as the call
    source the loop never reads it. A fed run must enter no client and open no
    socket, so the CLI passes this object rather than letting
    ``JevClient.from_settings`` build the real one: any attribute access raises,
    which makes "the client was entered after all" a loud failure instead of a
    silent live call. The loop types that slot as ``JevClient``; this object is
    passed precisely because the slot is never reached, and the test that makes
    every client factory raise is what keeps that true.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> object:
        raise AssertionError(
            "a fed run must not enter a Jev client; the answers come from the "
            f"transcript, and something reached for client.{name}"
        )


@dataclass(frozen=True, slots=True)
class _FedRun:
    """A validated transcript, the feed built from it, and the transcript's counts.

    ``calls_fired`` and ``answers_applied`` are read from the transcript's own
    bytes: the first counts its tick lines whose ``decision.jev_called`` is true,
    the second counts its tick lines that carry a non-null ``response`` -- exactly
    the lines :func:`recorded_answers_from_ticks` keeps. They are kept here because
    a fed run's summary counts *this* run, and a reader comparing the two run's
    numbers needs the transcript's beside them.
    """

    feed: RecordedAnswerFeed
    path: Path
    calls_fired: int
    answers_applied: int


def _load_fed_run(path: object) -> _FedRun:
    """Read and validate the ``--answers`` transcript, or fail with a clear reason.

    Three things make a transcript unusable, and each is refused *before* the run
    starts, because a fed run that quietly does nothing -- or, worse, falls back to
    a live call -- is the failure this flag exists to prevent:

    * the path does not exist;
    * it cannot be read, or is not a finished, well-formed journal (the strict
      reader is used, so a transcript without its summary trailer is refused);
    * it holds no applied answers -- every tick line has a ``null`` response, so
      there is nothing to replay and the feed would refuse every call.

    A transcript that holds *some* answers but runs out mid-run is a different case
    and is not refused here: the feed records ``answers_exhausted`` on the tick that
    tried, and the fidelity line printed after the run shows the transcript's
    counts beside the run's.
    """
    try:
        journal = read_journal(path)
    except FileNotFoundError as exc:
        raise _TranscriptError(f"transcript not found: {path}") from exc
    except JournalFormatError as exc:
        raise _TranscriptError(f"transcript is unreadable: {exc}") from exc
    except OSError as exc:
        raise _TranscriptError(f"cannot read transcript {path}: {exc}") from exc
    try:
        answers = recorded_answers_from_ticks(journal.ticks)
        feed = RecordedAnswerFeed(answers)
    except (ValueError, JevError) as exc:
        # A line that carries a response but loses its ``answer_age_ticks`` or
        # ``jev_call_ms``, or a response body the parser refuses. A fabricated
        # latency or a guessed answer would be worse than a refused transcript.
        raise _TranscriptError(f"transcript {path} is malformed: {exc}") from exc
    if not answers:
        raise _TranscriptError(
            f"transcript {path} holds no applied answers: every tick line has a "
            "null response, so there is nothing to replay"
        )
    fired = 0
    for tick in journal.ticks:
        decision = tick.get("decision")
        if isinstance(decision, Mapping) and decision.get("jev_called"):
            fired += 1
    return _FedRun(
        feed=feed,
        path=Path(path),
        calls_fired=fired,
        answers_applied=len(answers),
    )


def _summary_count(summary: object, key: str) -> object:
    """A summary count, or the words the readers use when it was not recorded."""
    if isinstance(summary, Mapping):
        value = summary.get(key)
        if value is not None:
            return value
    return "not recorded"


def _fidelity_report(fed: _FedRun, journal: object) -> str:
    """The transcript's own counts beside this run's, so a misalignment is visible.

    A transcript is built from the tick lines that **applied** an answer, because
    only those carry a ``response``. A call the recording fired but could not apply
    -- it failed, or its answer arrived too stale to use -- has no slot in the
    transcript, so the k-th recorded answer is the k-th answer the recording
    *applied*, not the k-th call it *fired*. A replay that fires its k-th call on a
    different tick can therefore receive another original call's answer, and when
    the replay fires **fewer** calls than the transcript holds the unused tail is
    never asked for: no mismatch and no exhaustion signal fires. That silent case is
    why these numbers are printed rather than left to the source.

    What a reader compares: ``transcript recorded`` against ``this run``. The
    transcript's two counts coming apart says the recording itself lost calls
    between firing and applying; the two fired counts coming apart says the replay
    asked a different number of questions. Either makes the positional alignment an
    assumption rather than a fact, and neither is named by any degradation on the
    ticks.
    """
    summary = getattr(journal, "summary", None)
    run_fired = _summary_count(summary, "decisions")
    run_applied = _summary_count(summary, "answers_applied")
    lines = [
        f"fed run from {fed.path}: this run's answers came from the transcript, "
        "not the API",
        f"  transcript recorded: {fed.calls_fired} calls fired, "
        f"{fed.answers_applied} answers applied",
        f"  this run: {run_fired} calls fired, {run_applied} answers applied",
    ]
    if fed.calls_fired != run_fired:
        lines.append(
            f"  note: the recording fired {fed.calls_fired} calls and this run "
            f"fired {run_fired}, so the k-th recorded answer need not be the k-th "
            "original call, and that misalignment is recorded nowhere else"
        )
    return "\n".join(lines)


def _play(
    settings: Settings,
    journal_path: object,
    *,
    artifacts_dir: object | None,
    max_ticks: int | None = None,
    observer: Callable[[TickRecord], None] | None = None,
    fed: _FedRun | None = None,
    render_mode: str | None = None,
) -> RunSummary:
    """Play one run into ``journal_path`` and return its summary.

    The writer, emulator, planner, client, advisor and loop are built here, so
    ``run`` and ``demo`` take the same path and produce the same journal. The
    summary trailer is written before returning: a journal without one is a run
    that did not finish, which is exactly what the live viewer reports.

    ``observer`` replaces the journal's own per-tick writer when given; ``None``
    keeps that writer, which is what both commands want. ``max_ticks`` overrides
    ``settings.max_ticks`` for the loop, as it does for ``run``.

    ``fed`` drives the run from a transcript instead of the live client: the loop's
    call source is the feed and its ``client`` slot holds :class:`_UnenteredClient`,
    so no client is built and none can be entered. When ``fed`` is ``None`` but
    ``settings.answers_journal`` is set, the transcript is read and validated here
    -- ``run`` pre-reads it so it can print its own refusal, and ``demo`` (which has
    no ``--answers`` flag but honours the environment) reaches this branch.

    ``render_mode`` is the display choice the ``live`` command will pass down to
    the emulator; ``run`` and ``demo`` pass nothing and stay headless. It changes
    only what is drawn, never what is recorded: the emulator draws the real
    timeline and nothing else, so the mode is no evidence and no caller that did
    not ask for a window sees one.

    The header is written from ``settings``, never from the ``loop_settings``
    below: a fed run with no credential still records ``jev_available: false`` and
    names its transcript under ``answers_journal``. The loop receives a substituted
    credential only because it refuses to fire a call when none is configured, and
    a fed run's call is answered by the transcript; the value reaches no client,
    because a fed run builds none.
    """
    if fed is None and settings.answers_journal is not None:
        fed = _load_fed_run(settings.answers_journal)
    try:
        writer = JournalWriter(
            journal_path, settings=settings, artifacts_dir=artifacts_dir
        )
    except OSError as exc:
        raise _JournalOpenError(str(exc)) from exc

    try:
        emulator = None
        try:
            # Only a caller that asked for a window passes the mode, so the
            # default path builds the emulator with the exact call it always
            # made -- same keyword set, same seam every test double replaces.
            emulator = (
                _build_emulator(settings)
                if render_mode is None
                else _build_emulator(settings, render_mode=render_mode)
            )
            loop_settings = settings
            if fed is None:
                client = _build_client(settings)
                dispatcher = None
            else:
                # A fed run opens no socket and enters no client. Any attempt to
                # reach one through this stand-in raises instead of quietly
                # calling Jev, which is the guarantee the flag exists to make.
                client = _UnenteredClient()
                if not settings.jev_available:
                    loop_settings = replace(
                        settings, jev_api_key=TRANSCRIPT_CREDENTIAL
                    )
                dispatcher = fed.feed
            loop = ExecutionLoop(
                loop_settings,
                emulator=emulator,
                planner=_build_planner(loop_settings),
                client=client,
                advisor=_build_advisor(loop_settings),
                dispatcher=dispatcher,
            )
            summary = loop.run(
                max_ticks=max_ticks,
                observer=writer.observer() if observer is None else observer,
            )
        finally:
            if emulator is not None:
                emulator.close()
        writer.write_summary(summary)
    finally:
        writer.close()
    return summary


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        settings = _apply_overrides(Settings.load(), args)
    except ValueError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    journal_path = Path(args.journal)
    artifacts_dir = Path(settings.artifacts_dir)

    # A fed run is validated before anything is written, so a bad transcript
    # cannot leave a half-started journal behind or, worse, let the run proceed
    # against the live client. ``_play`` re-checks only when ``run`` did not pass
    # a feed; here the check is what produces this command's message.
    fed: _FedRun | None = None
    if settings.answers_journal is not None:
        try:
            fed = _load_fed_run(settings.answers_journal)
        except _TranscriptError as exc:
            print(f"cannot replay recorded answers: {exc}", file=sys.stderr)
            return 1
        if settings.rollout_cost_source != "fixed":
            # The flag replays the answers; it does not make the horizon
            # clock-free. A user could otherwise believe the flag alone bought
            # reproducibility and get two runs that differ in the horizon.
            print(
                "warning: --answers replays the recorded Jev answers, but the "
                "rollout horizon still comes from the measured cost source, so it "
                "still follows this machine and two such runs are not "
                "reproducible. Also set MARIO_ROLLOUT_COST_SOURCE=fixed with "
                "MARIO_ROLLOUT_FIXED_BASE_MS and MARIO_ROLLOUT_FIXED_PER_FRAME_MS "
                "to make the horizon clock-free",
                file=sys.stderr,
            )
    elif not settings.jev_available:
        print(
            "warning: no JEV_API_KEY is configured; every decision will use the "
            "local fallback (fallback_no_key) and no Jev request will be sent",
            file=sys.stderr,
        )

    try:
        _play(settings, journal_path, artifacts_dir=artifacts_dir, fed=fed)
    except _JournalOpenError as exc:
        print(f"cannot open journal {journal_path}: {exc}", file=sys.stderr)
        return 1
    except _TranscriptError as exc:
        print(f"cannot replay recorded answers: {exc}", file=sys.stderr)
        return 1
    except (JevError, OSError) as exc:
        print(f"run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    try:
        journal = read_journal(journal_path)
    except (JournalFormatError, OSError) as exc:
        print(f"the journal just written could not be read back: {exc}", file=sys.stderr)
        return 1
    print(render_text(journal))
    if fed is not None:
        # The fidelity counts go beside the report, not inside it: the dashboard
        # is a separate, read-only reader, and this is the surface a ``run`` user
        # actually sees. A fed run that replayed a misaligned transcript must be
        # visible here without anyone opening the source.
        print()
        print(_fidelity_report(fed, journal))
    return 0


# -- replay ------------------------------------------------------------------------


def _read_or_report(path: object) -> tuple[object | None, int]:
    """Read a journal, or print an actionable message and return an exit code."""
    try:
        return read_journal(path), 0
    except FileNotFoundError:
        print(f"journal not found: {path}", file=sys.stderr)
        return None, 1
    except OSError as exc:
        print(f"cannot read journal {path}: {exc}", file=sys.stderr)
        return None, 1
    except JournalFormatError as exc:
        print(f"journal is unreadable: {exc}", file=sys.stderr)
        return None, 1


def _cmd_replay(args: argparse.Namespace) -> int:
    journal, code = _read_or_report(args.journal)
    if journal is None:
        return code
    print(render_text(journal))
    return 0


# -- dashboard ---------------------------------------------------------------------


def _cmd_dashboard(args: argparse.Namespace) -> int:
    journal, code = _read_or_report(args.journal)
    if journal is None:
        return code
    target = Path(args.out) if args.out else Path(args.journal).with_suffix(".html")
    try:
        written = write_html(target, journal)
    except OSError as exc:
        print(f"cannot write dashboard to {target}: {exc}", file=sys.stderr)
        return 1
    print(str(written))
    return 0


# -- serve -------------------------------------------------------------------------


def _cmd_serve(args: argparse.Namespace) -> int:
    """Serve the report for a journal until interrupted. The journal is read per request."""
    return serve_journal(
        args.journal,
        artifacts_dir=args.artifacts_dir,
        bind=args.bind,
        port=args.port,
        once=args.once,
    )


# -- demo --------------------------------------------------------------------------

#: Printed when this run returned after writing its own summary trailer. The run
#: finished, whatever the background thread is doing at this instant, because the
#: ``run_finished`` event is set only after ``_play`` returned. That return is the
#: one fact that can establish the trailer as this run's own: a trailer read off
#: disk while this run has not returned cannot be attributed to it. The sentence
#: therefore states the run's own write and not what the disk holds, which this
#: branch does not read.
DEMO_FINISHED_LINE = (
    "stopping the viewer; the run had already finished and wrote its summary "
    "trailer"
)
#: Printed when the journal is readable and holds no trailer while the run thread
#: is still going: the viewer was stopped by Ctrl-C and the daemon run died with
#: the process. The journal genuinely keeps every tick written so far and nothing
#: marked the end, so a reader really does report the run as in progress. Saying
#: "finished" here would be a lie.
DEMO_ABANDONED_LINE = (
    "stopping the viewer; the run is abandoned, not finished: the journal on disk "
    "keeps every tick written so far and has no summary trailer, so reading it "
    "reports the run as in progress"
)
#: Printed when the journal is readable and holds no trailer and the run thread has
#: ended: writing the trailer never happened, so the run did not finish (the error
#: itself went to stderr when it happened). The journal is evidence up to the
#: failure and the viewer reads it as in progress, which is what an unfinished run
#: is -- never a finished one.
DEMO_FAILED_LINE = (
    "stopping the viewer; the run stopped before writing its summary trailer: the "
    "journal on disk keeps every tick written so far and has no summary trailer, "
    "so reading it reports the run as in progress"
)
#: Printed when this run did not return and a readable journal carries a summary
#: trailer. The trailer may be this run's own -- ``write_summary`` succeeded and
#: ``writer.close()`` then raised -- or an earlier run's left at the same path, and
#: this process cannot tell the two apart. The sentence therefore asserts no
#: provenance: it must stay true in both states.
DEMO_TRAILER_NOT_OURS_LINE = (
    "stopping the viewer; this run did not end successfully, and the journal on "
    "disk carries a summary trailer this run may not have written"
)
#: Printed when the journal could not be read from here and the run thread has not
#: been reaped. "Could not be read from here" is the only thing this process
#: observed -- the file may be absent, be a directory, or have hidden the failure
#: inside the reader itself -- so it is the only thing the sentence may say. Nor
#: may it say whether the run finished: an unreadable journal is evidence neither
#: way, and a thread that is still alive is not evidence of unfinished work,
#: because the trailer it wrote may simply not have been reaped yet. Only this
#: run's own return establishes completion, and that branch never reaches here.
DEMO_ABANDONED_UNWRITTEN_LINE = (
    "stopping the viewer; its journal could not be read from here, so there is "
    "nothing to show, and whether the run finished is not known from here"
)
#: Printed when the journal could not be read from here and the run thread has
#: ended. Same absence, and the same refusal to name a cause this process did not
#: establish -- "stopped" is deliberate, since a dead thread cannot say whether the
#: run finished, failed or was killed.
DEMO_FAILED_UNWRITTEN_LINE = (
    "stopping the viewer; the run stopped and its journal could not be read from "
    "here, so there is nothing to show"
)


def _demo_exit_sentence(
    journal_path: object, *, run_finished: bool, run_alive: bool
) -> str:
    """The one honest exit line, decided by this run's own return and the journal.

    ``demo`` and ``live`` share this sentence unchanged. They differ only in where
    ``run_alive`` comes from: ``demo`` reads it off its background run thread,
    while ``live`` runs the game on the main thread and answers it from the
    interrupt -- a Ctrl-C that ends the run means the journal was still being
    written when the viewer stopped.

    FINISHED is derived from this run's own return, never from a trailer found on
    disk: ``run_finished`` is set only once ``_play`` returned, which happens only
    after ``write_summary`` wrote this run's trailer. A trailer found while this
    run has not finished cannot be attributed to it -- the path may still hold an
    earlier run's trailer, or this run's trailer may have been written just before
    its ``close()`` raised -- so the sentence reports the trailer's presence
    without claiming its author.

    The sentence is a claim about the evidence, so it may only be read off the
    evidence. Reading can fail -- the file may be missing, be a directory, be
    undecodable, or hold a line that is not JSON -- and that failure *is* the
    answer, so the two ``*_UNWRITTEN`` sentences report only what this process
    observed ("could not be read from here") and never what the file *is*. A read
    failure is never propagated: turning a working exit into a crash over its own
    status line would be strictly worse than the status line being vague,
    especially since the exit is what the user sees last.

    The cost of catching everything is stated rather than hidden: if
    ``read_journal`` itself is broken, this reports that the journal could not be
    read from here, which is true of what happened and silent about the file. The
    failure mode is therefore a vague sentence -- never a false claim about ticks,
    a trailer or completion, and never a traceback on the way out. Completion in
    particular is asserted only from this run's own return, never from a trailer
    on disk and never from the background thread's liveness, which the reaping
    window makes unreliable exactly when a run finishes during the viewing window.

    ``run_alive`` chooses only between the abandoned and failed wording once
    ``run_finished`` is False. Whether a trailer exists is read from the file, but
    a trailer with ``run_finished`` False selects the no-provenance sentence rather
    than a completion claim.
    """
    if run_finished:
        # This run returned after writing its own trailer, so the disk is no longer
        # the authority for it.
        return DEMO_FINISHED_LINE
    try:
        journal = read_journal(journal_path, allow_partial=True)
    except Exception:  # noqa: BLE001 - any read failure *is* the unreadable state
        # Missing, a directory, undecodable bytes, a torn line, or a bug inside
        # the reader: all this process knows is that it could not read the file,
        # which is exactly what the sentence says.
        if run_alive:
            return DEMO_ABANDONED_UNWRITTEN_LINE
        return DEMO_FAILED_UNWRITTEN_LINE
    if journal.summary is not None:
        # A trailer is present, but this run did not return: it may be this run's
        # (write_summary succeeded, close then raised) or an earlier run's at the
        # same path, and this process cannot tell them apart.
        return DEMO_TRAILER_NOT_OURS_LINE
    if run_alive:
        return DEMO_ABANDONED_LINE
    return DEMO_FAILED_LINE


def _start_run_in_background(run_callable: Callable[[], None]) -> threading.Thread:
    """Run the play in a daemon thread and leave the main thread to the server.

    This is a composition, not a concurrency change: the emulator is headless and
    the Jev call already runs on the dispatcher's worker, so nothing in the loop
    needs the main thread. The thread is a daemon because Ctrl-C stops the
    *viewer* by ending the process; waiting for a run that can take minutes would
    hold the terminal hostage, and the cost is stated on exit rather than hidden.
    """
    thread = threading.Thread(
        target=run_callable, name="mario-jev-demo-run", daemon=True
    )
    thread.start()
    return thread


def _open_browser(url: str) -> bool:
    """Open ``url`` with the platform browser; returns what ``webbrowser`` said."""
    return webbrowser.open(url)


def _open_in_browser(url: str) -> None:
    """Open ``url`` behind the seam and report only what this process observed.

    A browser that raises did not open, and neither did one that returned false;
    both are said on stdout, and the URL the viewer already printed stands as the
    fallback. Neither is a viewer failure, so nothing here propagates. ``demo`` and
    ``live`` share this so the two commands cannot report the same failure two
    different ways.
    """
    try:
        opened = _open_browser(url)
    except Exception as exc:  # noqa: BLE001 - the browser is outside our control
        # A browser that raises did not open. That is the browser seam failing,
        # never the viewer: say the observed failure and leave the already-printed
        # URL as the fallback.
        print(
            f"the browser did not open for {url}: {type(exc).__name__}: {exc}; "
            "open that address yourself",
            file=sys.stdout,
        )
        return
    if not opened:
        # The URL has already been printed by the viewer. A claim that a window
        # appeared would be about the user's machine and was not observed; this
        # says only what was observed.
        print(
            f"the browser did not open for {url}; open that address yourself",
            file=sys.stdout,
        )


def _cmd_demo(args: argparse.Namespace) -> int:
    """Play world 1-1, write the evidence, and serve the page, in one command."""
    try:
        settings = _apply_overrides(Settings.load(), args)
    except ValueError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    journal_path = Path(args.journal)
    artifacts_dir = Path(settings.artifacts_dir)

    if settings.answers_journal is None and not settings.jev_available:
        print(
            "warning: no JEV_API_KEY is configured; every decision will use the "
            "local fallback (fallback_no_key) and no Jev request will be sent",
            file=sys.stderr,
        )

    run_finished = threading.Event()

    def play() -> None:
        try:
            _play(
                settings,
                journal_path,
                artifacts_dir=artifacts_dir,
                max_ticks=args.max_ticks,
            )
        except _JournalOpenError as exc:
            print(f"cannot open journal {journal_path}: {exc}", file=sys.stderr)
        except (JevError, OSError) as exc:
            # A failed run does not end the viewer: the journal keeps the ticks
            # up to the failure and the page renders them. The error is said out
            # loud and the server keeps serving.
            print(f"run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        else:
            # The event is the only fact that can establish the trailer on disk
            # as this run's own: it is set only after ``_play`` returned, which is
            # only after ``write_summary`` wrote this run's trailer.
            run_finished.set()

    thread = _start_run_in_background(play)

    def on_ready(url: str) -> None:
        if args.no_open:
            return
        _open_in_browser(url)

    try:
        code = serve_journal(
            journal_path,
            artifacts_dir=artifacts_dir,
            bind=args.bind,
            port=args.port,
            on_ready=on_ready,
        )
    except KeyboardInterrupt:
        # ``serve`` catches its own Ctrl-C; this covers one raised before the
        # server was up. Either way the viewer was interrupted, not finished.
        code = 0
    except OSError as exc:
        # The browser callback handles its own failures, so what reaches here is an
        # ``OSError`` from serving itself. The message names the address and the
        # error and asserts no cause it did not establish. The sentence below is
        # the same journal-derived one as the normal path.
        print(
            f"the viewer failed on {args.bind}:{args.port}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(
            _demo_exit_sentence(
                journal_path,
                run_finished=run_finished.is_set(),
                run_alive=thread.is_alive(),
            ),
            file=sys.stdout,
        )
        return 1

    # This run's own return and what the journal holds now decide which sentence
    # is true. The run can finish inside the viewing window (a few hundred paced
    # ticks is on the order of a minute on this machine, and the cost is load- and
    # mode-dependent, so no tighter figure is claimed), and calling that
    # "abandoned, no trailer" would be a false claim about the evidence -- the one
    # thing this project never makes.
    print(
        _demo_exit_sentence(
            journal_path,
            run_finished=run_finished.is_set(),
            run_alive=thread.is_alive(),
        ),
        file=sys.stdout,
    )
    return code


# -- live --------------------------------------------------------------------------


def _cmd_live(args: argparse.Namespace) -> int:
    """Play the stage in its own window while the live board is served beside it.

    The same settings and overrides as ``demo``, with two deliberate differences
    that the parser's help states too: ``--max-ticks`` defaults to **900**, because
    a viewer means to sit and watch and 300 ticks is about 45 seconds, and
    ``--journal`` defaults to ``artifacts/live.jsonl``.

    The arrangement inverts ``demo``'s and that is the whole point: the run owns
    the main thread because nes-py refuses to render a window anywhere else, and
    the viewer is the daemon thread. The viewer starts -- configured with the live
    route -- before the first frame is played, and is stopped on every exit path.
    The exit line is ``demo``'s own :func:`_demo_exit_sentence`, with the three
    outcomes mapped deliberately: a clean return is finished; a ``(JevError,
    OSError)`` out of ``_play`` is printed as ``run failed`` and means the run did
    not finish; and Ctrl-C means the run was still going when the viewer stopped,
    so ``live`` answers ``run_alive`` from the interrupt rather than from a thread.
    A failing journal open keeps ``run``'s own message.
    """
    try:
        settings = _apply_overrides(Settings.load(), args)
    except ValueError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    journal_path = Path(args.journal)
    artifacts_dir = Path(settings.artifacts_dir)

    # The same warning on the same condition as ``demo``: no credential and no
    # transcript means every decision degrades to the local fallback.
    if settings.answers_journal is None and not settings.jev_available:
        print(
            "warning: no JEV_API_KEY is configured; every decision will use the "
            "local fallback (fallback_no_key) and no Jev request will be sent",
            file=sys.stderr,
        )

    def on_ready(url: str) -> None:
        if args.no_open:
            # No window is opened, so say the address as the thing to open rather
            # than let the user wait for one that will never appear.
            print(
                f"not opening a browser; open {url} to watch the decision board",
                file=sys.stdout,
            )
            return
        _open_in_browser(url)

    try:
        viewer = ThreadedViewer(
            journal_path,
            artifacts_dir=artifacts_dir,
            bind=args.bind,
            port=args.port,
            on_ready=on_ready,
            live=True,
        )
    except OSError as exc:
        # Binding failed before anything ran. The message names the address and the
        # error and asserts no cause it did not establish, exactly as ``demo`` does
        # for its viewer; the sentence is the journal-derived one.
        print(
            f"the viewer failed on {args.bind}:{args.port}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(
            _demo_exit_sentence(
                journal_path, run_finished=False, run_alive=False
            ),
            file=sys.stdout,
        )
        return 1

    run_finished = False
    run_interrupted = False
    try:
        try:
            viewer.start()
            try:
                # The run owns the main thread: ``human`` mode opens the game
                # window, which is this command's point and is not optional.
                _play(
                    settings,
                    journal_path,
                    artifacts_dir=artifacts_dir,
                    max_ticks=args.max_ticks,
                    render_mode="human",
                )
            except KeyboardInterrupt:
                # The run was still going when the interrupt arrived, which is what
                # run_alive asks; the journal has no trailer of this run's.
                run_interrupted = True
            except _JournalOpenError as exc:
                # ``run``'s own message for the one failure that happens before the
                # run proper starts.
                print(
                    f"cannot open journal {journal_path}: {exc}", file=sys.stderr
                )
            except (JevError, OSError) as exc:
                print(f"run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            else:
                run_finished = True
        except KeyboardInterrupt:
            # Ctrl-C before or during ``start``: the viewer was coming up and the
            # run never got going, which is the abandoned state too.
            run_interrupted = True
    finally:
        # Every path out stops the viewer: the socket and its daemon thread must
        # not outlive the command that opened them.
        viewer.stop()

    print(
        _demo_exit_sentence(
            journal_path,
            run_finished=run_finished,
            run_alive=run_interrupted,
        ),
        file=sys.stdout,
    )
    # The run is the point of this command: finishing or being interrupted is a
    # clean exit, and a run that never finished is a failure.
    return 0 if (run_finished or run_interrupted) else 1


# -- doctor command ----------------------------------------------------------------


def _cmd_doctor(args: argparse.Namespace) -> int:
    try:
        report = collect_doctor()
    except ValueError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 1
    print(format_doctor(report))
    return doctor_exit_code(report)


# -- parser and entry point --------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mario-jev",
        description="Play Super Mario Bros. with Jev, record the evidence, and read it back.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="play the stage and record a journal")
    run.add_argument("--stage", default=None, help="world-stage, e.g. 1-1")
    run.add_argument("--max-ticks", type=int, default=None, help="stop after N decision ticks")
    run.add_argument(
        "--journal", default=DEFAULT_JOURNAL, help="journal path to write"
    )
    run.add_argument(
        "--artifacts-dir", default=None, help="directory for per-tick PNG artifacts"
    )
    run.add_argument(
        "--answers",
        default=None,
        help=(
            "replay the recorded Jev answers in this journal instead of calling "
            "Jev: no network and no worker thread. Every tick is marked "
            "transcript_answers and the run header names this journal"
        ),
    )
    pacing = run.add_mutually_exclusive_group()
    pacing.add_argument(
        "--real-time",
        dest="real_time",
        action="store_true",
        default=None,
        help="pace the emulator at the configured frame rate",
    )
    pacing.add_argument(
        "--fast",
        dest="real_time",
        action="store_false",
        default=None,
        help="run as fast as possible",
    )
    run.set_defaults(handler=_cmd_run)

    replay = subparsers.add_parser("replay", help="print a recorded journal as a report")
    replay.add_argument("journal", help="journal path to read")
    replay.set_defaults(handler=_cmd_replay)

    dashboard = subparsers.add_parser(
        "dashboard", help="write the static HTML report for a journal"
    )
    dashboard.add_argument("journal", help="journal path to read")
    dashboard.add_argument(
        "--out", default=None, help="output HTML path (default: beside the journal)"
    )
    dashboard.set_defaults(handler=_cmd_dashboard)

    serve = subparsers.add_parser(
        "serve", help="serve the report for a journal, live, over a local HTTP port"
    )
    serve.add_argument("journal", help="journal path to read again on every request")
    serve.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"TCP port to listen on (default: {DEFAULT_PORT}; 0 picks a free one)",
    )
    serve.add_argument(
        "--bind",
        default=DEFAULT_BIND,
        help=(
            f"address to listen on (default: {DEFAULT_BIND}, loopback only; "
            "another address publishes this machine's run to the network)"
        ),
    )
    serve.add_argument(
        "--artifacts-dir",
        default=None,
        help="directory artifacts are served from (default: the one the run recorded)",
    )
    serve.add_argument(
        "--once",
        action="store_true",
        help="answer one request and exit",
    )
    serve.set_defaults(handler=_cmd_serve)

    demo = subparsers.add_parser(
        "demo",
        help="play world 1-1, write the run's evidence, and serve the page for it",
    )
    demo.add_argument(
        "--stage", default="1-1", help="world-stage to play (default: 1-1)"
    )
    demo.add_argument(
        "--max-ticks",
        type=int,
        default=300,
        help="stop after N decision ticks (default: 300)",
    )
    demo.add_argument(
        "--journal",
        default="artifacts/demo.jsonl",
        help="journal path to write (default: artifacts/demo.jsonl)",
    )
    demo.add_argument(
        "--artifacts-dir",
        default=None,
        help="directory for per-tick PNG artifacts (default: the configured one)",
    )
    demo.add_argument(
        "--seed",
        type=int,
        default=None,
        help="emulator seed (default: the configured one)",
    )
    demo.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"TCP port to listen on (default: {DEFAULT_PORT}; 0 picks a free one)",
    )
    demo.add_argument(
        "--bind",
        default=DEFAULT_BIND,
        help=(
            f"address to listen on (default: {DEFAULT_BIND}, loopback only; "
            "another address publishes this machine's run to the network)"
        ),
    )
    demo.add_argument(
        "--no-open",
        action="store_true",
        help="print the URL but do not open a browser window",
    )
    demo.set_defaults(handler=_cmd_demo)

    live = subparsers.add_parser(
        "live",
        help=(
            "play the stage in its own window while serving the live decision "
            "board beside it"
        ),
        description=(
            "Play the stage with a real game window and serve the live decision "
            "board at /live from a background thread. Two deliberate differences "
            "from demo: --max-ticks defaults to 900, because a viewer means to sit "
            "and watch and 300 ticks is about 45 seconds; and --journal defaults "
            "to artifacts/live.jsonl."
        ),
    )
    live.add_argument(
        "--stage", default="1-1", help="world-stage to play (default: 1-1)"
    )
    live.add_argument(
        "--max-ticks",
        type=int,
        default=900,
        help="stop after N decision ticks (default: 900)",
    )
    live.add_argument(
        "--journal",
        default="artifacts/live.jsonl",
        help="journal path to write (default: artifacts/live.jsonl)",
    )
    live.add_argument(
        "--artifacts-dir",
        default=None,
        help="directory for per-tick PNG artifacts (default: the configured one)",
    )
    live.add_argument(
        "--seed",
        type=int,
        default=None,
        help="emulator seed (default: the configured one)",
    )
    live.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"TCP port to listen on (default: {DEFAULT_PORT}; 0 picks a free one)",
    )
    live.add_argument(
        "--bind",
        default=DEFAULT_BIND,
        help=(
            f"address to listen on (default: {DEFAULT_BIND}, loopback only; "
            "another address publishes this machine's run to the network)"
        ),
    )
    live.add_argument(
        "--no-open",
        action="store_true",
        help="print the URL but do not open a browser window",
    )
    live.set_defaults(handler=_cmd_live)

    doctor = subparsers.add_parser("doctor", help="report environment readiness")
    doctor.set_defaults(handler=_cmd_doctor)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return its exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    return handler(args)
