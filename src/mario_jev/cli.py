"""The command line: play, replay, render the dashboard, and check readiness.

Four subcommands, one entry point. :func:`main` takes an explicit ``argv`` so the
tests drive every path in-process and never spawn a subprocess, and it returns an
exit code rather than raising: a missing journal, an unreadable journal or a
broken configuration becomes an actionable message, not a traceback.

* ``run`` plays the stage with the real emulator and the Jev client, records a
  journal, and prints the report for the run it just wrote. It works with no
  credential (degrading to ``fallback_no_key`` on every tick) and with no
  advisor, and it stops cleanly when the episode ends.
* ``replay`` reads a journal and prints the report. It constructs no client, no
  emulator and no planner, and touches no network; replaying the same file twice
  is byte-identical.
* ``dashboard`` writes the static HTML report.
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
from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

from mario_jev.advisor import PlayJevAdvisor
from mario_jev.config import Settings, redact_url_userinfo
from mario_jev.dashboard import render_text, write_html
from mario_jev.emulator import MarioEmulator
from mario_jev.jev import JevClient, JevError
from mario_jev.journal import JournalFormatError, JournalWriter, read_journal
from mario_jev.policy import ExecutionLoop
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


def _build_emulator(settings: Settings) -> MarioEmulator:
    """A reset real emulator for ``settings``."""
    emulator = MarioEmulator(
        stage=settings.stage,
        seed=settings.seed,
        real_time=settings.real_time,
        target_fps=settings.target_fps,
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
    changes: dict[str, object] = {}
    if args.stage is not None:
        changes["stage"] = args.stage
    if args.max_ticks is not None:
        changes["max_ticks"] = args.max_ticks
    if args.real_time is not None:
        changes["real_time"] = args.real_time
    if args.artifacts_dir is not None:
        changes["artifacts_dir"] = Path(args.artifacts_dir)
    if not changes:
        return settings
    updated = replace(settings, **changes)
    updated.validate()
    return updated


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        settings = _apply_overrides(Settings.load(), args)
    except ValueError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    journal_path = Path(args.journal)
    artifacts_dir = Path(settings.artifacts_dir)

    if not settings.jev_available:
        print(
            "warning: no JEV_API_KEY is configured; every decision will use the "
            "local fallback (fallback_no_key) and no Jev request will be sent",
            file=sys.stderr,
        )

    try:
        writer = JournalWriter(
            journal_path, settings=settings, artifacts_dir=artifacts_dir
        )
    except OSError as exc:
        print(f"cannot open journal {journal_path}: {exc}", file=sys.stderr)
        return 1

    summary = None
    try:
        emulator = None
        try:
            emulator = _build_emulator(settings)
            loop = ExecutionLoop(
                settings,
                emulator=emulator,
                planner=_build_planner(settings),
                client=_build_client(settings),
                advisor=_build_advisor(settings),
            )
            summary = loop.run(observer=writer.observer())
        finally:
            if emulator is not None:
                emulator.close()
        writer.write_summary(summary)
    except (JevError, OSError) as exc:
        print(f"run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        writer.close()

    try:
        journal = read_journal(journal_path)
    except (JournalFormatError, OSError) as exc:
        print(f"the journal just written could not be read back: {exc}", file=sys.stderr)
        return 1
    print(render_text(journal))
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

    doctor = subparsers.add_parser("doctor", help="report environment readiness")
    doctor.set_defaults(handler=_cmd_doctor)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return its exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    return handler(args)
