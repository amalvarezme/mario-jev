"""The credential bridge: verify its contract without ever reading the key.

These tests exercise ``scripts/jev-env.sh`` through a subprocess. They never read
the real Keychain entry, never make a network call, and never require a real key:
every case either removes ``JEV_API_KEY`` and points the bridge at a service name
that does not exist, or sets a fake value that short-circuits the Keychain.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "jev-env.sh"
REPO_ROOT = SCRIPT.parents[1]
ABSENT_SERVICE = "__mario_jev_absent_entry__"


def _env(**overrides: str) -> dict[str, str]:
    """The caller's environment with JEV_API_KEY removed, as the bridge expects."""
    env = {name: value for name, value in os.environ.items() if name != "JEV_API_KEY"}
    env.update(overrides)
    return env


def test_bridge_is_syntactically_valid_on_the_available_bash() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)], capture_output=True, text=True, timeout=30
    )

    assert result.returncode == 0, result.stderr


def test_bridge_is_executable() -> None:
    assert SCRIPT.stat().st_mode & 0o111, "the bridge must be executable"


def test_a_missing_keychain_entry_fails_loudly_and_leaks_nothing() -> None:
    result = subprocess.run(
        [str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=30,
        env=_env(JEV_KEYCHAIN_SERVICE=ABSENT_SERVICE),
    )

    assert result.returncode != 0
    assert result.stderr.strip()
    assert "apikey_" not in result.stdout
    assert "apikey_" not in result.stderr


def test_an_executed_command_gets_a_preset_key_without_the_keychain() -> None:
    result = subprocess.run(
        [str(SCRIPT), "/bin/sh", "-c", 'printf %s "$JEV_API_KEY"'],
        capture_output=True,
        text=True,
        timeout=30,
        env=_env(JEV_API_KEY="preset-value", JEV_KEYCHAIN_SERVICE=ABSENT_SERVICE),
    )

    # The absent service name is deliberate: a preset key must never touch it.
    assert result.returncode == 0
    assert result.stdout == "preset-value"


def test_sourcing_exports_the_key_into_the_caller_shell() -> None:
    relative = SCRIPT.relative_to(REPO_ROOT)
    result = subprocess.run(
        ["bash", "-c", f'source {relative}; printf %s "$JEV_API_KEY"'],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=REPO_ROOT,
        env=_env(JEV_API_KEY="preset-value"),
    )

    assert result.returncode == 0
    assert result.stdout == "preset-value"


def test_sourcing_does_not_leak_strict_mode_into_the_caller_shell() -> None:
    relative = SCRIPT.relative_to(REPO_ROOT)
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"source {relative}; case \"$-\" in *e*) echo LEAK_ERREXIT ;; esac; "
            'case "$-" in *u*) echo LEAK_NOUNSET ;; esac; '
            "if shopt -o -q pipefail; then echo LEAK_PIPEFAIL; fi",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=REPO_ROOT,
        env=_env(JEV_API_KEY="preset-value"),
    )

    assert result.returncode == 0
    assert result.stdout == "", f"sourcing leaked shell options: {result.stdout!r}"


def test_sourcing_preserves_strict_mode_the_caller_already_had() -> None:
    # The flags are read with a plain `case` on "$-", never inside a command
    # substitution: bash clears -e inside $( ), so "$(echo "$-")" would report
    # errexit as absent even when the caller has it on.
    relative = SCRIPT.relative_to(REPO_ROOT)
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"set -euo pipefail; source {relative}; "
            'case "$-" in *e*) echo ERREXIT=yes ;; *) echo ERREXIT=no ;; esac; '
            'case "$-" in *u*) echo NOUNSET=yes ;; *) echo NOUNSET=no ;; esac; '
            "if shopt -o -q pipefail; then echo PIPEFAIL=yes; else echo PIPEFAIL=no; fi",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=REPO_ROOT,
        env=_env(JEV_API_KEY="preset-value"),
    )

    assert result.returncode == 0
    assert result.stdout.splitlines() == ["ERREXIT=yes", "NOUNSET=yes", "PIPEFAIL=yes"], (
        f"sourcing dropped the caller's strict mode: {result.stdout!r}"
    )


def test_executing_without_arguments_prints_usage_and_exits_2() -> None:
    result = subprocess.run(
        [str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=30,
        env=_env(JEV_API_KEY="preset-value", JEV_KEYCHAIN_SERVICE=ABSENT_SERVICE),
    )

    assert result.returncode == 2
    assert "usage" in result.stderr.lower()
