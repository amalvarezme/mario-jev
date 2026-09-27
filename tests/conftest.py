"""Shared fixtures.

The emulator is a real, stateful, native object: every test that builds one owes
the process a ``close()``. The factory fixture collects them so a failing test
cannot leak a running emulator into the next one.
"""

from __future__ import annotations

from typing import Callable

import pytest

from mario_jev.emulator import MarioEmulator


@pytest.fixture
def emulator_factory() -> Callable[..., MarioEmulator]:
    """Build emulators that are closed automatically at the end of the test."""
    created: list[MarioEmulator] = []

    def make(**kwargs: object) -> MarioEmulator:
        kwargs.setdefault("real_time", False)
        emulator = MarioEmulator(**kwargs)  # type: ignore[arg-type]
        created.append(emulator)
        return emulator

    yield make

    for emulator in created:
        emulator.close()


@pytest.fixture
def emulator(emulator_factory: Callable[..., MarioEmulator]) -> MarioEmulator:
    """A reset emulator sitting on the first frame of 1-1."""
    instance = emulator_factory()
    instance.reset(seed=0)
    return instance
