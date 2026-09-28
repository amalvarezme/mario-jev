"""The optional advisor: it must be impossible for it to break the loop.

Nothing here is skipped or weakened. The optional stack really is absent on this
machine (``torch`` and ``transformers`` are not installed and their 2.2 GB of
weights are deliberately not downloaded), so the absent path is asserted *for
real* on the branch the machine is actually in, and the present path is asserted
through the injected loader seam the contract provides. The one thing that cannot
be exercised here is the real Hugging Face call inside :func:`default_loader`; it
is not faked, and no test pretends to cover it.

The property most of this file is about is negative: ``advise`` returns ``None``
and records a reason, and no exception ever escapes into the loop.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import time
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pytest

from mario_jev.advisor import (
    ABSENCE_REASONS,
    AdvisorAbsence,
    AdvisorAdvice,
    AdvisorLoadError,
    AdvisorMove,
    FRAME_SHAPE,
    LoadedAdvisor,
    PlayJevAdvisor,
    default_loader,
    matching_scores,
    normalise_move_name,
    rank_moves,
    renormalise,
    top_move,
    validate_frame,
)
from mario_jev.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
MODEL_ID = "OmniJev/PlayJev-0.8B"

#: The moves the policy would offer the advisor on a tick, in the emulator's own
#: spelling. ``"jump"`` is deliberately not here: the model's own vocabulary is
#: not the same list.
ACTION_NAMES: tuple[str, ...] = ("NOOP", "right", "right+A", "right+B", "left")

#: A distribution over the model's own vocabulary, with one move the tick does
#: not offer (``"jump"``), so the alignment has something real to drop.
DISTRIBUTION: dict[str, float] = {"right": 0.5, "right+A": 0.3, "left": 0.1, "jump": 0.1}


def frame() -> np.ndarray:
    """The frame the emulator delivers: 240 x 256 RGB, uint8."""
    return np.zeros(FRAME_SHAPE, dtype=np.uint8)


def enabled(**overrides: object) -> Settings:
    """Settings with the advisor switched on, as ``MARIO_ADVISOR=1`` would."""
    values: dict[str, object] = {"advisor_enabled": True}
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


class Predict:
    """A test backend: a fixed answer, an optional failure, and a call count."""

    def __init__(self, result: object = None, *, error: BaseException | None = None) -> None:
        self.result = {} if result is None else result
        self.error = error
        self.calls = 0

    def __call__(self, _frame: object) -> Mapping[str, float]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result  # type: ignore[return-value]


class Mimic:
    """An object that has a shape and a dtype, but is not a numpy array.

    This is the caller bug the boundary gate must catch rather than coerce: a
    numeric-string shape that duck-typing would turn into ``(240, 256, 3)``, and a
    plain sequence dressed the same way.
    """

    def __init__(self, shape: object, dtype: object) -> None:
        self.shape = shape
        self.dtype = dtype


def loaded(predict: object, backend: str = "test-double") -> LoadedAdvisor:
    """A ``LoadedAdvisor`` around any callable, which is all the seam requires."""
    return LoadedAdvisor(backend=backend, predict=predict)  # type: ignore[arg-type]


def loader_for(predict: object) -> Callable[[str], LoadedAdvisor]:
    """An injected loader that always succeeds with this backend."""
    return lambda _model_id: loaded(predict)


def loader_raising(error: BaseException) -> Callable[[str], LoadedAdvisor]:
    """An injected loader whose whole job is to fail in a chosen way."""

    def loader(_model_id: str) -> LoadedAdvisor:
        raise error

    return loader


def loader_returning(value: object) -> Callable[[str], LoadedAdvisor]:
    """An injected loader that breaks the seam by not returning a LoadedAdvisor."""

    def loader(_model_id: str) -> LoadedAdvisor:
        return value  # type: ignore[return-value]

    return loader


def advisor_with(predict: object, backend: str = "test-double") -> PlayJevAdvisor:
    """An enabled advisor backed by a test double instead of the real loader."""
    return PlayJevAdvisor.from_settings(enabled(), loader=loader_for(predict))


# --- 1. the disabled path ----------------------------------------------------


def test_disabled_settings_leave_the_advisor_absent_without_touching_torch() -> None:
    advisor = PlayJevAdvisor.from_settings(Settings())

    assert Settings().advisor_enabled is False
    assert advisor.available is False
    assert advisor.absence == AdvisorAbsence(MODEL_ID, "disabled")
    assert advisor.model_id == MODEL_ID
    # The package was imported at collection time; importing it must not have
    # pulled the optional stack in. torch is genuinely absent here, so an entry
    # could only have come from an import attempt.
    assert "torch" not in sys.modules
    assert advisor.advise(frame(), ACTION_NAMES) is None


# --- 2. the real absent path, on the branch this machine is in ---------------


def test_the_default_loader_reports_whatever_is_true_on_this_machine() -> None:
    advisor = PlayJevAdvisor.from_settings(enabled())

    if importlib.util.find_spec("torch") is None:
        # The true branch here: neither torch nor transformers is installed, so
        # the real loader stops at the first probe and says so.
        assert advisor.available is False
        assert advisor.absence == AdvisorAbsence(MODEL_ID, "torch_missing")
        with pytest.raises(AdvisorLoadError) as raised:
            default_loader(MODEL_ID)
        assert raised.value.reason == "torch_missing"
        assert advisor.advise(frame(), ACTION_NAMES) is None
    elif importlib.util.find_spec("transformers") is None:
        assert advisor.available is False
        assert advisor.absence == AdvisorAbsence(MODEL_ID, "transformers_missing")
        with pytest.raises(AdvisorLoadError) as raised:
            default_loader(MODEL_ID)
        assert raised.value.reason == "transformers_missing"
    else:
        # The stack is installed: then the loader either loaded the local weights
        # or named a load-side reason. Downloading is never part of the question.
        assert advisor.absence is None or advisor.absence.reason in {
            "weights_missing",
            "load_failed",
        }


# --- 3. import discipline, proven in a subprocess ----------------------------

BLOCKED_STACK_SCRIPT = textwrap.dedent(
    """
    import importlib.abc
    import json
    import sys

    class Blocker(importlib.abc.MetaPathFinder):
        blocked = ("torch", "transformers")

        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in self.blocked:
                raise ImportError("blocked on purpose: " + fullname)
            return None

    sys.meta_path.insert(0, Blocker())

    import numpy as np
    import mario_jev.advisor as advisor
    from mario_jev.config import Settings

    frame = np.zeros(advisor.FRAME_SHAPE, dtype=np.uint8)
    instance = advisor.PlayJevAdvisor.from_settings(Settings(advisor_enabled=True))
    advice = instance.advise(frame, ("NOOP", "right"))
    print(json.dumps({
        "torch_imported": "torch" in sys.modules,
        "transformers_imported": "transformers" in sys.modules,
        "available": instance.available,
        "reason": instance.absence.reason if instance.absence is not None else None,
        "advice_returned": advice is not None,
    }))
    """
)


def test_the_module_never_imports_the_optional_stack_not_even_transitively() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(SRC_ROOT), env.get("PYTHONPATH", "")])
    )
    result = subprocess.run(
        [sys.executable, "-c", BLOCKED_STACK_SCRIPT],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["torch_imported"] is False
    assert report["transformers_imported"] is False
    assert report["available"] is False
    assert report["reason"] == "torch_missing"
    assert report["advice_returned"] is False


# --- 4. the present path, through the injected loader ----------------------


def test_an_injected_loader_aligns_and_renormalises_the_distribution() -> None:
    advisor = advisor_with(Predict(DISTRIBUTION), backend="test-double")
    assert advisor.available is True
    assert advisor.absence is None

    advice = advisor.advise(frame(), ACTION_NAMES)

    assert isinstance(advice, AdvisorAdvice)
    assert advice.model_id == MODEL_ID
    assert advice.backend == "test-double"
    # "jump" is not offered by the tick, so it is dropped rather than reported.
    names = tuple(move.name for move in advice.moves)
    assert names == ("right", "right+A", "left")
    assert all(name in ACTION_NAMES for name in names)
    # The survivors sum to 1: these are the probabilities over action_names only.
    assert sum(move.probability for move in advice.moves) == pytest.approx(1.0)
    assert tuple(move.probability for move in advice.moves) == pytest.approx(
        (0.5 / 0.9, 0.3 / 0.9, 0.1 / 0.9)
    )
    assert tuple(move.probability for move in advice.moves) == tuple(
        sorted((move.probability for move in advice.moves), reverse=True)
    )
    assert advice.top.name == "right"
    assert advice.top.probability == pytest.approx(0.5 / 0.9)
    assert advice.top == advice.moves[0]
    assert advice.latency_ms >= 0.0


def test_latency_is_the_measured_wall_clock_of_the_backend_call() -> None:
    def predict(_frame: object) -> Mapping[str, float]:
        time.sleep(0.02)
        return {"right": 1.0}

    advice = advisor_with(predict).advise(frame(), ACTION_NAMES)

    assert advice is not None
    assert advice.latency_ms >= 20.0


# --- 5. determinism ----------------------------------------------------------


def test_equal_probabilities_resolve_the_top_move_by_name() -> None:
    tied = (AdvisorMove("right", 0.5), AdvisorMove("left", 0.5))

    assert rank_moves(tied) == (AdvisorMove("left", 0.5), AdvisorMove("right", 0.5))
    assert top_move(tied).name == "left"
    assert top_move(tied) == top_move((AdvisorMove("left", 0.5), AdvisorMove("right", 0.5)))

    advice = advisor_with(Predict({"left": 0.5, "right": 0.5})).advise(frame(), ("left", "right"))

    assert advice is not None
    assert tuple(move.name for move in advice.moves) == ("left", "right")
    assert advice.top.name == "left"


# --- 6. no matching moves ----------------------------------------------------


@pytest.mark.parametrize(
    "distribution",
    [
        pytest.param({"alpha": 0.4, "beta": 0.6}, id="a_foreign_vocabulary"),
        pytest.param({"jump": 1.0}, id="only_moves_this_tick_does_not_offer"),
    ],
)
def test_a_vocabulary_that_shares_nothing_with_the_tick_is_reported_as_such(
    distribution: dict[str, float],
) -> None:
    advisor = advisor_with(Predict(distribution))

    assert advisor.advise(frame(), ACTION_NAMES) is None
    assert advisor.available is False
    assert advisor.absence == AdvisorAbsence(MODEL_ID, "no_matching_moves")


# --- 7. load_failed from a throwing backend ---------------------------------


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        pytest.param(RuntimeError("the backend exploded"), "load_failed", id="runtime_error"),
        pytest.param(AdvisorLoadError("weights_missing"), "weights_missing", id="load_error"),
    ],
)
def test_a_backend_that_throws_is_recorded_and_never_escapes(
    error: BaseException, reason: str
) -> None:
    advisor = advisor_with(Predict(error=error))

    assert advisor.advise(frame(), ACTION_NAMES) is None
    assert advisor.available is False
    assert advisor.absence == AdvisorAbsence(MODEL_ID, reason)


# --- 8. load_failed from a degenerate distribution --------------------------


@pytest.mark.parametrize(
    "distribution",
    [
        pytest.param({"right": 0.0}, id="one_zero"),
        pytest.param({"right": 0.0, "left": 0.0}, id="all_zero"),
        pytest.param({"right": float("nan")}, id="nan"),
        pytest.param({"right": float("inf")}, id="infinity"),
        pytest.param({"right": -0.5}, id="negative"),
        pytest.param({"right": -0.5, "left": 0.0}, id="all_non_positive"),
    ],
)
def test_a_degenerate_distribution_is_a_backend_failure_not_a_mismatch(
    distribution: dict[str, float],
) -> None:
    advisor = advisor_with(Predict(distribution))

    assert advisor.advise(frame(), ACTION_NAMES) is None
    # "no_matching_moves" would accuse the vocabulary; the names did match.
    assert advisor.absence == AdvisorAbsence(MODEL_ID, "load_failed")


@pytest.mark.parametrize(
    "distribution",
    [
        pytest.param({"right": True}, id="bool_true"),
        pytest.param({"right": False, "left": 0.5}, id="bool_false_beside_a_positive_mass"),
        pytest.param({"right": "0.5"}, id="numeric_string_fraction"),
        pytest.param({"right": "1"}, id="numeric_string_one"),
    ],
)
def test_a_bool_or_string_probability_is_a_backend_failure(
    distribution: Mapping[str, object],
) -> None:
    predict = Predict(distribution)
    advisor = advisor_with(predict)

    assert advisor.advise(frame(), ACTION_NAMES) is None
    # The names matched and the backend ran, so this is a backend failure rather
    # than "no_matching_moves".
    assert advisor.absence == AdvisorAbsence(MODEL_ID, "load_failed")
    assert predict.calls == 1


# --- 9. AdvisorLoadError handling -------------------------------------------


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        pytest.param(
            AdvisorLoadError("weights_missing"), "weights_missing", id="reported_reason"
        ),
        pytest.param(
            AdvisorLoadError("transformers_missing"),
            "transformers_missing",
            id="reported_transformers",
        ),
        pytest.param(RuntimeError("no idea what happened"), "load_failed", id="bare_runtime"),
        pytest.param(ImportError("cannot import"), "load_failed", id="bare_import"),
    ],
)
def test_a_failing_loader_is_recorded_with_its_reason(error: BaseException, reason: str) -> None:
    advisor = PlayJevAdvisor.from_settings(enabled(), loader=loader_raising(error))

    assert advisor.available is False
    assert advisor.absence == AdvisorAbsence(MODEL_ID, reason)
    assert advisor.advise(frame(), ACTION_NAMES) is None


def test_a_loader_that_does_not_return_a_loaded_advisor_is_a_load_failure() -> None:
    advisor = PlayJevAdvisor.from_settings(enabled(), loader=loader_returning(None))

    assert advisor.available is False
    assert advisor.absence == AdvisorAbsence(MODEL_ID, "load_failed")


class ExplodingPredict(LoadedAdvisor):
    """A ``LoadedAdvisor`` whose ``predict`` member raises when it is inspected.

    A loader may *return* one of these instead of raising. ``from_settings`` reads
    the seam member to validate it (``callable(loaded.predict)``) and again while
    building the advisor, so a hostile returned member has to be guarded just like
    a loader that raises: otherwise the read itself is the escape.
    """

    def __init__(self, error: BaseException) -> None:
        # ``LoadedAdvisor`` is frozen, so the hostile error is the only state and
        # the ``predict`` property is the seam member the guard must inspect.
        self._error = error

    @property
    def predict(self) -> Callable[[object], Mapping[str, float]]:
        raise self._error


def test_a_loader_returning_a_hostile_predict_member_is_a_load_failure() -> None:
    # The loader behaves: it returns a ``LoadedAdvisor``. Reading that object's
    # seam member is what raises, and it happens inside ``from_settings``, so it
    # must degrade to an absent advisor rather than escape like a loader that
    # raises would not.
    hostile = ExplodingPredict(RuntimeError("the predict member is hostile"))
    advisor = PlayJevAdvisor.from_settings(enabled(), loader=loader_returning(hostile))

    assert advisor.available is False
    assert advisor.absence is not None
    assert advisor.absence.reason == "load_failed"
    assert type(advisor.absence.reason) is str
    assert advisor.advise(frame(), ACTION_NAMES) is None


def test_a_base_exception_from_a_hostile_predict_member_is_not_swallowed() -> None:
    # Only Exception is a failure to absorb: the interrupt belongs to the human.
    hostile = ExplodingPredict(KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        PlayJevAdvisor.from_settings(enabled(), loader=loader_returning(hostile))


class HostileLoadError(AdvisorLoadError):
    """An ``AdvisorLoadError`` whose ``reason`` cannot be read at all.

    ``AdvisorLoadError`` keeps ``reason`` read-only and inside the closed set, but
    a subclass is free to override the property with something that raises. That is
    the one way an ``AdvisorLoadError`` handler could itself throw, so reading the
    reason has to be part of the guarded path rather than trusted.
    """

    def __init__(self) -> None:
        super().__init__("weights_missing")

    @property
    def reason(self) -> str:
        raise RuntimeError("the reason property is hostile")


def test_a_hostile_reason_property_cannot_escape_from_settings() -> None:
    # The loader is well-behaved; it is reading ``.reason`` that fails. A reason
    # that cannot be read is not a reason to report, so it degrades honestly.
    advisor = PlayJevAdvisor.from_settings(enabled(), loader=loader_raising(HostileLoadError()))

    assert advisor.available is False
    assert advisor.absence == AdvisorAbsence(MODEL_ID, "load_failed")
    assert advisor.advise(frame(), ACTION_NAMES) is None


def test_a_hostile_reason_property_inside_a_call_cannot_escape_advise() -> None:
    advisor = advisor_with(Predict(error=HostileLoadError()))

    assert advisor.advise(frame(), ACTION_NAMES) is None
    assert advisor.available is False
    assert advisor.absence == AdvisorAbsence(MODEL_ID, "load_failed")


class StatefulHashReason(str):
    """A ``str`` reason that is a member once, and hostile on every later test.

    The first hash succeeds, so the value looks like a member of the closed set to
    the reason helper. The second hash, which the *old* code performed when
    :class:`AdvisorAbsence` re-tested the very same object, raises. That is the
    escape the helpers must close: what leaves them has to be one of their own
    constants, so this object never reaches that second test.
    """

    def __init__(self, value: str) -> None:
        super().__init__()
        self._hashes = 0

    def __hash__(self) -> int:
        self._hashes += 1
        if self._hashes > 1:
            raise RuntimeError("the reason refuses a second hash")
        return super().__hash__()


class StatefulEqReason(str):
    """A ``str`` reason that compares equal once, then raises on comparison."""

    __hash__ = str.__hash__

    def __init__(self, value: str) -> None:
        super().__init__()
        self._comparisons = 0

    def __eq__(self, other: object) -> bool:
        self._comparisons += 1
        if self._comparisons > 1:
            raise RuntimeError("the reason refuses a second comparison")
        return str.__eq__(self, other)  # type: ignore[return-value]


class StatefulReasonError(AdvisorLoadError):
    """An ``AdvisorLoadError`` that reports a hostile object as its reason.

    The constructor is handed a well-behaved reason, so the hostile object is only
    ever touched by the reason helpers, never by ``AdvisorLoadError.__init__``.
    """

    def __init__(self, hostile: str) -> None:
        super().__init__("weights_missing")
        self._hostile = hostile

    @property
    def reason(self) -> str:
        return self._hostile


class HostileShapeFrame(np.ndarray):
    """A real numpy array whose ``.shape`` getter raises, to reach a handler body.

    ``validate_frame`` accepts it as an array and then reads ``.shape``; the error
    is not a ``ValueError``, so the frame gate does not catch it. It reaches
    ``advise``'s own ``except AdvisorLoadError`` — a handler body, where a raising
    ``_record`` would escape the loop rather than be caught by a sibling except.
    """

    def __new__(cls, error: BaseException) -> "HostileShapeFrame":
        array = np.zeros(FRAME_SHAPE, dtype=np.uint8).view(cls)
        array._error = error
        return array

    @property
    def shape(self) -> tuple[int, ...]:
        raise self._error


def test_a_stateful_hostile_reason_cannot_escape_from_settings() -> None:
    # The reason is an in-set value whose second hash raises. The loader's own
    # construction never touches it; only the reason helper does.
    hostile = StatefulHashReason("weights_missing")
    advisor = PlayJevAdvisor.from_settings(
        enabled(), loader=loader_raising(StatefulReasonError(hostile))
    )

    assert advisor.available is False
    assert advisor.absence is not None
    assert advisor.absence.reason == "weights_missing"
    assert advisor.advise(frame(), ACTION_NAMES) is None


def test_a_stateful_hostile_reason_cannot_escape_the_backend_path() -> None:
    # Same hostile value, hostile on comparison instead of hashing, raised by the
    # backend rather than the loader.
    hostile = StatefulEqReason("weights_missing")
    advisor = advisor_with(Predict(error=StatefulReasonError(hostile)))

    assert advisor.advise(frame(), ACTION_NAMES) is None
    assert advisor.available is False
    assert advisor.absence is not None
    assert advisor.absence.reason == "weights_missing"


def test_a_stateful_hostile_reason_cannot_escape_advise_own_handler() -> None:
    # Route to ``advise``'s own ``except AdvisorLoadError`` (not the one inside
    # ``_advise``): the frame is a real numpy array whose ``.shape`` getter raises
    # the AdvisorLoadError. The frame gate only catches ``ValueError``, so the
    # error propagates to the handler whose body must not raise.
    hostile = StatefulHashReason("weights_missing")
    hostile_frame = HostileShapeFrame(StatefulReasonError(hostile))
    advisor = advisor_with(Predict({"right": 1.0}))

    assert advisor.advise(hostile_frame, ACTION_NAMES) is None
    assert advisor.available is False
    assert advisor.absence is not None
    assert advisor.absence.reason == "weights_missing"


@pytest.mark.parametrize(
    "reason",
    [
        pytest.param("weights_missing", id="weights_missing"),
        pytest.param("load_failed", id="load_failed"),
        pytest.param("torch_missing", id="torch_missing"),
        pytest.param("no_matching_moves", id="no_matching_moves"),
    ],
)
def test_a_well_behaved_reason_still_round_trips_on_both_paths(reason: str) -> None:
    from_loader = PlayJevAdvisor.from_settings(
        enabled(), loader=loader_raising(AdvisorLoadError(reason))
    )
    assert from_loader.absence is not None
    assert from_loader.absence.reason == reason

    from_backend = advisor_with(Predict(error=AdvisorLoadError(reason)))
    assert from_backend.advise(frame(), ACTION_NAMES) is None
    assert from_backend.absence is not None
    assert from_backend.absence.reason == reason


def test_a_hostile_reason_never_survives_into_the_record() -> None:
    # The invariant, stated directly: after a hostile value, the recorded reason is
    # one of this module's own exact ``str`` constants, not the caller's object.
    hostile = StatefulHashReason("weights_missing")
    advisor = PlayJevAdvisor.from_settings(
        enabled(), loader=loader_raising(StatefulReasonError(hostile))
    )

    assert advisor.absence is not None
    reason = advisor.absence.reason
    assert type(reason) is str
    assert reason in ABSENCE_REASONS
    assert not isinstance(reason, StatefulHashReason)


def test_a_base_exception_from_the_loader_is_not_swallowed() -> None:
    # Only Exception is a failure to absorb: an interrupt belongs to the human,
    # and swallowing it would make the advisor uncancellable.
    with pytest.raises(KeyboardInterrupt):
        PlayJevAdvisor.from_settings(enabled(), loader=loader_raising(KeyboardInterrupt()))


# --- 10. absence is sticky ---------------------------------------------------


def test_absence_is_sticky_and_the_backend_is_not_called_again() -> None:
    predict = Predict({"right": 0.0})
    advisor = advisor_with(predict)

    assert advisor.advise(frame(), ACTION_NAMES) is None
    assert predict.calls == 1
    assert advisor.available is False

    assert advisor.advise(frame(), ACTION_NAMES) is None
    assert predict.calls == 1
    # The first failure is the recorded one, and it is never replaced.
    assert advisor.absence == AdvisorAbsence(MODEL_ID, "load_failed")


# --- 11. advise never raises -------------------------------------------------


@pytest.mark.parametrize(
    ("loader", "frame_arg", "action_names", "reason"),
    [
        pytest.param(
            loader_raising(AdvisorLoadError("torch_missing")),
            frame(),
            ACTION_NAMES,
            "torch_missing",
            id="loader_reports_torch_missing",
        ),
        pytest.param(
            loader_raising(RuntimeError("boom")),
            frame(),
            ACTION_NAMES,
            "load_failed",
            id="loader_raises",
        ),
        pytest.param(
            loader_returning("not a LoadedAdvisor"),
            frame(),
            ACTION_NAMES,
            "load_failed",
            id="loader_returns_an_impostor",
        ),
        pytest.param(
            loader_for(Predict(error=RuntimeError("boom"))),
            frame(),
            ACTION_NAMES,
            "load_failed",
            id="predict_raises",
        ),
        pytest.param(
            loader_for(Predict(error=AdvisorLoadError("weights_missing"))),
            frame(),
            ACTION_NAMES,
            "weights_missing",
            id="predict_reports_a_reason",
        ),
        pytest.param(
            loader_for(Predict([("right", 1.0)])),
            frame(),
            ACTION_NAMES,
            "load_failed",
            id="predict_returns_a_sequence",
        ),
        pytest.param(
            loader_for(Predict({})), frame(), ACTION_NAMES, "no_matching_moves", id="predict_empty"
        ),
        pytest.param(
            loader_for(Predict({"right": "maybe"})),
            frame(),
            ACTION_NAMES,
            "load_failed",
            id="predict_returns_a_non_number",
        ),
        pytest.param(
            loader_for(Predict({"right": 0.0})),
            frame(),
            ACTION_NAMES,
            "load_failed",
            id="predict_returns_only_zero",
        ),
        pytest.param(
            loader_for(Predict({"jump": 1.0})),
            frame(),
            ACTION_NAMES,
            "no_matching_moves",
            id="predict_offers_only_unknown_moves",
        ),
        pytest.param(
            loader_for(Predict({"right": 1.0})),
            np.zeros((256, 240, 3), dtype=np.uint8),
            ACTION_NAMES,
            "load_failed",
            id="frame_is_transposed",
        ),
        pytest.param(
            loader_for(Predict({"right": 1.0})),
            np.zeros(FRAME_SHAPE, dtype=np.uint16),
            ACTION_NAMES,
            "load_failed",
            id="frame_dtype_is_wrong",
        ),
        pytest.param(
            loader_for(Predict({"right": 1.0})),
            [[0, 0, 0]],
            ACTION_NAMES,
            "load_failed",
            id="frame_is_not_an_array",
        ),
        pytest.param(
            loader_for(Predict({"right": 1.0})),
            frame(),
            (),
            "no_matching_moves",
            id="no_action_names_offered",
        ),
    ],
)
def test_advise_never_raises_and_names_the_failure(
    loader: Callable[[str], LoadedAdvisor],
    frame_arg: object,
    action_names: tuple[str, ...],
    reason: str,
) -> None:
    advisor = PlayJevAdvisor.from_settings(enabled(), loader=loader)

    assert advisor.advise(frame_arg, action_names) is None
    assert advisor.available is False
    assert advisor.absence == AdvisorAbsence(MODEL_ID, reason)


# --- 12. no optional import at module import time ---------------------------


def test_the_module_import_did_not_pull_in_the_optional_stack() -> None:
    if importlib.util.find_spec("torch") is None:
        # `find_spec` cannot import torch, so on this machine the absence of a
        # "torch" entry in sys.modules is evidence that importing
        # mario_jev.advisor did not import it (and did not fail trying, which
        # would have left the same absence). Where the optional stack *is*
        # installed, the assertion would be a claim about that machine rather
        # than about this module, so the honest branch asserts what it can see.
        assert "torch" not in sys.modules
        assert "transformers" not in sys.modules
    else:
        assert importlib.util.find_spec("torch") is not None


# --- the helpers, on their own ----------------------------------------------


def test_the_closed_reason_set_is_exactly_the_contracted_six() -> None:
    assert ABSENCE_REASONS == frozenset(
        {
            "disabled",
            "torch_missing",
            "transformers_missing",
            "weights_missing",
            "load_failed",
            "no_matching_moves",
        }
    )


def test_a_reason_outside_the_closed_set_is_a_bug_not_a_state() -> None:
    with pytest.raises(ValueError):
        AdvisorLoadError("probably_fine")


def test_an_absence_reason_outside_the_closed_set_is_refused_at_construction() -> None:
    with pytest.raises(ValueError):
        AdvisorAbsence("model", "invented_reason")
    # Every contracted reason still constructs: the closed set is the only gate.
    for reason in ABSENCE_REASONS:
        assert AdvisorAbsence("model", reason).reason == reason


def test_a_load_errors_reason_is_read_only_and_keeps_its_text() -> None:
    error = AdvisorLoadError("load_failed")

    assert error.reason == "load_failed"
    # The read-only property must not have changed the exception's own text.
    assert str(error) == "load_failed"
    assert error.args == ("load_failed",)
    with pytest.raises(AttributeError):
        error.reason = "bogus"  # type: ignore[misc]


def test_a_directly_constructed_advisor_refuses_an_untyped_absence() -> None:
    # The invariant is a property of the type, not of ``from_settings``: only
    # ``None`` or a real AdvisorAbsence can be stored, so ``.absence`` can never
    # carry a raw string.
    with pytest.raises(TypeError) as raised:
        PlayJevAdvisor(MODEL_ID, absence="bogus_string_not_an_absence")  # type: ignore[arg-type]
    # The message names the offending type, and never the offending value.
    assert "str" in str(raised.value)
    assert "bogus_string_not_an_absence" not in str(raised.value)

    assert PlayJevAdvisor(MODEL_ID, absence=None).absence is None

    absence = AdvisorAbsence(MODEL_ID, "disabled")
    assert PlayJevAdvisor(MODEL_ID, absence=absence).absence == absence


def test_the_advice_types_are_frozen_slotted_dataclasses() -> None:
    for kind in (AdvisorMove, AdvisorAdvice, AdvisorAbsence, LoadedAdvisor):
        assert kind.__dataclass_params__.frozen is True
        assert isinstance(kind.__slots__, tuple)
    assert AdvisorMove.__slots__ == ("name", "probability")

    move = AdvisorMove("right", 0.5)
    with pytest.raises(FrozenInstanceError):
        move.name = "left"


def test_the_package_exports_the_contracted_names_without_drift() -> None:
    import mario_jev.advisor as advisor
    from mario_jev.advisor import playjev

    required = {
        "AdvisorAbsence",
        "AdvisorAdvice",
        "AdvisorLoadError",
        "AdvisorMove",
        "LoadedAdvisor",
        "PlayJevAdvisor",
    }
    assert required <= set(advisor.__all__)
    assert all(getattr(advisor, name) is not None for name in required)
    # The package re-exports the module's contract exactly: no name exported but
    # undefined, and no defined name left out of the public surface.
    assert set(advisor.__all__) == set(playjev.__all__)
    assert len(advisor.__all__) == len(set(advisor.__all__))


def test_frame_validation_accepts_the_emulator_frame() -> None:
    assert validate_frame(frame()) is None


@pytest.mark.parametrize(
    "bad_frame",
    [
        pytest.param(np.zeros((256, 240, 3), dtype=np.uint8), id="transposed"),
        pytest.param(np.zeros((240, 256), dtype=np.uint8), id="no_channels"),
        pytest.param(np.zeros((240, 256, 4), dtype=np.uint8), id="rgba"),
        pytest.param(np.zeros(FRAME_SHAPE, dtype=np.uint16), id="uint16"),
        pytest.param(np.zeros(FRAME_SHAPE, dtype=np.float32), id="float32"),
        pytest.param([0, 1, 2], id="a_plain_list"),
        pytest.param(None, id="none"),
        pytest.param(object(), id="no_shape_attribute"),
    ],
)
def test_frame_validation_rejects_anything_but_the_contracted_frame(bad_frame: object) -> None:
    with pytest.raises(ValueError):
        validate_frame(bad_frame)


def test_frame_validation_is_exact_not_coercive() -> None:
    # The real emulator frame is accepted, and nothing that merely resembles it is.
    assert validate_frame(frame()) is None
    # A numeric-string shape is not a (240, 256, 3) frame, however coercible.
    with pytest.raises(ValueError):
        validate_frame(Mimic(("240", "256", "3"), "uint8"))
    # A plain list/tuple-based mimic is not a numpy array either.
    with pytest.raises(ValueError):
        validate_frame(Mimic(list(FRAME_SHAPE), "uint8"))
    with pytest.raises(ValueError):
        validate_frame(Mimic(tuple(FRAME_SHAPE), "uint8"))
    # Real shape and dtype, but not an array: still refused.
    with pytest.raises(ValueError):
        validate_frame(Mimic(FRAME_SHAPE, "uint8"))


def test_a_mimicked_frame_is_refused_before_the_backend_runs() -> None:
    predict = Predict({"right": 1.0})
    advisor = advisor_with(predict)

    result = advisor.advise(Mimic(("240", "256", "3"), "uint8"), ACTION_NAMES)

    assert result is None
    assert advisor.absence == AdvisorAbsence(MODEL_ID, "load_failed")
    # The boundary gate precedes the backend, so the mimic never reaches predict.
    assert predict.calls == 0


def test_move_names_are_matched_across_case_and_surrounding_space() -> None:
    assert normalise_move_name(" Right+A ") == "right+a"
    assert matching_scores(
        {" Right ": 0.4, "RIGHT+A": 0.6}, ("right", "right+A")
    ) == {"right": 0.4, "right+A": 0.6}


def test_two_spellings_of_one_move_are_added_together() -> None:
    assert matching_scores({"right": 0.25, " Right ": 0.25}, ("right",)) == {"right": 0.5}


def test_alignment_keeps_mass_unnormalised_so_the_two_decisions_stay_apart() -> None:
    assert matching_scores({"right": 0.2, "jump": 0.8}, ("right",)) == {"right": 0.2}


def test_alignment_rejects_a_backend_that_does_not_return_a_mapping() -> None:
    with pytest.raises(ValueError):
        matching_scores([("right", 1.0)], ("right",))  # type: ignore[arg-type]


def test_renormalise_scales_the_survivors_and_drops_the_rest() -> None:
    moves = renormalise({"right": 0.5, "left": 0.3, "down": -0.2, "up": float("nan")})

    assert tuple(move.name for move in moves) == ("right", "left")
    assert tuple(move.probability for move in moves) == pytest.approx((0.625, 0.375))


@pytest.mark.parametrize(
    "matches",
    [
        pytest.param({}, id="empty"),
        pytest.param({"right": 0.0}, id="zero"),
        pytest.param({"right": float("nan")}, id="nan"),
        pytest.param({"right": -1.0}, id="negative"),
    ],
)
def test_renormalise_rejects_mass_that_is_not_positive_and_finite(
    matches: Mapping[str, float],
) -> None:
    with pytest.raises(ValueError):
        renormalise(matches)
