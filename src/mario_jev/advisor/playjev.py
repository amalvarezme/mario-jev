"""The optional local PlayJev-0.8B advisor, and the wall it can never climb.

PlayJev reads the frame directly and answers with a probability per move. It is
*designed to be* a useful *feature* and never the *authority*: its published
agreement with the teacher on Infinite Mario is 0.32 (1757 vs 4229), the second
worst of its ten games. The policy is *designed to* consume its distribution as
one more input and to keep the deterministic decision path in charge; that policy
is WU4 slice C and is not written yet.

Three properties make that safe, and each one is structural rather than a
convention:

* **It cannot break the loop it will feed.** :meth:`PlayJevAdvisor.advise` never
  raises an :class:`Exception`; the loop itself is WU4 slice C, not written yet.
  Every such failure — a missing optional import, a load failure, a backend that
  throws, a vocabulary that shares nothing with the offered moves — becomes a
  recorded :class:`AdvisorAbsence` and a ``None`` return. A
  :class:`BaseException` such as :class:`KeyboardInterrupt` is deliberately not
  swallowed.
* **Asking costs nothing.** Availability is answered with
  :func:`importlib.util.find_spec`, which does not import ``torch``. ``torch`` and
  ``transformers`` are imported inside the real loader's body only, so
  ``import mario_jev.advisor`` works on a machine where neither is installed —
  which is the normal state here, not an error state.
* **It is never retried once broken.** The first absence is recorded and kept, so
  a broken backend cannot be probed again on every tick.

The 2.2 GB of weights are deliberately absent and the real loader never downloads
them: it resolves the local Hugging Face snapshot only, and reports
``"weights_missing"`` when there is none.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from mario_jev.config import Settings

#: The frame shape the emulator delivers: 240 x 256 RGB.
FRAME_SHAPE: tuple[int, int, int] = (240, 256, 3)
#: The frame dtype the emulator delivers.
FRAME_DTYPE = "uint8"

#: The settings said the advisor is off.
REASON_DISABLED = "disabled"
#: ``torch`` is not importable, so there is no backend at all.
REASON_TORCH_MISSING = "torch_missing"
#: ``torch`` is importable but the model library is not.
REASON_TRANSFORMERS_MISSING = "transformers_missing"
#: The optional stack is present but the local snapshot is not.
REASON_WEIGHTS_MISSING = "weights_missing"
#: A load, a call, or a distribution that no honest reading can accept.
REASON_LOAD_FAILED = "load_failed"
#: The backend answered, but about a vocabulary this tick does not offer.
REASON_NO_MATCHING_MOVES = "no_matching_moves"

#: The closed reason set. A reason outside it is a bug, not a new state.
ABSENCE_REASONS: frozenset[str] = frozenset(
    {
        REASON_DISABLED,
        REASON_TORCH_MISSING,
        REASON_TRANSFORMERS_MISSING,
        REASON_WEIGHTS_MISSING,
        REASON_LOAD_FAILED,
        REASON_NO_MATCHING_MOVES,
    }
)

#: The short backend label reported in :attr:`AdvisorAdvice.backend`.
BACKEND = "playjev-torch"

#: Config keys a snapshot may use for its own move vocabulary, in priority order.
_VOCABULARY_KEYS: tuple[str, ...] = ("id2label", "labels", "move_labels", "buttons")

__all__ = [
    "ABSENCE_REASONS",
    "BACKEND",
    "FRAME_DTYPE",
    "FRAME_SHAPE",
    "REASON_DISABLED",
    "REASON_LOAD_FAILED",
    "REASON_NO_MATCHING_MOVES",
    "REASON_TORCH_MISSING",
    "REASON_TRANSFORMERS_MISSING",
    "REASON_WEIGHTS_MISSING",
    "AdvisorAbsence",
    "AdvisorAdvice",
    "AdvisorLoadError",
    "AdvisorMove",
    "LoadedAdvisor",
    "PlayJevAdvisor",
    "default_loader",
    "matching_scores",
    "normalise_move_name",
    "rank_moves",
    "renormalise",
    "top_move",
    "validate_frame",
]


@dataclass(frozen=True, slots=True)
class AdvisorMove:
    """One move and the probability the advisor gave it."""

    name: str
    probability: float


@dataclass(frozen=True, slots=True)
class AdvisorAdvice:
    """One frame's advice, already aligned to the moves the tick offered.

    ``moves`` holds every surviving move in descending probability order and
    ``top`` is the single maximum, ties broken by name so two runs on the same
    frame agree. ``latency_ms`` is the measured wall-clock of the backend call
    alone; it is recorded, never repaired.
    """

    model_id: str
    backend: str
    moves: tuple[AdvisorMove, ...]
    top: AdvisorMove
    latency_ms: float


@dataclass(frozen=True, slots=True)
class AdvisorAbsence:
    """Why the advisor has no answer. Recorded once, and never cleared.

    ``reason`` is enforced against :data:`ABSENCE_REASONS` here, at construction,
    the same way :class:`AdvisorLoadError` enforces it: a reason outside the set is
    a bug, not a new state, and it should not survive to the journal.
    """

    model_id: str
    reason: str

    def __post_init__(self) -> None:
        if self.reason not in ABSENCE_REASONS:
            raise ValueError(f"unknown advisor reason: {self.reason!r}")


@dataclass(frozen=True, slots=True)
class LoadedAdvisor:
    """A loaded backend, and the single call the advisor makes on it.

    ``predict`` takes the frame and returns probabilities over the *model's own*
    move vocabulary. It is not told which moves the tick offers: alignment is the
    advisor's job, which is what makes it testable on its own.
    """

    backend: str
    predict: Callable[[object], Mapping[str, float]]


class AdvisorLoadError(Exception):
    """Raised by a loader; always carries a reason from :data:`ABSENCE_REASONS`.

    ``reason`` is enforced at construction and is read-only afterwards, so the
    reason a loader reported cannot be rewritten into an invented state on the
    way to a journal.
    """

    def __init__(self, reason: str) -> None:
        if reason not in ABSENCE_REASONS:
            raise ValueError(f"unknown advisor reason: {reason!r}")
        super().__init__(reason)
        self._reason = reason

    @property
    def reason(self) -> str:
        """The reported reason, from the closed set. No setter, by design."""
        return self._reason


#: The canonical plain-``str`` constant for each reason. Both values and keys are
#: this module's own constants, so a lookup can return one of them instead of the
#: caller's object; a hostile ``str`` subclass therefore never survives past the
#: reason helpers to be re-tested downstream.
_CANONICAL_REASONS: dict[str, str] = {reason: reason for reason in ABSENCE_REASONS}


def _canonical_reason(reason: object) -> str:
    """One of this module's own plain-``str`` constants, or a fallback.

    The lookup both tests membership and returns the mapped constant, so the value
    returned is never the argument: a ``str`` subclass that behaves while it is
    tested the first time and turns hostile on a later test — the membership test
    :class:`AdvisorAbsence` would otherwise perform on the same object downstream —
    can no longer reach that later test. A reason the lookup cannot accept is
    ``"load_failed"``. Only :class:`Exception` is absorbed; a :class:`BaseException`
    still propagates.
    """
    try:
        return _CANONICAL_REASONS[reason]
    except Exception:
        return REASON_LOAD_FAILED


def _resolved_reason(reason: object) -> str:
    """A reason that is guaranteed to be one of this module's own constants.

    Nothing that reaches a journal should be able to invent a state, and nothing
    here may raise on the way. The value returned is always the canonical plain
    ``str`` constant, never the argument, so a hostile ``str`` subclass cannot
    survive to be re-tested by :class:`AdvisorAbsence`.
    """
    return _canonical_reason(reason)


def _reported_reason(error: AdvisorLoadError) -> str:
    """The reason an :class:`AdvisorLoadError` reports, as one of our constants.

    On the real class ``reason`` is a plain read-only property, but a subclass may
    override it with something that raises. Reading the reason is part of the guarded
    path, and so is canonicalising it: the value returned is always one of this
    module's own plain ``str`` constants, never the caller's object, so a hostile
    value cannot be re-tested downstream. A reason that cannot be read — or that the
    lookup refuses — is not a reason this layer may trust, so it degrades to
    ``"load_failed"``. Only :class:`Exception` is absorbed; a :class:`BaseException`
    still propagates.
    """
    try:
        return _canonical_reason(error.reason)
    except Exception:
        return REASON_LOAD_FAILED


def validate_frame(frame: object) -> None:
    """Reject anything that is not a ``(240, 256, 3)`` ``uint8`` RGB frame.

    The frame is the numpy array :meth:`MarioEmulator.frame` produces, so the gate
    checks that type directly rather than duck-typing: an object that merely *has*
    a shape and a dtype — a numeric-string shape, or a list dressed up the same way
    — is not the contracted frame, and accepting it would let a caller bug surface
    later as a backend failure. It is the boundary gate, and a frame that fails it
    is a bug in the caller, which is why it raises :class:`ValueError` and lets
    :meth:`PlayJevAdvisor.advise` turn that into a recorded absence instead of a
    crash in the loop that will consume the advisor.
    """
    if not isinstance(frame, np.ndarray):
        raise ValueError(f"frame must be a numpy array, got {type(frame).__name__}")
    if frame.shape != FRAME_SHAPE:
        raise ValueError(f"frame must have shape {FRAME_SHAPE}, got {frame.shape}")
    if frame.dtype != np.dtype(FRAME_DTYPE):
        raise ValueError(f"frame must have dtype {FRAME_DTYPE}, got {frame.dtype}")


def normalise_move_name(name: object) -> str:
    """The comparison key for a move name: trimmed and lowercased.

    The emulator spells its actions ``"right+A"`` while a model card may spell the
    same move ``" Right+A "``. Case and surrounding space are the only differences
    this layer forgives; the move itself has to match.
    """
    return str(name).strip().lower()


def matching_scores(
    scores: Mapping[str, float],
    action_names: Sequence[str],
) -> dict[str, float]:
    """The backend's distribution restricted to the moves the tick offers.

    Returns a mapping from the tick's own spelling of a move to the mass the
    backend gave it. A model vocabulary that spells one move twice has its masses
    added, because both entries are the same move. Values are not yet filtered or
    renormalised: that is :func:`renormalise`, so the two decisions stay separate.
    """
    if not isinstance(scores, Mapping):
        raise ValueError("the backend must return a mapping of move name to probability")

    offered: dict[str, str] = {}
    for name in action_names:
        offered.setdefault(normalise_move_name(name), str(name))

    matched: dict[str, float] = {}
    for name, value in scores.items():
        canonical = offered.get(normalise_move_name(name))
        if canonical is None:
            continue
        # The mass comes from a softmax over the model's logits, so it is a real
        # number. bool is an int subclass, and a JSON true where a number belongs
        # is a shape violation, not a 1; a string is a backend bug too. Either way
        # it is a backend failure, not a probability to add.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"probability for {name!r} is not a number")
        matched[canonical] = matched.get(canonical, 0.0) + float(value)
    return matched


def renormalise(matches: Mapping[str, float]) -> tuple[AdvisorMove, ...]:
    """Surviving positive probabilities, scaled to sum to 1.

    A non-positive or non-finite value carries no probability and is dropped. If
    that leaves no mass at all the distribution is degenerate, and the caller
    records ``"load_failed"``: a backend that answers with zeros has failed, while
    ``"no_matching_moves"`` would accuse the wrong thing.
    """
    positive = {
        name: value for name, value in matches.items() if math.isfinite(value) and value > 0.0
    }
    total = math.fsum(positive.values())
    if not positive or not math.isfinite(total) or total <= 0.0:
        raise ValueError("advisor probability mass is not positive and finite")
    return tuple(
        AdvisorMove(name=name, probability=value / total) for name, value in positive.items()
    )


def rank_moves(moves: Sequence[AdvisorMove]) -> tuple[AdvisorMove, ...]:
    """Descending probability, ties broken by name so the order is total."""
    return tuple(sorted(moves, key=lambda move: (-move.probability, move.name)))


def top_move(moves: Sequence[AdvisorMove]) -> AdvisorMove:
    """The single highest-probability move, or a ``ValueError`` if there is none."""
    ranked = rank_moves(moves)
    if not ranked:
        raise ValueError("there is no move to rank")
    return ranked[0]


def _module_available(name: str) -> bool:
    """Whether an importable module exists, without importing it.

    A meta-path finder that refuses the name, or a parent package that cannot be
    resolved, means the module is not importable — which is exactly the question.
    ``find_spec`` never executes the module, so this cannot cost seconds.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


class PlayJevAdvisor:
    """The optional advisor, absent by default and harmless when broken.

    Build it with :meth:`from_settings`. Once absent it is absent for the rest of
    the run: ``available`` never returns to true.
    """

    __slots__ = ("_model_id", "_absence", "_backend", "_predict")

    def __init__(
        self,
        model_id: str,
        *,
        absence: AdvisorAbsence | None = None,
        loaded: LoadedAdvisor | None = None,
    ) -> None:
        if absence is not None and not isinstance(absence, AdvisorAbsence):
            raise TypeError(
                "absence must be None or an AdvisorAbsence, "
                f"got {type(absence).__name__}"
            )
        self._model_id = model_id
        self._absence = absence
        self._backend = loaded.backend if loaded is not None else ""
        self._predict = loaded.predict if loaded is not None else None

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        loader: Callable[[str], LoadedAdvisor] | None = None,
    ) -> "PlayJevAdvisor":
        """Probe, load, and return an advisor that is either usable or absent.

        A disabled advisor is decided from the setting alone: no probe, no import,
        no cost. An injected ``loader`` replaces the whole probe-and-load step, so
        the present path is testable without the optional stack; the default loader
        is the only thing that knows about ``torch``.
        """
        model_id = settings.advisor_model_id
        if not settings.advisor_enabled:
            return cls(model_id, absence=AdvisorAbsence(model_id, REASON_DISABLED))

        resolve = default_loader if loader is None else loader
        try:
            loaded = resolve(model_id)
        except AdvisorLoadError as exc:
            # Reading ``.reason`` is guarded, and so is what is recorded: the
            # value is canonicalised to one of our own constants before storage.
            return cls(model_id, absence=AdvisorAbsence(model_id, _reported_reason(exc)))
        except Exception:
            return cls(model_id, absence=AdvisorAbsence(model_id, REASON_LOAD_FAILED))

        try:
            # The seam is inspected inside a guard: a loader that *returns* an
            # object whose members explode on inspection must degrade exactly as
            # a loader that raises does, so no ``Exception`` may escape here.
            if not isinstance(loaded, LoadedAdvisor) or not callable(loaded.predict):
                return cls(model_id, absence=AdvisorAbsence(model_id, REASON_LOAD_FAILED))
            return cls(model_id, loaded=loaded)
        except Exception:
            return cls(model_id, absence=AdvisorAbsence(model_id, REASON_LOAD_FAILED))

    @property
    def model_id(self) -> str:
        """The model this advisor was configured for."""
        return self._model_id

    @property
    def available(self) -> bool:
        """Whether the advisor can still answer. False forever once absent."""
        return self._absence is None and self._predict is not None

    @property
    def absence(self) -> AdvisorAbsence | None:
        """The first failure that made the advisor unusable, or ``None``."""
        return self._absence

    def advise(
        self,
        frame: object,
        action_names: Sequence[str],
    ) -> AdvisorAdvice | None:
        """Ask the advisor about one frame. Never raises an :class:`Exception`.

        Returns ``None`` when the advisor is absent, when the backend fails, or
        when its answer cannot honestly be read; the reason is recorded in
        :attr:`absence`. A :class:`BaseException` such as
        :class:`KeyboardInterrupt` is deliberately not swallowed.
        """
        if not self.available:
            return None
        try:
            return self._advise(frame, action_names)
        except AdvisorLoadError as exc:
            # A loader-flavoured failure raised from inside the backend call.
            self._record(_reported_reason(exc))
            return None
        except Exception:
            # Belt and braces: no unforeseen failure may escape to the caller.
            self._record(REASON_LOAD_FAILED)
            return None

    def _advise(
        self,
        frame: object,
        action_names: Sequence[str],
    ) -> AdvisorAdvice | None:
        """The guarded body of :meth:`advise`, with one reason per failure."""
        predict = self._predict
        if predict is None:
            # Guarded by `available`, and kept explicit so that optimised runs
            # behave the same as ordinary ones.
            return self._record(REASON_LOAD_FAILED)

        try:
            validate_frame(frame)
        except ValueError:
            return self._record(REASON_LOAD_FAILED)

        started = time.perf_counter()
        try:
            scores = predict(frame)
        except AdvisorLoadError as exc:
            return self._record(_reported_reason(exc))
        except Exception:
            return self._record(REASON_LOAD_FAILED)
        latency_ms = (time.perf_counter() - started) * 1000.0

        try:
            matches = matching_scores(scores, action_names)
        except Exception:
            return self._record(REASON_LOAD_FAILED)
        if not matches:
            return self._record(REASON_NO_MATCHING_MOVES)

        try:
            moves = rank_moves(renormalise(matches))
        except Exception:
            return self._record(REASON_LOAD_FAILED)

        return AdvisorAdvice(
            model_id=self._model_id,
            backend=self._backend,
            moves=moves,
            top=moves[0],
            latency_ms=latency_ms,
        )

    def _record(self, reason: str) -> None:
        """Record the first absence and keep it. Returns ``None`` for chaining.

        The first failure is the one a journal should carry: later attempts cannot
        happen anyway, because ``available`` is false from here on.
        """
        if self._absence is None:
            self._absence = AdvisorAbsence(self._model_id, _resolved_reason(reason))
        return None


def _cache_roots() -> tuple[Path, ...]:
    """The Hugging Face cache directories to look in, in order, without network.

    Both the modern ``HF_HUB_CACHE``/``HF_HOME`` variables and the historic
    ``TRANSFORMERS_CACHE`` are honoured, because a machine may have either.
    """
    roots: list[Path] = []
    for name in ("HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        raw = os.environ.get(name)
        if raw:
            roots.append(Path(raw))
    home = os.environ.get("HF_HOME")
    roots.append(Path(home) / "hub" if home else Path.home() / ".cache" / "huggingface" / "hub")
    return tuple(dict.fromkeys(roots))


def _has_weight_file(snapshot: Path) -> bool:
    """Whether a snapshot directory actually holds model weights."""
    for pattern in ("*.safetensors", "*.bin", "*.pt"):
        if any(snapshot.glob(pattern)):
            return True
    return False


def _local_snapshot(model_id: str) -> Path | None:
    """The local weights for ``model_id``, or ``None``. Never downloads.

    A bare local directory is accepted as a model id, then the standard
    ``models--<owner>--<name>/snapshots/<revision>`` cache layout. A snapshot only
    counts when it has both a ``config.json`` and weight files, so a half-evicted
    cache reads as absent instead of failing later.
    """
    direct = Path(model_id)
    if direct.is_dir() and (direct / "config.json").is_file() and _has_weight_file(direct):
        return direct

    slug = "models--" + model_id.replace("/", "--")
    for cache in _cache_roots():
        snapshots = cache / slug / "snapshots"
        if not snapshots.is_dir():
            continue
        for snapshot in sorted(snapshots.iterdir()):
            if (snapshot / "config.json").is_file() and _has_weight_file(snapshot):
                return snapshot
    return None


def default_loader(model_id: str) -> LoadedAdvisor:
    """Load PlayJev from the local cache, or name the reason it cannot be loaded.

    The two probes come first and they are ``find_spec`` lookups, not imports, so
    the answer to "is the optional stack here?" costs nothing. The weights are
    resolved locally and loaded with ``local_files_only=True``: this loader never
    reaches the network and never pulls the 2.2 GB.
    """
    if not _module_available("torch"):
        raise AdvisorLoadError(REASON_TORCH_MISSING)
    if not _module_available("transformers"):
        raise AdvisorLoadError(REASON_TRANSFORMERS_MISSING)

    snapshot = _local_snapshot(model_id)
    if snapshot is None:
        raise AdvisorLoadError(REASON_WEIGHTS_MISSING)

    try:
        return _load_snapshot(snapshot)
    except AdvisorLoadError:
        raise
    except Exception as exc:
        raise AdvisorLoadError(REASON_LOAD_FAILED) from exc


def _move_vocabulary(snapshot: Path) -> tuple[str, ...]:
    """The model's own move vocabulary, read from its snapshot config.

    Which config key holds it depends on how the repository was published, so the
    documented candidates are tried in order. A snapshot that names no vocabulary
    cannot be aligned against anything, which is a load failure.
    """
    config = json.loads((snapshot / "config.json").read_text())
    for key in _VOCABULARY_KEYS:
        raw = config.get(key)
        if isinstance(raw, Mapping) and raw:
            return tuple(str(raw[index]) for index in sorted(raw, key=_label_index))
        if isinstance(raw, (list, tuple)) and raw:
            return tuple(str(item) for item in raw)
    raise AdvisorLoadError(REASON_LOAD_FAILED)


def _label_index(key: object) -> int:
    """Sort ``{"0": "NOOP", ...}`` numerically, and anything else as text."""
    try:
        return int(str(key))
    except ValueError:
        return 0


def _label_token_ids(processor: object, labels: tuple[str, ...]) -> list[int]:
    """The token ids the model uses for its own move labels."""
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise AdvisorLoadError(REASON_LOAD_FAILED)
    return [int(tokenizer.convert_tokens_to_ids(label)) for label in labels]


def _make_predict(
    torch: object,
    model: object,
    processor: object,
    labels: tuple[str, ...],
) -> Callable[[object], Mapping[str, float]]:
    """Wrap the loaded model in the ``frame -> probabilities`` seam.

    Everything here is a best-effort reading of the published assets. It is the
    one part of this slice that cannot be exercised without the 2.2 GB of weights,
    deliberately not downloaded here. Two kinds of disagreement are worth telling
    apart. Every disagreement that *raises* — a missing attribute, a shape the
    library refuses, a token id it cannot resolve — is absorbed as ``"load_failed"``
    and never leaves the ``advise`` call. A disagreement that runs and answers with
    finite numbers is *not* detected: a wrong vocabulary key order, an unknown label
    that resolves to an ``unk`` token id, or a logits position read from the wrong
    slot all produce a structurally valid distribution that will look like real
    signal to the policy of WU4 slice C, which is not written yet. That is exactly
    why the advisor is *designed to be* a feature and never the authority, and why
    this call is disclosed as unverified rather than claimed to be tested.
    """

    def predict(frame: object) -> Mapping[str, float]:
        try:
            with torch.no_grad():
                inputs = processor(images=frame, return_tensors="pt")
                logits = model(**inputs).logits
                token_ids = _label_token_ids(processor, labels)
                probabilities = torch.softmax(logits[0, -1][token_ids], dim=-1)
            return {label: float(value) for label, value in zip(labels, probabilities)}
        except AdvisorLoadError:
            raise
        except Exception as exc:
            raise AdvisorLoadError(REASON_LOAD_FAILED) from exc

    return predict


def _load_snapshot(snapshot: Path) -> LoadedAdvisor:
    """Import the optional stack and wrap the model. This is the only importer.

    The imports live inside the function body on purpose: importing
    :mod:`mario_jev.advisor` must not pay for ``torch``, and must not fail when
    ``torch`` is not installed.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    labels = _move_vocabulary(snapshot)
    processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True)
    model.eval()
    return LoadedAdvisor(backend=BACKEND, predict=_make_predict(torch, model, processor, labels))
