"""Typed runtime settings.

The Jev credential is the one value in this project that must never reach a log,
a traceback, a journal entry, or a commit. It is therefore held in a field with
``repr=False`` so that ``repr(settings)``, f-string interpolation, and dataclass
equality debugging cannot print it, and it is exposed to the rest of the program
only through :meth:`Settings.jev_available`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

DEFAULT_ENDPOINT = "https://thejevai.com/v1/systemone"
DEFAULT_MODEL = "jev-latest"
DEFAULT_PLAYJEV_MODEL = "OmniJev/PlayJev-0.8B"

#: Environment variable holding the Jev API key. Never logged.
JEV_API_KEY_ENV = "JEV_API_KEY"


def _int_env(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _float_env(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _bool_env(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    normalised = raw.strip().lower()
    if normalised in {"1", "true", "yes", "on"}:
        return True
    if normalised in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


@dataclass(frozen=True, slots=True)
class Settings:
    """Resolved settings for one agent run.

    Attributes are grouped by the phase that reads them.
    """

    # --- Jev decision layer -------------------------------------------------
    # repr=False and compare=False: the key must not appear in diagnostics.
    jev_api_key: str | None = field(default=None, repr=False, compare=False)
    jev_endpoint: str = DEFAULT_ENDPOINT
    jev_model: str = DEFAULT_MODEL
    jev_timeout_s: float = 20.0
    jev_max_attempts: int = 3

    # --- decision cadence ---------------------------------------------------
    # How many emulator frames pass between two Jev consultations. One request
    # per frame is impossible over HTTP, so the loop decouples them.
    decision_cadence_frames: int = 8
    # Below this confidence the Jev answer is not trusted and the deterministic
    # safe fallback runs instead.
    confidence_threshold: float = 0.6

    # --- trajectory rollout -------------------------------------------------
    # Wall-clock budget for the whole candidate rollout sweep. The horizon is
    # shortened to fit this budget instead of being fixed, because a fixed
    # horizon cannot hold at every machine speed.
    rollout_budget_ms: float = 120.0
    rollout_max_frames: int = 48
    rollout_min_frames: int = 6

    # --- real-time behaviour ------------------------------------------------
    real_time: bool = True
    target_fps: float = 60.0
    # Hard cap on decision ticks; None means "until the episode ends".
    max_ticks: int | None = None
    stage: str = "1-1"
    seed: int = 0

    # --- optional advisor ---------------------------------------------------
    advisor_enabled: bool = False
    advisor_model_id: str = DEFAULT_PLAYJEV_MODEL

    # --- evidence -----------------------------------------------------------
    artifacts_dir: Path = Path("artifacts")

    @classmethod
    def load(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """Build settings from the environment, with documented defaults."""
        source: Mapping[str, str] = os.environ if env is None else env

        raw_key = source.get(JEV_API_KEY_ENV) or None
        if raw_key is not None:
            raw_key = raw_key.strip() or None

        raw_max_ticks = source.get("MARIO_MAX_TICKS") or None
        max_ticks = _int_env(source, "MARIO_MAX_TICKS", 0) if raw_max_ticks else None
        if max_ticks is not None and max_ticks <= 0:
            # A non-positive cap means "no cap", not "finish immediately".
            max_ticks = None

        settings = cls(
            jev_api_key=raw_key,
            jev_endpoint=source.get("JEV_ENDPOINT") or DEFAULT_ENDPOINT,
            jev_model=source.get("JEV_MODEL") or DEFAULT_MODEL,
            jev_timeout_s=_float_env(source, "JEV_TIMEOUT_S", 20.0),
            jev_max_attempts=_int_env(source, "JEV_MAX_ATTEMPTS", 3),
            decision_cadence_frames=_int_env(source, "MARIO_DECISION_CADENCE", 8),
            confidence_threshold=_float_env(source, "MARIO_CONFIDENCE_THRESHOLD", 0.6),
            rollout_budget_ms=_float_env(source, "MARIO_ROLLOUT_BUDGET_MS", 120.0),
            rollout_max_frames=_int_env(source, "MARIO_ROLLOUT_MAX_FRAMES", 48),
            rollout_min_frames=_int_env(source, "MARIO_ROLLOUT_MIN_FRAMES", 6),
            real_time=_bool_env(source, "MARIO_REAL_TIME", True),
            target_fps=_float_env(source, "MARIO_TARGET_FPS", 60.0),
            max_ticks=max_ticks,
            stage=source.get("MARIO_STAGE") or "1-1",
            seed=_int_env(source, "MARIO_SEED", 0),
            advisor_enabled=_bool_env(source, "MARIO_ADVISOR", False),
            advisor_model_id=source.get("MARIO_ADVISOR_MODEL") or DEFAULT_PLAYJEV_MODEL,
            artifacts_dir=Path(source.get("MARIO_ARTIFACTS_DIR") or "artifacts"),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Reject values that would make the loop silently misbehave."""
        if self.decision_cadence_frames < 1:
            raise ValueError("decision_cadence_frames must be >= 1")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be within [0, 1]")
        if self.target_fps <= 0:
            raise ValueError("target_fps must be > 0")
        if self.rollout_max_frames < self.rollout_min_frames:
            raise ValueError("rollout_max_frames must be >= rollout_min_frames")
        if self.rollout_min_frames < 1:
            raise ValueError("rollout_min_frames must be >= 1")
        if self.rollout_budget_ms <= 0:
            raise ValueError("rollout_budget_ms must be > 0")
        if self.jev_max_attempts < 1:
            raise ValueError("jev_max_attempts must be >= 1")

    @property
    def jev_available(self) -> bool:
        """Whether a Jev credential is configured. Never returns the credential."""
        return bool(self.jev_api_key)

    @property
    def redacted(self) -> dict[str, object]:
        """A diagnostics-safe view of the settings, with no credential material."""
        return {
            "jev_endpoint": self.jev_endpoint,
            "jev_model": self.jev_model,
            "jev_available": self.jev_available,
            "decision_cadence_frames": self.decision_cadence_frames,
            "confidence_threshold": self.confidence_threshold,
            "rollout_budget_ms": self.rollout_budget_ms,
            "rollout_max_frames": self.rollout_max_frames,
            "real_time": self.real_time,
            "target_fps": self.target_fps,
            "max_ticks": self.max_ticks,
            "stage": self.stage,
            "seed": self.seed,
            "advisor_enabled": self.advisor_enabled,
            "advisor_model_id": self.advisor_model_id,
            "artifacts_dir": str(self.artifacts_dir),
        }
