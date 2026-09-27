"""Settings, and the one property that really matters: the key never leaks."""

from __future__ import annotations

import pytest

from mario_jev.config import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    JEV_API_KEY_ENV,
    Settings,
)

SECRET = "sk_jev_this_must_never_appear_anywhere_1234567890"


def test_defaults_are_documented_values() -> None:
    settings = Settings.load(env={})

    assert settings.jev_endpoint == DEFAULT_ENDPOINT
    assert settings.jev_model == DEFAULT_MODEL
    assert settings.jev_api_key is None
    assert settings.decision_cadence_frames == 8
    assert settings.confidence_threshold == 0.6
    assert settings.rollout_min_frames == 6
    assert settings.rollout_max_frames == 48
    assert settings.real_time is True
    assert settings.target_fps == 60.0
    assert settings.max_ticks is None
    assert settings.stage == "1-1"
    assert settings.advisor_enabled is False


def test_environment_overrides_every_setting_it_claims() -> None:
    settings = Settings.load(
        env={
            JEV_API_KEY_ENV: SECRET,
            "JEV_ENDPOINT": "https://example.invalid/v1/systemone",
            "JEV_MODEL": "jev-test",
            "MARIO_DECISION_CADENCE": "16",
            "MARIO_CONFIDENCE_THRESHOLD": "0.75",
            "MARIO_ROLLOUT_BUDGET_MS": "50",
            "MARIO_REAL_TIME": "false",
            "MARIO_TARGET_FPS": "30",
            "MARIO_MAX_TICKS": "250",
            "MARIO_STAGE": "2-1",
            "MARIO_SEED": "7",
            "MARIO_ADVISOR": "true",
            "MARIO_ARTIFACTS_DIR": "out",
        }
    )

    assert settings.jev_endpoint == "https://example.invalid/v1/systemone"
    assert settings.jev_model == "jev-test"
    assert settings.decision_cadence_frames == 16
    assert settings.confidence_threshold == 0.75
    assert settings.rollout_budget_ms == 50
    assert settings.real_time is False
    assert settings.target_fps == 30.0
    assert settings.max_ticks == 250
    assert settings.stage == "2-1"
    assert settings.seed == 7
    assert settings.advisor_enabled is True
    assert str(settings.artifacts_dir) == "out"


def test_the_api_key_never_appears_in_repr() -> None:
    settings = Settings.load(env={JEV_API_KEY_ENV: SECRET})

    assert settings.jev_available is True
    assert SECRET not in repr(settings)
    assert SECRET not in str(settings)
    assert SECRET not in f"{settings}"
    assert SECRET not in repr(settings.redacted)


def test_the_api_key_does_not_participate_in_equality() -> None:
    # Two settings that differ only by credential must compare equal, so that a
    # diff of two settings objects can never print the secret.
    first = Settings.load(env={JEV_API_KEY_ENV: SECRET})
    second = Settings.load(env={JEV_API_KEY_ENV: "sk_jev_a_different_key_value"})

    assert first == second


def test_redacted_view_reports_availability_not_material() -> None:
    redacted = Settings.load(env={JEV_API_KEY_ENV: SECRET}).redacted

    assert redacted["jev_available"] is True
    assert not any(SECRET in str(value) for value in redacted.values())


def test_missing_or_blank_key_is_reported_unavailable() -> None:
    assert Settings.load(env={}).jev_available is False
    assert Settings.load(env={JEV_API_KEY_ENV: ""}).jev_available is False
    assert Settings.load(env={JEV_API_KEY_ENV: "   "}).jev_available is False


def test_a_non_positive_tick_cap_means_no_cap() -> None:
    # A cap of 0 would otherwise mean "finish before starting".
    assert Settings.load(env={"MARIO_MAX_TICKS": "0"}).max_ticks is None
    assert Settings.load(env={"MARIO_MAX_TICKS": "-5"}).max_ticks is None


@pytest.mark.parametrize(
    ("env", "field"),
    [
        ({"MARIO_DECISION_CADENCE": "0"}, "decision_cadence_frames"),
        ({"MARIO_CONFIDENCE_THRESHOLD": "1.5"}, "confidence_threshold"),
        ({"MARIO_TARGET_FPS": "0"}, "target_fps"),
        ({"MARIO_ROLLOUT_MIN_FRAMES": "100"}, "rollout_min_frames"),
        ({"MARIO_ROLLOUT_BUDGET_MS": "0"}, "rollout_budget_ms"),
        ({"JEV_MAX_ATTEMPTS": "0"}, "jev_max_attempts"),
    ],
)
def test_invalid_settings_are_rejected(env: dict[str, str], field: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        Settings.load(env=env)

    assert field in str(excinfo.value) or "must be" in str(excinfo.value)


def test_malformed_numbers_are_rejected_with_the_variable_name() -> None:
    with pytest.raises(ValueError, match="MARIO_TARGET_FPS"):
        Settings.load(env={"MARIO_TARGET_FPS": "fast"})

    with pytest.raises(ValueError, match="MARIO_REAL_TIME"):
        Settings.load(env={"MARIO_REAL_TIME": "maybe"})
