"""Settings, and the one property that really matters: the key never leaks."""

from __future__ import annotations

import json

import pytest

from mario_jev.config import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    JEV_API_KEY_ENV,
    Settings,
    redact_url_userinfo,
)

SECRET = "sk_jev_this_must_never_appear_anywhere_1234567890"

#: A credential smuggled into the endpoint's userinfo instead of the key field.
#: The endpoint is a URL, and a URL has a documented place to put a password; the
#: redaction has to cover that place too, or the key field's guarantee is a
#: guarantee about one field rather than about the file.
URL_PASSWORD = "url_pw_this_must_never_appear_anywhere_0987654321"
ENDPOINT_WITH_USERINFO = f"https://alice:{URL_PASSWORD}@example.invalid/v1/systemone"


def test_defaults_are_documented_values() -> None:
    settings = Settings.load(env={})

    assert settings.jev_endpoint == DEFAULT_ENDPOINT
    assert settings.jev_model == DEFAULT_MODEL
    assert settings.jev_api_key is None
    assert settings.decision_cadence_frames == 8
    assert settings.confidence_threshold == 0.6
    assert settings.rollout_min_frames == 6
    assert settings.rollout_max_frames == 48
    assert settings.rollout_cost_source == "measured"
    assert settings.rollout_fixed_base_ms == 0.0
    assert settings.rollout_fixed_per_frame_ms == 0.0
    assert settings.real_time is True
    assert settings.target_fps == 60.0
    assert settings.max_ticks is None
    assert settings.stage == "1-1"
    assert settings.advisor_enabled is False


def test_default_endpoint_is_the_verified_typesafe_host() -> None:
    # Verified live 2026-09-27 against https://api.typesafe.ai/openapi.json
    # (title "TypeSafe", version 0.2.0). The old host belongs to a different
    # product and answers this account's key with
    # 401 {"code":-1,"message":"Invalid API key."}. Pin the host literally so a
    # regression is caught here instead of at runtime.
    assert DEFAULT_ENDPOINT == "https://api.typesafe.ai/v1/systemone"
    assert Settings.load(env={}).jev_endpoint == "https://api.typesafe.ai/v1/systemone"
    assert "thejevai.com" not in DEFAULT_ENDPOINT
    assert "thejevai.com" not in Settings.load(env={}).jev_endpoint


def test_environment_overrides_every_setting_it_claims() -> None:
    settings = Settings.load(
        env={
            JEV_API_KEY_ENV: SECRET,
            "JEV_ENDPOINT": "https://example.invalid/v1/systemone",
            "JEV_MODEL": "jev-test",
            "MARIO_DECISION_CADENCE": "16",
            "MARIO_CONFIDENCE_THRESHOLD": "0.75",
            "MARIO_ROLLOUT_BUDGET_MS": "50",
            "MARIO_ROLLOUT_COST_SOURCE": "fixed",
            "MARIO_ROLLOUT_FIXED_BASE_MS": "0.5",
            "MARIO_ROLLOUT_FIXED_PER_FRAME_MS": "1.1",
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
    assert settings.rollout_cost_source == "fixed"
    assert settings.rollout_fixed_base_ms == 0.5
    assert settings.rollout_fixed_per_frame_ms == 1.1
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


def test_the_redacted_view_carries_the_budget_that_defines_slow() -> None:
    """The header has to say what "slow" meant for the run.

    A tick that spent 20 s is only reconstructible from the journal when the
    header says the timeout was 20 s and the attempt budget was 3: without those
    two numbers a reader cannot tell a slow call from a call that hit its
    deadline and was retried. ``rollout_min_frames`` rides along because the
    horizon bounds only mean anything as a pair.
    """
    redacted = Settings.load(
        env={JEV_API_KEY_ENV: SECRET, "JEV_TIMEOUT_S": "20", "JEV_MAX_ATTEMPTS": "3"}
    ).redacted

    assert redacted["jev_timeout_s"] == 20.0
    assert redacted["jev_max_attempts"] == 3
    assert redacted["rollout_min_frames"] == 6
    assert redacted["rollout_max_frames"] == 48
    # The whole view, serialised the way the journal serialises it, and scanned
    # for the sentinel: a new key must not become a new way out for the key.
    assert SECRET not in json.dumps(redacted)


def test_the_cost_source_defaults_to_measured_and_ignores_the_fixed_figures() -> None:
    settings = Settings.load(env={})

    assert settings.rollout_cost_source == "measured"
    # "Measured ignores them": the figures are never read, so nonsense in them must
    # not stop a measured run that does not use them. Asserted through ``load``
    # because validation lives there -- a directly constructed ``Settings`` is never
    # validated, so building one could not falsify this.
    ignored = Settings.load(
        env={
            "MARIO_ROLLOUT_COST_SOURCE": "measured",
            "MARIO_ROLLOUT_FIXED_BASE_MS": "-1",
            "MARIO_ROLLOUT_FIXED_PER_FRAME_MS": "0",
        }
    )

    assert ignored.rollout_cost_source == "measured"
    assert ignored.rollout_fixed_base_ms == -1.0
    assert ignored.rollout_fixed_per_frame_ms == 0.0


def test_the_fixed_cost_source_is_accepted_with_two_positive_finite_figures() -> None:
    settings = Settings.load(
        env={
            "MARIO_ROLLOUT_COST_SOURCE": "fixed",
            "MARIO_ROLLOUT_FIXED_BASE_MS": "0.5",
            "MARIO_ROLLOUT_FIXED_PER_FRAME_MS": "1.1",
        }
    )

    assert settings.rollout_cost_source == "fixed"
    assert settings.rollout_fixed_base_ms == 0.5
    assert settings.rollout_fixed_per_frame_ms == 1.1


def test_an_unknown_cost_source_is_rejected_with_the_field_name() -> None:
    with pytest.raises(ValueError, match="rollout_cost_source"):
        Settings.load(env={"MARIO_ROLLOUT_COST_SOURCE": "guessed"})


def test_a_fixed_source_without_the_figures_is_rejected() -> None:
    # The defaults are 0.0, so opting in to "fixed" without stating the figures is
    # rejected rather than silently sized from a figure nobody chose.
    with pytest.raises(ValueError, match="rollout_fixed_base_ms"):
        Settings.load(env={"MARIO_ROLLOUT_COST_SOURCE": "fixed"})


@pytest.mark.parametrize("value", ["0", "-1", "inf", "nan", ""])
def test_fixed_base_ms_must_be_positive_and_finite(value: str) -> None:
    with pytest.raises(ValueError, match="rollout_fixed_base_ms"):
        Settings.load(
            env={
                "MARIO_ROLLOUT_COST_SOURCE": "fixed",
                "MARIO_ROLLOUT_FIXED_BASE_MS": value,
                "MARIO_ROLLOUT_FIXED_PER_FRAME_MS": "1.1",
            }
        )


@pytest.mark.parametrize("value", ["0", "-1", "inf", "nan", ""])
def test_fixed_per_frame_ms_must_be_positive_and_finite(value: str) -> None:
    with pytest.raises(ValueError, match="rollout_fixed_per_frame_ms"):
        Settings.load(
            env={
                "MARIO_ROLLOUT_COST_SOURCE": "fixed",
                "MARIO_ROLLOUT_FIXED_BASE_MS": "0.5",
                "MARIO_ROLLOUT_FIXED_PER_FRAME_MS": value,
            }
        )


def test_the_redacted_view_carries_the_source_that_decided_the_horizon() -> None:
    redacted = Settings.load(
        env={
            "MARIO_ROLLOUT_COST_SOURCE": "fixed",
            "MARIO_ROLLOUT_FIXED_BASE_MS": "0.5",
            "MARIO_ROLLOUT_FIXED_PER_FRAME_MS": "1.1",
        }
    ).redacted

    assert redacted["rollout_cost_source"] == "fixed"
    assert redacted["rollout_fixed_base_ms"] == 0.5
    assert redacted["rollout_fixed_per_frame_ms"] == 1.1
    json.dumps(redacted)  # the run header serialises this view


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


# -- a credential in the endpoint URL ----------------------------------------------


def test_the_redacted_endpoint_carries_no_userinfo() -> None:
    # The leak this closes: ``redacted`` used to emit ``jev_endpoint`` verbatim, so
    # a password in the URL was written straight into the journal's run header.
    settings = Settings.load(env={"JEV_ENDPOINT": ENDPOINT_WITH_USERINFO})

    redacted = settings.redacted

    assert redacted["jev_endpoint"] == "https://example.invalid/v1/systemone"
    assert URL_PASSWORD not in json.dumps(redacted)
    assert "alice" not in json.dumps(redacted)
    # And the host is still there: this is a redaction, not a deletion. A reader
    # must still be able to tell which endpoint the run talked to.
    assert "example.invalid" in str(redacted["jev_endpoint"])


def test_settings_still_accepts_an_endpoint_with_userinfo() -> None:
    # Deliberately permissive: userinfo is a legitimate form for a URL to take, and
    # refusing it would break a working deployment to protect a log. The guarantee
    # is about what gets *written*, so the field itself keeps what it was given.
    settings = Settings.load(env={"JEV_ENDPOINT": ENDPOINT_WITH_USERINFO})

    assert settings.jev_endpoint == ENDPOINT_WITH_USERINFO


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        # Nothing to strip: returned untouched, including the default endpoint.
        (DEFAULT_ENDPOINT, DEFAULT_ENDPOINT),
        ("https://example.invalid/v1", "https://example.invalid/v1"),
        # A bare username carries no secret, but it goes too: a rule that depended
        # on a colon being present would hold only sometimes.
        ("https://alice@example.invalid/v1", "https://example.invalid/v1"),
        # The port, path, query and fragment all survive.
        ("https://a:b@example.invalid:8443/v1?q=1#f", "https://example.invalid:8443/v1?q=1#f"),
        # An IPv6 literal keeps its brackets.
        ("https://a:b@[::1]:9000/v1", "https://[::1]:9000/v1"),
        # Not a URL at all: printed as what it is, so a misconfiguration stays visible.
        ("not a url", "not a url"),
        ("", ""),
    ],
)
def test_redact_url_userinfo_cases(url: str, expected: str) -> None:
    assert redact_url_userinfo(url) == expected


def test_malformed_numbers_are_rejected_with_the_variable_name() -> None:
    with pytest.raises(ValueError, match="MARIO_TARGET_FPS"):
        Settings.load(env={"MARIO_TARGET_FPS": "fast"})

    with pytest.raises(ValueError, match="MARIO_REAL_TIME"):
        Settings.load(env={"MARIO_REAL_TIME": "maybe"})
