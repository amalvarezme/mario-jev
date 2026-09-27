"""Perception: terrain and the player, plus the RAM cross-check.

The synthetic frames here matter as much as the live ones. They are the only way
to assert behaviour the game cannot be asked to produce on demand — a pit, a
missing player, red pixels hidden in the HUD.
"""

from __future__ import annotations

import numpy as np
import pytest

from mario_jev.emulator import MarioEmulator
from mario_jev.perception import (
    BLACK,
    EXPECTED_MARIO_RED_PIXELS,
    GROUND,
    GROUND_HIGHLIGHT,
    HUD_ROWS,
    KNOWN_COLOURS,
    MARIO_RED,
    MARIO_RED_PIXELS_RANGE,
    MIN_MARIO_RED_PIXELS,
    SKY,
    SOLID_COLOURS,
    audit,
    detect,
    dominant_colour,
    mask_for,
    ram_player_box,
)

HEIGHT, WIDTH = 240, 256
FLOOR_TOP = 208


# -- helpers ---------------------------------------------------------------------


def blank_frame(colour: tuple[int, int, int] = SKY) -> np.ndarray:
    return np.tile(np.array(colour, dtype=np.uint8), (HEIGHT, WIDTH, 1))


def frame_with_floor(
    gap_columns: range | None = None, floor_top: int = FLOOR_TOP
) -> np.ndarray:
    """A frame whose bottom is solid floor, optionally with a pit."""
    frame = blank_frame(SKY)
    frame[floor_top:, :] = np.array(GROUND, dtype=np.uint8)
    frame[floor_top, :] = np.array(GROUND_HIGHLIGHT, dtype=np.uint8)
    if gap_columns is not None:
        frame[floor_top:, gap_columns] = np.array(SKY, dtype=np.uint8)
    return frame


def paint_red(frame: np.ndarray, x0: int, y0: int, size: int = 9) -> np.ndarray:
    frame[y0 : y0 + size, x0 : x0 + size] = np.array(MARIO_RED, dtype=np.uint8)
    return frame


# -- the palette is the measured palette -----------------------------------------


def test_the_sky_is_the_measured_dominant_colour_of_a_real_frame(emulator: MarioEmulator) -> None:
    frame = emulator.frame()

    assert dominant_colour(frame) == SKY


def test_the_measured_palette_entries_all_really_appear_in_a_real_frame(
    emulator: MarioEmulator,
) -> None:
    # If SMB1's palette ever changed, or a colour constant was mistyped, this is
    # where it surfaces: every documented gameplay colour must be present.
    for _ in range(120):
        emulator.step(1)
    frame = emulator.frame()

    present = {
        name
        for name, colour in KNOWN_COLOURS.items()
        if mask_for(frame, colour).any()
    }
    # White (clouds) and the blacks are positional; the ones the detectors depend
    # on must be there.
    for required in ("sky", "ground", "ground_highlight", "mario_red", "pipe_green"):
        assert required in present, f"{required} never appeared in a real frame"


def test_marios_red_in_the_playfield_stays_inside_the_measured_range(
    emulator: MarioEmulator,
) -> None:
    low, high = MARIO_RED_PIXELS_RANGE
    sampled = 0
    for index in range(240):
        result = emulator.step(1 if index < 90 else 0)
        if result.done:
            break
        if index % 20:
            continue
        red = mask_for(emulator.frame(), MARIO_RED)
        gameplay = red[HUD_ROWS:, :].sum()
        sampled += 1
        assert low <= gameplay <= high, f"frame {index}: {gameplay} red pixels"

    assert sampled >= 5


def test_the_hud_can_also_carry_red_which_is_why_it_is_excluded(
    emulator: MarioEmulator,
) -> None:
    # The HUD's coin icon is drawn from the same red as Mario on some frames, so a
    # "find the red" detector that skipped the HUD exclusion would report a second
    # Mario at the top of the screen. Asserted on a synthetic frame because the
    # icon's visibility varies frame to frame in the game itself.
    frame = frame_with_floor()
    frame[20:28, 88:96] = np.array(MARIO_RED, dtype=np.uint8)

    assert mask_for(frame, MARIO_RED)[:HUD_ROWS, :].sum() > 0
    assert not detect(frame).mario_detected


# -- terrain ---------------------------------------------------------------------


def test_the_floor_line_of_world_1_1_is_measured_at_row_208(emulator: MarioEmulator) -> None:
    perception = detect(emulator.frame())

    assert perception.terrain.ground_line == FLOOR_TOP
    assert perception.terrain.hole_columns == ()


def test_black_seams_do_not_break_the_floor_scan(emulator: MarioEmulator) -> None:
    # Regression: without counting the black seam as solid, the scan stops on the
    # first seam and reports the whole floor as a single row at 239.
    for _ in range(120):
        emulator.step(1)
    perception = detect(emulator.frame())

    assert perception.terrain.ground_line == FLOOR_TOP
    assert perception.terrain.ground_line != HEIGHT - 1


def test_a_synthetic_pit_is_detected() -> None:
    frame = frame_with_floor(gap_columns=range(120, 160))

    terrain = detect(frame).terrain

    assert terrain.ground_line == FLOOR_TOP
    assert terrain.hole_columns == tuple(range(120, 160))
    assert terrain.is_hole(120)
    assert not terrain.is_hole(119)


def test_pit_geometry_helpers() -> None:
    terrain = detect(frame_with_floor(gap_columns=range(120, 160))).terrain

    assert terrain.hole_distance(100, lookahead=64) == 20
    assert terrain.hole_distance(100, lookahead=10) is None
    assert terrain.hole_width_ahead(120, lookahead=64) == 40
    assert terrain.hole_width_ahead(60, lookahead=40) == 0


def test_surface_offset_reports_raised_ground() -> None:
    frame = frame_with_floor()
    # A raised platform 16 px higher, from column 100 to 120.
    frame[FLOOR_TOP - 16 : FLOOR_TOP, 100:121] = np.array(GROUND, dtype=np.uint8)

    terrain = detect(frame).terrain

    assert terrain.surface_offset(100, lookahead=16) == 16
    assert terrain.surface_offset(200, lookahead=16) == 0


def test_detect_rejects_a_malformed_frame() -> None:
    with pytest.raises(ValueError, match="expected an"):
        detect(np.zeros((240, 256), dtype=np.uint8))

    with pytest.raises(ValueError, match="expected an"):
        detect(np.zeros((240, 256, 4), dtype=np.uint8))


# -- the player ------------------------------------------------------------------


def test_mario_is_detected_in_a_real_frame(emulator: MarioEmulator) -> None:
    for _ in range(120):
        emulator.step(1)
    perception = detect(emulator.frame())

    low, high = MARIO_RED_PIXELS_RANGE
    assert perception.mario_detected
    assert perception.mario_box is not None
    # The walk animation changes how much of Mario's red is visible, so the count
    # is bounded by the measured range rather than compared to one pose.
    assert low <= perception.mario_red_pixels <= high
    assert perception.mario_confidence == pytest.approx(1.0)


def test_mario_is_not_invented_from_a_synthetic_frame_without_him() -> None:
    perception = detect(frame_with_floor())

    assert not perception.mario_detected
    assert perception.mario_box is None
    assert perception.mario_confidence == 0.0


def test_red_hidden_in_the_hud_is_not_mario() -> None:
    frame = frame_with_floor()
    # The HUD's coin icon lives here. Without the HUD exclusion this becomes a
    # second "Mario" in the top rows.
    frame[20:28, 88:96] = np.array(MARIO_RED, dtype=np.uint8)

    perception = detect(frame)

    assert not perception.mario_detected

    # ... and the same red in the playfield *is* Mario.
    assert detect(paint_red(frame_with_floor(), 100, 180)).mario_detected


def test_too_few_red_pixels_is_not_mario() -> None:
    frame = frame_with_floor()
    # Six pixels, below the threshold of eight.
    frame[180:183, 100:102] = np.array(MARIO_RED, dtype=np.uint8)

    perception = detect(frame)

    assert perception.mario_red_pixels == 6
    assert perception.mario_red_pixels < MIN_MARIO_RED_PIXELS
    assert not perception.mario_detected


def test_confidence_is_the_measured_median_ratio_and_is_capped() -> None:
    # No Mario at all scores zero.
    assert detect(frame_with_floor()).mario_confidence == 0.0

    # Exactly the median count scores 1.0.
    median = frame_with_floor()
    median[180:189, 100:105] = np.array(MARIO_RED, dtype=np.uint8)  # 45 px
    assert detect(median).mario_confidence == pytest.approx(1.0)

    # More red than the median (a walk pose carries up to 55) must still cap at 1.0.
    more = frame_with_floor()
    more[180:189, 100:109] = np.array(MARIO_RED, dtype=np.uint8)  # 81 px
    assert detect(more).mario_confidence == pytest.approx(1.0)

    assert EXPECTED_MARIO_RED_PIXELS == 45
    assert MARIO_RED_PIXELS_RANGE == (37, 55)


def test_ram_player_box_uses_the_measured_offset() -> None:
    # Measured: 0x03AD = 112 and 0x03B8 = 176 put the sprite at rows 192..207.
    assert ram_player_box(112, 176) == (112, 192, 127, 207)


# -- the audit -------------------------------------------------------------------


def test_the_frame_and_ram_agree_about_mario_in_a_real_episode(
    emulator: MarioEmulator,
) -> None:
    agree = 0
    ticks = 0
    for index in range(200):
        result = emulator.step(1 if index < 80 else 0)
        if result.done:
            break
        if index % 5:
            continue
        perception = detect(emulator.frame(), tick=index)
        verdict = audit(perception, emulator.ram())
        ticks += 1
        if verdict.mario_inside_ram_box:
            agree += 1

    assert ticks >= 20, "sampled too few ticks for this to mean anything"
    # Measured agreement in this episode is 100%; the bound leaves room for an
    # occluded frame without letting a real regression through.
    assert agree / ticks >= 0.9, f"vision/RAM agreement collapsed: {agree}/{ticks}"


def test_the_players_feet_rest_on_the_floor_line(emulator: MarioEmulator) -> None:
    # Independent of the red mask: RAM and the frame's floor line must agree about
    # where the ground is, and that is the geometric relation the state layer uses.
    for _ in range(150):
        result = emulator.step(0)
        if result.done:
            break
    perception = detect(emulator.frame())
    verdict = audit(perception, emulator.ram())

    assert verdict.mario_box_from_ram is not None
    assert verdict.mario_box_from_ram[3] == perception.terrain.ground_line - 1
    assert verdict.feet_on_ground is True


def test_audit_reports_no_verdict_when_mario_was_not_seen() -> None:
    class FakeRam:
        @staticmethod
        def byte(address: int) -> int:
            return 112 if address == 0x03AD else 176

        @staticmethod
        def active_enemies() -> tuple[object, ...]:
            return ()

    verdict = audit(detect(frame_with_floor()), FakeRam())

    assert verdict.mario_detected_from_frame is False
    assert verdict.mario_inside_ram_box is None
    assert verdict.offset is None
    assert verdict.mario_box_from_ram == (112, 192, 127, 207)


def test_audit_reports_the_offset_when_the_frame_and_ram_disagree() -> None:
    class FakeRam:
        @staticmethod
        def byte(address: int) -> int:
            return 10 if address == 0x03AD else 176

        @staticmethod
        def active_enemies() -> tuple[object, ...]:
            return ()

    perception = detect(paint_red(frame_with_floor(), 100, 190))

    verdict = audit(perception, FakeRam())

    assert verdict.mario_detected_from_frame is True
    assert verdict.mario_inside_ram_box is False
    # The painted box starts at (100, 190); the RAM box at (10, 192).
    assert verdict.offset == (90, -2)


def test_audit_counts_active_enemies_from_ram(emulator: MarioEmulator) -> None:
    for _ in range(120):
        emulator.step(1)

    verdict = audit(detect(emulator.frame()), emulator.ram())

    assert verdict.ram_active_enemies == len(emulator.ram().active_enemies())


def test_black_is_documented_as_a_seam_colour() -> None:
    # The floor profile depends on this, so it is asserted rather than left to a
    # reader to notice that BLACK is in the solid set.
    assert BLACK in SOLID_COLOURS
    assert GROUND in SOLID_COLOURS and GROUND_HIGHLIGHT in SOLID_COLOURS
