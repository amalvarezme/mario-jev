"""Per-tick pictures, drawn with Pillow and nothing else.

Two pure functions, no clock and no globals. Both create their parent
directories and both return the path they wrote.

* :func:`write_frame_png` draws the real 240x256 RGB frame with the perception
  overlay on top: Mario's bounding box, the floor line, and the pit columns.
* :func:`write_trajectory_png` draws the rollout table as a *chart*, one row per
  candidate, with labelled axes and units and the outcome written next to each
  candidate. It is deliberately not a photo, and it is deliberately drawn
  without ``matplotlib``: the project does not depend on it.

Neither function decides anything. They are given data and they draw it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mario_jev.perception import FramePerception

#: The player's detection box, drawn as an outline so the sprite stays visible.
MARIO_BOX_COLOUR = (255, 255, 0)
#: The row the level's floor sits at, drawn across the whole frame.
FLOOR_LINE_COLOUR = (0, 255, 255)
#: The pit columns, drawn as markers from the floor line to the bottom edge.
HOLE_COLOUR = (255, 0, 255)

#: Chart palette. High contrast on a near-white card.
CHART_BACKGROUND = (250, 250, 250)
CHART_TEXT = (25, 25, 25)
CHART_MUTED = (110, 110, 110)
CHART_AXIS = (150, 150, 150)
CHART_GRID = (225, 225, 225)
CHART_BAR = (70, 130, 180)
CHART_BAR_NEGATIVE = (205, 95, 95)
CHART_CHOSEN = (205, 120, 0)
CHART_CHOSEN_ROW = (255, 241, 176)

#: Chart geometry, in pixels at scale 1.
CHART_WIDTH = 860
LABEL_COLUMN = 320
PLOT_RIGHT_MARGIN = 30
PLOT_TOP = 64
ROW_HEIGHT = 26
PLOT_BOTTOM_MARGIN = 40


def _png_path(path: object) -> Path:
    """Normalize ``path`` and make sure its parent directory exists."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _as_rgb_frame(frame: object) -> np.ndarray:
    array = np.ascontiguousarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB frame, got {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"expected a uint8 frame, got {array.dtype}")
    return array


def _source_label(source: object) -> str:
    value = getattr(source, "value", None)
    return str(value) if value is not None else str(source)


def write_frame_png(
    path: object,
    frame: object,
    perception: FramePerception,
    *,
    scale: int = 2,
) -> Path:
    """Draw ``frame`` with its perception overlay and save it as a PNG.

    The overlay is drawn from ``perception`` alone: Mario's box
    (``perception.mario_box``), the floor line (``terrain.ground_line``) and the
    pit columns (``terrain.hole_columns``). Nothing is inferred from pixels here;
    the perception already did that, and this function only makes it visible.
    """
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")

    array = _as_rgb_frame(frame)
    target = _png_path(path)

    height, width = int(array.shape[0]), int(array.shape[1])
    image = Image.fromarray(array).convert("RGB")
    if scale != 1:
        image = image.resize((width * scale, height * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)

    terrain = perception.terrain
    ground_line = int(terrain.ground_line)
    floor_y = ground_line * scale
    stroke = max(1, scale)

    draw.line(
        [(0, floor_y), (image.width - 1, floor_y)],
        fill=FLOOR_LINE_COLOUR,
        width=stroke,
    )
    for column in terrain.hole_columns:
        left = int(column) * scale
        draw.line(
            [(left, floor_y), (left, image.height - 1)],
            fill=HOLE_COLOUR,
            width=stroke,
        )

    box = perception.mario_box
    if box is not None:
        x0, y0, x1, y1 = (int(value) for value in box)
        draw.rectangle(
            [
                x0 * scale,
                y0 * scale,
                (x1 + 1) * scale - 1,
                (y1 + 1) * scale - 1,
            ],
            outline=MARIO_BOX_COLOUR,
            width=stroke,
        )

    image.save(target, format="PNG")
    return target


def _bar_x(value: float, lo: float, hi: float, left: int, right: int) -> int:
    if hi <= lo:
        return left if value <= lo else right
    fraction = (value - lo) / (hi - lo)
    return int(round(left + fraction * (right - left)))


def _draw_empty_chart(target: Path, table: object) -> None:
    image = Image.new("RGB", (CHART_WIDTH, 200), CHART_BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text(
        (16, 16),
        f"Trajectory candidates - tick {int(table.tick)}",
        fill=CHART_TEXT,
    )
    draw.text(
        (16, 40),
        "no candidates were rolled out for this tick",
        fill=CHART_MUTED,
    )
    image.save(target, format="PNG")


def write_trajectory_png(
    path: object,
    table: object,
    *,
    decision: object | None = None,
) -> Path:
    """Draw the rollout table as a chart and save it as a PNG.

    One row per candidate, ordered as the table is. Each row carries the action
    name, its ``dx`` in in-level pixels, its risk in ``[0, 1]`` and its outcome
    label; the bar spans ``dx`` on a labelled axis. When ``decision`` is given,
    the executed action's row is highlighted and marked ``chosen``. An empty
    candidate list draws a ``no candidates`` card instead of failing.
    """
    target = _png_path(path)
    candidates = tuple(getattr(table, "candidates", ()))
    if not candidates:
        _draw_empty_chart(target, table)
        return target

    rows = len(candidates)
    height = PLOT_TOP + ROW_HEIGHT * rows + PLOT_BOTTOM_MARGIN
    image = Image.new("RGB", (CHART_WIDTH, height), CHART_BACKGROUND)
    draw = ImageDraw.Draw(image)

    dxs = [int(candidate.dx) for candidate in candidates]
    lo = float(min(0, min(dxs)))
    hi = float(max(0, max(dxs)))
    if hi <= lo:
        hi = lo + 1.0
    left = LABEL_COLUMN
    right = CHART_WIDTH - PLOT_RIGHT_MARGIN

    tick_value = int(getattr(table, "tick", 0))
    horizon = int(getattr(table, "horizon", 0))
    draw.text(
        (16, 14),
        f"Trajectory candidates - tick {tick_value}, horizon {horizon} frames",
        fill=CHART_TEXT,
    )
    draw.text(
        (16, 34),
        "dx = forward progress in in-level pixels; risk 0 (safe) .. 1 (lethal)",
        fill=CHART_MUTED,
    )

    chosen = None if decision is None else str(decision.action)

    # Axis gridlines and labels under the plot.
    axis_y = PLOT_TOP + ROW_HEIGHT * rows
    draw.line([(left, axis_y), (right, axis_y)], fill=CHART_AXIS)
    for value in (lo, 0.0, hi):
        x = _bar_x(value, lo, hi, left, right)
        draw.line([(x, PLOT_TOP), (x, axis_y)], fill=CHART_GRID)
        draw.text((x - 10, axis_y + 6), f"{value:+.0f}", fill=CHART_MUTED)
    draw.text((left, axis_y + 22), "dx (in-level pixels)", fill=CHART_MUTED)

    zero_x = _bar_x(0.0, lo, hi, left, right)

    for index, candidate in enumerate(candidates):
        name = str(candidate.name)
        dx = int(candidate.dx)
        risk = float(candidate.risk)
        outcome = str(getattr(candidate.outcome, "value", candidate.outcome))
        top = PLOT_TOP + index * ROW_HEIGHT
        middle = top + ROW_HEIGHT // 2

        if name == chosen:
            draw.rectangle(
                [(0, top), (CHART_WIDTH - 1, top + ROW_HEIGHT - 1)],
                fill=CHART_CHOSEN_ROW,
            )

        label = f"{name}  dx={dx:+d}  risk={risk:.2f}  {outcome}"
        draw.text((12, middle - 5), label, fill=CHART_TEXT if name != chosen else CHART_CHOSEN)

        value_x = _bar_x(float(dx), lo, hi, left, right)
        bar_colour = CHART_CHOSEN if name == chosen else (
            CHART_BAR_NEGATIVE if dx < 0 else CHART_BAR
        )
        draw.line([(zero_x, middle), (value_x, middle)], fill=bar_colour, width=9)
        if name == chosen:
            draw.text((right - 60, middle - 5), "chosen", fill=CHART_CHOSEN)

    if decision is not None:
        draw.text(
            (16, height - 22),
            f"executed: {chosen} ({_source_label(decision.source)})",
            fill=CHART_TEXT,
        )

    image.save(target, format="PNG")
    return target


__all__ = ["write_frame_png", "write_trajectory_png"]
