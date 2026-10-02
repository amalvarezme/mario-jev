"""The one-slot frame sink: written by the run's thread, read by the viewer's.

The sink exists because the two threads that matter never meet: the run owns the
emulator and hands over a frame it is about to move past, and the viewer answers
HTTP requests whenever a browser asks. What is pinned here is what makes that
hand-off safe and cheap -- the sink keeps its own copy, so the emulator mutating
its screen afterwards cannot rewrite a frame already published; one published
frame is encoded once, so a burst of requests does not pay for the same PNG
twice; and a frame that is not an RGB frame is refused where it is published
rather than encoded into nonsense.

Nothing here measures time. The two-thread test is bounded by a frame count, not
by a deadline, because a sleep-based race test would measure the machine.
"""

from __future__ import annotations

import threading
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from mario_jev import live_frame as live_frame_module
from mario_jev.live_frame import LiveFrame

#: The NES frame shape this sink is for, spelled out rather than imported: the
#: sink is deliberately not an emulator module and does not depend on one.
SHAPE = (240, 256, 3)


def _frame(seed: int) -> np.ndarray:
    """A deterministic frame whose pixels differ from every other seed's."""
    return np.random.default_rng(seed).integers(0, 256, size=SHAPE, dtype=np.uint8)


def _decode(data: bytes) -> np.ndarray:
    """The pixels a PNG really carries, read back through Pillow."""
    with Image.open(BytesIO(data)) as image:
        return np.array(image.convert("RGB"))


def test_nothing_is_published_before_the_first_frame() -> None:
    sink = LiveFrame()

    # ``None`` is the honest answer: the viewer says "no frame yet" rather than
    # being handed an empty image that would look like a black game.
    assert sink.png() is None


def test_a_published_frame_comes_back_as_a_png_of_the_same_pixels() -> None:
    sink = LiveFrame()
    frame = _frame(1)

    sink.publish(frame)

    printed = sink.png()
    assert isinstance(printed, bytes)
    assert printed.startswith(b"\x89PNG\r\n\x1a\n")
    assert np.array_equal(_decode(printed), frame)


def test_a_burst_of_requests_for_one_frame_reuses_the_same_bytes() -> None:
    sink = LiveFrame()
    sink.publish(_frame(1))

    first = sink.png()
    second = sink.png()

    # The very same object, not merely equal bytes: that is the encode-once rule
    # observed without counting calls. A sink that re-encoded everywhere would
    # pass an equality assertion and still make the page pay per request.
    assert first is second


def test_a_later_publish_replaces_the_frame_and_encodes_again() -> None:
    sink = LiveFrame()
    sink.publish(_frame(1))
    before = sink.png()

    sink.publish(_frame(2))

    after = sink.png()
    assert after is not before
    assert np.array_equal(_decode(after), _frame(2))
    assert not np.array_equal(_decode(before), _frame(2))


def test_the_sink_keeps_its_own_copy_of_the_published_buffer() -> None:
    """The step's buffer is not the sink's, and the emulator goes on mutating its own.

    The run hands over the same fresh array its ``StepResult`` carries, and the
    emulator's own screen buffer keeps changing after that. A sink holding the
    reference would show the run's *latest* frame in every read, silently.
    """
    sink = LiveFrame()
    frame = _frame(1)
    expected = frame.copy()

    sink.publish(frame)
    frame[:] = 0  # the emulator mutating the buffer it handed over

    assert np.array_equal(_decode(sink.png()), expected)


def test_a_non_contiguous_frame_is_stored_as_a_contiguous_one() -> None:
    sink = LiveFrame()
    # A reversed view: the same pixels in a different memory order, which is the
    # shape ``ascontiguousarray`` exists for on the emulator's own screen buffer.
    view = _frame(1)[:, ::-1]

    sink.publish(view)

    assert np.array_equal(_decode(sink.png()), np.ascontiguousarray(view))


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((240, 256), dtype=np.uint8),  # no channel axis at all
        np.zeros((240, 256, 4), dtype=np.uint8),  # RGBA, not RGB
        np.zeros((240, 256, 3), dtype=np.float32),  # not the uint8 of a frame
    ],
)
def test_something_that_is_not_an_rgb_frame_is_refused(frame: np.ndarray) -> None:
    sink = LiveFrame()

    with pytest.raises(ValueError):
        sink.publish(frame)


def test_a_refused_frame_leaves_the_slot_and_its_encoding_alone() -> None:
    sink = LiveFrame()
    sink.publish(_frame(1))
    published = sink.png()

    with pytest.raises(ValueError):
        sink.publish(np.zeros((240, 256, 4), dtype=np.uint8))

    # The sequence did not move either, which shows here as the cached bytes still
    # being reused instead of a re-encode of the frame still in the slot.
    assert sink.png() is published


def test_an_encoding_overtaken_by_a_publish_is_not_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The frame that was newest when the request arrived is the frame answered.

    A publish landing **while** a request encodes is the interesting race: the
    bytes describe the frame the request asked for, and caching them against the
    newer sequence number would make every later request for that newer frame
    answer with the older picture -- permanently frozen, with everything looking
    healthy. The extra publish inside the encoding is how the race is made
    deterministic here instead of hoped for.
    """
    sink = LiveFrame()
    sink.publish(_frame(1))
    real_encode = live_frame_module._encode_png

    def encode_while_a_newer_frame_arrives(frame: np.ndarray) -> bytes:
        sink.publish(_frame(2))
        return real_encode(frame)

    monkeypatch.setattr(live_frame_module, "_encode_png", encode_while_a_newer_frame_arrives)

    overtaken = sink.png()

    assert np.array_equal(_decode(overtaken), _frame(1))
    # The newer frame is not answered from the stale encoding.
    assert np.array_equal(_decode(sink.png()), _frame(2))


def test_publishing_while_another_thread_reads_stays_readable() -> None:
    """One writer, one reader, a bounded number of frames, no exception anywhere.

    The two-thread shape is the whole point of the sink, so it is exercised rather
    than argued: the writer publishes, the reader reads and decodes whatever it
    gets, and both failures are collected to be asserted here instead of being
    printed to a thread's stderr where a test cannot see them.
    """
    sink = LiveFrame()
    frames = 100
    writer_done = threading.Event()
    failures: list[BaseException] = []

    def write() -> None:
        try:
            for index in range(frames):
                sink.publish(_frame(index))
        except BaseException as exc:  # reported below, never swallowed
            failures.append(exc)
        finally:
            writer_done.set()

    def read() -> None:
        try:
            while not writer_done.is_set():
                data = sink.png()
                if data is None:
                    continue
                if _decode(data).shape != SHAPE:
                    raise AssertionError(f"decoded a {_decode(data).shape} PNG")
        except BaseException as exc:  # reported below, never swallowed
            failures.append(exc)

    writer = threading.Thread(target=write, name="publisher", daemon=True)
    reader = threading.Thread(target=read, name="reader", daemon=True)
    writer.start()
    reader.start()
    writer.join(timeout=30.0)
    reader.join(timeout=30.0)

    assert not writer.is_alive()
    assert not reader.is_alive()
    assert failures == []
    # And after the burst, the newest frame is readable and is the newest frame.
    assert np.array_equal(_decode(sink.png()), _frame(frames - 1))
