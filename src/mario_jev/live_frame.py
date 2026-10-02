"""The newest frame a run produced, in one slot, for the page that watches it.

The run and the viewer that serves the board page are two threads that never
meet: the run steps the emulator on the main thread, and the viewer answers HTTP
requests from a daemon thread whenever a browser polls. :class:`LiveFrame` is the
whole of what they share -- a slot holding the newest frame, a sequence number
saying how many times that slot has been written, and the PNG encoding of the
newest frame, cached so one published frame is encoded once.

Three decisions shape it, and each one is here because the obvious alternative
costs something real:

* **The copy on publish.** ``publish`` is handed the same fresh array the run's
  ``StepResult`` carries. It copies it anyway. The sink outlives the step that
  published it, and the emulator goes on mutating the buffer it handed over --
  ``screen_view()`` is mutated in place by every ``step()`` -- so a slot holding
  the reference would show the run's *latest* frame in every read, silently, with
  no exception anywhere. The step is already copying that array once; this is the
  second copy, and it is the one that makes the slot safe to keep.

* **The encoding outside the lock.** ``png()`` reads the slot under the lock and
  encodes after releasing it. Encoding takes milliseconds; holding the lock
  through it would block ``publish`` on the run's thread for that long, which is
  exactly the delay a 60 fps loop cannot afford. The lock is held only long
  enough to read the slot, and the frame read is safe to use outside the lock
  because a publish never mutates a frame it has already stored -- it replaces
  the slot with a fresh copy. The honest cost is that two requests arriving for
  the same never-encoded frame at the same instant may each encode it; making one
  of them wait would need the lock held across the encoding, which is the thing
  worth avoiding. A burst of sequential requests, which is what a polling page
  produces, encodes once.

* **The refusal.** A frame that is not a ``(H, W, 3)`` ``uint8`` array is refused
  where it is published, loudly, rather than being encoded into nonsense that the
  browser would show as a broken image with no clue where it came from. The sink
  checks that the array *is* an RGB frame, not that it is exactly 240x256: that
  shape is the emulator's business, and a size check here would make this module
  depend on the emulator for nothing.

Deliberately not here: no queue and no history (the page wants the newest frame,
and old frames have no reader), nothing written to disk (the journal's per-tick
PNGs stay the only frames the run records, perception overlay and all), no JPEG
and no scaling (the browser sizes the image), and no wait-for-a-frame call. A
reader asking before the first publish gets ``None`` and says so.
"""

from __future__ import annotations

import io
import threading

import numpy as np
from PIL import Image

__all__ = ["LiveFrame"]


def _as_rgb_frame(frame: object) -> np.ndarray:
    """``frame`` as an array, or a ``ValueError`` naming what it is instead."""
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB frame, got shape {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"expected a uint8 frame, got dtype {array.dtype}")
    return array


def _encode_png(frame: np.ndarray) -> bytes:
    """The frame as PNG bytes. Lossless, so the pixels read back identical."""
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="PNG")
    return buffer.getvalue()


class LiveFrame:
    """The newest published frame and its PNG, guarded by one lock.

    One lock guards all three pieces of state -- the slot, the sequence number
    and the cached bytes -- because they are one fact: *this is frame number N,
    and these are its pixels as a PNG*. Splitting them across two locks would
    let a reader take the pixels of one frame and the sequence number of the
    next and cache an encoding it wrongly believes is current. Every read and
    every write of the tuple happens under that lock; the encoding, and nothing
    else, happens outside it.

    Safe to write from one thread and read from another. ``publish`` is meant for
    the run's thread, ``png()`` for every viewer thread that asks.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._sequence = 0
        # ``None`` means "no encoding cached"; otherwise it is
        # ``(sequence, bytes)``, and the bytes are reused only while the sequence
        # still matches the slot.
        self._encoded: tuple[int, bytes] | None = None

    def publish(self, frame: object) -> None:
        """Store a copy of ``frame`` as the newest frame.

        The copy is deliberate: see the module docstring. Nothing about a refused
        frame reaches the slot, so a bad publish leaves the previous frame and its
        cached PNG exactly as they were.
        """
        copied = _as_rgb_frame(frame).copy(order="C")
        with self._lock:
            self._frame = copied
            self._sequence += 1

    def png(self) -> bytes | None:
        """The newest frame as PNG bytes, or ``None`` if none was ever published.

        At most one encoding per published frame for sequential callers: the
        second request for a frame returns the very same bytes object rather than
        equal bytes, so a burst of requests from a polling page costs one
        encoding. A later publish defines a new frame and is encoded again.

        A caller that wants the newest frame should call this again rather than
        keep the result: the bytes are the frame that was newest when it was
        encoded, and the run has not stopped while the page was reading.
        """
        with self._lock:
            if self._frame is None:
                return None
            if self._encoded is not None and self._encoded[0] == self._sequence:
                return self._encoded[1]
            frame = self._frame
            sequence = self._sequence

        # Under no lock: the run's thread must not wait on an encoding, and the
        # frame read above is safe to use here because publish replaces the slot
        # with a new array instead of mutating the one already in it.
        encoded = _encode_png(frame)

        with self._lock:
            # Only cache it if the slot still holds the frame that was encoded. A
            # publish that landed while this one encoded already invalidated it,
            # and caching now would make that newer frame answer with stale bytes.
            if self._sequence == sequence:
                self._encoded = (sequence, encoded)
        return encoded
