"""The journal writer: one run, one file, one flushed line at a time.

The writer is the object ``ExecutionLoop.run(observer=...)`` wants: call
:meth:`JournalWriter.observer` and hand the result to the loop. Each tick is
serialized and **flushed immediately**, so a crashed or killed run still leaves a
complete, parseable prefix on disk — that is the whole point of an evidence
journal.

Shape of a run
--------------

* The header is written **on construction**. The clock is read exactly once, for
  that header, so a run without ticks still has a timestamp.
* Each observed tick appends one line.
* :meth:`write_summary` appends the trailer; a tick after it is refused.
* :meth:`close` is idempotent, and the writer is a context manager.

Artifacts
---------

When ``artifacts_dir`` is given, each tick also gets
``<dir>/frame_<frame_index:06d>.png`` and ``<dir>/trajectory_<tick:06d>.png``,
and the tick line records those **relative** names (the directory itself is in
the header, under ``artifacts_dir``).

The frame and its perception are read straight off the ``TickRecord``: the record
carries the tick's own frame and exactly what ``detect()`` saw in it, so the
writer never has to guess which frame a tick described. There is no frame source
to inject, because the evidence is already on the record.

Writing an artifact is best-effort by construction: a failure is recorded in the
tick's ``degradations`` list and the run continues. A journal missing a picture is
still evidence; a run stopped by a full disk is not.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from mario_jev.config import Settings
from mario_jev.policy import RunSummary, TickRecord

from .artifacts import write_frame_png, write_trajectory_png
from .records import run_record, summary_record, tick_record


def _utcnow() -> datetime:
    """The default clock: an aware UTC instant."""
    return datetime.now(timezone.utc)


class JournalWriter:
    """Write one JSONL journal for one run."""

    def __init__(
        self,
        path: object,
        *,
        settings: Settings,
        clock: Callable[[], datetime] = _utcnow,
        artifacts_dir: object | None = None,
    ) -> None:
        self._path = Path(path)
        self._settings = settings
        self._clock = clock
        self._artifacts_dir = None if artifacts_dir is None else Path(artifacts_dir)
        self._closed = False
        self._trailer_written = False

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self._path, "w", encoding="utf-8")
        # The header is written now, on construction, so even a run that never
        # ticks has a timestamped, self-describing first line.
        self._write(
            run_record(
                settings,
                started_at=clock(),
                artifacts_dir=self._artifacts_dir,
            )
        )

    # -- writing -------------------------------------------------------------

    def _write(self, record: dict[str, object]) -> None:
        if self._closed:
            raise RuntimeError("the journal is closed")
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()

    def observer(self) -> Callable[[TickRecord], None]:
        """The callable to pass to ``ExecutionLoop.run(observer=...)``."""
        return self._observe

    def _observe(self, record: TickRecord) -> None:
        if self._trailer_written:
            raise RuntimeError("a tick cannot be written after the summary trailer")

        frame_png: str | None = None
        trajectory_png: str | None = None
        degradations: list[str] = []
        if self._artifacts_dir is not None:
            frame_png, trajectory_png, degradations = self._write_artifacts(record)

        entry = tick_record(record, frame_png=frame_png, trajectory_png=trajectory_png)
        if degradations:
            entry["degradations"] = [*entry["degradations"], *degradations]
        self._write(entry)

    # -- artifacts (best effort, never fatal) --------------------------------

    def _write_artifacts(
        self, record: TickRecord
    ) -> tuple[str | None, str | None, list[str]]:
        degradations: list[str] = []

        frame_name = f"frame_{int(record.frame_index):06d}.png"
        trajectory_name = f"trajectory_{int(record.tick):06d}.png"
        directory = self._artifacts_dir
        assert directory is not None  # only called when artifacts are enabled

        # The frame and its perception are the tick's own; they are read off the
        # record, never re-derived, so the picture always matches the tick line.
        frame_png: str | None = None
        try:
            write_frame_png(directory / frame_name, record.frame, record.perception)
        except Exception as exc:  # noqa: BLE001 - an artifact never fails the run
            degradations.append(f"artifact_frame_failed: {type(exc).__name__}")
        else:
            frame_png = frame_name

        trajectory_png: str | None = None
        try:
            write_trajectory_png(
                directory / trajectory_name,
                record.rollout,
                decision=record.decision,
            )
        except Exception as exc:  # noqa: BLE001 - an artifact never fails the run
            degradations.append(f"artifact_trajectory_failed: {type(exc).__name__}")
        else:
            trajectory_png = trajectory_name

        return frame_png, trajectory_png, degradations

    # -- closing -------------------------------------------------------------

    def write_summary(self, summary: RunSummary) -> None:
        """Append the one summary trailer. A second call is a bug."""
        if self._trailer_written:
            raise RuntimeError("the summary trailer was already written")
        self._write(summary_record(summary))
        self._trailer_written = True

    def close(self) -> None:
        """Flush and close. Idempotent."""
        if self._closed:
            return
        self._handle.flush()
        self._handle.close()
        self._closed = True

    def __enter__(self) -> "JournalWriter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = ["JournalWriter"]
