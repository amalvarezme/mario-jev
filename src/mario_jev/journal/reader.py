"""Reading a journal back: line-level validation, then whole-file validation.

Two readers, one validation core:

* :func:`iter_journal` validates every line — it is parseable JSON, it is an
  object, and its ``kind`` is one of the three — and yields the raw lines.
* :func:`read_journal` additionally enforces the run *shape*: one header first,
  zero or more ticks, one trailer last, and nothing after the trailer. With
  ``allow_partial=True`` it enforces the same shape for every line it can see and
  tolerates exactly one absence: the trailer. That is what a run still in
  progress looks like, and the only thing it may leave out is the end.

A malformed line is never skipped. The reader raises :class:`JournalFormatError`
with the 1-based line number instead, because a journal that silently drops a
line is worse evidence than one that refuses to be read. A line that is not
UTF-8 is one of those malformed lines: the file is decoded line by line so an
undecodable one is reported with its number rather than letting a
``UnicodeDecodeError`` escape. Partial reading does not relax any of that: it
forgives a missing trailer, never a corrupt line, a second header or a tick that
follows a trailer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .records import (
    KIND_RUN,
    KIND_SUMMARY,
    KIND_TICK,
    KINDS,
    Journal,
    JournalFormatError,
)


def _iter_raw_lines(path: object) -> Iterator[tuple[int, str]]:
    """Yield ``(line_number, decoded_line)``, or raise on the first undecodable line.

    Read as bytes and decoded per line on purpose: splitting on the newline byte
    keeps the line numbering honest, and an undecodable line is a malformed line in
    exactly the sense the module docstring means, so it raises
    :class:`JournalFormatError` naming it instead of letting the decoder's own
    exception escape.
    """
    with open(Path(path), "rb") as handle:
        for number, raw in enumerate(handle, start=1):
            try:
                yield number, raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise JournalFormatError(
                    number, f"not UTF-8 ({exc.reason} at byte {exc.start} of the line)"
                ) from exc


def _iter_entries(path: object) -> Iterator[tuple[int, dict[str, object]]]:
    """Yield ``(line_number, record)`` for every line, or raise on the first bad one."""
    for number, raw in _iter_raw_lines(path):
        if raw.strip() == "":
            raise JournalFormatError(
                number, "a blank line; every line must be one JSON object"
            )
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise JournalFormatError(
                number, f"not valid JSON ({exc.msg})"
            ) from exc
        if not isinstance(entry, dict):
            raise JournalFormatError(number, "not a JSON object")
        kind = entry.get("kind")
        if kind not in KINDS:
            raise JournalFormatError(number, f"unknown kind {kind!r}")
        yield number, entry


def iter_journal(path: object) -> Iterator[dict[str, object]]:
    """Yield every line of a journal as a plain object.

    Line-level validation only: parseability, object shape, and a known ``kind``.
    The run shape is not enforced here; use :func:`read_journal` for that.
    """
    for _, entry in _iter_entries(path):
        yield entry


def read_journal(path: object, *, allow_partial: bool = False) -> Journal:
    """Read and validate a journal.

    With ``allow_partial=False`` (the default) the whole run shape is required:
    exactly one header, zero or more ticks, one trailer last, nothing after it.

    With ``allow_partial=True`` every line that is present is held to exactly the
    same rules; the one thing that becomes optional is the trailer, because a run
    still in progress has not written it yet. The returned journal then carries
    ``summary=None`` -- "not written yet", which is a different fact from a
    recorded summary that is missing one of its keys. The trailer is the only
    absence forgiven: a malformed line, a second header, or a tick after a trailer
    still raises, because a reader that skipped one of those would be showing a
    different run than the one on disk.

    Raises:
        JournalFormatError: a line is not parseable, or is not an object, or has
            an unknown ``kind``, or the sequence is not one header, zero or more
            ticks and (unless ``allow_partial``) one trailer.
    """
    entries = list(_iter_entries(path))
    if not entries:
        raise JournalFormatError(
            1, "the journal is empty; it has no run header and no summary trailer"
        )

    first_number, first = entries[0]
    if first["kind"] != KIND_RUN:
        raise JournalFormatError(
            first_number,
            f"expected the run header on the first line, found {first['kind']!r}",
        )

    summary_at: int | None = None
    for index in range(1, len(entries)):
        number, entry = entries[index]
        kind = entry["kind"]
        if kind == KIND_RUN:
            raise JournalFormatError(number, "a second run header; exactly one is allowed")
        if kind == KIND_SUMMARY:
            if summary_at is not None:
                raise JournalFormatError(
                    number, "a second summary trailer; exactly one is allowed"
                )
            summary_at = index
        elif kind == KIND_TICK and summary_at is not None:
            raise JournalFormatError(number, "a tick after the summary trailer")

    if summary_at is None:
        if not allow_partial:
            last_number = entries[-1][0]
            raise JournalFormatError(
                last_number, "missing summary trailer; the journal must end with one"
            )
        # A run in progress: the header and every tick so far were validated
        # above, and only the trailer is absent. There is no summary to report,
        # so the field is ``None`` rather than an empty mapping -- an empty
        # mapping would read as a summary that recorded nothing, and a summary-less
        # journal must be labelled as a run still going, never as lost evidence.
        return Journal(
            run=first,
            ticks=tuple(entry for _, entry in entries[1:]),
            summary=None,
        )

    ticks = tuple(entry for _, entry in entries[1:summary_at])
    return Journal(run=first, ticks=ticks, summary=entries[summary_at][1])


__all__ = ["iter_journal", "read_journal"]
