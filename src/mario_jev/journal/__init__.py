"""The evidence journal: a JSONL trace a human can read to judge the run.

The journal is the project's evidence artifact, so its schema is fixed here
rather than invented per call site. One file holds one run:

* exactly one ``run`` header (first line),
* zero or more ``tick`` lines,
* exactly one ``summary`` trailer (last line).

Every line is one UTF-8 JSON object terminated by ``\\n``, and every line is
flushed as it is written, so a killed run still leaves a complete prefix. The
reader validates the whole file and raises :class:`JournalFormatError` with the
1-based line number instead of skipping anything.

What each line carries, and why
-------------------------------

* **run** — ``started_at`` (aware UTC ISO-8601), ``stage``, ``seed`` and
  ``settings``. The settings are recorded through :attr:`Settings.redacted`, so
  the Jev credential cannot reach the file.
* **tick** — its ``kind``, its own ``tick`` number, the loop's ``frame_index``
  *and* the raw ``emulator_tick`` (they differ; conflating them is a silent bug
  the project already had), the recorded ``state`` projection, the ``rollout``
  table, the ``questions`` and ``response`` exactly as the client produced them,
  the ``decision`` with its ``source`` as a plain string, a typed ``advisor``
  block that says whether an advice was present and, when none was, whether its
  absence was *measured* or the advisor was simply disabled, ``latencies_ms``,
  ``degradations``, the post-step ``terminated``/``truncated`` flags, and the
  relative ``frame_png``/``trajectory_png`` paths.
* **summary** — its ``kind`` and the counted run shape: ``ticks``, ``frames``,
  ``decisions``, ``sources`` and ``degradations`` as plain objects keyed by
  string names, ``elapsed_s``, ``decision_rate_hz``, and the ``termination``
  that says why the run stopped.

Those two enumerations are the schema, and they are checked against the real
key sets by ``tests/test_journal.py``: a stale enumeration is a false statement
about the evidence file, so it fails the suite rather than the reader.

See :mod:`mario_jev.journal.artifacts` for the two Pillow-only drawings.
"""

from .artifacts import write_frame_png, write_trajectory_png
from .reader import iter_journal, read_journal
from .records import (
    KINDS,
    KIND_RUN,
    KIND_SUMMARY,
    KIND_TICK,
    Journal,
    JournalFormatError,
    advisor_record,
    run_record,
    summary_record,
    tick_record,
)
from .writer import JournalWriter

__all__ = [
    "KINDS",
    "KIND_RUN",
    "KIND_SUMMARY",
    "KIND_TICK",
    "Journal",
    "JournalFormatError",
    "JournalWriter",
    "advisor_record",
    "iter_journal",
    "read_journal",
    "run_record",
    "summary_record",
    "tick_record",
    "write_frame_png",
    "write_trajectory_png",
]
