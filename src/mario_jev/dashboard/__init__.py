"""The evidence dashboard: one static, self-contained HTML file, and one live page.

:func:`build_html` turns one :class:`~mario_jev.journal.Journal` into one
self-contained HTML file — inline CSS and SVG, PNGs embedded as ``data:`` URIs —
and :func:`render_text` prints the same facts for the terminal. :func:`build_live_html`
turns the same journal into the live page of the newest decision tick. All are pure
functions of the journal, so the same evidence always renders the same bytes.
"""

from .build import DEFAULT_TITLE, build_html, render_text, write_html
from .live import build_live_html

__all__ = [
    "DEFAULT_TITLE",
    "build_html",
    "build_live_html",
    "render_text",
    "write_html",
]
