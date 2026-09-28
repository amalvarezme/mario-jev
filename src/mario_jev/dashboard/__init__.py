"""The evidence dashboard: a static HTML report built from a recorded journal.

There is no live server and no web framework. :func:`build_html` turns one
:class:`~mario_jev.journal.Journal` into one self-contained HTML file — inline CSS
and SVG, PNGs embedded as ``data:`` URIs — and :func:`render_text` prints the same
facts for the terminal. Both are pure functions of the journal, so the same
evidence always renders the same bytes.
"""

from .build import DEFAULT_TITLE, build_html, render_text, write_html

__all__ = [
    "DEFAULT_TITLE",
    "build_html",
    "render_text",
    "write_html",
]
