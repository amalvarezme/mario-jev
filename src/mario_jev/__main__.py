"""``python -m mario_jev``: the module entry point for the command line."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
