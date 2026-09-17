"""Make the tests directory importable whatever import mode pytest is using.

Seven test modules share one hand-written Qt stub by importing a sibling by bare
name::

    import test_simulation_controls_panel as panel_stubs

That only resolves if the tests directory is on ``sys.path``, which pytest's
default ``prepend`` import mode arranges as a side effect of how it loads test
modules. Nothing in the suite asks for it, so the whole pattern rests on a
default. Running with ``--import-mode=importlib`` breaks all seven modules at
collection time, today, on any pytest version.

This is defensive hardening, not a fix for anything currently failing.

The sharing is still a smell: a stub used by seven modules wants to be a helper
module rather than another test module. That is a refactor; this is the one line
that makes the existing arrangement not depend on a default.
"""

from __future__ import annotations

from pathlib import Path
import sys

_TESTS_DIR = str(Path(__file__).resolve().parent)

if _TESTS_DIR not in sys.path:
    # Appended rather than prepended: a test module must never shadow a real
    # package that happens to share its name.
    sys.path.append(_TESTS_DIR)
