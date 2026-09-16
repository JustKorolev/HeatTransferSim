"""Locating data files that ship with the application.

``materials.json`` used to be found as ``Path(__file__).parents[1] /
"materials.json"`` -- a sibling of the package. That resolves in a checkout and
nowhere else: installed as a wheel the package sits in ``site-packages``, whose
parent holds no such file, and the failure is silent. The material library would
quietly fall back to five built-in defaults and every cell in a freshly built
graph would get generic properties, with no error anywhere to say so.

The file now lives in ``graph_visualizer/data/`` and is shipped with the package,
so there is exactly one copy and it is always present. :func:`materials_file`
still prefers a project-local override, because a user working on their own
assembly should be able to keep their own material table beside it without
editing anything inside the installation.

Lookup order, first hit wins:

1. ``$HEATTRANSFERSIM_MATERIALS`` -- an explicit path, for scripted runs.
2. ``./materials.json`` -- the working directory, so a project folder can carry
   its own table.
3. the copy shipped inside this package.
"""

from __future__ import annotations

import os
from pathlib import Path

MATERIALS_FILENAME = "materials.json"
MATERIALS_ENV_VAR = "HEATTRANSFERSIM_MATERIALS"

#: The copy installed with the package. Always exists.
PACKAGED_MATERIALS_FILE = Path(__file__).resolve().parent / "data" / MATERIALS_FILENAME


def materials_file() -> Path:
    """The material table to use, by the order documented above.

    Never raises and never returns a path that does not exist: the packaged copy
    is the floor, so callers do not each need a fallback.
    """
    override = os.environ.get(MATERIALS_ENV_VAR, "").strip()
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file():
            return candidate

    local = Path.cwd() / MATERIALS_FILENAME
    if local.is_file():
        return local

    return PACKAGED_MATERIALS_FILE


def materials_file_description() -> str:
    """Which table is in use and why, for a log line or a status bar."""
    chosen = materials_file()
    if chosen == PACKAGED_MATERIALS_FILE:
        return f"{chosen} (shipped with the application)"
    if os.environ.get(MATERIALS_ENV_VAR, "").strip():
        return f"{chosen} (from ${MATERIALS_ENV_VAR})"
    return f"{chosen} (found in the working directory)"
