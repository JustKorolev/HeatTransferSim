"""Command-line entry points, inside the package on purpose.

These five used to sit at the repository root, and the application launched them
as subprocesses by resolving a sibling path:

    Path(__file__).resolve().parent.parent / "run_simulation.py"

That works from a checkout and nowhere else. Installed as a wheel,
``graph_visualizer`` lands in ``site-packages`` and its parent contains no such
script, so headless runs, G-matrix builds, modal-controller builds, fast-load
refreshes and controller exports would every one of them fail on a machine that
had installed the app rather than cloned it.

Living here, they are importable modules, so the launchers use
``[sys.executable, "-m", "graph_visualizer.cli.<name>", ...]`` -- which resolves
identically from a checkout, a wheel, a virtualenv or a frozen interpreter.
:func:`module_command` builds that list so no caller has to remember.

The root-level scripts are kept as thin shims, so ``python run_simulation.py``
still works for anyone with the muscle memory.
"""

from __future__ import annotations

import sys

#: Module path per script, so a rename has one place to change.
RUN_SIMULATION = "graph_visualizer.cli.run_simulation"
EXPORT_CONTROLLER = "graph_visualizer.cli.export_controller"
BUILD_G_MATRIX = "graph_visualizer.cli.build_g_matrix"
BUILD_MODAL_CONTROLLER = "graph_visualizer.cli.build_modal_controller"
REFRESH_FAST_LOAD = "graph_visualizer.cli.refresh_fast_load"


def module_command(module: str, *args: str) -> list[str]:
    """``[python, -m, module, *args]`` for a subprocess launch.

    ``sys.executable`` rather than a bare ``python``: the application may well be
    running under a virtualenv or a conda environment that is not first on PATH,
    and a run launched into the wrong interpreter fails later and confusingly --
    at an import, halfway through a graph load.
    """
    return [sys.executable, "-m", str(module), *[str(a) for a in args]]
