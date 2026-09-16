"""Rebuild a graph folder's low-memory load artifacts.

Thin shim. The implementation moved into the package so the application can
launch it as ``python -m graph_visualizer.cli.refresh_fast_load``, which resolves whether
the project was cloned or pip-installed; a path to a script beside the package
only resolves in a checkout.

Equivalent, once installed:

    hts-refresh-fast-load [options]
    python -m graph_visualizer.cli.refresh_fast_load [options]
"""

from graph_visualizer.cli.refresh_fast_load import main

if __name__ == "__main__":
    raise SystemExit(main())
