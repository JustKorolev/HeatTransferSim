"""Headless closed-loop simulation runner.

Thin shim. The implementation moved into the package so the application can
launch it as ``python -m graph_visualizer.cli.run_simulation``, which resolves whether
the project was cloned or pip-installed; a path to a script beside the package
only resolves in a checkout.

Equivalent, once installed:

    hts-run [options]
    python -m graph_visualizer.cli.run_simulation [options]
"""

from graph_visualizer.cli.run_simulation import main

if __name__ == "__main__":
    raise SystemExit(main())
