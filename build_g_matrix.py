"""Solve the plant's DC gain matrix G from the fast-load artifacts.

Thin shim. The implementation moved into the package so the application can
launch it as ``python -m graph_visualizer.cli.build_g_matrix``, which resolves whether
the project was cloned or pip-installed; a path to a script beside the package
only resolves in a checkout.

Equivalent, once installed:

    hts-build-g [options]
    python -m graph_visualizer.cli.build_g_matrix [options]
"""

from graph_visualizer.cli.build_g_matrix import main

if __name__ == "__main__":
    raise SystemExit(main())
