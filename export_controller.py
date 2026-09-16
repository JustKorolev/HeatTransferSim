"""Write the MIMO PI controller out as constants.

Thin shim. The implementation moved into the package so the application can
launch it as ``python -m graph_visualizer.cli.export_controller``, which resolves whether
the project was cloned or pip-installed; a path to a script beside the package
only resolves in a checkout.

Equivalent, once installed:

    hts-export-controller [options]
    python -m graph_visualizer.cli.export_controller [options]
"""

from graph_visualizer.cli.export_controller import main

if __name__ == "__main__":
    raise SystemExit(main())
