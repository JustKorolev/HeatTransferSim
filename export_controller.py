"""Write the MIMO PI controller out as constants: a C header plus a JSON twin.

Example:
    python export_controller.py --graph graphs/CRYOSTAT_V2
    python export_controller.py --graph graphs/CRYOSTAT_V2 --prefix CRYO -o firmware/inc

With no --gain-matrix, the matrix named by the graph's own
``simulation_parameters.json`` is used, so the export describes the controller the
app is configured to run rather than a different one. --list shows what is
available when that is not set.

The header is data only -- gains, limits, G, the regularized inverse, provenance.
The twelve-line control law it implements is printed in the header's comment
block; see ``graph_visualizer/controller_export.py`` for exactly where the
shipped inverse and the app's bounded QP agree, and where they do not.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from graph_visualizer.controller_export import (
    ControllerExportError,
    export_controller,
)
from graph_visualizer.simulation_parameters import load_simulation_parameters


def _load_graph_model(graph_folder: Path):
    """Roles only, by the cheapest path that still carries them.

    The compact loader skips graph.json (1.6 GB at 471k nodes), but it only has
    heaters, sensors and setpoints when nodes.csv was written with a role_json
    column. Without roles this export would silently describe a plant with no
    actuators, so anything short of certainty falls back to the full loader.
    """
    from graph_visualizer.fast_graph_io import fast_load_has_roles, load_graph_for_simulation
    from graph_visualizer.graph_io import load_graph_folder

    if fast_load_has_roles(graph_folder):
        try:
            model, _matrices, _report = load_graph_for_simulation(graph_folder)
            return model
        except Exception as exc:  # noqa: BLE001 - fall back rather than fail the export
            print(f"Compact load unavailable ({exc}); reading graph.json instead.")
    model, _matrices = load_graph_folder(str(graph_folder))
    return model


def _list_gain_matrices(graph_folder: Path) -> int:
    from graph_visualizer.sys_id_artifacts import list_sys_id_gain_matrices

    matrices = list_sys_id_gain_matrices(graph_folder)
    if not matrices:
        print(f"No sys-id gain matrices under {graph_folder / 'sys_id'}.")
        print("Run a sys ID from the Heat Transfer Simulation tab first: Kp and Ki")
        print("alone do not define this controller, because the decoupling lives in G.")
        return 1
    print(f"Gain matrices under {graph_folder}:")
    for info in matrices:
        print(f"  {info.name:<30} {info.created_at}  {info.path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--graph", required=True, help="path to graphs/<name>")
    parser.add_argument(
        "--gain-matrix",
        default=None,
        help="sys-id run folder holding G. Default: whatever the graph's "
             "simulation_parameters.json names.",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="output folder (default: <graph>/controller_export)",
    )
    parser.add_argument(
        "--prefix", default="HTS_CTRL",
        help="identifier prefix for every #define and array in the header",
    )
    parser.add_argument(
        "--list", action="store_true", help="list this graph's gain matrices and exit"
    )
    args = parser.parse_args()

    graph_folder = Path(args.graph)
    if not graph_folder.is_dir():
        print(f"error: no such graph folder: {graph_folder}", file=sys.stderr)
        return 2
    if args.list:
        return _list_gain_matrices(graph_folder)

    params, _raw = load_simulation_parameters(graph_folder / "simulation_parameters.json")
    output = Path(args.output) if args.output else graph_folder / "controller_export"

    print(f"Loading {graph_folder} ...")
    try:
        model = _load_graph_model(graph_folder)
    except Exception as exc:  # noqa: BLE001 - a load failure is the user's problem to see
        print(f"error: could not load the graph: {exc}", file=sys.stderr)
        return 1

    try:
        json_path, header_path, constants = export_controller(
            model,
            params,
            output,
            gain_matrix_path=args.gain_matrix,
            graph_name=graph_folder.name,
            prefix=args.prefix,
        )
    except ControllerExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"\n{constants.n_sensors} controlled sensor(s) x {constants.n_heaters} heater(s), "
        f"cond(G) = {constants.cond_G:.4g}"
    )
    if constants.pinv_is_exact:
        print("The exported inverse reproduces the app's allocator exactly.")
    else:
        print(
            f"The exported inverse APPROXIMATES the app's allocator "
            f"(max deviation {constants.pinv_max_error_W:.3g} W at a representative command)."
        )
    for note in constants.notes:
        print(f"  - {note}")
    print(f"\nWrote:\n  {header_path}\n  {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
