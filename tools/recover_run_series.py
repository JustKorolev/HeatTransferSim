"""Rebuild a crashed run's timeseries from its snapshots and checkpoints.

A run killed at the C level never reaches ``_finalize``, so it writes no
timeseries and no plots -- but every snapshot and checkpoint it did write holds a
FULL temperature field. On a run that died at step 760 that is 19 fields spanning
the whole run, which is plenty to see what the controller was doing.

This reconstructs the sensor series from those fields and writes the same
``timeseries.npz`` / ``timeseries.csv`` a clean finish would have, so
``regenerate_plots`` can then draw the normal figures from it.

The result is SPARSE IN TIME -- one sample per snapshot/checkpoint rather than one
per step -- so it is for seeing the trajectory, not for measuring step-to-step
behaviour like command jitter. Sample times are exact; nothing is interpolated.

    python tools/recover_run_series.py <run_dir> --graph graphs/<name>

Reads the graph only for the node ordering and each sensor's readout cells; it
does not assemble the operator or prepare a simulation.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _samples(run_dir: Path) -> list[tuple[float, Path, str]]:
    """(time_s, path, kind) for every full field the run left, oldest first."""
    out: list[tuple[float, Path, str]] = []
    for path in (run_dir / "snapshots").glob("T_*.npy"):
        try:
            out.append((float(path.stem.split("_", 1)[1]), path, "snapshot"))
        except (IndexError, ValueError):
            continue
    for path in (run_dir / "checkpoints").glob("ckpt_*.npz"):
        try:
            with np.load(path) as data:
                out.append((float(data["time_s"]), path, "checkpoint"))
        except (OSError, KeyError, ValueError):
            continue
    out.sort(key=lambda item: item[0])
    return out


def _read_field(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path, mmap_mode="r"), dtype=float)
    with np.load(path) as data:
        return np.asarray(data["temperatures_K"], dtype=float)


def _sensor_manifest(run_dir: Path):
    """(node_ids, setpoints, controlled mask) in the run's own series order."""
    rows = list(csv.DictReader((run_dir / "sensors.csv").open(encoding="utf-8")))
    ids = np.array([int(r["node_id"]) for r in rows], dtype=int)
    setpoints = np.array([float(r["setpoint_K"]) for r in rows], dtype=float)
    controlled = np.array(
        [str(r.get("monitor_only", "")).strip().lower() == "false" for r in rows], dtype=bool
    )
    return ids, setpoints, controlled


def recover(run_dir: Path, graph_dir: Path) -> dict:
    from graph_visualizer.simulation_runner import SimulationRunner

    samples = _samples(run_dir)
    if not samples:
        raise FileNotFoundError(f"{run_dir} has no snapshots or checkpoints to recover from.")
    sensor_ids, setpoints, controlled = _sensor_manifest(run_dir)

    print(f"loading {graph_dir} for the node ordering and sensor readout cells ...")
    from graph_visualizer.fast_graph_io import load_graph_for_simulation

    model, matrices, _report = load_graph_for_simulation(graph_dir)
    node_ids = np.asarray(matrices["node_ids"], dtype=int).reshape(-1)
    index_by_id = {int(v): i for i, v in enumerate(node_ids)}
    sensor_ix = np.array([index_by_id[int(s)] for s in sensor_ids], dtype=int)

    # Reuse the runner's own readout operator so a recovered sensor value is the
    # SAME weighted mean over body cells the run itself reported -- a second
    # definition here would quietly disagree with the live one.
    helper = object.__new__(SimulationRunner)
    helper._build_sensor_readout_operator(
        SimpleNamespace(model=model, node_ids=node_ids, node_index_by_id=index_by_id),
        sensor_ids.tolist(),
        sensor_ix,
    )

    series: dict[str, list[float]] = {"time_s": []}
    heater_ids: list[int] = []
    print(f"reading {len(samples)} field(s) ...")
    for time_s, path, kind in samples:
        temps = _read_field(path)
        if temps.shape != node_ids.shape:
            print(f"  skipping {path.name}: {temps.shape} does not match the graph {node_ids.shape}")
            continue
        y = helper._sensor_readout_temperatures(temps, sensor_ix)
        series["time_s"].append(float(time_s))
        for j, value in enumerate(y):
            series.setdefault(f"sensor_{j}_K", []).append(float(value))
            if np.isfinite(setpoints[j]):
                series.setdefault(f"sensor_{j}_err_K", []).append(float(value - setpoints[j]))
        err = y - setpoints
        for key, mask in (
            ("rms_tracking_error_K", np.isfinite(err)),
            ("rms_tracking_error_controlled_K", np.isfinite(err) & controlled),
            ("rms_tracking_error_monitor_K", np.isfinite(err) & ~controlled),
        ):
            if mask.any():
                series.setdefault(key, []).append(float(np.sqrt(np.mean(err[mask] ** 2))))
        series.setdefault("avg_temp_K", []).append(float(temps.mean()))
        series.setdefault("max_temp_K", []).append(float(temps.max()))
        series.setdefault("min_temp_K", []).append(float(temps.min()))
        # Heater commands were checkpointed; snapshots have none, so carry the last
        # known value forward rather than inventing one.
        if kind == "checkpoint":
            with np.load(path) as data:
                if "controller_heater_ids" in data:
                    heater_ids = [int(v) for v in data["controller_heater_ids"]]
                    powers = [float(v) for v in data["controller_last_power_W"]]
                    for hid, watts in zip(heater_ids, powers):
                        series.setdefault(f"heater_{hid}_W", []).append(watts)
        print(f"  t={time_s:8.1f}s  {kind:10} max={temps.max():8.2f} K")

    # Heater series are shorter than the sensor ones (checkpoints only). Pad from the
    # front so they still line up with the tail of the time axis when plotted.
    n = len(series["time_s"])
    for key, values in series.items():
        if key.startswith("heater_") and len(values) < n:
            series[key] = [values[0]] * (n - len(values)) + values

    out = object.__new__(SimulationRunner)
    out.out_dir = run_dir
    out._series = series
    out._write_timeseries()
    print(f"\nwrote {run_dir / 'timeseries.npz'} with {n} sample(s)")
    return series


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--graph", type=Path, required=True, help="the graph folder the run used")
    ap.add_argument("--plots", action="store_true", help="also redraw plots/ afterwards")
    args = ap.parse_args()
    recover(args.run_dir, args.graph)
    if args.plots:
        from graph_visualizer.simulation_runner import regenerate_plots

        print("plots:", regenerate_plots(args.run_dir))


if __name__ == "__main__":
    main()
