"""Preflight: how much thermal authority does each heater actually have?

Answers the question that decides whether a run is worth starting: for every
heater, can the power it deposits get *out* of the cell it lands in?

A heater whose deposition cell has near-zero conductance is worse than useless.
The controller's DC gain correctly sees its tiny authority, the regularized
pseudo-inverse responds by assigning it enormous watts-per-kelvin, the command
clips at the heater's max, and it stays pinned there for the whole run while the
cell cooks. On ``no_mli_high_res`` (2026-08-10) two heaters did exactly this: 60 W
of a 95 W run, indefinitely, into a cell that reached 4894 K.

There are two distinct failure shapes and they need different fixes, which is why
this reports them separately:

- **Off the main body.** The cell is in a component with no path to a cryocooler.
  Caught automatically at load by ``cell_quarantine`` -- nothing to configure.
  Note the modal build already drops these from ``F``, so such a heater gets a
  zero DC-gain column and commands ~0 W rather than pinning at max.
- **On the main body but weakly connected.** The cell is technically reachable, so
  reachability cannot see it, but its conductance is so low that the power cannot
  leave. This is the case that pins a heater at its clamp, and it needs
  ``quarantine_min_conductance_W_per_K`` set deliberately for this graph.

Reads only the light files (C.npy, L_sparse.json, node_ids.npy, nodes.csv) -- it
never loads graph.json, so it runs in seconds on a 3M-cell graph.

    python tools/check_heater_authority.py graphs/no_mli_high_res
    python tools/check_heater_authority.py graphs/no_mli_high_res --max-power 30
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np

from analyze_plant_modes import load_sparse_operator


def _parse_list(value):
    try:
        return ast.literal_eval(value) if isinstance(value, str) and value.strip() else []
    except Exception:  # noqa: BLE001
        return []


def cell_descriptions(folder: Path, node_ids_wanted):
    """{node_id: (component_name, material_name, confidence)} from nodes.csv.

    A weak deposition cell is usually weak for a *nameable* reason -- an
    unassigned or low-confidence material falls back to a default that may be far
    less conductive than the real part, and a heater bonded to it looks fine in
    CAD while having almost no thermal authority. Reporting the material next to
    the conductance turns "this cell is weak" into something actionable.
    """
    import pandas as pd

    wanted = {int(v) for v in node_ids_wanted}
    if not wanted:
        return {}
    available = set(pd.read_csv(folder / "nodes.csv", nrows=0).columns)
    columns = [c for c in ("node_id", "component_name", "material_name", "confidence")
               if c in available]
    if "node_id" not in columns:
        return {}
    out: dict[int, tuple[str, str, str]] = {}
    for chunk in pd.read_csv(folder / "nodes.csv", usecols=columns, chunksize=200_000):
        hit = chunk[chunk["node_id"].isin(wanted)]
        for _, row in hit.iterrows():
            out[int(row["node_id"])] = (
                str(row.get("component_name", "") or ""),
                str(row.get("material_name", "") or "(no material_name column)"),
                str(row.get("confidence", "") or ""),
            )
    return out


def _heater_rows(folder: Path):
    """(heater_node_id, [(deposition_node_id, weight), ...]) from nodes.csv."""
    import pandas as pd

    columns = [
        "node_id", "is_heater", "heater_valid",
        "power_deposition_node_ids", "power_deposition_weights",
    ]
    frame = pd.read_csv(folder / "nodes.csv", usecols=columns)
    heaters = frame[(frame["is_heater"] == True)].reset_index(drop=True)  # noqa: E712
    out = []
    for _, row in heaters.iterrows():
        ids = _parse_list(row["power_deposition_node_ids"])
        weights = _parse_list(row["power_deposition_weights"])
        if not ids:
            ids, weights = [int(row["node_id"])], [1.0]
        if len(weights) != len(ids) or sum(float(w) for w in weights) <= 0.0:
            weights = [1.0 / len(ids)] * len(ids)
        total = float(sum(float(w) for w in weights))
        out.append((int(row["node_id"]), [
            (int(i), float(w) / total) for i, w in zip(ids, weights)
        ]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("graph", help="path to graphs/<name>")
    parser.add_argument("--max-power", type=float, default=30.0,
                        help="heater max power [W], for the implied steady-state rise")
    parser.add_argument("--rise-warn-K", type=float, default=100.0,
                        help="flag a heater whose cell would rise more than this at max power")
    args = parser.parse_args()

    folder = Path(args.graph)
    C, L, _Grad, node_ids = load_sparse_operator(folder)[:4]
    n = C.size
    print(f"graph: {folder}  ({n:,} cells)")

    adjacency = abs(L.tocsr())
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    total_G = np.asarray(adjacency.sum(axis=1)).reshape(-1)

    from scipy.sparse.csgraph import connected_components

    n_components, labels = connected_components(adjacency, directed=False)
    sizes = np.bincount(labels)
    main_label = int(np.argmax(sizes))
    print(f"conduction components: {n_components}  "
          f"(main = {sizes[main_label]:,} cells, {n - sizes[main_label]:,} elsewhere)")

    row_of = {int(v): i for i, v in enumerate(node_ids)}
    findings = []
    worst_cells: dict[int, list[tuple[int, float]]] = {}
    for heater_id, targets in _heater_rows(folder):
        # Series conductance of the weighted deposition set, as seen by the heater.
        conductance = 0.0
        off_main = 0
        for node_id, weight in targets:
            row = row_of.get(node_id)
            if row is None:
                continue
            if int(labels[row]) != main_label:
                off_main += 1
                continue
            conductance += weight * float(total_G[row])
        rise = args.max_power / conductance if conductance > 0 else float("inf")
        findings.append((heater_id, conductance, rise, off_main, len(targets)))
        # Keep this heater's least-conductive cells so the material report can say
        # WHY it is weak, not just that it is.
        ranked = []
        for node_id, weight in targets:
            row = row_of.get(node_id)
            if row is not None:
                ranked.append((node_id, float(total_G[row])))
        worst_cells[heater_id] = sorted(ranked, key=lambda item: item[1])[:3]

    findings.sort(key=lambda item: item[2], reverse=True)
    print()
    print(f"{'heater':>12} {'sum G [W/K]':>13} {'rise @%.0fW [K]' % args.max_power:>16} "
          f"{'off-main':>9} {'targets':>8}")
    for heater_id, conductance, rise, off_main, count in findings:
        flag = ""
        if off_main == count:
            flag = "  <- fully off-main (quarantine catches this)"
        elif rise > args.rise_warn_K:
            flag = "  <- WEAKLY CONNECTED (needs a conductance floor)"
        rise_text = "inf" if not np.isfinite(rise) else f"{rise:,.1f}"
        print(f"{heater_id:>12} {conductance:>13.4g} {rise_text:>16} "
              f"{off_main:>9} {count:>8}{flag}")

    weak = [f for f in findings if f[3] != f[4] and f[2] > args.rise_warn_K]
    dead = [f for f in findings if f[3] == f[4]]
    print()
    if dead:
        print(f"{len(dead)} heater(s) deposit only off the main body. Nothing to configure -- "
              "cell_quarantine excludes those cells at load.")
    if weak:
        print(f"{len(weak)} heater(s) are on the main body but poorly connected. Reachability "
              "CANNOT catch these.")
        # Report the gap honestly instead of manufacturing a threshold. These
        # distributions are usually a smooth continuum, in which case any floor is
        # a judgement call and saying otherwise would be misleading.
        values = np.array(sorted(f[1] for f in findings))
        if values.size > 1:
            ratios = values[1:] / np.maximum(values[:-1], 1.0e-30)
            split = int(np.argmax(ratios))
            gap = float(ratios[split])
            print(f"  conductance spread: {values[0]:.4g} .. {values[-1]:.4g} W/K")
            print(f"  largest gap in the sorted list: {gap:.2f}x "
                  f"(between {values[split]:.4g} and {values[split + 1]:.4g} W/K)")
            if gap >= 3.0:
                floor = float(np.sqrt(values[split] * values[split + 1]))
                print(f"  => bimodal enough to threshold: set "
                      f"quarantine_min_conductance_W_per_K = {floor:.3g}")
            else:
                print("  => NOT bimodal: this is a smooth continuum, so any floor is arbitrary "
                      "and WILL be wrong for some cell. Prefer fixing the contacts/geometry, or "
                      "pick a floor knowing it is a judgement call, not a detected boundary.")
    if not weak and not dead:
        print("Every heater can shed its power. Nothing to configure.")

    _report_materials(folder, weak + dead, worst_cells)
    _report_dc_gain(folder)
    return 0


def _report_materials(folder: Path, flagged, worst_cells) -> None:
    """What the flagged heaters are actually depositing into."""
    if not flagged:
        return
    wanted = [nid for f in flagged for nid, _g in worst_cells.get(f[0], [])]
    try:
        described = cell_descriptions(folder, wanted)
    except Exception as exc:  # noqa: BLE001 - reporting must not break the check
        print(f"\n(could not read cell materials from nodes.csv: {exc})")
        return
    if not described:
        return
    print()
    print("least-conductive deposition cell(s) of each flagged heater:")
    print(f"{'heater':>12} {'cell':>10} {'G [W/K]':>10}  {'material':<28} {'component':<24} conf")
    for heater_id, *_ in flagged:
        for node_id, conductance in worst_cells.get(heater_id, []):
            component, material, confidence = described.get(node_id, ("?", "?", ""))
            print(f"{heater_id:>12} {node_id:>10} {conductance:>10.4g}  "
                  f"{material[:28]:<28} {component[:24]:<24} {confidence}")
    materials = {described[n][1] for n in described}
    suspect = {m for m in materials
               if not m or m.lower().startswith(("not assigned", "unassigned", "none"))}
    if suspect:
        print()
        print(f"NOTE: {', '.join(sorted(suspect))} appears among these cells. An unassigned "
              "material falls back to a default that may be far less conductive than the real "
              "part, which would make a heater look bonded in CAD while having almost no "
              "thermal authority. Fixing the assignment is a better fix than any threshold.")


def _report_dc_gain(folder: Path) -> None:
    """The authoritative authority measure, if a built controller is available.

    Total edge conductance is only a proxy: it counts conduction between the
    deposition cells themselves, which does nothing to move heat to the sink. The
    real quantity is the plant DC gain column ``G[:, j]`` = steady-state kelvin at
    each controlled sensor per watt into heater j, which the modal build already
    computes with a full sparse solve. A heater with a near-zero column has no
    authority, and the regularized pseudo-inverse will pin it at its clamp."""
    artifacts = sorted(folder.glob("modal_controller*.npz"))
    if not artifacts:
        print()
        print("No modal_controller*.npz here yet. After you rebuild, re-run this: it will use "
              "the artifact's exact DC gain, which is authoritative where the conductance "
              "figures above are only a proxy.")
        return
    path = artifacts[-1]
    data = np.load(path, allow_pickle=False)
    if "dc_gain" not in data or "heater_ids" not in data:
        return
    G = np.asarray(data["dc_gain"], dtype=float)
    heater_ids = [int(v) for v in data["heater_ids"]]
    norms = np.linalg.norm(G, axis=0)
    order = np.argsort(norms)
    print()
    print(f"exact DC gain from {path.name} -- K per W at the controlled sensors:")
    for j in order[:8]:
        share = norms[j] / max(float(np.max(norms)), 1.0e-30)
        note = "  <- no authority; will pin at its clamp" if share < 0.02 else ""
        print(f"{heater_ids[j]:>12} {norms[j]:>13.4g}  ({share * 100:5.1f}% of the strongest){note}")
    print(f"  strongest: {float(np.max(norms)):.4g} K/W    ratio strongest/weakest: "
          f"{float(np.max(norms) / max(np.min(norms), 1.0e-30)):.3g}")


if __name__ == "__main__":
    raise SystemExit(main())
