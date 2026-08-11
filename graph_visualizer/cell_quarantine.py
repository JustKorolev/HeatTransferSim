"""Quarantine cells that can absorb heat but never shed it.

A cell that has no conduction path to a heat sink is a thermal dead end: any
power deposited into it can only raise its temperature, forever. That is not a
small local error. In the ``no_mli_high_res`` run of 2026-08-10, two heaters sat
pinned at their 30 W clamp for the whole run depositing into a detached 28-cell
CAD solid (``solid_3755``). Those two heaters carried 60 W of the 95 W total --
about 92% of the run's net power imbalance -- the solid reached 4894 K, and every
whole-graph metric (max temperature, temperature rate, energy drift, the
autoscale colour range) was pinned to a body that occupies 0.0009% of the graph.
The temperature field was fine. The diagnostics and 2 of 27 actuators were not.

The test here is deliberately narrow, because a false positive silently deletes
real geometry:

1. **Sink reachability** (the actual bug). Build the conduction adjacency, label
   its connected components, and quarantine every component that cannot reach a
   cryocooler. This catches a detached solid whole -- including its interior
   cells, which have perfectly healthy conductance *to each other* and so are
   invisible to any per-node test.
2. **Zero conductance.** A cell with no conduction edges at all can only
   accumulate. Unambiguous, and no real geometry looks like this.
3. **A conductance floor** -- opt-in, defaulting to off. A per-node conductance
   threshold is *not* safe as a default: cell conductance scales with both
   material and cell size, so a floor that is generous for a 1 mm copper cell
   (~0.5 W/K) would quarantine an entirely legitimate 1 mm G10 cell (~1e-4 W/K).
   Exposed for when you know your graph, off unless you ask.

Quarantined cells stay in the state vector -- reindexing a 3M-node graph would
invalidate every ``node_index_by_id`` in the codebase for no thermal gain. What
changes is that heater power is never deposited into them and they are excluded
from whole-graph metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix, issparse, triu


@dataclass
class QuarantineResult:
    """Which cells are inert, and why -- enough detail to report it honestly."""

    mask: np.ndarray
    """Boolean over matrix rows; True = quarantined."""

    unreachable_rows: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    """Rows quarantined for failing to reach a sink."""

    isolated_rows: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    """Rows quarantined for having no conduction edges at all."""

    below_floor_rows: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    """Rows quarantined by the opt-in conductance floor."""

    component_count: int = 0
    """Total connected components in the conduction graph."""

    quarantined_component_count: int = 0
    """How many of those could not reach a sink."""

    skipped_reason: str = ""
    """Set when sink reachability could not be evaluated (so callers can say so)."""

    @property
    def count(self) -> int:
        return int(np.count_nonzero(self.mask))

    @property
    def any_quarantined(self) -> bool:
        return bool(np.any(self.mask))

    def summary(self, node_ids: np.ndarray | None = None) -> str:
        """One-line human summary for events.log / report.md."""
        if self.skipped_reason:
            return f"cell quarantine skipped: {self.skipped_reason}"
        if not self.any_quarantined:
            return (
                f"cell quarantine: no inert cells "
                f"({self.component_count} conduction component(s), all sink-reachable)"
            )
        parts = []
        if self.unreachable_rows.size:
            parts.append(
                f"{self.unreachable_rows.size} in {self.quarantined_component_count} "
                f"sink-unreachable component(s)"
            )
        if self.isolated_rows.size:
            parts.append(f"{self.isolated_rows.size} with no conduction edges")
        if self.below_floor_rows.size:
            parts.append(f"{self.below_floor_rows.size} below the conductance floor")
        detail = ""
        if node_ids is not None and self.count:
            sample = [int(v) for v in np.asarray(node_ids)[self.mask][:8]]
            detail = f" e.g. node ids {sample}" + (", ..." if self.count > 8 else "")
        return (
            f"cell quarantine: {self.count} cell(s) marked inert ({'; '.join(parts)}). "
            f"They receive no heater power and are excluded from whole-graph metrics.{detail}"
        )


def _conduction_adjacency(L) -> csr_matrix:
    """Off-diagonal conduction structure of the Laplacian, as a symmetric pattern.

    ``L`` carries ``L_ij = -G_ij`` off the diagonal, so a nonzero off-diagonal
    entry *is* a conduction edge. Take magnitudes and drop the diagonal.
    """
    A = csr_matrix(L) if issparse(L) else csr_matrix(np.asarray(L, dtype=float))
    A = abs(A)
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A.tocsr()


def find_quarantined_cells(
    L,
    *,
    sink_rows=(),
    radiation_grounded=None,
    min_conductance_W_per_K: float = 0.0,
) -> QuarantineResult:
    """Identify cells that can absorb heat but have no path to shed it.

    ``L`` is the conduction Laplacian (sparse or dense). ``sink_rows`` are the
    matrix rows of cryocooler cells. ``radiation_grounded`` is an optional boolean
    mask of cells with a radiative path to the environment -- radiation is a
    diagonal sink that does not appear in ``L`` at all, so a cell that radiates is
    grounded even with no conduction edges and must never be quarantined.
    ``min_conductance_W_per_K`` is the opt-in per-node floor; leave it at 0.0 to
    quarantine only cells with literally no conduction edges.

    Everything is skipped, rather than failed, in the two cases where the question
    is not answerable:

    - **No cryocooler cells.** A radiation-only or open-loop model is legitimate,
      and there is no sink to be reachable from.
    - **The result would quarantine every cell.** That means the model has no
      conduction structure to reason about (a pure lumped model, or a load path
      that produced no edges) -- not that the whole graph is a dead end.
    """
    A = _conduction_adjacency(L)
    n = A.shape[0]
    mask = np.zeros(n, dtype=bool)
    result = QuarantineResult(mask=mask)

    # Sink reachability is the load-bearing test; without a sink, none of the
    # others mean anything either (a cell with no conduction edges may still be
    # radiatively grounded, and we have no way to rank it against the rest).
    sink_rows = np.asarray(list(sink_rows), dtype=int).reshape(-1)
    sink_rows = sink_rows[(sink_rows >= 0) & (sink_rows < n)]
    if sink_rows.size == 0:
        result.skipped_reason = (
            "no cryocooler cells in this model, so there is no sink to be reachable from "
            "(radiation-only or open-loop run)"
        )
        return result

    # Total conduction out of each cell. Sum of |off-diagonal| over the row.
    total_G = np.asarray(A.sum(axis=1)).reshape(-1)

    # Rule 2: no conduction edges at all -- can only accumulate.
    isolated = total_G <= 0.0

    # Rule 3: opt-in conductance floor.
    floor = float(min_conductance_W_per_K)
    below = (total_G > 0.0) & (total_G < floor) if floor > 0.0 else np.zeros(n, dtype=bool)

    # Rule 1: sink reachability -- the one that catches a detached solid whole.
    from scipy.sparse.csgraph import connected_components

    n_components, labels = connected_components(A, directed=False)
    result.component_count = int(n_components)
    sink_labels = np.unique(labels[sink_rows])
    unreachable = ~np.isin(labels, sink_labels)
    result.quarantined_component_count = int(n_components - sink_labels.size)

    mask = isolated | below | unreachable

    # Radiation grounds a cell without touching L, so exempt anything that can
    # radiate to the environment -- it has a path to shed heat after all.
    if radiation_grounded is not None:
        grounded = np.asarray(radiation_grounded, dtype=bool).reshape(-1)
        if grounded.shape == mask.shape:
            mask &= ~grounded

    # Never quarantine the entire graph: that is a statement about the model
    # having no conduction structure, not about every cell being a dead end.
    if bool(np.all(mask)):
        result.skipped_reason = (
            "every cell would be quarantined, which means this model carries no conduction "
            "structure to evaluate (a lumped model, or a load that produced no edges) rather "
            "than a graph made entirely of dead ends"
        )
        return result

    result.isolated_rows = np.flatnonzero(isolated & mask)
    result.below_floor_rows = np.flatnonzero(below & mask)
    result.unreachable_rows = np.flatnonzero(unreachable & mask)
    result.mask = mask
    return result


def deposition_targets_lost(
    model,
    heater_node_ids,
    node_index_by_id: dict[int, int],
    mask: np.ndarray,
) -> dict[int, list[int]]:
    """Heaters whose power-deposition cells were quarantined.

    Returns ``{heater_node_id: [quarantined deposition node ids]}`` for every
    heater that lost at least one target. A heater that lost *all* of its targets
    is now commanding power into nothing -- it stays in the controller (killing an
    actuator is a bigger decision than killing a cell) but the caller should say
    so loudly, because its commands no longer reach the plant.
    """
    lost: dict[int, list[int]] = {}
    for heater_id in heater_node_ids or ():
        heater = getattr(model, "nodes", {}).get(int(heater_id))
        if heater is None:
            continue
        targets = [int(v) for v in getattr(heater, "power_deposition_node_ids", []) or []]
        if not targets:
            targets = [int(heater_id)]
        dropped = [
            node_id
            for node_id in targets
            if node_id in node_index_by_id and bool(mask[node_index_by_id[node_id]])
        ]
        if dropped:
            lost[int(heater_id)] = dropped
    return lost


def fully_orphaned_heaters(
    model,
    heater_node_ids,
    node_index_by_id: dict[int, int],
    mask: np.ndarray,
) -> list[int]:
    """Heaters that have no surviving deposition target at all."""
    orphans: list[int] = []
    for heater_id in heater_node_ids or ():
        heater = getattr(model, "nodes", {}).get(int(heater_id))
        if heater is None:
            continue
        targets = [int(v) for v in getattr(heater, "power_deposition_node_ids", []) or []]
        if not targets:
            targets = [int(heater_id)]
        known = [node_id for node_id in targets if node_id in node_index_by_id]
        if known and all(bool(mask[node_index_by_id[node_id]]) for node_id in known):
            orphans.append(int(heater_id))
    return orphans
