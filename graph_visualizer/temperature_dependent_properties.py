"""Rebuild C(T) and the conduction Laplacian L(T) from NIST cryogenic properties.

When temperature-dependent properties are enabled, per-node heat capacity and
pairwise conductances become temperature dependent:

    C_i(T)  = mass_i * cp(material_i, T_i)                          (exact, per node)
    G_ij(T) = 1 / [ L_i / (k_i(T_i) * A) + L_j / (k_j(T_j) * A) + R_interface(T) ]

where L_i, L_j are the per-cell conduction half-lengths along the contact normal
(actual cell half-sizes s_i/2, s_j/2 -- not distance/2 -- so unequal-size
neighbours are split correctly), A is the shared face area, and the interface
term is

    R_interface(T) = 1 / (h(T) * A),   h(T) = h_ref * (T_face / T_ref) ** n

for a bolted joint, or omitted entirely for a bonded joint (same component name,
h_ref = 0). h_ref is the room-temperature contact conductance; the T^n factor is
a rough cryogenic contact-conductance model (n tunable, data is joint-specific).
T_face is the mean of the two node temperatures.

Each edge's decomposition (A, L_i, L_j, h_ref) is taken from the graph when the
builder saved it, otherwise reconstructed from node geometry + component names
using the default bolted conductance -- so existing graphs work without a rebuild.

The two half-cell resistances in series realise the harmonic mean of the two
conductivities, which is the correct finite-volume face conductance across a
material interface.

Role/marker nodes and manual-capacitance nodes keep constant C. Visual
role-contact edges are excluded. Assembly reuses a fixed COO (row, col) pattern
and only recomputes the data array each step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags

from . import material_properties_cryo as mp
from .matrix_builder import (
    DEFAULT_CONTACT_INTERFACE_CONDUCTANCE_W_M2K,
    _is_cad_role_node,
    _is_visual_role_contact_edge,
)

DEFAULT_BOLTED_CONTACT_CONDUCTANCE_W_M2K = 3000.0
_MIN_FACE_TEMPERATURE_K = 1.0  # floor for the h(T) scaling to avoid h -> 0


@dataclass
class TemperatureDependentOperator:
    """Precomputed structure that rebuilds (C, inv_C, L) for a temperature field."""

    n: int
    copper_rrr: int
    # h(T) = h_ref * (T_face / T_ref) ** temp_exponent
    contact_temp_exponent: float
    contact_reference_temperature_K: float
    # Per-node capacitance basis
    thermal_mass_kg: np.ndarray
    fixed_C_mask: np.ndarray
    fixed_C_values: np.ndarray
    cp_fallback: np.ndarray
    k0: np.ndarray
    # material -> row indices (only materials with a NIST curve)
    cp_groups: dict[str, np.ndarray]
    k_groups: dict[str, np.ndarray]
    # Edge arrays (conduction edges only, in matrix-index space)
    edge_i: np.ndarray
    edge_j: np.ndarray
    edge_area_m2: np.ndarray  # shared face area A
    edge_len_i_m: np.ndarray  # conduction half-length on the i side
    edge_len_j_m: np.ndarray  # conduction half-length on the j side
    edge_h_ref: np.ndarray  # interface conductance W/m2K at T_ref; 0 => bonded
    # Fixed CSR assembly pattern (off-diagonals then diagonal)
    _rows: np.ndarray = field(repr=False, default=None)
    _cols: np.ndarray = field(repr=False, default=None)

    def specific_heat(self, temperatures_K: np.ndarray) -> np.ndarray:
        temperatures = np.asarray(temperatures_K, dtype=float).reshape(-1)
        cp = self.cp_fallback.copy()
        for name, rows in self.cp_groups.items():
            cp[rows] = mp.specific_heat_J_kgK(name, temperatures[rows])
        return cp

    def conductivity(self, temperatures_K: np.ndarray) -> np.ndarray:
        temperatures = np.asarray(temperatures_K, dtype=float).reshape(-1)
        k = self.k0.copy()
        for name, rows in self.k_groups.items():
            k[rows] = mp.thermal_conductivity_W_mK(name, temperatures[rows], rrr=self.copper_rrr)
        return np.where(k > 0.0, k, self.k0)

    def capacitance(self, temperatures_K: np.ndarray) -> np.ndarray:
        C = self.thermal_mass_kg * self.specific_heat(temperatures_K)
        if np.any(self.fixed_C_mask):
            C = np.where(self.fixed_C_mask, self.fixed_C_values, C)
        return np.where(np.isfinite(C) & (C > 0.0), C, np.maximum(self.fixed_C_values, 1.0e-12))

    def edge_conductance(self, temperatures_K: np.ndarray) -> np.ndarray:
        """G_ij(T) for every conduction edge, from resistance components."""
        temperatures = np.asarray(temperatures_K, dtype=float).reshape(-1)
        k = self.conductivity(temperatures)
        k_i = np.maximum(k[self.edge_i], 1.0e-30)
        k_j = np.maximum(k[self.edge_j], 1.0e-30)
        area = np.maximum(self.edge_area_m2, 1.0e-30)
        resistance = self.edge_len_i_m / (k_i * area) + self.edge_len_j_m / (k_j * area)
        bolted = self.edge_h_ref > 0.0
        if np.any(bolted):
            t_face = np.maximum(
                0.5 * (temperatures[self.edge_i] + temperatures[self.edge_j]),
                _MIN_FACE_TEMPERATURE_K,
            )
            scale = (t_face / max(self.contact_reference_temperature_K, 1.0e-6)) ** self.contact_temp_exponent
            h = np.maximum(self.edge_h_ref * scale, 1.0e-30)
            resistance = resistance + np.where(bolted, 1.0 / (h * area), 0.0)
        return np.where(resistance > 0.0, 1.0 / np.where(resistance > 0.0, resistance, 1.0), 0.0)

    def laplacian(self, temperatures_K: np.ndarray) -> csr_matrix:
        if self.edge_i.size == 0:
            # An all-zero Laplacian means NO conduction anywhere: every node becomes
            # thermally isolated and the run silently produces nonsense (heaters cook
            # their own cells while the rest of the graph never moves). Only legitimate
            # for a genuinely edgeless model, so refuse when the graph has nodes --
            # prepare_simulation catches this and falls back to the prebuilt L.
            if self.n > 1:
                raise ValueError(
                    f"temperature-dependent operator has no conduction edges for {self.n} nodes; "
                    "rebuilding L(T) would yield an all-zero Laplacian (every node thermally "
                    "isolated). The model was likely loaded without edges (low-memory nodes.csv "
                    "path), which cannot support T-dependent properties."
                )
            return csr_matrix((self.n, self.n))
        g = np.maximum(0.0, self.edge_conductance(temperatures_K))
        diag = np.zeros(self.n, dtype=float)
        np.add.at(diag, self.edge_i, g)
        np.add.at(diag, self.edge_j, g)
        data = np.concatenate([-g, -g, diag])
        # The sparsity pattern is FIXED -- only the values change with temperature.
        # Rebuilding the COO and converting it every step allocated ~0.5 GB of
        # transient index/data arrays per step (int64 row/col pairs plus a fresh
        # CSR) and ran coo_tocsr over 20M entries, on top of an 18.5 GiB resident
        # model. Build the structure once, then reorder the values into it.
        order = getattr(self, "_csr_order", None)
        if order is None:
            positions = coo_matrix(
                (np.arange(1, data.size + 1, dtype=float), (self._rows, self._cols)),
                shape=(self.n, self.n),
            ).tocsr()
            if positions.nnz == data.size:
                # 1:1 COO->CSR (no coincident entries), so the permutation is exact.
                self._csr_order = positions.data.astype(np.intp) - 1
                self._csr_indices = positions.indices
                self._csr_indptr = positions.indptr
            else:
                # Coincident entries would have to be summed; keep the exact path.
                self._csr_order = False
            order = self._csr_order
        if order is False:
            return coo_matrix((data, (self._rows, self._cols)), shape=(self.n, self.n)).tocsr()
        # A FRESH data array every call: callers legitimately hold two Laplacians at
        # once (e.g. comparing cold vs warm conductance), so returning a reused
        # buffer would silently make them the same matrix. The index arrays are
        # immutable here and shared, so only the values are allocated.
        return csr_matrix(
            (np.take(data, order), self._csr_indices, self._csr_indptr),
            shape=(self.n, self.n),
            copy=False,
        )

    def rebuild(self, temperatures_K: np.ndarray) -> tuple[np.ndarray, np.ndarray, csr_matrix]:
        C = self.capacitance(temperatures_K)
        inv_C = 1.0 / C
        L = self.laplacian(temperatures_K)
        return C, inv_C, L


def _contact_axis(center_a: np.ndarray, center_b: np.ndarray) -> int:
    delta = np.abs(np.asarray(center_a, dtype=float) - np.asarray(center_b, dtype=float))
    return int(np.argmax(delta)) if delta.size == 3 else 0


def _face_area_m2(size_mm: np.ndarray, axis: int) -> float:
    other = [a for a in range(3) if a != axis]
    return float(size_mm[other[0]] * size_mm[other[1]] * 1.0e-6)


def build_temperature_dependent_operator(
    model: Any,
    node_ids: np.ndarray,
    *,
    copper_rrr: int = mp.DEFAULT_COPPER_RRR,
    default_bolted_conductance_W_m2K: float = DEFAULT_BOLTED_CONTACT_CONDUCTANCE_W_M2K,
    contact_temp_exponent: float = 1.0,
    contact_reference_temperature_K: float = 293.15,
) -> TemperatureDependentOperator:
    node_ids = np.asarray(node_ids, dtype=int)
    n = int(node_ids.size)
    index = {int(node_id): row for row, node_id in enumerate(node_ids)}

    thermal_mass_kg = np.zeros(n, dtype=float)
    fixed_C_mask = np.zeros(n, dtype=bool)
    fixed_C_values = np.zeros(n, dtype=float)
    cp_fallback = np.zeros(n, dtype=float)
    k0 = np.zeros(n, dtype=float)
    material_by_row: list[str] = [""] * n
    node_by_row: list[Any] = [None] * n

    for node_id in node_ids:
        row = index[int(node_id)]
        node = model.nodes[int(node_id)]
        node_by_row[row] = node
        material_by_row[row] = str(getattr(node, "material", "") or "")
        cp0 = float(getattr(node, "cp_J_kgK", 0.0) or 0.0)
        cp_fallback[row] = cp0
        k0[row] = float(getattr(node, "k_W_mK", 0.0) or 0.0)
        C_build = float(getattr(node, "C_J_K", 0.0) or 0.0)
        fixed_C_values[row] = C_build
        fixed_C_mask[row] = bool(getattr(node, "C_manual_override", False)) or _is_cad_role_node(node)
        mass = float(getattr(node, "mass_kg", 0.0) or 0.0)
        if not (np.isfinite(mass) and mass > 0.0):
            mass = C_build / cp0 if cp0 > 0.0 else 0.0
        thermal_mass_kg[row] = mass

    def _groups(kind: str) -> dict[str, np.ndarray]:
        rows_by_material: dict[str, list[int]] = {}
        for row, material in enumerate(material_by_row):
            if kind == "cp" and fixed_C_mask[row]:
                continue
            if mp.has_curve(material):
                rows_by_material.setdefault(material, []).append(row)
        return {name: np.asarray(rows, dtype=int) for name, rows in rows_by_material.items()}

    cp_groups = _groups("cp")
    k_groups = _groups("k")

    # Edge geometry is computed VECTORIZED. The per-edge Python path below
    # (_edge_geometry / _edge_interface_conductance) allocates several small numpy
    # arrays per edge, which costs minutes on a multi-million-edge graph -- paid on
    # every run's startup. One cheap attribute pass to gather flat arrays, then all
    # of the geometry at once, is equivalent and orders of magnitude faster.
    edge_i: list[int] = []
    edge_j: list[int] = []
    saved_area_list: list[float] = []
    distance_list: list[float] = []
    scalar_rows: list[int] = []  # edges that must use the exact per-edge fallback
    for edge in model.edges.values():
        if _is_visual_role_contact_edge(edge):
            continue
        if edge.source not in index or edge.target not in index:
            continue
        if max(0.0, float(edge.Gij_W_K)) <= 0.0:
            continue
        i = index[edge.source]
        j = index[edge.target]
        edge_i.append(i)
        edge_j.append(j)
        saved_area_list.append(float(getattr(edge, "shared_area_m2", 0.0) or 0.0))
        distance_list.append(float(getattr(edge, "distance_m", 0.0) or 0.0))
        # Builder-saved conduction lengths / interface conductance are not fields on
        # EdgeProperties, but a future graph could carry them; such edges fall back
        # to the exact scalar helpers so behaviour never silently changes.
        if (
            getattr(edge, "conduction_length_a_m", None) is not None
            or getattr(edge, "conduction_length_b_m", None) is not None
            or getattr(edge, "interface_conductance_W_m2K", None) is not None
        ):
            scalar_rows.append(len(edge_i) - 1)

    edge_i_arr = np.asarray(edge_i, dtype=int)
    edge_j_arr = np.asarray(edge_j, dtype=int)
    edge_area, edge_len_i, edge_len_j, edge_h_ref = _vectorized_edge_geometry(
        node_by_row,
        edge_i_arr,
        edge_j_arr,
        np.asarray(saved_area_list, dtype=float),
        np.asarray(distance_list, dtype=float),
        float(default_bolted_conductance_W_m2K),
    )
    if scalar_rows:
        edge_list = [
            edge
            for edge in model.edges.values()
            if not _is_visual_role_contact_edge(edge)
            and edge.source in index
            and edge.target in index
            and max(0.0, float(edge.Gij_W_K)) > 0.0
        ]
        for row in scalar_rows:
            edge = edge_list[row]
            node_a = node_by_row[edge_i_arr[row]]
            node_b = node_by_row[edge_j_arr[row]]
            area, len_a, len_b = _edge_geometry(node_a, node_b, edge)
            edge_area[row] = area
            edge_len_i[row] = len_a
            edge_len_j[row] = len_b
            edge_h_ref[row] = _edge_interface_conductance(
                node_a, node_b, edge, default_bolted_conductance_W_m2K
            )
    diag_index = np.arange(n, dtype=int)
    rows = np.concatenate([edge_i_arr, edge_j_arr, diag_index])
    cols = np.concatenate([edge_j_arr, edge_i_arr, diag_index])

    return TemperatureDependentOperator(
        n=n,
        copper_rrr=int(copper_rrr),
        contact_temp_exponent=float(contact_temp_exponent),
        contact_reference_temperature_K=float(contact_reference_temperature_K),
        thermal_mass_kg=thermal_mass_kg,
        fixed_C_mask=fixed_C_mask,
        fixed_C_values=fixed_C_values,
        cp_fallback=cp_fallback,
        k0=k0,
        cp_groups=cp_groups,
        k_groups=k_groups,
        edge_i=edge_i_arr,
        edge_j=edge_j_arr,
        edge_area_m2=np.asarray(edge_area, dtype=float),
        edge_len_i_m=np.asarray(edge_len_i, dtype=float),
        edge_len_j_m=np.asarray(edge_len_j, dtype=float),
        edge_h_ref=np.asarray(edge_h_ref, dtype=float),
        _rows=rows,
        _cols=cols,
    )


def _vectorized_edge_geometry(
    node_by_row: list[Any],
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    saved_area_m2: np.ndarray,
    distance_m: np.ndarray,
    default_bolted: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(area, half_length_i, half_length_j, h_ref) for every edge, all at once.

    Exactly mirrors ``_edge_geometry`` + ``_edge_interface_conductance`` for edges
    whose endpoints both carry ``center_mm``/``size_mm`` (every octree cell), and
    reproduces their fallbacks elementwise for any that do not.
    """
    count = int(edge_i.size)
    if count == 0:
        empty = np.zeros(0, dtype=float)
        return empty, empty.copy(), empty.copy(), empty.copy()

    n = len(node_by_row)
    centers = np.zeros((n, 3), dtype=float)
    sizes = np.zeros((n, 3), dtype=float)
    side = np.zeros(n, dtype=float)
    has_geometry = np.zeros(n, dtype=bool)
    components: list[str] = [""] * n
    for row, node in enumerate(node_by_row):
        center = getattr(node, "center_mm", None)
        size = getattr(node, "size_mm", None)
        if center is not None and size is not None:
            centers[row] = np.asarray(center, dtype=float)
            sizes[row] = np.asarray(size, dtype=float)
            has_geometry[row] = True
        side[row] = float(getattr(node, "side_length_m", 0.0) or 0.0)
        components[row] = str(getattr(node, "component_name", "") or "")
    component_code = np.unique(np.asarray(components, dtype=object).astype(str), return_inverse=True)[1]

    size_i = sizes[edge_i]
    size_j = sizes[edge_j]
    # Contact axis: the one with the largest centre separation (ties -> lowest axis,
    # matching np.argmax in the scalar helper).
    axis = np.argmax(np.abs(centers[edge_i] - centers[edge_j]), axis=1)
    rows = np.arange(count)
    len_i = size_i[rows, axis] * 0.5e-3
    len_j = size_j[rows, axis] * 0.5e-3

    # Face area on that axis is the product of the OTHER two extents.
    def face_area(size: np.ndarray) -> np.ndarray:
        total = size[:, 0] * size[:, 1] * size[:, 2]
        return total / np.where(size[rows, axis] > 0.0, size[rows, axis], 1.0) * 1.0e-6

    area = np.where(
        saved_area_m2 > 0.0,
        saved_area_m2,
        np.minimum(face_area(size_i), face_area(size_j)),
    )

    # Endpoints lacking geometry use the legacy distance/2 model.
    geometric = has_geometry[edge_i] & has_geometry[edge_j]
    half = 0.5 * distance_m
    len_i = np.where(geometric, len_i, half)
    len_j = np.where(geometric, len_j, half)
    area = np.where(geometric, area, saved_area_m2)

    # Same final fallbacks as the scalar helper, keyed on node_a's side length.
    side_i = side[edge_i]
    bad_area = ~(area > 0.0)
    if bad_area.any():
        area = np.where(bad_area, np.where(side_i > 0.0, side_i * side_i, 1.0e-9), area)
    bad_len = ~((len_i > 0.0) & (len_j > 0.0))
    if bad_len.any():
        replacement = np.maximum(0.5 * side_i, 1.0e-9)
        len_i = np.where(bad_len, replacement, len_i)
        len_j = np.where(bad_len, replacement, len_j)

    # Bonded (same component) => no interface resistance; otherwise bolted.
    h_ref = np.where(component_code[edge_i] == component_code[edge_j], 0.0, float(default_bolted))
    return area, len_i, len_j, h_ref


def _edge_geometry(node_a: Any, node_b: Any, edge: Any) -> tuple[float, float, float]:
    """Return (area_m2, half_length_a_m, half_length_b_m) for an edge.

    Prefers builder-saved values, then reconstructs from node geometry, then
    falls back to the edge's stored distance/area (the legacy distance/2 model).
    """
    saved_len_a = getattr(edge, "conduction_length_a_m", None)
    saved_len_b = getattr(edge, "conduction_length_b_m", None)
    saved_area = float(getattr(edge, "shared_area_m2", 0.0) or 0.0)
    center_a = getattr(node_a, "center_mm", None)
    center_b = getattr(node_b, "center_mm", None)
    size_a = getattr(node_a, "size_mm", None)
    size_b = getattr(node_b, "size_mm", None)
    if center_a is not None and center_b is not None and size_a is not None and size_b is not None:
        size_a = np.asarray(size_a, dtype=float)
        size_b = np.asarray(size_b, dtype=float)
        axis = _contact_axis(np.asarray(center_a, dtype=float), np.asarray(center_b, dtype=float))
        len_a = float(size_a[axis]) * 0.5 * 1.0e-3
        len_b = float(size_b[axis]) * 0.5 * 1.0e-3
        area = saved_area if saved_area > 0.0 else min(_face_area_m2(size_a, axis), _face_area_m2(size_b, axis))
    else:
        half = 0.5 * float(getattr(edge, "distance_m", 0.0) or 0.0)
        len_a = len_b = half
        area = saved_area
    if saved_len_a is not None and saved_len_b is not None:
        len_a = float(saved_len_a)
        len_b = float(saved_len_b)
    if not (area > 0.0):
        side = float(getattr(node_a, "side_length_m", 0.0) or 0.0)
        area = side * side if side > 0.0 else 1.0e-9
    if not (len_a > 0.0 and len_b > 0.0):
        side = float(getattr(node_a, "side_length_m", 0.0) or 0.0)
        len_a = len_b = max(0.5 * side, 1.0e-9)
    return area, len_a, len_b


def _edge_interface_conductance(node_a: Any, node_b: Any, edge: Any, default_bolted: float) -> float:
    """Interface conductance h_ref (W/m2K); 0 => bonded (same component)."""
    saved = getattr(edge, "interface_conductance_W_m2K", None)
    if saved is not None:
        return max(0.0, float(saved))
    component_a = str(getattr(node_a, "component_name", "") or "")
    component_b = str(getattr(node_b, "component_name", "") or "")
    if component_a == component_b:
        return 0.0  # bonded: same component (matches the octree builder's rule)
    return float(default_bolted)
