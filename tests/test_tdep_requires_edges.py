"""T-dependent properties must never silently degrade to a zero Laplacian.

The low-memory (nodes.csv) loader does not populate ``model.edges`` -- it assumes
conduction always comes from the prebuilt L. That assumption breaks when
``use_temperature_dependent_properties`` is on, because L(T) is rebuilt from
model.edges each step. With no edges the rebuilt Laplacian is all zeros, so every
node is thermally isolated: heaters cook their own deposition cells to thousands
of K, sensors never move off their initial temperature, and the controller winds
up to saturation forever. That is exactly what no_mli_high_res did -- 99.4% of its
heat capacity sat at exactly its start temperature for the whole run.
"""

from __future__ import annotations

import numpy as np
import pytest

from graph_visualizer.temperature_dependent_properties import TemperatureDependentOperator


def _edgeless_operator(n: int) -> TemperatureDependentOperator:
    empty_i = np.zeros(0, dtype=int)
    op = TemperatureDependentOperator.__new__(TemperatureDependentOperator)
    op.n = n
    op.edge_i = empty_i
    op.edge_j = empty_i
    return op


def test_edgeless_operator_refuses_to_build_a_zero_laplacian() -> None:
    op = _edgeless_operator(1000)
    with pytest.raises(ValueError, match="all-zero Laplacian"):
        op.laplacian(np.full(1000, 40.0))


def test_single_node_model_is_still_allowed() -> None:
    op = _edgeless_operator(1)
    L = op.laplacian(np.array([40.0]))
    assert L.shape == (1, 1)
    assert L.nnz == 0
