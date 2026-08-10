"""The cooler must not route cooling into zero-conductance CAD markers.

A role marker is a VISUAL annotation: its edges carry G = 0 W/K and exchange no
heat. _receiving_nodes_for_device classified an edge as a real thermal interface
with `"contact" in edge_text`, and a marker's source_metadata is
"cad_role_node_contact" -- so markers were promoted to explicit interfaces. They
then took the majority of the contact-area weight, which both dragged the tip
temperature the PT60 curve is evaluated at far below the true cold tip AND applied
that share of the cooling to nodes conducting nowhere.

Measured on no_mli_high_res: 4 marker cells held 72.1% of the weight, the tip read
33.75 K instead of 49.46 K, and delivered cooling collapsed to 2.9 W of an
available 30.2 W -- against 10.5 W of heater power, so the body warmed without
bound for 17,600 s.
"""

from __future__ import annotations

from graph_visualizer.cryocooler import _receiving_nodes_for_device
from graph_visualizer.models import EdgeProperties, NodeProperties, ThermalGraphModel


def _model() -> ThermalGraphModel:
    model = ThermalGraphModel()
    for node_id, component, cryo in (
        (1, "COLDHEAD", True),    # the cooler itself
        (2, "STRAP", False),      # real conducting interface
        (3, "COLDHEAD", False),   # same component, ordinary edge -> skipped
        (4, "COO_MARK", False),   # zero-conductance CAD role marker
    ):
        node = NodeProperties(node_id=node_id, coord=(node_id, 0, 0), component_name=component)
        node.has_cryocooler = cryo
        model.nodes[node_id] = node
    model.edges = {
        # real bolted interface, conducts
        (1, 2): EdgeProperties(1, 2, 3.5, "auto", "e1", "uncertain_contact", 4.0e-4, 5e-3, "low", []),
        # same component, not an explicit interface -> skipped by the component rule
        (1, 3): EdgeProperties(1, 3, 8.0, "auto", "e2", "internal_conduction", 9.0e-4, 5e-3, "high", []),
        # the trap: "cad_role_node_contact" contains "contact", and G = 0
        (1, 4): EdgeProperties(1, 4, 0.0, "cad_role_node_contact", "e3", "role_node_contact",
                               9.9e-3, 0.0, "medium", []),
    }
    return model


def test_zero_conductance_marker_is_not_a_receiving_node() -> None:
    ids, areas, _explicit = _receiving_nodes_for_device(_model(), "COLDHEAD", (1,), {1, 2, 3, 4})
    assert 4 not in ids, "a G=0 CAD marker must never receive cooling"
    assert 4 not in areas, "and must never take contact-area weight"


def test_the_real_interface_still_receives() -> None:
    ids, areas, _explicit = _receiving_nodes_for_device(_model(), "COLDHEAD", (1,), {1, 2, 3, 4})
    assert 2 in ids
    assert areas[2] == 4.0e-4


def test_marker_does_not_dominate_the_weighting() -> None:
    """The marker's 9.9e-3 m2 dwarfs the real 4.0e-4 m2 strap: if it were counted it
    would take 96% of the weight, which is exactly how 72.1% of the real cooler's
    capacity ended up going nowhere."""
    _ids, areas, _explicit = _receiving_nodes_for_device(_model(), "COLDHEAD", (1,), {1, 2, 3, 4})
    assert sum(areas.values()) == 4.0e-4


def test_dc_grounding_uses_the_lift_curve_slope_not_a_fixed_tip() -> None:
    """1000 W/K gave a cryocooler-mounted heater a DC gain ~170x too small, so the
    controller believed heat dumped there was free."""
    from graph_visualizer.modal_reduction import cryocooler_ground_conductance_W_K

    at_50 = cryocooler_ground_conductance_W_K(50.0)
    assert 0.9 < at_50 < 1.3, f"PT60 dQ/dT at 50 K is ~1.07 W/K, got {at_50}"
    # Stiffer when colder (the curve steepens), and always positive so the DC
    # operator stays non-singular even below the 27.669 K floor.
    assert cryocooler_ground_conductance_W_K(30.0) > at_50
    assert cryocooler_ground_conductance_W_K(80.0) < at_50
    assert cryocooler_ground_conductance_W_K(5.0) > 0.0
