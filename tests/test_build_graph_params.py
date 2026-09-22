"""The Build Graph tab's parameters, and the command line they produce.

The failure this guards is silent: a form that passes a flag the builder no
longer has, or sets a destination the builder does not read, produces a build
that runs happily with the wrong settings. So every field is checked against the
real parser, and every generated argv is parsed BACK and compared to what the
form held.
"""

from __future__ import annotations

import pytest

from graph_visualizer.build_graph_params import (
    BUILD_FIELDS,
    all_fields,
    build_argv,
    default_values,
    step_folders,
    validate_against_parser,
)
from octree_graph.cli import build_parser


# --------------------------------------------------------------------------- #
# The spec agrees with the CLI
# --------------------------------------------------------------------------- #
def test_every_field_names_a_real_parser_option() -> None:
    """The whole point. Several dests differ from their flag -- for instance
    --low-k-refine-threshold-w-mk sets low_conductivity_refine_threshold -- so
    this is not a formality."""
    problems = validate_against_parser(build_parser())
    assert not problems, "spec has drifted from the CLI:\n  " + "\n  ".join(problems)


def test_no_field_is_listed_twice() -> None:
    dests = [field.dest for field in all_fields()]
    assert len(dests) == len(set(dests)), "a parameter appears in two sections"


def test_every_field_has_a_label_and_a_tooltip() -> None:
    """These names are jargon; a field with no explanation is a field nobody
    touches, or touches wrongly."""
    for field in all_fields():
        assert field.label, f"{field.dest} has no label"
        assert len(field.tooltip) > 30, f"{field.dest} has no real tooltip"


def test_sections_are_non_empty_and_ordered_as_the_build_runs() -> None:
    titles = [title for title, _fields in BUILD_FIELDS]
    assert titles[0] == "Octree resolution"
    assert "Heaters and sensors" in titles
    for title, fields in BUILD_FIELDS:
        assert fields, f"section {title} is empty"


# --------------------------------------------------------------------------- #
# The argv round-trips
# --------------------------------------------------------------------------- #
def test_the_generated_argv_parses_back_to_the_same_values() -> None:
    """What the user sees in the form must be what the builder receives."""
    values = default_values()
    argv = build_argv("meshes/step_test", "MY_GRAPH", "graphs", values)
    args = build_parser().parse_args(argv)

    assert args.mesh_dir == "meshes/step_test"
    assert args.graph_name == "MY_GRAPH"
    assert args.output_root == "graphs"

    for field in all_fields():
        got = getattr(args, field.dest)
        if field.kind == "bool":
            assert bool(got) == bool(field.default), field.dest
        elif field.kind == "text":
            expected = [p.strip() for p in str(field.default).split(",") if p.strip()]
            assert got == (expected or None), field.dest
        elif field.kind == "int":
            assert int(got) == int(field.default), field.dest
        else:
            assert float(got) == pytest.approx(float(field.default)), field.dest


def test_the_defaults_match_the_command_that_built_step_full_medium() -> None:
    """These are the settings that produced a real graph, not the CLI's own
    conservative defaults. Pinned so a later edit has to be deliberate."""
    values = default_values()
    expected = {
        "min_cell_size_mm": 10.0,
        "max_cell_size_mm": 20.0,
        "max_depth": 10,
        "samples_per_cell": 27,
        "step_deflection_mm": 1.5,
        "voxel_workers": 8,
        "voxel_worker_memory_fraction": 0.8,
        "low_conductivity_refine_threshold": 10.0,
        "surface_complexity_refine_threshold": 0,
        "multi_surface_refine_count": 0,
        "crowded_component_refine_count": 0,
        "contact_detection_distance_mm": 2.0,
        "contact_gap_tolerance_mm": 0.2,
        "max_heater_sensor_pair_distance_mm": 50.0,
        "max_heaters_per_sensor": 2,
        "role_refine_distance_mm": 20.0,
        "role_refine_max_depth": 10,
        "max_leaf_cells": 2_000_000,
        "heater_name_substring": "SAFE-HEATER",
        "sensor_name_substring": "COO-0001-P0003",
        "boundary_refine": False,
    }
    for dest, want in expected.items():
        assert values[dest] == want, f"{dest}: {values[dest]!r} != {want!r}"


def test_boundary_refine_off_emits_the_negative_flag() -> None:
    """BooleanOptionalAction: the off state is a DIFFERENT flag, not an omission,
    and the reference build turns it off."""
    argv = build_argv("m", "g", "graphs", {"boundary_refine": False})
    assert "--no-boundary-refine" in argv
    assert "--boundary-refine" not in argv
    assert build_parser().parse_args(["--mesh-dir", "m", "--graph-name", "g", *argv[6:]]).boundary_refine is False

    argv_on = build_argv("m", "g", "graphs", {"boundary_refine": True})
    assert "--boundary-refine" in argv_on
    assert "--no-boundary-refine" not in argv_on


def test_several_substrings_become_repeated_flags() -> None:
    """The CLI appends, so two heaters need the flag twice."""
    argv = build_argv("m", "g", "graphs", {"heater_name_substring": "SAFE-HEATER, TRIM-HEATER"})
    args = build_parser().parse_args(argv)
    assert args.heater_name_substring == ["SAFE-HEATER", "TRIM-HEATER"]


def test_a_blank_substring_is_omitted_not_passed_empty() -> None:
    """An empty --heater-name-substring would substring-match EVERY component,
    which is the opposite of leaving it unset."""
    argv = build_argv("m", "g", "graphs", {"heater_name_substring": "   "})
    assert "--heater-name-substring" not in argv
    assert build_parser().parse_args(argv).heater_name_substring is None


def test_whole_numbers_are_not_passed_in_float_notation() -> None:
    """So a command copied out of the tab reads like the one in someone's notes."""
    argv = build_argv("m", "g", "graphs", {"min_cell_size_mm": 10.0, "max_depth": 10})
    assert "10" in argv
    assert "10.000000" not in argv


def test_an_unknown_value_is_ignored_rather_than_guessed() -> None:
    argv = build_argv("m", "g", "graphs", {"not_a_field": 1})
    build_parser().parse_args(argv)  # must still parse


# --------------------------------------------------------------------------- #
# Folder discovery
# --------------------------------------------------------------------------- #
def test_only_folders_holding_a_step_file_are_offered(tmp_path) -> None:
    """Listing a folder that cannot be built, then failing on the button, is worse
    than not listing it."""
    (tmp_path / "with_step").mkdir()
    (tmp_path / "with_step" / "a.step").write_text("ISO-10303-21;", encoding="utf-8")
    (tmp_path / "with_stp").mkdir()
    (tmp_path / "with_stp" / "b.STP").write_text("ISO-10303-21;", encoding="utf-8")
    (tmp_path / "mesh_only").mkdir()
    (tmp_path / "mesh_only" / "c.glb").write_bytes(b"glTF")
    (tmp_path / "empty").mkdir()

    found = step_folders(tmp_path)
    assert set(found) == {"with_step", "with_stp"}
    assert "mesh_only" not in found, "GLB is no longer an input format"
    assert "empty" not in found


def test_a_missing_mesh_root_is_empty_not_an_error(tmp_path) -> None:
    assert step_folders(tmp_path / "nope") == []


def test_the_shipped_sample_is_discoverable() -> None:
    """A fresh clone must have something in the dropdown."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "meshes"
    if not root.is_dir():
        pytest.skip("meshes/ not present")
    assert "step_test" in step_folders(root)
