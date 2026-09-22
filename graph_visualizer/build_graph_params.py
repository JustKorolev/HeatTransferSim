"""The build parameters the Build Graph tab exposes, and the argv they produce.

Kept separate from the tab, and free of Qt, so the part that can be wrong
silently -- the mapping from a form to a command line -- is testable without a
display.

Two properties matter and are both pinned by tests:

* every field here names a REAL option on ``octree_graph.cli.build_parser``, so a
  renamed flag fails loudly instead of the tab quietly passing something the
  builder ignores;
* the argv this produces parses back to the values the form held, so what the
  user sees is what the builder runs.

The defaults are the ones from the command that actually produced
STEP_FULL_MEDIUM, not the CLI's own defaults. The CLI has to stay conservative
for arbitrary input; a user opening this tab is pointing it at a cryostat
assembly, and starting them at the settings that worked is more useful than
starting them at the settings that are safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Where the tab looks for assemblies, relative to the working directory.
MESH_ROOT = "meshes"
STEP_SUFFIXES = (".step", ".stp")


@dataclass(frozen=True)
class BuildField:
    """One editable parameter.

    ``dest`` is the argparse destination, which is what ties this to the real
    parser. ``option`` is the flag to emit -- they differ for several of these
    (``--low-k-refine-threshold-w-mk`` sets ``low_conductivity_refine_threshold``).
    """

    dest: str
    option: str
    label: str
    kind: str  # "float" | "int" | "text" | "bool"
    default: Any
    tooltip: str = ""
    minimum: float = 0.0
    maximum: float = 1.0e12
    step: float = 1.0


#: Grouped the way the build actually proceeds: what to build, how finely, how
#: hard to work, what to refine, how contact is decided, and which parts are
#: heaters and sensors.
BUILD_FIELDS: tuple[tuple[str, tuple[BuildField, ...]], ...] = (
    (
        "Octree resolution",
        (
            BuildField(
                "min_cell_size_mm", "--min-cell-size-mm", "min cell size mm", "float", 10.0,
                "Smallest cell the octree may create. The floor on spatial resolution, and "
                "the dominant control on cell count -- halving it can multiply the cells by "
                "eight.",
                minimum=0.001, maximum=1.0e6, step=1.0,
            ),
            BuildField(
                "max_cell_size_mm", "--max-cell-size-mm", "max cell size mm", "float", 20.0,
                "Largest cell allowed. Caps how coarse empty or uniform regions may go.",
                minimum=0.001, maximum=1.0e6, step=1.0,
            ),
            BuildField(
                "max_depth", "--max-depth", "max depth", "int", 10,
                "Hard limit on subdivision levels, whatever the size limits would allow. "
                "The backstop against a runaway refine.",
                minimum=1, maximum=24, step=1,
            ),
            BuildField(
                "samples_per_cell", "--samples-per-cell", "samples per cell", "int", 27,
                "Point-in-solid samples per cell used to decide occupancy. 27 is a 3x3x3 "
                "grid. More is slower and less prone to missing thin features.",
                minimum=1, maximum=1000, step=1,
            ),
            BuildField(
                "max_leaf_cells", "--max-leaf-cells", "max leaf cells", "int", 2_000_000,
                "Abort if the octree would exceed this many leaves. A guard against a "
                "build that would not fit in memory -- raise it deliberately, not by "
                "reflex.",
                minimum=1000, maximum=100_000_000, step=100_000,
            ),
        ),
    ),
    (
        "STEP tessellation",
        (
            BuildField(
                "step_deflection_mm", "--step-deflection-mm", "deflection mm", "float", 1.5,
                "How far the tessellated triangles may deviate from the true B-rep surface. "
                "Smaller is a truer solid and many more triangles; the whole build scales "
                "with triangle count.",
                minimum=0.001, maximum=100.0, step=0.1,
            ),
        ),
    ),
    (
        "Workers",
        (
            BuildField(
                "voxel_workers", "--voxel-workers", "voxel workers", "int", 8,
                "Processes classifying cells. 1 is sequential; 0 auto-selects "
                "conservatively. Each worker gets its own COPY of the triangle data, so "
                "raise this only if there is memory to spare.",
                minimum=0, maximum=256, step=1,
            ),
            BuildField(
                "voxel_worker_memory_fraction", "--voxel-worker-memory-fraction",
                "worker memory fraction", "float", 0.8,
                "Share of available memory the workers may take between them, which is "
                "what bounds how many are actually started.",
                minimum=0.05, maximum=1.0, step=0.05,
            ),
        ),
    ),
    (
        "Refinement",
        (
            BuildField(
                "boundary_refine", "--boundary-refine", "boundary refine", "bool", False,
                "Subdivide cells straddling a component boundary. Off in the reference "
                "build: with a fine enough floor it multiplies cells for little gain.",
            ),
            BuildField(
                "low_conductivity_refine_threshold", "--low-k-refine-threshold-w-mk",
                "low-k refine below W/mK", "float", 10.0,
                "Refine around materials conducting worse than this. Insulators set the "
                "gradients, so resolving them matters more than resolving the metal.",
                minimum=0.0, maximum=10_000.0, step=1.0,
            ),
            BuildField(
                "surface_complexity_refine_threshold", "--surface-complexity-refine-threshold",
                "surface complexity", "int", 0,
                "Refine cells holding more than this many surfaces. 0 disables it.",
                minimum=0, maximum=100_000, step=1,
            ),
            BuildField(
                "multi_surface_refine_count", "--multi-surface-refine-count",
                "multi-surface count", "int", 0,
                "Refine cells touching at least this many distinct surfaces. 0 disables it.",
                minimum=0, maximum=1000, step=1,
            ),
            BuildField(
                "crowded_component_refine_count", "--crowded-component-refine-count",
                "crowded component count", "int", 0,
                "Refine where a cell's padded bounds overlap at least this many "
                "components, to preserve small air gaps. 0 disables it.",
                minimum=0, maximum=1000, step=1,
            ),
        ),
    ),
    (
        "Contact detection",
        (
            BuildField(
                "contact_detection_distance_mm", "--contact-detection-distance-mm",
                "detection distance mm", "float", 2.0,
                "How far apart two parts may be and still be treated as touching. This is "
                "the single most consequential thermal setting here: it decides which "
                "conduction paths exist at all.",
                minimum=0.0, maximum=1000.0, step=0.1,
            ),
            BuildField(
                "contact_gap_tolerance_mm", "--contact-gap-tolerance-mm",
                "gap tolerance mm", "float", 0.2,
                "Gap treated as closed when computing contact area, absorbing CAD "
                "imprecision between nominally mating faces.",
                minimum=0.0, maximum=100.0, step=0.01,
            ),
        ),
    ),
    (
        "Heaters and sensors",
        (
            BuildField(
                "heater_name_substring", "--heater-name-substring", "heater substring", "text",
                "SAFE-HEATER",
                "Components whose name contains this are heaters. Matching is by SUBSTRING, "
                "not pattern. Separate several with commas.\n\nLeave blank and the graph "
                "has no heaters at all -- nothing to control.",
            ),
            BuildField(
                "sensor_name_substring", "--sensor-name-substring", "sensor substring", "text",
                "COO-0001-P0003",
                "Components whose name contains this are sensors. Separate several with "
                "commas.\n\nLeave blank and there is nothing to regulate.",
            ),
            BuildField(
                "max_heater_sensor_pair_distance_mm", "--max-heater-sensor-pair-distance-mm",
                "max pair distance mm", "float", 50.0,
                "How far a sensor may be from a heater and still be paired with it.",
                minimum=0.0, maximum=10_000.0, step=1.0,
            ),
            BuildField(
                "max_heaters_per_sensor", "--max-heaters-per-sensor", "max heaters per sensor",
                "int", 2,
                "How many heaters one sensor may be assigned to.",
                minimum=1, maximum=64, step=1,
            ),
            BuildField(
                "role_refine_distance_mm", "--role-refine-distance-mm", "role refine mm",
                "float", 20.0,
                "Extra refinement within this distance of a heater or sensor, where the "
                "gradients are steepest and the controller's own measurements live.",
                minimum=0.0, maximum=10_000.0, step=1.0,
            ),
            BuildField(
                "role_refine_max_depth", "--role-refine-max-depth", "role refine max depth",
                "int", 10,
                "Depth cap for that local refinement.",
                minimum=1, maximum=24, step=1,
            ),
        ),
    ),
)


#: Row keys for the tab's own controls -- the ones that are not BUILD_FIELDS.
#: Prefixed ``build_``, because a tutorial's "Show me" resolves a bare row key
#: against the first tab that offers it and this tab is first: an unprefixed
#: ``graph_name`` here would shadow the simulation panel's row of that name.
ROW_ASSEMBLY_FOLDER = "build_assembly_folder"
ROW_GRAPH_NAME = "build_graph_name"
ROW_OUTPUT_ROOT = "build_output_root"
ROW_START = "build_start"
ROW_STOP = "build_stop"

EXTRA_HELP_ROWS: tuple[str, ...] = (
    ROW_ASSEMBLY_FOLDER,
    ROW_GRAPH_NAME,
    ROW_OUTPUT_ROOT,
    ROW_START,
    ROW_STOP,
)


def help_row_keys() -> set[str]:
    """Every row key the Build Graph tab offers to the help index.

    One source of truth, so a tutorial step or a shadowing check cannot be
    written against a key list that has drifted from the tab.
    """
    return {field.dest for field in all_fields()} | set(EXTRA_HELP_ROWS)


def all_fields() -> tuple[BuildField, ...]:
    return tuple(field for _section, fields in BUILD_FIELDS for field in fields)


def validate_against_parser(parser: Any) -> list[str]:
    """Problems found by checking every field against the real parser.

    Returns a list of human-readable complaints; empty means the spec and the CLI
    agree. This is what stops the tab from silently passing a flag the builder no
    longer has, or setting a dest the builder does not read.
    """
    actions = {action.dest: action for action in parser._actions}
    option_to_dest = {
        option: action.dest for action in parser._actions for option in action.option_strings
    }
    problems: list[str] = []
    for field in all_fields():
        action = actions.get(field.dest)
        if action is None:
            problems.append(f"{field.dest}: no such parser destination")
            continue
        if field.option not in option_to_dest:
            problems.append(f"{field.dest}: parser has no option {field.option}")
        elif option_to_dest[field.option] != field.dest:
            problems.append(
                f"{field.option} sets {option_to_dest[field.option]}, not {field.dest}"
            )
    return problems


def build_argv(
    mesh_dir: str,
    graph_name: str,
    output_root: str,
    values: dict[str, Any],
) -> list[str]:
    """The argv for a build, from the form's values.

    Emits only what the form holds, in a fixed order, so a command can be read
    back and compared against what was asked for. Blank text fields are omitted
    rather than passed empty: an empty --heater-name-substring would match every
    component, which is the opposite of leaving it unset.
    """
    argv: list[str] = [
        "--mesh-dir", str(mesh_dir),
        "--graph-name", str(graph_name),
        "--output-root", str(output_root),
    ]
    for field in all_fields():
        if field.dest not in values:
            continue
        value = values[field.dest]
        if field.kind == "bool":
            # BooleanOptionalAction: the negative form is a distinct flag.
            argv.append(field.option if value else _negated(field.option))
        elif field.kind == "text":
            for piece in _split_substrings(value):
                argv += [field.option, piece]
        else:
            argv += [field.option, _format_number(value, field.kind)]
    return argv


def _negated(option: str) -> str:
    return option.replace("--", "--no-", 1)


def _split_substrings(value: Any) -> list[str]:
    """Comma-separated text into repeated flag values, blanks dropped."""
    return [piece.strip() for piece in str(value or "").split(",") if piece.strip()]


def _format_number(value: Any, kind: str) -> str:
    if kind == "int":
        return str(int(round(float(value))))
    # %g so 10.0 is passed as "10" rather than "10.000000", which is what a person
    # comparing the command against their notes expects to see.
    return f"{float(value):g}"


def default_values() -> dict[str, Any]:
    return {field.dest: field.default for field in all_fields()}


def step_folders(mesh_root: Any) -> list[str]:
    """Subfolders of ``meshes/`` that hold a STEP file, newest first.

    Only folders with something to build are offered: listing one that cannot be
    built and failing on the button is worse than not listing it.
    """
    from pathlib import Path

    root = Path(mesh_root)
    if not root.is_dir():
        return []
    found = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if any(
            item.is_file() and item.suffix.lower() in STEP_SUFFIXES for item in child.iterdir()
        ):
            found.append(child)
    found.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [path.name for path in found]
