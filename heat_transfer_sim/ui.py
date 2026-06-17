"""PyVista/Qt UI for the two-cube thermal simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .color_scale import TemperatureColorScale
from .cube import Cube, zero_heater
from .plotting import plot_simulation_result
from .settings import ParameterStore
from .simulation import SimulationResult, TwoCubeSimulation, make_pulsed_heater


@dataclass
class UIParameters:
    cube_1_side_length: float = 0.1
    cube_1_mass: float = 0.25
    cube_1_specific_heat: float = 900.0
    cube_1_conductivity: float = 205.0
    cube_1_initial_temperature: float = 20.0
    cube_1_min_x: float = -0.1
    cube_1_min_y: float = -0.05
    cube_1_min_z: float = -0.05
    cube_1_heater_power: float = 30.0
    cube_1_heater_stop_time: float = 120.0

    cube_2_side_length: float = 0.1
    cube_2_mass: float = 0.25
    cube_2_specific_heat: float = 900.0
    cube_2_conductivity: float = 205.0
    cube_2_initial_temperature: float = 20.0
    cube_2_min_x: float = 0.0
    cube_2_min_y: float = -0.05
    cube_2_min_z: float = -0.05
    cube_2_heater_power: float = 0.0
    cube_2_heater_stop_time: float = 0.0

    interface_resistance: float = 0.0
    contact_area: float = 0.0
    simulation_duration: float = 600.0
    num_points: int = 1001
    simulated_seconds_per_update: float = 1.0
    solver_max_step: float = 0.05
    display_update_interval_ms: float = 1000.0


class ThermalUI:
    """Interactive Qt/PyVista viewer and parameter panel."""

    def __init__(self) -> None:
        self._load_ui_dependencies()
        self.parameter_store = ParameterStore(UIParameters, migrate=self._migrate_parameters)
        self.params = self.parameter_store.load()
        self.result = None
        self.live_simulation = None
        self.live_time = 0.0
        self.live_history: dict[str, list[float]] = {}
        self.cube_actors: dict[str, Any] = {}
        self.center_marker_actors: list[Any] = []
        self.label_actors: list[Any] = []
        self.temperature_color_scale = TemperatureColorScale(20.0, 30.0)
        self.inputs: dict[str, Any] = {}

        self.app = self.QtWidgets.QApplication.instance() or self.QtWidgets.QApplication([])
        self.window = self.QtWidgets.QMainWindow()
        self.window.setWindowTitle("Two-Cube Thermal RC Simulation")
        self._build_layout()
        self.run_simulation()

    def _load_ui_dependencies(self) -> None:
        try:
            import pyvista as pv
            from pyvistaqt import QtInteractor
            from qtpy import QtCore, QtWidgets
        except ImportError as exc:
            raise RuntimeError(
                "ThermalUI requires pyvista, pyvistaqt, and a Qt binding. "
                "Install the UI dependencies with: pip install -r requirements.txt"
            ) from exc

        self.pv = pv
        self.QtInteractor = QtInteractor
        self.QtCore = QtCore
        self.QtWidgets = QtWidgets

    def _build_layout(self) -> None:
        central = self.QtWidgets.QWidget()
        layout = self.QtWidgets.QHBoxLayout(central)

        self.controls = self.QtWidgets.QScrollArea()
        self.controls.setWidgetResizable(True)
        controls_content = self.QtWidgets.QWidget()
        self.controls.setWidget(controls_content)
        form = self.QtWidgets.QFormLayout(controls_content)
        self._add_numeric_inputs(form)

        run_button = self.QtWidgets.QPushButton("Run Full Simulation")
        run_button.clicked.connect(self.run_simulation)
        form.addRow(run_button)

        live_button = self.QtWidgets.QPushButton("Start Live")
        live_button.clicked.connect(self.start_live_simulation)
        form.addRow(live_button)

        pause_button = self.QtWidgets.QPushButton("Pause Live")
        pause_button.clicked.connect(self.pause_live_simulation)
        form.addRow(pause_button)

        reset_button = self.QtWidgets.QPushButton("Reset Live")
        reset_button.clicked.connect(self.reset_live_simulation)
        form.addRow(reset_button)

        plot_button = self.QtWidgets.QPushButton("Open Matplotlib Plots")
        plot_button.clicked.connect(self.open_plots)
        form.addRow(plot_button)

        self.status_label = self.QtWidgets.QLabel("")
        form.addRow(self.status_label)

        self.plotter = self.QtInteractor(central)
        self.plotter.set_background("white")

        self.live_timer = self.QtCore.QTimer(self.window)
        self.live_timer.timeout.connect(self.step_live_simulation)

        layout.addWidget(self.controls, 0)
        layout.addWidget(self.plotter.interactor, 1)
        self.window.setCentralWidget(central)
        self.window.resize(1200, 760)

    def _add_numeric_inputs(self, form: Any) -> None:
        for section_title, field_names in self._parameter_sections():
            section_label = self.QtWidgets.QLabel(section_title)
            section_label.setStyleSheet("font-weight: 700; margin-top: 10px;")
            form.addRow(section_label)

            for field_name in field_names:
                value = getattr(self.params, field_name)
                spin_box = self.QtWidgets.QDoubleSpinBox()
                spin_box.setDecimals(6)
                spin_box.setRange(*self._range_for(field_name))
                spin_box.setValue(float(value))
                spin_box.setSingleStep(self._step_for(field_name))
                spin_box.setToolTip(self._tooltip_for(field_name))
                spin_box.valueChanged.connect(self._handle_parameter_change)
                self.inputs[field_name] = spin_box

                label = self.QtWidgets.QLabel(self._label_for(field_name))
                label.setToolTip(self._tooltip_for(field_name))
                form.addRow(label, spin_box)

    def _read_params(self) -> UIParameters:
        values = {}
        for field_name, widget in self.inputs.items():
            value = widget.value()
            if field_name == "num_points":
                value = int(value)
            values[field_name] = value
        return UIParameters(**values)

    def _save_parameters_from_inputs(self) -> None:
        if not self.inputs:
            return
        self.params = self._read_params()
        self.parameter_store.save(self.params)

    def _handle_parameter_change(self, *_: Any) -> None:
        """Persist edited parameters and refresh cube geometry immediately."""
        if not self.inputs:
            return
        self._save_parameters_from_inputs()
        self.pause_live_simulation()
        try:
            preview_simulation = self._build_simulation()
            self.live_simulation = None
            self.result = None
            self.temperature_color_scale = TemperatureColorScale.from_temperatures(
                [
                    preview_simulation.cube_1.temperature,
                    preview_simulation.cube_2.temperature,
                ]
            )
            self._build_cube_scene(preview_simulation, reset_camera=False)
            self.status_label.setText(
                "Parameters saved. Geometry preview updated. "
                f"A = {self._safe_contact_area_text(preview_simulation)}"
            )
        except Exception as exc:
            self.status_label.setText(str(exc))

    def _build_simulation(self) -> TwoCubeSimulation:
        self.params = self._read_params()
        self.parameter_store.save(self.params)
        cube_1 = Cube(
            name="Cube 1",
            side_length=self.params.cube_1_side_length,
            mass=self.params.cube_1_mass,
            specific_heat=self.params.cube_1_specific_heat,
            thermal_conductivity=self.params.cube_1_conductivity,
            temperature=self.params.cube_1_initial_temperature,
            position=self._center_from_min_corner(
                self.params.cube_1_min_x,
                self.params.cube_1_min_y,
                self.params.cube_1_min_z,
                self.params.cube_1_side_length,
            ),
            heater_power=make_pulsed_heater(
                self.params.cube_1_heater_power, 0.0, self.params.cube_1_heater_stop_time
            ),
        )
        cube_2 = Cube(
            name="Cube 2",
            side_length=self.params.cube_2_side_length,
            mass=self.params.cube_2_mass,
            specific_heat=self.params.cube_2_specific_heat,
            thermal_conductivity=self.params.cube_2_conductivity,
            temperature=self.params.cube_2_initial_temperature,
            position=self._center_from_min_corner(
                self.params.cube_2_min_x,
                self.params.cube_2_min_y,
                self.params.cube_2_min_z,
                self.params.cube_2_side_length,
            ),
            heater_power=(
                make_pulsed_heater(
                    self.params.cube_2_heater_power,
                    0.0,
                    self.params.cube_2_heater_stop_time,
                )
                if self.params.cube_2_heater_power > 0.0
                else zero_heater
            ),
        )
        return TwoCubeSimulation(
            cube_1=cube_1,
            cube_2=cube_2,
            interface_resistance=self.params.interface_resistance,
            contact_area=self.params.contact_area if self.params.contact_area > 0.0 else None,
        )

    def run_simulation(self) -> None:
        try:
            simulation = self._build_simulation()
            self.result = simulation.solve(
                (0.0, self.params.simulation_duration),
                num_points=max(2, self.params.num_points),
            )
            self.temperature_color_scale = self._color_scale_from_result(self.result)
            self._draw_cubes(simulation, reset_camera=True)
            self.status_label.setText(
                "Final: T1 = "
                f"{self.result.final_temperature_1:.2f}, "
                f"T2 = {self.result.final_temperature_2:.2f}, "
                f"A = {simulation.effective_contact_area:.4g} m^2, "
                f"R12 = {self.result.contact_resistance:.4g} K/W"
            )
        except Exception as exc:
            self.status_label.setText(str(exc))

    def start_live_simulation(self) -> None:
        try:
            self.reset_live_simulation(draw=True)
            interval_ms = max(10, int(self.params.display_update_interval_ms))
            self.live_timer.start(interval_ms)
        except Exception as exc:
            self.status_label.setText(str(exc))

    def pause_live_simulation(self) -> None:
        self.live_timer.stop()

    def reset_live_simulation(self, draw: bool = True) -> None:
        self.live_timer.stop()
        self.live_simulation = self._build_simulation()
        self.live_time = 0.0
        self.temperature_color_scale = self._predict_live_color_scale()
        cube_1 = self.live_simulation.cube_1
        cube_2 = self.live_simulation.cube_2
        self.live_history = {
            "time": [0.0],
            "temperature_1": [cube_1.temperature],
            "temperature_2": [cube_2.temperature],
            "temperature_difference": [cube_1.temperature - cube_2.temperature],
            "heat_flow_1_to_2": [
                cube_1.heat_exchange_with(cube_2, self.live_simulation.contact_resistance)
            ],
            "heater_power_1": [cube_1.heater_power_at(0.0)],
            "heater_power_2": [cube_2.heater_power_at(0.0)],
        }
        self.result = self._result_from_live_history()
        if draw:
            self._build_cube_scene(self.live_simulation, reset_camera=True)
            self.status_label.setText(self._live_status_text())

    def step_live_simulation(self) -> None:
        if self.live_simulation is None:
            self.reset_live_simulation(draw=False)
        if self.live_time >= self.params.simulation_duration:
            self.pause_live_simulation()
            return

        time_step = min(
            self.params.simulated_seconds_per_update,
            self.params.simulation_duration - self.live_time,
        )
        step_result = self.live_simulation.step(
            self.live_time,
            time_step,
            max_solver_step=self.params.solver_max_step,
        )
        self.live_time = float(step_result.time[-1])
        cube_1 = self.live_simulation.cube_1
        cube_2 = self.live_simulation.cube_2
        self.live_history["time"].append(self.live_time)
        self.live_history["temperature_1"].append(cube_1.temperature)
        self.live_history["temperature_2"].append(cube_2.temperature)
        self.live_history["temperature_difference"].append(cube_1.temperature - cube_2.temperature)
        self.live_history["heat_flow_1_to_2"].append(
            cube_1.heat_exchange_with(cube_2, self.live_simulation.contact_resistance)
        )
        self.live_history["heater_power_1"].append(cube_1.heater_power_at(self.live_time))
        self.live_history["heater_power_2"].append(cube_2.heater_power_at(self.live_time))
        self.result = self._result_from_live_history()
        self._update_cube_colors(self.live_simulation)
        self.status_label.setText(self._live_status_text())

    def _result_from_live_history(self) -> SimulationResult:
        if self.live_simulation is None:
            raise RuntimeError("Live simulation has not been initialized")
        return SimulationResult(
            time=np.array(self.live_history["time"]),
            temperature_1=np.array(self.live_history["temperature_1"]),
            temperature_2=np.array(self.live_history["temperature_2"]),
            temperature_difference=np.array(self.live_history["temperature_difference"]),
            heat_flow_1_to_2=np.array(self.live_history["heat_flow_1_to_2"]),
            heater_power_1=np.array(self.live_history["heater_power_1"]),
            heater_power_2=np.array(self.live_history["heater_power_2"]),
            contact_resistance=self.live_simulation.contact_resistance,
        )

    def _live_status_text(self) -> str:
        if self.live_simulation is None or self.result is None:
            return ""
        return (
            f"Live t = {self.live_time:.1f} s, "
            f"T1 = {self.result.final_temperature_1:.2f}, "
            f"T2 = {self.result.final_temperature_2:.2f}, "
            f"A = {self.live_simulation.effective_contact_area:.4g} m^2, "
            f"R12 = {self.live_simulation.contact_resistance:.4g} K/W"
        )

    @staticmethod
    def _safe_contact_area_text(simulation: TwoCubeSimulation) -> str:
        try:
            return f"{simulation.effective_contact_area:.4g} m^2"
        except ValueError:
            return "no touching face overlap"

    def _draw_cubes(self, simulation: TwoCubeSimulation, reset_camera: bool) -> None:
        self._build_cube_scene(simulation, reset_camera=reset_camera)

    def _build_cube_scene(self, simulation: TwoCubeSimulation, reset_camera: bool) -> None:
        camera_position = self.plotter.camera_position if not reset_camera else None
        self.plotter.clear()
        self.cube_actors = {}
        self.center_marker_actors = []
        self.label_actors = []
        self._add_contact_patch(simulation)
        for cube in (simulation.cube_1, simulation.cube_2):
            mesh = self.pv.Box(bounds=cube.flat_bounds)
            actor = self.plotter.add_mesh(
                mesh,
                color=self.temperature_color_scale.hex_color(cube.temperature),
                show_edges=True,
                edge_color="black",
            )
            self.cube_actors[cube.name] = actor
            center_actor = self.plotter.add_points(
                self._point_array(cube.position),
                color="black",
                point_size=12,
                render_points_as_spheres=True,
            )
            self.center_marker_actors.append(center_actor)
            label_actor = self.plotter.add_point_labels(
                self._point_array(cube.position),
                [self._cube_position_label(cube)],
                font_size=14,
                text_color="black",
                shape_opacity=0.0,
                always_visible=True,
            )
            self.label_actors.append(label_actor)

        self.plotter.add_axes()
        self.plotter.show_bounds(
            grid="front",
            location="outer",
            xlabel="x [m]",
            ylabel="y [m]",
            zlabel="z [m]",
        )
        if reset_camera:
            self.plotter.camera_position = "iso"
            self.plotter.reset_camera()
        elif camera_position is not None:
            self.plotter.camera_position = camera_position
        self.plotter.render()
        self._force_render_update()

    def _update_cube_colors(self, simulation: TwoCubeSimulation) -> None:
        if not self.cube_actors:
            self._build_cube_scene(simulation, reset_camera=True)
            return

        for cube in (simulation.cube_1, simulation.cube_2):
            actor = self.cube_actors.get(cube.name)
            if actor is not None:
                actor.prop.color = self.temperature_color_scale.rgb_color(cube.temperature)
                actor.GetProperty().Modified()
        self.plotter.render()
        self._force_render_update()

    def _add_contact_patch(self, simulation: TwoCubeSimulation) -> None:
        patch_bounds = self._contact_patch_bounds(simulation)
        if patch_bounds is None:
            return
        patch_mesh = self.pv.Box(bounds=patch_bounds)
        self.plotter.add_mesh(
            patch_mesh,
            color="#ffd84d",
            opacity=0.55,
            show_edges=True,
            edge_color="#8a6a00",
        )

    @staticmethod
    def _contact_patch_bounds(
        simulation: TwoCubeSimulation,
    ) -> tuple[float, float, float, float, float, float] | None:
        cube_1_bounds = simulation.cube_1.bounds
        cube_2_bounds = simulation.cube_2.bounds
        thickness = max(
            1.0e-6,
            0.01 * min(simulation.cube_1.side_length, simulation.cube_2.side_length),
        )

        for normal_axis in range(3):
            first_min, first_max = cube_1_bounds[normal_axis]
            second_min, second_max = cube_2_bounds[normal_axis]
            if abs(first_max - second_min) <= 1.0e-9:
                contact_coordinate = first_max
            elif abs(second_max - first_min) <= 1.0e-9:
                contact_coordinate = second_max
            else:
                continue

            ranges = [[0.0, 0.0] for _ in range(3)]
            ranges[normal_axis] = [
                contact_coordinate - 0.5 * thickness,
                contact_coordinate + 0.5 * thickness,
            ]
            overlap_is_positive = True
            for tangent_axis in range(3):
                if tangent_axis == normal_axis:
                    continue
                lower = max(cube_1_bounds[tangent_axis][0], cube_2_bounds[tangent_axis][0])
                upper = min(cube_1_bounds[tangent_axis][1], cube_2_bounds[tangent_axis][1])
                if upper <= lower:
                    overlap_is_positive = False
                    break
                ranges[tangent_axis] = [lower, upper]
            if overlap_is_positive:
                return (
                    ranges[0][0],
                    ranges[0][1],
                    ranges[1][0],
                    ranges[1][1],
                    ranges[2][0],
                    ranges[2][1],
                )

        return None

    @staticmethod
    def _cube_position_label(cube: Cube) -> str:
        bounds = cube.flat_bounds
        return (
            f"{cube.name}\n"
            f"min=({bounds[0]:.3g}, {bounds[2]:.3g}, {bounds[4]:.3g}) m\n"
            f"max=({bounds[1]:.3g}, {bounds[3]:.3g}, {bounds[5]:.3g}) m"
        )

    def _predict_live_color_scale(self) -> TemperatureColorScale:
        preview_simulation = self._build_simulation()
        preview_result = preview_simulation.solve(
            (0.0, self.params.simulation_duration),
            num_points=max(2, self.params.num_points),
            update_cubes=False,
            max_step=self.params.solver_max_step,
        )
        return self._color_scale_from_result(preview_result)

    def _force_render_update(self) -> None:
        """Flush Qt/VTK rendering after geometry or color edits."""
        try:
            self.plotter.interactor.Render()
        except AttributeError:
            pass
        try:
            self.plotter.interactor.update()
        except AttributeError:
            pass

    @staticmethod
    def _point_array(point: tuple[float, float, float]) -> np.ndarray:
        """Return a PyVista-friendly one-point coordinate array."""
        return np.array([point], dtype=float)

    @staticmethod
    def _center_from_min_corner(
        min_x: float, min_y: float, min_z: float, side_length: float
    ) -> tuple[float, float, float]:
        half_side = 0.5 * side_length
        return (min_x + half_side, min_y + half_side, min_z + half_side)

    @staticmethod
    def _migrate_parameters(raw_settings: dict[str, Any]) -> dict[str, Any]:
        """Convert older center-coordinate settings to min-corner settings."""
        migrated = dict(raw_settings)
        for cube_prefix in ("cube_1", "cube_2"):
            side_length = float(migrated.get(f"{cube_prefix}_side_length", 0.1))
            half_side = 0.5 * side_length
            for axis in ("x", "y", "z"):
                old_center_field = f"{cube_prefix}_{axis}"
                new_min_field = f"{cube_prefix}_min_{axis}"
                if new_min_field not in migrated and old_center_field in migrated:
                    migrated[new_min_field] = float(migrated[old_center_field]) - half_side

        if "interface_resistance" not in migrated:
            contact_length = migrated.get("contact_length")
            contact_conductivity = migrated.get("contact_conductivity")
            contact_area = migrated.get("contact_area")
            if contact_length and contact_conductivity and contact_area:
                migrated["interface_resistance"] = (
                    float(contact_length)
                    / (float(contact_conductivity) * float(contact_area))
                )
        return migrated

    @staticmethod
    def _color_scale_from_result(result: SimulationResult) -> TemperatureColorScale:
        all_temperatures = np.concatenate([result.temperature_1, result.temperature_2])
        min_temperature = float(np.min(all_temperatures))
        max_temperature = float(np.max(all_temperatures))
        if max_temperature <= min_temperature:
            max_temperature = min_temperature + 1.0
        return TemperatureColorScale(min_temperature, max_temperature)

    def open_plots(self) -> None:
        if self.result is not None:
            plot_simulation_result(self.result, show=True)

    def show(self) -> None:
        self.window.show()
        self.app.exec_()

    @staticmethod
    def _parameter_sections() -> list[tuple[str, tuple[str, ...]]]:
        return [
            (
                "Size and Mass",
                (
                    "cube_1_side_length",
                    "cube_1_mass",
                    "cube_2_side_length",
                    "cube_2_mass",
                ),
            ),
            (
                "Thermal Properties",
                (
                    "cube_1_specific_heat",
                    "cube_1_conductivity",
                    "cube_1_initial_temperature",
                    "cube_2_specific_heat",
                    "cube_2_conductivity",
                    "cube_2_initial_temperature",
                ),
            ),
            (
                "Position and Contact Geometry",
                (
                    "cube_1_min_x",
                    "cube_1_min_y",
                    "cube_1_min_z",
                    "cube_2_min_x",
                    "cube_2_min_y",
                    "cube_2_min_z",
                    "interface_resistance",
                    "contact_area",
                ),
            ),
            (
                "Heater Inputs",
                (
                    "cube_1_heater_power",
                    "cube_1_heater_stop_time",
                    "cube_2_heater_power",
                    "cube_2_heater_stop_time",
                ),
            ),
            (
                "Simulation Controls",
                (
                    "simulation_duration",
                    "num_points",
                    "simulated_seconds_per_update",
                    "solver_max_step",
                    "display_update_interval_ms",
                ),
            ),
        ]

    @staticmethod
    def _label_for(field_name: str) -> str:
        labels = {
            "contact_area": "contact area override, 0 = auto [m^2]",
            "interface_resistance": "extra interface resistance [K/W]",
            "cube_1_side_length": "cube 1 side length [m]",
            "cube_1_mass": "cube 1 mass [kg]",
            "cube_1_specific_heat": "cube 1 specific heat [J/(kg K)]",
            "cube_1_conductivity": "cube 1 bulk conductivity [W/(m K)]",
            "cube_1_initial_temperature": "cube 1 initial temperature",
            "cube_1_min_x": "cube 1 min corner x [m]",
            "cube_1_min_y": "cube 1 min corner y [m]",
            "cube_1_min_z": "cube 1 min corner z [m]",
            "cube_1_heater_power": "cube 1 heater power [W]",
            "cube_1_heater_stop_time": "cube 1 heater stop time [s]",
            "cube_2_side_length": "cube 2 side length [m]",
            "cube_2_mass": "cube 2 mass [kg]",
            "cube_2_specific_heat": "cube 2 specific heat [J/(kg K)]",
            "cube_2_conductivity": "cube 2 bulk conductivity [W/(m K)]",
            "cube_2_initial_temperature": "cube 2 initial temperature",
            "cube_2_min_x": "cube 2 min corner x [m]",
            "cube_2_min_y": "cube 2 min corner y [m]",
            "cube_2_min_z": "cube 2 min corner z [m]",
            "cube_2_heater_power": "cube 2 heater power [W]",
            "cube_2_heater_stop_time": "cube 2 heater stop time [s]",
            "simulation_duration": "simulation duration [s]",
            "simulated_seconds_per_update": "simulated seconds per display update [s]",
            "solver_max_step": "max solver internal step [s]",
            "display_update_interval_ms": "display update interval [ms]",
            "num_points": "full-run plot points",
        }
        return labels.get(field_name, field_name.replace("_", " "))

    @staticmethod
    def _tooltip_for(field_name: str) -> str:
        tooltips = {
            "interface_resistance": (
                "Extra resistance for imperfect contact between faces. Total resistance "
                "is (s1/2)/(k1 A) + R_interface + (s2/2)/(k2 A). Use 0 for ideal contact."
            ),
            "contact_area": (
                "Set to 0 to compute area from overlapping touching faces. Enter a "
                "positive value to override the geometry-derived area."
            ),
            "cube_1_conductivity": (
                "Bulk conductivity of cube 1. The model uses it in the half-cube "
                "resistance term (s1/2)/(k1 A)."
            ),
            "cube_2_conductivity": (
                "Bulk conductivity of cube 2. The model uses it in the half-cube "
                "resistance term (s2/2)/(k2 A)."
            ),
            "simulated_seconds_per_update": (
                "How much simulated time advances each time the 3D display refreshes."
            ),
            "solver_max_step": (
                "Maximum adaptive ODE step inside each display update. Lower values can "
                "improve accuracy without slowing the visible refresh rate."
            ),
            "display_update_interval_ms": (
                "Real wall-clock delay between visual updates. 1000 ms means one screen "
                "update per second."
            ),
        }
        if "_min_" in field_name:
            return "Minimum-corner coordinate of the cube in meters."
        return tooltips.get(field_name, "")

    @staticmethod
    def _step_for(field_name: str) -> float:
        if "_min_" in field_name:
            return 0.01
        if "temperature" in field_name:
            return 1.0
        if "specific_heat" in field_name or "conductivity" in field_name:
            return 10.0
        if "duration" in field_name or "stop_time" in field_name:
            return 10.0
        if "power" in field_name:
            return 5.0
        if "num_points" in field_name:
            return 100.0
        if "update_interval" in field_name:
            return 100.0
        if "seconds_per_update" in field_name:
            return 1.0
        if "solver_max_step" in field_name:
            return 0.01
        return 0.01

    @staticmethod
    def _range_for(field_name: str) -> tuple[float, float]:
        if "_min_" in field_name:
            return (-1.0e9, 1.0e9)
        if field_name in {"simulated_seconds_per_update", "solver_max_step"}:
            return (1.0e-9, 1.0e9)
        if field_name == "display_update_interval_ms":
            return (10.0, 1.0e9)
        return (0.0, 1.0e9)
