"""The shared control panel, exercised against a Qt stub.

PySide6 is not installed in CI, so the panel is built against hand-written stub
widgets. Each widget type is a DISTINCT class: the panel dispatches on
``isinstance``/``hasattr``, so collapsing them into one do-everything stub would
make every branch look correct and test nothing.

What matters here is what a screenshot would otherwise have to prove: both tabs
build the same rows in the same order, the mode only hides rows (never reorders
or omits them), and a set/read round-trip preserves fields the panel has no
widget for.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from graph_visualizer.simulation_controls_panel import (
    MODE_HEADLESS,
    MODE_LIVE,
    PID_QP_LABEL,
    SimulationControlsPanel,
)
from graph_visualizer.simulation_parameters import SimulationParameters


# --- Qt stubs ------------------------------------------------------------- #
class _Signal:
    def __init__(self) -> None:
        self.slots: list = []

    def connect(self, slot) -> None:
        self.slots.append(slot)

    def emit(self, *args) -> None:
        for slot in list(self.slots):
            slot(*args)


class _Widget:
    def __init__(self, *args, **kwargs) -> None:
        self.visible = True
        self.enabled = True
        self.tooltip = ""
        self._signals_blocked = False

    def setVisible(self, value) -> None:
        self.visible = bool(value)

    def setEnabled(self, value) -> None:
        self.enabled = bool(value)

    def isEnabled(self) -> bool:
        return self.enabled

    def setToolTip(self, text) -> None:
        self.tooltip = text

    def blockSignals(self, value) -> None:
        self._signals_blocked = bool(value)

    def setStyleSheet(self, _text) -> None:
        pass

    def setMinimumWidth(self, _value) -> None:
        pass

    def setMaximumWidth(self, _value) -> None:
        pass

    def setMaximumHeight(self, _value) -> None:
        pass

    def setSizePolicy(self, *_args) -> None:
        pass

    def setFixedHeight(self, _value) -> None:
        pass

    def setWordWrap(self, _value) -> None:
        pass

    def setAlignment(self, _value) -> None:
        pass

    def fontMetrics(self):
        return _FontMetrics()


class _FontMetrics:
    def lineSpacing(self) -> int:
        return 14


class QLabel(_Widget):
    def __init__(self, text: str = "", *args, **kwargs) -> None:
        super().__init__()
        self.text_value = text

    def setText(self, text) -> None:
        self.text_value = text

    def text(self) -> str:
        return self.text_value


class QPushButton(_Widget):
    def __init__(self, text: str = "", *args, **kwargs) -> None:
        super().__init__()
        self.text_value = text
        self.clicked = _Signal()

    def text(self) -> str:
        return self.text_value


class QCheckBox(_Widget):
    def __init__(self, text: str = "", *args, **kwargs) -> None:
        super().__init__()
        self.text_value = text
        self.checked = False
        self.stateChanged = _Signal()

    def setChecked(self, value) -> None:
        self.checked = bool(value)
        if not self._signals_blocked:
            self.stateChanged.emit(int(self.checked))

    def isChecked(self) -> bool:
        return self.checked

    def text(self) -> str:
        return self.text_value


class QDoubleSpinBox(_Widget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self._value = 0.0
        self.valueChanged = _Signal()
        self.decimals = 2
        self.minimum = 0.0
        self.maximum = 0.0

    def setDecimals(self, value) -> None:
        self.decimals = int(value)

    def setRange(self, minimum, maximum) -> None:
        self.minimum, self.maximum = float(minimum), float(maximum)

    def setSingleStep(self, _value) -> None:
        pass

    def setValue(self, value) -> None:
        self._value = float(value)
        if not self._signals_blocked:
            self.valueChanged.emit(self._value)

    def value(self) -> float:
        return self._value


class QSpinBox(_Widget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self._value = 0
        self.valueChanged = _Signal()
        self.minimum = 0
        self.maximum = 0

    def setRange(self, minimum, maximum) -> None:
        self.minimum, self.maximum = int(minimum), int(maximum)

    def setSingleStep(self, _value) -> None:
        pass

    def setValue(self, value) -> None:
        self._value = int(value)
        if not self._signals_blocked:
            self.valueChanged.emit(self._value)

    def value(self) -> int:
        return self._value


class QComboBox(_Widget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.items: list[tuple[str, object]] = []
        self.index = -1
        self.currentTextChanged = _Signal()
        self.currentIndexChanged = _Signal()

    def addItem(self, text, data=None) -> None:
        self.items.append((str(text), data))
        if self.index < 0:
            self.index = 0

    def addItems(self, texts) -> None:
        for text in texts:
            self.addItem(text)

    def clear(self) -> None:
        self.items = []
        self.index = -1

    def count(self) -> int:
        return len(self.items)

    def setCurrentIndex(self, index) -> None:
        self.index = int(index)
        if not self._signals_blocked:
            self.currentTextChanged.emit(self.currentText())

    def currentIndex(self) -> int:
        return self.index

    def currentText(self) -> str:
        if 0 <= self.index < len(self.items):
            return self.items[self.index][0]
        return ""

    def currentData(self):
        if 0 <= self.index < len(self.items):
            return self.items[self.index][1]
        return None

    def setCurrentText(self, text) -> None:
        for position, (label, _data) in enumerate(self.items):
            if label == text:
                self.setCurrentIndex(position)
                return


class QSlider(_Widget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.valueChanged = _Signal()

    def setRange(self, *_args) -> None:
        pass


class QTableWidget(_Widget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.itemChanged = _Signal()
        self.itemSelectionChanged = _Signal()
        self.headers: list[str] = []

    def setHorizontalHeaderLabels(self, labels) -> None:
        self.headers = list(labels)

    def verticalHeader(self):
        return _Widget()

    def setEditTriggers(self, _value) -> None:
        pass

    def setSelectionBehavior(self, _value) -> None:
        pass


class QTreeWidget(_Widget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.itemSelectionChanged = _Signal()
        self.headers: list[str] = []

    def setHeaderLabels(self, labels) -> None:
        self.headers = list(labels)

    def setSelectionMode(self, _value) -> None:
        pass


class QGroupBox(_Widget):
    def __init__(self, title: str = "", *args, **kwargs) -> None:
        super().__init__()
        self.title = title


class QFormLayout:
    """Records rows in order; that ordering is what the tests compare."""

    def __init__(self, parent=None) -> None:
        self.rows: list[tuple[object, object]] = []

    def addRow(self, first, second=None):
        if second is None:
            self.rows.append((None, first))
        else:
            self.rows.append((QLabel(first) if isinstance(first, str) else first, second))
        return None

    def labelForField(self, widget):
        for label, field in self.rows:
            if field is widget:
                return label
        return None


class QHBoxLayout:
    def __init__(self, parent=None) -> None:
        self.widgets: list = []

    def setContentsMargins(self, *_args) -> None:
        pass

    def addWidget(self, widget, *_args) -> None:
        self.widgets.append(widget)

    def addStretch(self, *_args) -> None:
        pass


class QVBoxLayout(QHBoxLayout):
    pass


class QLineEdit(_Widget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.text_value = ""

    def setPlaceholderText(self, _text) -> None:
        pass

    def text(self) -> str:
        return self.text_value


class QProgressBar(_Widget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self._value = 0

    def setRange(self, *_args) -> None:
        pass

    def setValue(self, value) -> None:
        self._value = int(value)


class QPlainTextEdit(_Widget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.lines: list[str] = []

    def setReadOnly(self, _value) -> None:
        pass

    def setMaximumBlockCount(self, _value) -> None:
        pass

    def clear(self) -> None:
        self.lines = []

    def appendPlainText(self, text) -> None:
        self.lines.append(text)


class QScrollArea(_Widget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.inner = None

    def setWidgetResizable(self, _value) -> None:
        pass

    def setWidget(self, widget) -> None:
        self.inner = widget


class QTimer(_Widget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.timeout = _Signal()
        self.interval = None

    def start(self, interval=None) -> None:
        self.interval = interval

    def stop(self) -> None:
        self.interval = None


class _QtWidgets:
    QWidget = _Widget
    QLabel = QLabel
    QPushButton = QPushButton
    QCheckBox = QCheckBox
    QDoubleSpinBox = QDoubleSpinBox
    QSpinBox = QSpinBox
    QComboBox = QComboBox
    QSlider = QSlider
    QTableWidget = QTableWidget
    QTreeWidget = QTreeWidget
    QGroupBox = QGroupBox
    QFormLayout = QFormLayout
    QHBoxLayout = QHBoxLayout
    QVBoxLayout = QVBoxLayout
    QLineEdit = QLineEdit
    QProgressBar = QProgressBar
    QPlainTextEdit = QPlainTextEdit
    QScrollArea = QScrollArea

    class QSizePolicy:
        Fixed = "fixed"
        Preferred = "preferred"

    class QAbstractItemView:
        NoEditTriggers = "no-edit"
        SelectRows = "select-rows"
        SingleSelection = "single"


class _Qt:
    Horizontal = "horizontal"
    AlignTop = 1
    AlignLeft = 2


class _QtCore:
    Qt = _Qt
    QTimer = QTimer


class _QtStub:
    QtWidgets = _QtWidgets
    QtCore = _QtCore


def _build(mode: str, params: SimulationParameters | None = None):
    panel = SimulationControlsPanel(_QtStub, params=params, mode=mode)
    form = QFormLayout()
    panel.build(form)
    return panel, form


def _section_titles(form: QFormLayout) -> list[str]:
    return [field.title for _label, field in form.rows if isinstance(field, QGroupBox)]


def _row_labels(panel: SimulationControlsPanel) -> list[str]:
    """Every (label, widget) row the panel built, in build order, with the section
    it belongs to -- the panel's layout reduced to something comparable."""
    out = []
    for key, (form, widget) in panel._rows.items():
        label = form.labelForField(widget)
        out.append((key, label.text() if isinstance(label, QLabel) else None))
    return out


# --- tests ----------------------------------------------------------------- #
def test_both_modes_build_identical_rows_in_identical_order():
    live, _ = _build(MODE_LIVE)
    headless, _ = _build(MODE_HEADLESS)
    assert _row_labels(live) == _row_labels(headless)


def test_both_modes_build_identical_sections_in_identical_order():
    _live, live_form = _build(MODE_LIVE)
    _headless, headless_form = _build(MODE_HEADLESS)
    titles = _section_titles(live_form)
    assert titles == _section_titles(headless_form)
    assert titles == [
        "Run",
        "Environment",
        "Material Properties",
        "Cryocooler",
        "Controller (global limits)",
        "Modal LQR Design",
        "MIMO Thermal-Rate QP",
        "Solver",
        "Display",
        "Enabled Simulation I/O",
        "Simulation Sys ID for Controller Gain Matrix",
        "Solver Diagnostic",
        "Thermal I/O Readouts",
    ]


def test_live_mode_hides_only_the_headless_extras():
    panel, _form = _build(MODE_LIVE)
    hidden = {key for key, (_form, widget) in panel._rows.items() if not widget.visible}
    assert hidden == {
        "snapshot_interval_s",
        "checkpoint_interval_s",
        "run_setpoint_K",
        "run_initial_temperature_K",
        "open_output",
        # The Solver section and everything in it.
        "implicit_method",
        "implicit_sparse_simulation_rtol",
        "implicit_sparse_simulation_maxiter",
        "implicit_sparse_adaptive_substeps_enabled",
        "implicit_sparse_adaptive_target_delta_K",
        "implicit_sparse_adaptive_max_substeps",
        "implicit_sparse_residual_check_enabled",
        "gpu_solver_enabled",
        # Not a mode difference: the readout box starts hidden in both tabs and the
        # simulation tab reveals it once a simulation is initialized.
        "sensor_readouts",
    }
    assert panel._sections["solver"].visible is False
    assert panel._sections["display"].visible is True


def test_headless_mode_hides_playback_and_graph_dependent_controls():
    panel, _form = _build(MODE_HEADLESS)
    for key in (
        "initialize",
        "playback_speed",
        "simulation_history_limit",
        "loop_playback",
        "transport",
        "time_slider",
        "component",
        "legend",
        # Inside the hidden Display section.
        "autoscale_temperature",
        "color_min_K",
        "color_max_K",
    ):
        assert panel._rows[key][1].visible is False, key
    for key in (
        "dt_s",
        "t_final_s",
        "input_mode",
        "controller",
        "headless_run",
        "open_output",
        # Kept: config features useful even for a non-live run.
        "initial_temperature_all",
        "randomize_setpoints",
    ):
        assert panel._rows[key][1].visible is True, key
    for key in ("display", "enabled_io", "sys_id", "stepper_diagnostic"):
        assert panel._sections[key].visible is False, key
    assert panel._sections["solver"].visible is True


def test_hiding_a_row_hides_its_label_too():
    panel, _form = _build(MODE_HEADLESS)
    form, widget = panel._rows["playback_speed"]
    label = form.labelForField(widget)
    assert label.text() == "playback speed"
    assert label.visible is False


def test_headless_keeps_the_run_and_output_buttons():
    panel, _form = _build(MODE_HEADLESS)
    assert panel.run_headless_button.text() == "Start Headless Run"
    assert panel.stop_headless_button.text() == "Stop Run"
    assert panel.open_output_button.text() == "Open Output Folder"


def test_live_keeps_the_original_headless_button_labels():
    panel, _form = _build(MODE_LIVE)
    assert panel.run_headless_button.text() == "Run Headless (save, no viz)"
    assert panel.stop_headless_button.text() == "Stop Headless Run"


def test_read_round_trips_set_params():
    params = replace(
        SimulationParameters(),
        dt_s=0.25,
        t_final_s=7200.0,
        T_env_K=301.0,
        copper_rrr=150,
        cryocooler_enabled=False,
        use_temperature_dependent_properties=True,
        mimo_integral_abs_max=12.5,
        implicit_sparse_simulation_maxiter=42,
        gpu_solver_enabled=False,
        modal_integral_gain=0.075,
        input_mode="heater_inputs",
        implicit_sparse_simulation_method="backward_euler",
    )
    panel, _form = _build(MODE_HEADLESS)
    panel.set_params(params)
    result = panel.read(params)
    for field in (
        "dt_s",
        "t_final_s",
        "T_env_K",
        "copper_rrr",
        "cryocooler_enabled",
        "use_temperature_dependent_properties",
        "mimo_integral_abs_max",
        "implicit_sparse_simulation_maxiter",
        "gpu_solver_enabled",
        "modal_integral_gain",
        "input_mode",
        "implicit_sparse_simulation_method",
    ):
        assert getattr(result, field) == getattr(params, field), field


def test_read_keeps_fields_the_panel_has_no_widget_for():
    """The panel covers a subset of SimulationParameters; everything else must
    survive a read instead of reverting to a dataclass default."""
    base = replace(
        SimulationParameters(),
        colormap="viridis",
        implicit_sparse_block_jacobi_size=999,
        enabled_heater_node_ids=(3, 4),
    )
    panel, _form = _build(MODE_HEADLESS)
    result = panel.read(base)
    assert result.colormap == "viridis"
    assert result.implicit_sparse_block_jacobi_size == 999
    assert result.enabled_heater_node_ids == (3, 4)


def test_read_reports_the_selected_controller_artifact():
    panel, _form = _build(MODE_HEADLESS)
    panel.controller_scheme_combo.addItem(PID_QP_LABEL, None)
    panel.controller_scheme_combo.addItem("Modal LQR r=40", r"C:\graphs\g\modal_controller.npz")

    panel.controller_scheme_combo.setCurrentIndex(0)
    pid = panel.read(SimulationParameters())
    assert pid.mimo_controller_scheme == "pid_qp"
    assert pid.modal_controller_path == ""

    panel.controller_scheme_combo.setCurrentIndex(1)
    modal = panel.read(SimulationParameters())
    assert modal.mimo_controller_scheme == "modal_lqr"
    assert modal.modal_controller_path == r"C:\graphs\g\modal_controller.npz"


def test_widget_edits_report_the_field_that_changed():
    changed: list[str] = []
    panel = SimulationControlsPanel(
        _QtStub, mode=MODE_LIVE, on_parameter_change=changed.append
    )
    panel.build(QFormLayout())
    changed.clear()
    panel.inputs["dt_s"].setValue(3.0)
    panel.inputs["cryocooler_enabled"].setChecked(not panel.inputs["cryocooler_enabled"].isChecked())
    assert changed == ["dt_s", "cryocooler_enabled"]


def test_set_params_does_not_report_changes():
    changed: list[str] = []
    panel = SimulationControlsPanel(
        _QtStub, mode=MODE_LIVE, on_parameter_change=changed.append
    )
    panel.build(QFormLayout())
    changed.clear()
    panel.set_params(replace(SimulationParameters(), dt_s=9.0, cryocooler_enabled=False))
    assert changed == []


def test_actions_are_wired_to_the_buttons():
    fired: list[str] = []
    panel = SimulationControlsPanel(
        _QtStub,
        mode=MODE_LIVE,
        actions={
            "initialize": lambda: fired.append("initialize"),
            "play": lambda: fired.append("play"),
            "start_headless": lambda: fired.append("start_headless"),
        },
    )
    panel.build(QFormLayout())
    panel._rows["initialize"][1].clicked.emit()
    panel.run_headless_button.clicked.emit()
    assert fired == ["initialize", "start_headless"]


def test_export_to_gives_the_owner_direct_widget_handles():
    class _Owner:
        pass

    owner = _Owner()
    panel, _form = _build(MODE_LIVE)
    panel.export_to(owner)
    assert owner.inputs is panel.inputs
    for name in ("input_mode", "time_slider", "enabled_io_table", "stats_label", "snapshot_spin"):
        assert getattr(owner, name) is getattr(panel, name)


@pytest.mark.parametrize("mode", [MODE_LIVE, MODE_HEADLESS])
def test_every_input_maps_to_a_real_parameter(mode):
    params = SimulationParameters()
    panel, _form = _build(mode)
    unknown = [name for name in panel.inputs if not hasattr(params, name)]
    assert unknown == []


@pytest.mark.parametrize("mode", [MODE_LIVE, MODE_HEADLESS])
def test_readout_editor_builds_the_same_heater_sensor_options(mode):
    panel, _form = _build(mode)
    panel.build_readout_editor()
    # setpoint, heater power/PID and cryocooler options exist in both tabs.
    for field in (
        "controller_setpoint_K",
        "sensor_control_mode",
        "heater_max_power_W",
        "heater_efficiency",
        "controller_kp_coarse",
        "cryocooler_max_power_W",
        "cryocooler_enabled",
    ):
        assert field in panel.readout_editor_inputs, field
    owner = type("O", (), {})()
    panel.export_to(owner)
    assert owner.readout_editor_inputs is panel.readout_editor_inputs
    assert owner.readout_sensor_editor is panel.readout_sensor_editor


def test_readout_editor_change_routes_to_the_owner_action():
    seen: list[str] = []
    panel = SimulationControlsPanel(
        _QtStub,
        mode=MODE_LIVE,
        actions={"readout_heater_change": seen.append},
    )
    panel.build(QFormLayout())
    panel.build_readout_editor()
    panel.readout_editor_inputs["heater_max_power_W"].setValue(12.0)
    assert seen == ["heater_max_power_W"]


def test_headless_randomize_and_set_all_drive_the_whole_run_fields(tmp_path):
    tab = _headless_tab(tmp_path / "graphs")
    tab.initial_temperature_all_spin.setValue(77.0)
    tab._set_all_initial_temperatures()
    assert tab.initial_spin.value() == pytest.approx(77.0)
    assert tab.use_initial.isChecked() is True

    tab.sensor_random_center_spin.setValue(50.0)
    tab.sensor_random_spread_mK_spin.setValue(0.0)  # zero spread -> deterministic
    tab._randomize_setpoints()
    assert tab.setpoint_spin.value() == pytest.approx(50.0)
    assert tab.use_setpoint.isChecked() is True


# --- the headless tab, built on the same stubs ----------------------------- #
def _headless_tab(graphs_root):
    from graph_visualizer.headless_run_tab import HeadlessRunTab

    return HeadlessRunTab(_QtStub, None, graphs_root=lambda: graphs_root)


def test_headless_tab_builds_and_uses_the_shared_panel(tmp_path):
    tab = _headless_tab(tmp_path / "graphs")
    assert tab.panel.mode == MODE_HEADLESS
    # app.py puts this in the window's side-panel stack, next to the simulation
    # tab's, which is what makes the two panels comparable side by side.
    assert tab.controls_scroll.inner is not None
    # export_to must satisfy everything start_run() reaches for.
    for name in (
        "snapshot_spin",
        "checkpoint_spin",
        "setpoint_spin",
        "use_setpoint",
        "initial_spin",
        "use_initial",
        "run_headless_button",
        "stop_headless_button",
        "open_output_button",
        "controller_scheme_combo",
        "input_mode",
    ):
        assert getattr(tab, name) is getattr(tab.panel, name), name


def test_headless_tab_lists_graphs_without_loading_them(tmp_path):
    root = tmp_path / "graphs"
    (root / "assembly").mkdir(parents=True)
    (root / "assembly" / "node_ids.npy").write_bytes(b"")
    (root / "no_nodes").mkdir(parents=True)
    tab = _headless_tab(root)
    assert [label for label, _data in tab.graph_combo.items] == ["assembly"]
    # No controller artifacts on disk, so only the always-available scheme.
    assert [label for label, _data in tab.controller_scheme_combo.items] == [PID_QP_LABEL]
    assert tab.panel.selected_controller_artifact() == ""


def test_headless_tab_reads_the_graphs_saved_parameters(tmp_path):
    from graph_visualizer.simulation_parameters import save_simulation_parameters

    root = tmp_path / "graphs"
    folder = root / "assembly"
    folder.mkdir(parents=True)
    (folder / "node_ids.npy").write_bytes(b"")
    saved = replace(
        SimulationParameters(), dt_s=0.5, t_final_s=1234.0, colormap="viridis", copper_rrr=150
    )
    save_simulation_parameters(folder / "simulation_parameters.json", saved)

    tab = _headless_tab(root)
    assert tab.inputs["dt_s"].value() == pytest.approx(0.5)
    assert tab.inputs["t_final_s"].value() == pytest.approx(1234.0)
    assert tab.inputs["copper_rrr"].value() == 150

    tab.inputs["t_final_s"].setValue(4321.0)
    collected = tab._collect_parameters()
    assert collected.t_final_s == pytest.approx(4321.0)
    assert collected.dt_s == pytest.approx(0.5)
    # A field with no widget still comes from the graph's saved file.
    assert collected.colormap == "viridis"
