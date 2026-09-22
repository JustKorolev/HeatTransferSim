"""The Qt half of the help system: find a control, go to it, flash it.

:mod:`help_search` ranks; this reveals. Revealing is the part that is easy to get
almost right and still useless -- switching to the correct tab and leaving the
control forty rows below the fold is not an answer. So :meth:`HelpCenter.reveal`
does all four things: shows the tab, swaps the side panel to match, scrolls the
control into the middle of its scroll area, and flashes a border around it for a
moment so the eye lands on the right row rather than the right neighbourhood.

The index is built by walking the live panels (:meth:`HelpCenter.build_index`),
never by hand. Two hundred controls listed by hand would be wrong within the
month -- this repo's own module README claimed five tabs were two.
"""

from __future__ import annotations

from typing import Any

from .help_search import TUTORIALS, HelpTarget, Tutorial, search, tutorial_by_key

#: How long the flash lasts, and how many times it pulses. Long enough to find
#: with your eyes, short enough not to sit there looking like an error state.
HIGHLIGHT_MS = 1800
HIGHLIGHT_PULSES = 3
_HIGHLIGHT_STYLE = "border: 2px solid #e85d04; border-radius: 4px;"


class HelpCenter:
    """Owns the search index, the highlight animation and the Help dialog."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self.QtCore = app.QtCore
        self.QtWidgets = app.QtWidgets
        self._targets: list[HelpTarget] = []
        self._flash_timer: Any = None
        self._flash_widget: Any = None
        self._flash_original_style: str = ""
        self._flash_remaining = 0
        self._dialog: Any = None

    # -- index -------------------------------------------------------------- #
    def build_index(self) -> list[HelpTarget]:
        """Walk the panels and return one target per reachable control.

        Rebuilt on demand rather than cached for the session: the controller
        dropdown, the graph list and the per-node editors all change as graphs are
        loaded, and a stale index sends the user to a row that has moved.
        """
        targets: list[HelpTarget] = []
        tabs = getattr(self.app, "view_tabs", None)
        if tabs is None:
            return targets

        # Every tab is itself a destination: "where is validation?" is a fair
        # question and the answer is a tab, not a control.
        for index in range(tabs.count()):
            title = tabs.tabText(index)
            targets.append(
                HelpTarget(
                    key=f"tab:{index}",
                    label=title,
                    tab=title,
                    tab_index=index,
                    kind="tab",
                    keywords=_TAB_KEYWORDS.get(title, ()),
                )
            )

        for tab_index, panel_owner in self._panel_owners():
            title = tabs.tabText(tab_index) if tab_index < tabs.count() else ""
            # A tab whose controls are not a SimulationControlsPanel describes
            # itself. Without this the Build Graph tab contributed only the tab.
            describe = getattr(panel_owner, "help_targets", None)
            if callable(describe):
                targets.extend(self._targets_from_rows(describe(), title, tab_index))
                continue
            panel = getattr(panel_owner, "panel", None)
            if panel is None:
                continue
            targets.extend(self._targets_from_panel(panel, title, tab_index))
        return targets

    def _panel_owners(self) -> list[tuple[int, Any]]:
        """(tab index, tab object) for every tab that owns a controls panel.

        The index is looked up from the tab bar rather than hardcoded. It used to
        be written in literally -- 2, 3, 4 -- and adding the Build Graph tab in
        front of them silently made every search result point one tab to the left.
        """
        tabs = getattr(self.app, "view_tabs", None)
        pairs: list[tuple[int, Any]] = []
        for attribute in ("build_graph_tab", "simulation_tab", "thermal_validation_tab",
                          "headless_run_tab"):
            owner = getattr(self.app, attribute, None)
            if owner is None or tabs is None:
                continue
            widget = getattr(owner, "widget", None)
            index = tabs.indexOf(widget) if widget is not None else -1
            if index >= 0:
                pairs.append((index, owner))
        return pairs

    def _targets_from_panel(self, panel: Any, tab_title: str, tab_index: int) -> list[HelpTarget]:
        rows = getattr(panel, "_rows", None) or {}
        row_labels = getattr(panel, "_row_labels", None) or {}
        row_sections = getattr(panel, "_row_sections", None) or {}
        section_titles = getattr(panel, "_section_titles", None) or {}
        described: list[tuple[str, str, str, Any]] = []
        for key, entry in rows.items():
            try:
                _form, widget = entry
            except (TypeError, ValueError):
                continue
            label = str(row_labels.get(key) or "") or _widget_text(widget) or key.replace("_", " ")
            section = str(section_titles.get(row_sections.get(key, ""), ""))
            described.append((key, label, section, widget))
        return self._targets_from_rows(described, tab_title, tab_index)

    def _targets_from_rows(
        self, rows: Any, tab_title: str, tab_index: int
    ) -> list[HelpTarget]:
        """Turn ``(row key, label, section, widget)`` tuples into targets.

        This is the shape a tab returns from ``help_targets()``, and what the
        panel walk above reduces to, so both kinds of tab are indexed by exactly
        the same rules -- including the visibility check.
        """
        out: list[HelpTarget] = []
        for row in rows or ():
            try:
                key, label, section, widget = row
            except (TypeError, ValueError):
                continue
            # A hidden row belongs to the other tab's mode. Offering it would send
            # the user to a control that is not on screen, which reads as a bug.
            if not _is_visible(widget):
                continue
            out.append(
                HelpTarget(
                    key=f"{tab_index}:{key}",
                    label=str(label) or str(key).replace("_", " "),
                    tab=tab_title,
                    section=str(section),
                    tooltip=_widget_tooltip(widget),
                    keywords=_row_keywords(key),
                    tab_index=tab_index,
                    widget=widget,
                    kind="readout" if _is_readout(widget) else "control",
                )
            )
        return out

    # -- reveal ------------------------------------------------------------- #
    def reveal(self, target: HelpTarget) -> bool:
        """Show the tab, scroll the control into view, and flash it.

        Returns False when the target cannot be shown, so a caller can say so
        rather than appearing to do nothing.
        """
        tabs = getattr(self.app, "view_tabs", None)
        if tabs is not None and target.tab_index is not None:
            if 0 <= target.tab_index < tabs.count():
                tabs.setCurrentIndex(target.tab_index)
        widget = target.widget
        if widget is None:
            return target.kind == "tab"
        if not _is_visible(widget):
            return False
        self._scroll_into_view(widget)
        self.flash(widget)
        return True

    def _scroll_into_view(self, widget: Any) -> None:
        """Put the widget in the MIDDLE of its scroll area, not just barely on it.

        ensureWidgetVisible's default margin leaves the target hard against an
        edge, which is where the eye is least likely to look. Half the viewport
        height puts it where a person would centre it themselves.
        """
        scroll = _enclosing_scroll_area(widget, self.QtWidgets)
        if scroll is None:
            return
        try:
            viewport = scroll.viewport()
            margin = max(0, int(viewport.height() // 2) - int(widget.height() // 2))
            scroll.ensureWidgetVisible(widget, 0, margin)
        except Exception:  # noqa: BLE001 - never let a scroll hint break the jump
            pass

    def flash(self, widget: Any) -> None:
        """Pulse a border around ``widget`` for :data:`HIGHLIGHT_MS`.

        A stylesheet swap rather than a QPropertyAnimation: the targets are stock
        spin boxes, combos and buttons with no animatable colour property, and a
        graphics effect on a widget inside a scrolling form repaints badly.
        """
        self._restore_flashed_widget()
        if widget is None:
            return
        try:
            self._flash_original_style = widget.styleSheet()
        except Exception:  # noqa: BLE001 - a stub or an exotic widget
            return
        self._flash_widget = widget
        self._flash_remaining = HIGHLIGHT_PULSES * 2
        if self._flash_timer is None:
            self._flash_timer = self.QtCore.QTimer(getattr(self.app, "window", None))
            self._flash_timer.timeout.connect(self._flash_tick)
        self._flash_tick()
        self._flash_timer.start(max(1, HIGHLIGHT_MS // (HIGHLIGHT_PULSES * 2)))

    def _flash_tick(self) -> None:
        widget = self._flash_widget
        if widget is None:
            self._stop_flash_timer()
            return
        if self._flash_remaining <= 0:
            self._restore_flashed_widget()
            return
        on = self._flash_remaining % 2 == 0
        try:
            widget.setStyleSheet(
                self._flash_original_style + _HIGHLIGHT_STYLE if on else self._flash_original_style
            )
        except Exception:  # noqa: BLE001
            self._restore_flashed_widget()
            return
        self._flash_remaining -= 1

    def _restore_flashed_widget(self) -> None:
        widget = self._flash_widget
        self._flash_widget = None
        self._flash_remaining = 0
        self._stop_flash_timer()
        if widget is None:
            return
        try:
            widget.setStyleSheet(self._flash_original_style)
        except Exception:  # noqa: BLE001
            pass

    def _stop_flash_timer(self) -> None:
        if self._flash_timer is not None:
            try:
                self._flash_timer.stop()
            except Exception:  # noqa: BLE001
                pass

    # -- search, for the dialog --------------------------------------------- #
    def find(self, query: str, limit: int = 12) -> list[Any]:
        self._targets = self.build_index()
        return search(query, self._targets, limit=limit)

    def go_to_row_key(self, row_key: str) -> bool:
        """Reveal a control by its PANEL row key, for a tutorial's 'Show me'.

        The index keys targets as '<tab>:<row>' because the two simulation tabs
        share row keys; a tutorial names only the row, so the first tab offering
        it wins. That is the live tab, which is where a tutorial means.
        """
        for target in self.build_index():
            if target.key.split(":", 1)[-1] == row_key:
                return self.reveal(target)
        return False

    # -- dialog ------------------------------------------------------------- #
    def show_dialog(self, tutorial_key: str = "", focus_search: bool = False) -> Any:
        dialog = HelpDialog(self)
        self._dialog = dialog
        if tutorial_key:
            dialog.open_tutorial(tutorial_key)
        dialog.show_dialog(focus_search=focus_search)
        return dialog


_TAB_KEYWORDS = {
    "Build Graph": ("cad", "step", "assembly", "octree", "voxel", "mesh",
                    "convert", "import", "geometry", "new graph"),
    "3D Octree Graph Editor": ("cells", "geometry", "materials", "roles", "editor", "octree"),
    "2D Network Graph": ("adjacency", "network", "edges", "topology"),
    "Heat Transfer Simulation": ("run", "play", "live", "controller", "sys id", "simulate"),
    "Thermal Validation": ("verify", "analytical", "benchmark", "accuracy", "error"),
    "Headless Run": ("overnight", "batch", "background", "detached", "long run"),
}

#: Extra words for rows whose label is jargon. Only where the label alone would
#: not be found by someone describing the PROBLEM rather than naming the control.
_ROW_KEYWORDS: dict[str, tuple[str, ...]] = {
    "input_mode": ("open loop", "closed loop", "heater inputs", "zero"),
    "controller": ("scheme", "mimo", "pi", "modal", "lqr", "none"),
    "mimo_pi_kp": ("proportional", "gain", "tuning"),
    "mimo_pi_ki": ("integral", "gain", "tuning", "lambda"),
    "mimo_default_heater_max_power_W": ("watt", "ceiling", "limit", "cap", "saturation"),
    "mimo_heater_slew_rate_W_per_s": ("rate limit", "ramp", "driver"),
    "mimo_undershoot_weight": ("asymmetry", "overshoot", "cold"),
    "mimo_lambda_u": ("regularization", "effort", "damping", "conditioning"),
    "mimo_integral_abs_max": ("windup", "clamp"),
    "run_sys_id": ("gain matrix", "identify", "identification", "G", "step test"),
    "modal_build": ("reduce", "reduction", "lqr", "artifact"),
    "export_controller": ("firmware", "header", "deploy", "constants", "mcu", "stm32"),
    "initialize": ("prepare", "start", "setup"),
    "transport": ("play", "pause", "step", "run"),
    "dt_s": ("timestep", "step size", "sample rate", "period"),
    "t_final_s": ("duration", "length", "how long"),
    "T_env_K": ("ambient", "room", "exterior", "background"),
    "interior_environment_temperature_K": ("vacuum", "shield", "cold background"),
    "use_radiative_coupling": ("view factor", "surface to surface", "radiation"),
    "cryocooler_enabled": ("cooler", "pt60", "cooling", "lift"),
    "checkpoint_interval_s": ("resume", "restart", "recovery"),
    "snapshot_interval_s": ("series", "sampling", "output"),
    "use_temperature_dependent_properties": ("cp", "conductivity", "cryo", "nist"),
    "implicit_method": ("solver", "integrator", "tr_bdf2", "backward euler"),
    "autoscale_temperature": ("colour", "color", "scale", "range"),
    "enabled_io": ("enable", "disable", "exclude", "channel"),
    "save_trajectory": ("export", "csv", "save"),
    "reset_integrators": ("windup", "clear", "integral"),
}


def _row_keywords(key: str) -> tuple[str, ...]:
    return _ROW_KEYWORDS.get(key, ())


# --------------------------------------------------------------------------- #
# Small Qt helpers, each tolerant of a stub widget.
# --------------------------------------------------------------------------- #
def _is_readout(widget: Any) -> bool:
    """True for a row that only DISPLAYS something.

    A status label is still worth finding -- "where does it tell me the run
    failed?" is a real question -- but it must not outrank the control it
    describes. The label beside Export Controller Constants is captioned
    "export", and it beat that button on the query "export controller".
    """
    for klass in type(widget).__mro__:
        if klass.__name__ in {"QLabel", "QProgressBar"}:
            return True
    return False


def _is_visible(widget: Any) -> bool:
    """Would this row be on screen with its own tab open?

    isHidden(), deliberately, and not isVisible(). Qt reports isVisible() False
    for an ordinary control sitting on a tab the user is not currently looking
    at, so asking that dropped every control outside the open tab from the
    index -- which is exactly what someone searches for. isHidden() is true only
    for a row hidden in its own right, which is what "the other mode owns this
    one" means, so it is the question actually being asked here.
    """
    method = getattr(widget, "isHidden", None)
    if callable(method):
        try:
            return not bool(method())
        except Exception:  # noqa: BLE001
            pass
    return bool(getattr(widget, "visible", True))


def _widget_text(widget: Any) -> str:
    for name in ("text", "title"):
        method = getattr(widget, name, None)
        if method is None:
            continue
        try:
            value = method() if callable(method) else method
        except Exception:  # noqa: BLE001
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _widget_tooltip(widget: Any) -> str:
    method = getattr(widget, "toolTip", None)
    if method is None:
        return ""
    try:
        value = method() if callable(method) else method
    except Exception:  # noqa: BLE001
        return ""
    return value if isinstance(value, str) else ""


def _enclosing_scroll_area(widget: Any, QtWidgets: Any) -> Any:
    """The QScrollArea this widget sits in, or None."""
    scroll_type = getattr(QtWidgets, "QScrollArea", None)
    node = widget
    for _ in range(40):  # bounded: a cycle here would hang the UI
        try:
            node = node.parentWidget()
        except Exception:  # noqa: BLE001
            return None
        if node is None:
            return None
        if scroll_type is not None and isinstance(node, scroll_type):
            return node
    return None


class HelpDialog:
    """Search box, results list, and the tutorials, in one non-modal window."""

    def __init__(self, center: HelpCenter) -> None:
        self.center = center
        self.QtCore = center.QtCore
        self.QtWidgets = center.QtWidgets
        self._results: list[Any] = []
        self._build()

    def _build(self) -> None:
        Qt = self.QtCore.Qt
        self.widget = self.QtWidgets.QDialog(getattr(self.center.app, "window", None))
        self.widget.setWindowTitle("Help and Search")
        self.widget.resize(760, 560)
        # Non-modal on purpose: the whole point is to watch the main window jump
        # to the control and flash it, which a modal dialog would block.
        self.widget.setModal(False)
        layout = self.QtWidgets.QVBoxLayout(self.widget)

        self.tabs = self.QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)

        # --- search ---
        search_page = self.QtWidgets.QWidget()
        search_layout = self.QtWidgets.QVBoxLayout(search_page)
        self.search_box = self.QtWidgets.QLineEdit()
        self.search_box.setPlaceholderText(
            "Search every control -- try 'heater power', 'checkpoint', 'export', 'ambient'"
        )
        self.search_box.textChanged.connect(self._handle_query_changed)
        self.search_box.returnPressed.connect(self._activate_first)
        search_layout.addWidget(self.search_box)
        self.results = self.QtWidgets.QListWidget()
        self.results.itemActivated.connect(self._handle_item_activated)
        self.results.itemClicked.connect(self._handle_item_activated)
        search_layout.addWidget(self.results, 1)
        self.search_hint = self.QtWidgets.QLabel(
            "Pick a result to jump to that control. It is scrolled into view and "
            "outlined for a moment."
        )
        self.search_hint.setWordWrap(True)
        search_layout.addWidget(self.search_hint)
        self.tabs.addTab(search_page, "Search")

        # --- tutorials ---
        tutorial_page = self.QtWidgets.QWidget()
        tutorial_layout = self.QtWidgets.QHBoxLayout(tutorial_page)
        self.tutorial_list = self.QtWidgets.QListWidget()
        self.tutorial_list.setMaximumWidth(230)
        for tutorial in TUTORIALS:
            item = self.QtWidgets.QListWidgetItem(tutorial.title)
            item.setData(Qt.UserRole, tutorial.key)
            item.setToolTip(tutorial.summary)
            self.tutorial_list.addItem(item)
        self.tutorial_list.currentItemChanged.connect(self._handle_tutorial_selected)
        tutorial_layout.addWidget(self.tutorial_list)

        right = self.QtWidgets.QWidget()
        right_layout = self.QtWidgets.QVBoxLayout(right)
        self.tutorial_body = self.QtWidgets.QTextBrowser()
        self.tutorial_body.setOpenExternalLinks(False)
        # The anchors are 'show:<row key>'; clicking one reveals that control.
        self.tutorial_body.anchorClicked.connect(self._handle_anchor)
        right_layout.addWidget(self.tutorial_body, 1)
        tutorial_layout.addWidget(right, 1)
        self.tabs.addTab(tutorial_page, "Tutorials")

        buttons = self.QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        close = self.QtWidgets.QPushButton("Close")
        close.clicked.connect(self.widget.close)
        buttons.addWidget(close)
        layout.addLayout(buttons)

        if self.tutorial_list.count():
            self.tutorial_list.setCurrentRow(0)

    # -- search ------------------------------------------------------------- #
    def _handle_query_changed(self, text: str) -> None:
        self.results.clear()
        self._results = self.center.find(text)
        if not str(text).strip():
            self.search_hint.setText(
                "Pick a result to jump to that control. It is scrolled into view and "
                "outlined for a moment."
            )
            return
        if not self._results:
            self.search_hint.setText(
                f"Nothing matches {text!r}. Every word has to match something, so try "
                "fewer words."
            )
            return
        for scored in self._results:
            item = self.QtWidgets.QListWidgetItem(scored.target.describe())
            tooltip = scored.target.tooltip
            if tooltip:
                item.setToolTip(tooltip)
            self.results.addItem(item)
        self.search_hint.setText(f"{len(self._results)} match(es). Enter goes to the first.")

    def _activate_first(self) -> None:
        if self._results:
            self._go_to(0)

    def _handle_item_activated(self, item: Any) -> None:
        self._go_to(self.results.row(item))

    def _go_to(self, index: int) -> None:
        if not (0 <= index < len(self._results)):
            return
        target = self._results[index].target
        if self.center.reveal(target):
            self.search_hint.setText(f"Showing: {target.describe()}")
        else:
            self.search_hint.setText(
                f"{target.describe()} is not available right now -- it belongs to a "
                "mode or a graph that is not loaded."
            )

    # -- tutorials ---------------------------------------------------------- #
    def open_tutorial(self, key: str) -> None:
        for row in range(self.tutorial_list.count()):
            item = self.tutorial_list.item(row)
            if item.data(self.QtCore.Qt.UserRole) == key:
                self.tutorial_list.setCurrentRow(row)
                self.tabs.setCurrentIndex(1)
                return

    def _handle_tutorial_selected(self, current: Any, _previous: Any = None) -> None:
        if current is None:
            return
        tutorial = tutorial_by_key(current.data(self.QtCore.Qt.UserRole))
        if tutorial is None:
            return
        self.tutorial_body.setHtml(render_tutorial_html(tutorial))

    def _handle_anchor(self, url: Any) -> None:
        reference = url.toString() if hasattr(url, "toString") else str(url)
        if not reference.startswith("show:"):
            return
        row_key = reference.split(":", 1)[1]
        if not self.center.go_to_row_key(row_key):
            self.tutorial_body.append(
                "<p style='color:#b23'>That control is not on screen right now -- it "
                "belongs to a mode or a graph that is not loaded.</p>"
            )

    def show_dialog(self, focus_search: bool = False) -> None:
        self.widget.show()
        self.widget.raise_()
        if focus_search:
            self.tabs.setCurrentIndex(0)
            self.search_box.setFocus()
            self.search_box.selectAll()


def render_tutorial_html(tutorial: Tutorial) -> str:
    """A tutorial as HTML, with a 'Show me' link per step that names a control.

    Kept a free function so it can be checked without a Qt widget: the link
    scheme is the contract between the tutorial data and
    :meth:`HelpDialog._handle_anchor`, and a typo in it fails silently on screen.
    """
    parts = [
        "<html><body style='font-family: sans-serif;'>",
        f"<h2>{_escape(tutorial.title)}</h2>",
        f"<p style='color:#555'>{_escape(tutorial.summary)}</p>",
    ]
    for number, step in enumerate(tutorial.steps, start=1):
        parts.append(f"<h3>{number}. {_escape(step.title)}</h3>")
        for paragraph in step.body.split("\n\n"):
            parts.append(f"<p>{_escape(paragraph)}</p>")
        if step.image:
            parts.append(f"<p><img src='{_escape(step.image)}' width='640'></p>")
        if step.target_key:
            parts.append(
                f"<p><a href='show:{_escape(step.target_key)}'>Show me this control</a></p>"
            )
    parts.append("</body></html>")
    return "".join(parts)


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
