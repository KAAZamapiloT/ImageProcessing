import cv2
import numpy as np

from cli_parser import parse_command
from fft_engine import FFTEngine
from filters import (
    butterworth_highpass,
    butterworth_lowpass,
    gaussian_blur_transfer,
    gaussian_highpass,
    gaussian_lowpass,
    ideal_highpass,
    ideal_lowpass,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QActionGroup, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from utils import ensure_grayscale, numpy_to_qimage


class NumericControl(QWidget):
    valueChanged = Signal(float)

    def __init__(
        self,
        minimum,
        maximum,
        value,
        *,
        is_float=False,
        decimals=2,
        step=1.0,
        scale=100,
        parent=None,
    ):
        super().__init__(parent)
        self.is_float = is_float
        self.scale = int(scale)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.slider = QSlider(Qt.Horizontal)
        if is_float:
            self.spin = QDoubleSpinBox()
            self.spin.setDecimals(int(decimals))
            self.spin.setSingleStep(float(step))
            self.spin.setRange(float(minimum), float(maximum))
            self.slider.setRange(int(round(minimum * self.scale)), int(round(maximum * self.scale)))
            self.slider.setValue(int(round(value * self.scale)))
            self.spin.setValue(float(value))
        else:
            self.spin = QSpinBox()
            self.spin.setSingleStep(int(max(step, 1)))
            self.spin.setRange(int(minimum), int(maximum))
            self.slider.setRange(int(minimum), int(maximum))
            self.slider.setValue(int(value))
            self.spin.setValue(int(value))

        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)

        self.slider.valueChanged.connect(self._slider_to_spin)
        self.spin.valueChanged.connect(self._spin_to_slider)

    def _slider_to_spin(self, raw):
        if self.is_float:
            value = float(raw) / self.scale
            self.spin.blockSignals(True)
            self.spin.setValue(value)
            self.spin.blockSignals(False)
            self.valueChanged.emit(value)
        else:
            value = int(raw)
            self.spin.blockSignals(True)
            self.spin.setValue(value)
            self.spin.blockSignals(False)
            self.valueChanged.emit(float(value))

    def _spin_to_slider(self, value):
        if self.is_float:
            slider_value = int(round(float(value) * self.scale))
            self.slider.blockSignals(True)
            self.slider.setValue(slider_value)
            self.slider.blockSignals(False)
            self.valueChanged.emit(float(value))
        else:
            slider_value = int(value)
            self.slider.blockSignals(True)
            self.slider.setValue(slider_value)
            self.slider.blockSignals(False)
            self.valueChanged.emit(float(slider_value))

    def value(self):
        return float(self.spin.value()) if self.is_float else int(self.spin.value())

    def set_value(self, value):
        self.spin.setValue(value)


class FourierEditorWidget(QLabel):
    frequencyClicked = Signal(int, int)
    frequencyHovered = Signal(int, int)
    hoverExited = Signal()

    def __init__(self, parent=None):
        super().__init__("Fourier Spectrum", parent)
        self.setAlignment(Qt.AlignCenter)
        self.setFrameShape(QFrame.Box)
        self.setMouseTracking(True)
        self.setMinimumSize(320, 320)
        self.setScaledContents(True)
        self.fft_shape = None

    def set_fft_shape(self, shape):
        self.fft_shape = shape

    def _map_to_fft(self, x, y):
        if self.fft_shape is None:
            return None
        h, w = self.fft_shape
        wh = max(1, self.height())
        ww = max(1, self.width())
        cx = int(np.clip(x, 0, ww - 1))
        cy = int(np.clip(y, 0, wh - 1))
        u = int(np.clip(int(cy * h / wh), 0, h - 1))
        v = int(np.clip(int(cx * w / ww), 0, w - 1))
        return u, v

    def mousePressEvent(self, event):
        mapped = self._map_to_fft(event.position().x(), event.position().y())
        if mapped is not None:
            self.frequencyClicked.emit(*mapped)

    def mouseMoveEvent(self, event):
        mapped = self._map_to_fft(event.position().x(), event.position().y())
        if mapped is not None:
            self.frequencyHovered.emit(*mapped)

    def leaveEvent(self, event):
        self.hoverExited.emit()
        super().leaveEvent(event)


class PresetFilterPanel(QWidget):
    applyRequested = Signal()
    paramsChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.family_combo = QComboBox()
        self.family_combo.addItem("Gaussian", "gaussian")
        self.family_combo.addItem("Butterworth", "butterworth")
        self.family_combo.addItem("Ideal", "ideal")
        self.family_combo.addItem("Wiener", "wiener")

        self.response_combo = QComboBox()
        self.response_combo.addItem("Lowpass", "lowpass")
        self.response_combo.addItem("Highpass", "highpass")

        self.live_preview = QCheckBox("Live Preview")

        self.cutoff_control = NumericControl(1, 400, 60, is_float=False)
        self.order_control = NumericControl(1, 12, 2, is_float=False)

        self.wiener_k_control = NumericControl(
            0.0,
            1.0,
            0.005,
            is_float=True,
            decimals=4,
            step=0.001,
            scale=10000,
        )
        self.blur_sigma_control = NumericControl(
            0.1,
            20.0,
            2.0,
            is_float=True,
            decimals=2,
            step=0.1,
            scale=100,
        )
        self.blur_kernel_control = NumericControl(3, 101, 15, is_float=False)

        self.apply_button = QPushButton("Apply Preset")
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("StatusLabel")

        self.rows = {}
        self.rows["family"] = self._row("Family", self.family_combo)
        self.rows["response"] = self._row("Response", self.response_combo)
        self.rows["cutoff"] = self._row("Cutoff", self.cutoff_control)
        self.rows["order"] = self._row("Order", self.order_control)
        self.rows["k"] = self._row("K", self.wiener_k_control)
        self.rows["sigma"] = self._row("Blur Sigma", self.blur_sigma_control)
        self.rows["kernel"] = self._row("Kernel Size", self.blur_kernel_control)

        for key in ["family", "response", "cutoff", "order", "k", "sigma", "kernel"]:
            layout.addWidget(self.rows[key])
        layout.addWidget(self.live_preview)
        layout.addWidget(self.apply_button)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

        self.family_combo.currentIndexChanged.connect(self._on_family_changed)
        self.response_combo.currentIndexChanged.connect(self._emit_params_changed)
        self.live_preview.toggled.connect(self._emit_params_changed)

        for control in [
            self.cutoff_control,
            self.order_control,
            self.wiener_k_control,
            self.blur_sigma_control,
            self.blur_kernel_control,
        ]:
            control.valueChanged.connect(self._emit_params_changed)

        self.apply_button.clicked.connect(self.applyRequested.emit)
        self.blur_kernel_control.valueChanged.connect(self._enforce_odd_kernel)

        self._on_family_changed()

    def _emit_params_changed(self, *_):
        self.paramsChanged.emit()

    def _row(self, label, widget):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        text = QLabel(label)
        text.setMinimumWidth(84)
        row_layout.addWidget(text)
        row_layout.addWidget(widget, 1)
        return row

    def _enforce_odd_kernel(self):
        value = int(self.blur_kernel_control.value())
        if value % 2 == 0:
            value += 1
            if value > 101:
                value = 101
            self.blur_kernel_control.set_value(value)

    def _on_family_changed(self):
        family = self.current_family()
        is_wiener = family == "wiener"
        is_butterworth = family == "butterworth"

        self.rows["response"].setVisible(not is_wiener)
        self.rows["cutoff"].setVisible(not is_wiener)
        self.rows["order"].setVisible(is_butterworth)

        self.rows["k"].setVisible(is_wiener)
        self.rows["sigma"].setVisible(is_wiener)
        self.rows["kernel"].setVisible(is_wiener)

        self.paramsChanged.emit()

    def current_family(self):
        return self.family_combo.currentData()

    def live_preview_enabled(self):
        return self.live_preview.isChecked()

    def get_settings(self):
        return {
            "family": self.current_family(),
            "response": self.response_combo.currentData(),
            "cutoff": float(self.cutoff_control.value()),
            "order": int(self.order_control.value()),
            "k": float(self.wiener_k_control.value()),
            "sigma": float(self.blur_sigma_control.value()),
            "kernel_size": int(self.blur_kernel_control.value()),
        }

    def set_pending(self, pending):
        self.apply_button.setText("Apply Preset (Pending)" if pending else "Apply Preset")

    def set_status(self, text):
        self.status_label.setText(text)


class FourierToolsPanel(QWidget):
    applyToolRequested = Signal()
    undoRequested = Signal()
    redoRequested = Signal()
    resetRequested = Signal()
    reconstructRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.radius_control = NumericControl(1, 120, 12, is_float=False)
        self.boost_control = NumericControl(1.0, 20.0, 5.0, is_float=True, decimals=2, step=0.5, scale=100)
        self.rect_width_control = NumericControl(1, 220, 30, is_float=False)
        self.rect_height_control = NumericControl(1, 220, 30, is_float=False)
        self.line_length_control = NumericControl(10, 400, 120, is_float=False)
        self.line_thickness_control = NumericControl(1, 25, 2, is_float=False)
        self.suppression_cutoff_control = NumericControl(1, 300, 40, is_float=False)
        self.phase_random_control = NumericControl(
            0.0,
            1.0,
            0.6,
            is_float=True,
            decimals=2,
            step=0.05,
            scale=100,
        )

        self.line_orientation_combo = QComboBox()
        self.line_orientation_combo.addItem("Horizontal", "horizontal")
        self.line_orientation_combo.addItem("Vertical", "vertical")
        self.line_orientation_combo.addItem("Diagonal Down", "diag_down")
        self.line_orientation_combo.addItem("Diagonal Up", "diag_up")

        for row in [
            self._row("Radius", self.radius_control),
            self._row("Boost", self.boost_control),
            self._row("Rect Width", self.rect_width_control),
            self._row("Rect Height", self.rect_height_control),
            self._row("Line Length", self.line_length_control),
            self._row("Line Thick", self.line_thickness_control),
            self._row("Line Orient", self.line_orientation_combo),
            self._row("Cutoff", self.suppression_cutoff_control),
            self._row("Rand Phase", self.phase_random_control),
        ]:
            layout.addWidget(row)

        self.apply_tool_button = QPushButton("Apply Selected Tool")
        self.undo_button = QPushButton("Undo")
        self.redo_button = QPushButton("Redo")
        self.reset_button = QPushButton("Reset FFT")
        self.reconstruct_button = QPushButton("Reconstruct")

        layout.addWidget(self.apply_tool_button)
        layout.addWidget(self.undo_button)
        layout.addWidget(self.redo_button)
        layout.addWidget(self.reset_button)
        layout.addWidget(self.reconstruct_button)

        self.active_tool_label = QLabel("Active tool: Circle Zero Tool")
        self.active_tool_label.setObjectName("StatusLabel")
        layout.addWidget(self.active_tool_label)
        layout.addStretch(1)

        self.apply_tool_button.clicked.connect(self.applyToolRequested.emit)
        self.undo_button.clicked.connect(self.undoRequested.emit)
        self.redo_button.clicked.connect(self.redoRequested.emit)
        self.reset_button.clicked.connect(self.resetRequested.emit)
        self.reconstruct_button.clicked.connect(self.reconstructRequested.emit)

    def _row(self, label, widget):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        text = QLabel(label)
        text.setMinimumWidth(84)
        row_layout.addWidget(text)
        row_layout.addWidget(widget, 1)
        return row

    def set_active_tool_text(self, text):
        self.active_tool_label.setText(f"Active tool: {text}")

    def get_settings(self):
        return {
            "radius": int(self.radius_control.value()),
            "boost_factor": float(self.boost_control.value()),
            "rect_half_width": max(1, int(self.rect_width_control.value()) // 2),
            "rect_half_height": max(1, int(self.rect_height_control.value()) // 2),
            "line_length": int(self.line_length_control.value()),
            "line_thickness": int(self.line_thickness_control.value()),
            "line_orientation": self.line_orientation_combo.currentData(),
            "suppression_cutoff": int(self.suppression_cutoff_control.value()),
            "phase_random_amount": float(self.phase_random_control.value()),
        }

class CLIDock(QDockWidget):
    commandSubmitted = Signal(str)

    def __init__(self, parent=None):
        super().__init__("CLI Terminal", parent)
        self.setObjectName("clidock")

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setObjectName("TerminalOutput")

        input_row = QWidget()
        input_layout = QHBoxLayout(input_row)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(6)

        self.input = QLineEdit()
        self.input.setPlaceholderText("Enter command...")
        self.run_button = QPushButton("Run")

        input_layout.addWidget(self.input, 1)
        input_layout.addWidget(self.run_button)

        layout.addWidget(self.output, 1)
        layout.addWidget(input_row)

        self.setWidget(root)

        self.input.returnPressed.connect(self._emit_command)
        self.run_button.clicked.connect(self._emit_command)

    def _emit_command(self):
        cmd = self.input.text().strip()
        if not cmd:
            return
        self.input.clear()
        self.commandSubmitted.emit(cmd)

    def append_output(self, text):
        self.output.appendPlainText(text)
        self.output.verticalScrollBar().setValue(self.output.verticalScrollBar().maximum())


class InspectorDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Inspector Panel", parent)
        self.setObjectName("inspectordock")

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.coord_label = QLabel("(u, v): -")
        self.mag_label = QLabel("Magnitude: -")
        self.phase_label = QLabel("Phase: -")
        self.distance_label = QLabel("D(u,v): -")

        for lbl in [self.coord_label, self.mag_label, self.phase_label, self.distance_label]:
            lbl.setObjectName("StatusLabel")
            layout.addWidget(lbl)

        layout.addStretch(1)
        self.setWidget(root)

    def update_info(self, u, v, magnitude, phase, distance):
        self.coord_label.setText(f"(u, v): ({u}, {v})")
        self.mag_label.setText(f"Magnitude: {magnitude:.4f}")
        self.phase_label.setText(f"Phase: {phase:.4f} rad")
        self.distance_label.setText(f"D(u,v): {distance:.3f}")

    def clear_info(self):
        self.coord_label.setText("(u, v): -")
        self.mag_label.setText("Magnitude: -")
        self.phase_label.setText("Phase: -")
        self.distance_label.setText("D(u,v): -")


class MainWindow(QMainWindow):
    TOOL_SPECS = [
        ("Circle Zero Tool", "circle_zero"),
        ("Circle Boost Tool", "circle_boost"),
        ("Rectangular Mask Tool", "rect_mask"),
        ("Line Suppression Tool", "line_suppress"),
        ("DC Removal Tool", "dc_remove"),
        ("High-Frequency Suppression Tool", "high_suppress"),
        ("Low-Frequency Suppression Tool", "low_suppress"),
        ("Phase Randomizer Tool", "phase_randomizer"),
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fourier Domain Image Analyzer")
        self.resize(1600, 980)

        self.engine = FFTEngine()
        self.displayed_fft = None
        self.last_output = None
        self.last_cursor_uv = None

        self.active_tool = "circle_zero"
        self.tool_actions = {}

        self._build_ui()
        self._apply_theme()

    # =====================================================
    # UI Build
    # =====================================================

    def _build_ui(self):
        self._build_menus()
        self._build_central_splitter()
        self._build_docks()
        self._build_toolbar()
        self.statusBar().showMessage("Load an image to begin.")

    def _build_menus(self):
        mb = self.menuBar()

        self.file_menu = mb.addMenu("File")
        self.view_menu = mb.addMenu("View")
        self.tools_menu = mb.addMenu("Tools")
        self.presets_menu = mb.addMenu("Presets")
        self.help_menu = mb.addMenu("Help")

        open_action = QAction("Open Image", self)
        open_action.triggered.connect(self.load_image)
        self.file_menu.addAction(open_action)

        open_secondary_action = QAction("Open Secondary Image", self)
        open_secondary_action.triggered.connect(self.load_secondary_image)
        self.file_menu.addAction(open_secondary_action)

        save_action = QAction("Save Output", self)
        save_action.triggered.connect(self.save_output)
        self.file_menu.addAction(save_action)

        self.file_menu.addSeparator()
        self.file_menu.addAction("Exit", self.close)

        self.help_menu.addAction("Help", self.show_help)

    def _build_central_splitter(self):
        self.original_view = QLabel("Original")
        self.original_view.setAlignment(Qt.AlignCenter)
        self.original_view.setMinimumSize(300, 300)
        self.original_view.setFrameShape(QFrame.Box)

        self.spectrum_view = FourierEditorWidget()
        self.spectrum_view.setMinimumSize(300, 300)

        self.reconstructed_view = QLabel("Reconstructed")
        self.reconstructed_view.setAlignment(Qt.AlignCenter)
        self.reconstructed_view.setMinimumSize(300, 300)
        self.reconstructed_view.setFrameShape(QFrame.Box)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.original_view)
        splitter.addWidget(self.spectrum_view)
        splitter.addWidget(self.reconstructed_view)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)
        self.setCentralWidget(splitter)

        self.spectrum_view.frequencyClicked.connect(self.on_spectrum_clicked)
        self.spectrum_view.frequencyHovered.connect(self.on_spectrum_hovered)
        self.spectrum_view.hoverExited.connect(self.on_spectrum_hover_exited)

    def _build_docks(self):
        self.preset_panel = PresetFilterPanel()
        self.preset_dock = QDockWidget("Preset Filters Panel", self)
        self.preset_dock.setObjectName("presetdock")
        self.preset_dock.setWidget(self.preset_panel)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.preset_dock)

        self.tools_panel = FourierToolsPanel()
        self.tools_dock = QDockWidget("Fourier Tools Panel", self)
        self.tools_dock.setObjectName("toolsdock")
        self.tools_dock.setWidget(self.tools_panel)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.tools_dock)
        self.splitDockWidget(self.preset_dock, self.tools_dock, Qt.Vertical)

        self.cli_dock = CLIDock(self)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.cli_dock)
        self.cli_dock.hide()

        self.inspector_dock = InspectorDock(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.inspector_dock)

        self.view_menu.addAction(self.preset_dock.toggleViewAction())
        self.view_menu.addAction(self.tools_dock.toggleViewAction())
        self.view_menu.addAction(self.cli_dock.toggleViewAction())
        self.view_menu.addAction(self.inspector_dock.toggleViewAction())

        self.preset_panel.applyRequested.connect(lambda: self.apply_preset_filter(preview=False))
        self.preset_panel.paramsChanged.connect(self.on_preset_params_changed)

        self.tools_panel.applyToolRequested.connect(self.apply_selected_tool_at_cursor)
        self.tools_panel.undoRequested.connect(self.undo_fft)
        self.tools_panel.redoRequested.connect(self.redo_fft)
        self.tools_panel.resetRequested.connect(self.reset_fft)
        self.tools_panel.reconstructRequested.connect(self.reconstruct_current_fft)

        self.cli_dock.commandSubmitted.connect(self.on_cli_command)

    def _build_toolbar(self):
        self.tool_bar = QToolBar("Tools", self)
        self.tool_bar.setObjectName("main_toolbar")
        self.tool_bar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(self.tool_bar)

        action_group = QActionGroup(self)
        action_group.setExclusive(True)

        for index, (label, key) in enumerate(self.TOOL_SPECS):
            action = QAction(label, self)
            action.setCheckable(True)
            if index == 0:
                action.setChecked(True)
            action.triggered.connect(lambda checked, k=key, name=label: self.set_active_tool(k, name))
            self.tool_actions[key] = action
            action_group.addAction(action)
            self.tool_bar.addAction(action)
            self.tools_menu.addAction(action)

        self.tools_menu.addSeparator()
        self.tools_menu.addAction("Apply Selected Tool", self.apply_selected_tool_at_cursor)

        self._build_presets_menu()

    def _build_presets_menu(self):
        self.presets_menu.addAction("Remove DC Component", self.preset_remove_dc)
        self.presets_menu.addAction("Show Phase Only", self.preset_show_phase_only)
        self.presets_menu.addAction("Show Magnitude Only", self.preset_show_magnitude_only)
        self.presets_menu.addAction("Randomize Phase", self.preset_randomize_phase)
        self.presets_menu.addAction("Swap Magnitude", self.preset_swap_magnitude)
        self.presets_menu.addAction("Apply Motion Blur", self.preset_apply_motion_blur)
        self.presets_menu.addAction("Restore with Wiener", self.preset_restore_wiener)

    def _apply_theme(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
            QMenuBar, QMenu {
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
            QMenuBar::item:selected, QMenu::item:selected {
                background-color: #094771;
            }
            QLabel {
                background-color: #252526;
                border: 1px solid #333333;
                padding: 3px;
            }
            QLabel#StatusLabel {
                background-color: #1e1e1e;
                border: 1px solid #333333;
                color: #9cdcfe;
            }
            QPushButton {
                background-color: #2d2d30;
                border: 1px solid #333333;
                border-radius: 0px;
                padding: 5px 8px;
                color: #d4d4d4;
            }
            QPushButton:hover {
                border-color: #007acc;
            }
            QPushButton:checked, QPushButton:pressed {
                background-color: #007acc;
                color: #ffffff;
            }
            QComboBox, QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {
                background-color: #252526;
                border: 1px solid #333333;
                border-radius: 0px;
                color: #d4d4d4;
                padding: 3px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #333333;
                height: 6px;
                background: #3c3c3c;
            }
            QSlider::handle:horizontal {
                background: #007acc;
                border: 1px solid #0060a8;
                width: 12px;
                margin: -4px 0;
                border-radius: 0px;
            }
            QToolBar {
                background-color: #252526;
                border: 1px solid #333333;
                spacing: 4px;
            }
            QDockWidget::title {
                background: #252526;
                border: 1px solid #333333;
                padding: 4px;
            }
            QPlainTextEdit#TerminalOutput {
                background-color: #111111;
                color: #c5c5c5;
                border: 1px solid #333333;
                font-family: Consolas, 'Courier New', monospace;
            }
            """
        )

    # =====================================================
    # Display Helpers
    # =====================================================

    def set_label_image(self, label, img, keep_aspect=True):
        qimg = numpy_to_qimage(img)
        mode = Qt.KeepAspectRatio if keep_aspect else Qt.IgnoreAspectRatio
        pixmap = QPixmap.fromImage(qimg).scaled(
            label.width(),
            label.height(),
            mode,
            Qt.SmoothTransformation,
        )
        label.setPixmap(pixmap)

    def refresh_from_engine(self):
        if self.engine.fft_shifted is None:
            return
        self.displayed_fft = self.engine.get_fft_copy()
        self.set_label_image(self.spectrum_view, self.engine.magnitude(self.displayed_fft), keep_aspect=False)
        self.spectrum_view.set_fft_shape(self.displayed_fft.shape)

        recon = self.engine.reconstruct(self.displayed_fft)
        self.last_output = recon
        self.set_label_image(self.reconstructed_view, recon, keep_aspect=True)

    def show_preview_fft(self, preview_fft):
        self.displayed_fft = preview_fft
        self.set_label_image(self.spectrum_view, self.engine.magnitude(preview_fft), keep_aspect=False)
        self.spectrum_view.set_fft_shape(preview_fft.shape)

        recon = self.engine.reconstruct(preview_fft)
        self.last_output = recon
        self.set_label_image(self.reconstructed_view, recon, keep_aspect=True)

    # =====================================================
    # File Actions
    # =====================================================

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not path:
            return

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            QMessageBox.warning(self, "Error", "Unable to load image.")
            return

        img = ensure_grayscale(img)
        self.engine.load_image(img)
        self.last_cursor_uv = None

        self.set_label_image(self.original_view, img, keep_aspect=True)
        self.refresh_from_engine()
        self.preset_panel.set_pending(True)
        self.preset_panel.set_status("Preset filters ready.")
        self.tools_panel.set_active_tool_text("Circle Zero Tool")
        self.statusBar().showMessage("Image loaded.")

    def load_secondary_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Secondary Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not path:
            return

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            QMessageBox.warning(self, "Error", "Unable to load secondary image.")
            return

        img = ensure_grayscale(img)
        if self.engine.original is not None and img.shape != self.engine.original.shape:
            target_h, target_w = self.engine.original.shape
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        self.engine.load_secondary_image(img)
        self.statusBar().showMessage("Secondary image loaded.")

    def save_output(self):
        if self.last_output is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Output",
            "",
            "PNG (*.png);;JPG (*.jpg);;BMP (*.bmp);;TIF (*.tif)",
        )
        if not path:
            return

        cv2.imwrite(path, np.clip(self.last_output, 0, 255).astype(np.uint8))
        self.statusBar().showMessage("Output saved.")

    # =====================================================
    # Preset Filters
    # =====================================================

    def on_preset_params_changed(self):
        self.preset_panel.set_pending(True)
        if self.engine.fft_shifted is None:
            return

        if self.preset_panel.live_preview_enabled():
            self.apply_preset_filter(preview=True)
        else:
            self.refresh_from_engine()
            self.preset_panel.set_status("Preset parameters updated.")

    def build_filter_mask(self, shape, settings):
        family = settings["family"]
        response = settings["response"]
        cutoff = settings["cutoff"]
        order = settings["order"]

        if family == "gaussian":
            return gaussian_lowpass(shape, cutoff) if response == "lowpass" else gaussian_highpass(shape, cutoff)

        if family == "butterworth":
            return butterworth_lowpass(shape, cutoff, order) if response == "lowpass" else butterworth_highpass(shape, cutoff, order)

        if family == "ideal":
            return ideal_lowpass(shape, cutoff) if response == "lowpass" else ideal_highpass(shape, cutoff)

        raise ValueError("Filter mask requested for non-mask family.")

    def compute_wiener_preview(self, source_fft, settings):
        shape = source_fft.shape
        sigma = settings["sigma"]
        kernel_size = settings["kernel_size"]
        K = settings["k"]

        H = gaussian_blur_transfer(shape, sigma=sigma, kernel_size=kernel_size)
        G = H * source_fft
        denom = np.maximum(np.abs(H) ** 2 + max(float(K), 0.0), 1e-12)
        restored = (np.conj(H) / denom) * G
        return restored, G, H

    def apply_preset_filter(self, preview=False):
        if self.engine.fft_shifted is None:
            return

        settings = self.preset_panel.get_settings()
        base_fft = self.engine.get_fft_copy()

        try:
            if settings["family"] == "wiener":
                restored, degraded, transfer = self.compute_wiener_preview(base_fft, settings)
                if preview:
                    self.show_preview_fft(restored)
                    self.preset_panel.set_status("Live preview (Wiener)")
                else:
                    self.engine.set_fft(restored, push_undo=True)
                    self.engine.degraded_fft_shifted = degraded
                    self.engine.blur_transfer = transfer
                    self.engine.degraded_image = self.engine.reconstruct_from(degraded)
                    self.refresh_from_engine()
                    self.preset_panel.set_pending(False)
                    self.preset_panel.set_status("Wiener preset applied")
            else:
                H = self.build_filter_mask(base_fft.shape, settings)
                if preview:
                    self.show_preview_fft(base_fft * H)
                    self.preset_panel.set_status("Live preview")
                else:
                    self.engine.apply_filter(H, push_undo=True)
                    self.refresh_from_engine()
                    self.preset_panel.set_pending(False)
                    self.preset_panel.set_status("Preset applied")
        except Exception as exc:
            QMessageBox.warning(self, "Preset Error", str(exc))

    # =====================================================
    # Toolbar Tools
    # =====================================================

    def set_active_tool(self, key, label):
        self.active_tool = key
        self.tools_panel.set_active_tool_text(label)
        self.statusBar().showMessage(f"Tool selected: {label}")

    def on_spectrum_clicked(self, u, v):
        self.last_cursor_uv = (u, v)
        self.apply_selected_tool(u, v)

    def apply_selected_tool_at_cursor(self):
        if self.last_cursor_uv is None:
            QMessageBox.information(self, "Tool", "Click on Fourier spectrum first.")
            return
        self.apply_selected_tool(*self.last_cursor_uv)

    def apply_selected_tool(self, u, v):
        if self.engine.fft_shifted is None:
            return

        p = self.tools_panel.get_settings()
        tool = self.active_tool

        try:
            if tool == "circle_zero":
                self.engine.apply_circle_zero(u, v, p["radius"])
            elif tool == "circle_boost":
                self.engine.apply_circle_boost(u, v, p["radius"], p["boost_factor"])
            elif tool == "rect_mask":
                self.engine.apply_rectangular_mask(u, v, p["rect_half_height"], p["rect_half_width"])
            elif tool == "line_suppress":
                self.engine.apply_line_suppression(
                    u,
                    v,
                    length=p["line_length"],
                    thickness=p["line_thickness"],
                    orientation=p["line_orientation"],
                )
            elif tool == "dc_remove":
                self.engine.remove_dc_component()
            elif tool == "high_suppress":
                self.engine.remove_high_frequencies(p["suppression_cutoff"])
            elif tool == "low_suppress":
                self.engine.remove_low_frequencies(p["suppression_cutoff"])
            elif tool == "phase_randomizer":
                self.engine.apply_phase_randomizer(
                    u,
                    v,
                    radius=p["radius"],
                    amount=p["phase_random_amount"],
                )
            else:
                return

            self.refresh_from_engine()
            self.preset_panel.set_pending(True)
        except Exception as exc:
            QMessageBox.warning(self, "Tool Error", str(exc))

    def undo_fft(self):
        if self.engine.undo():
            self.refresh_from_engine()
            self.statusBar().showMessage("Undo")

    def redo_fft(self):
        if self.engine.redo():
            self.refresh_from_engine()
            self.statusBar().showMessage("Redo")

    def reset_fft(self):
        if self.engine.original is None:
            return
        self.engine.reset()
        self.refresh_from_engine()
        self.statusBar().showMessage("FFT reset")

    def reconstruct_current_fft(self):
        if self.engine.fft_shifted is None:
            return
        recon = self.engine.reconstruct()
        self.last_output = recon
        self.set_label_image(self.reconstructed_view, recon, keep_aspect=True)

    # =====================================================
    # Inspector
    # =====================================================

    def on_spectrum_hovered(self, u, v):
        if self.displayed_fft is None:
            self.inspector_dock.clear_info()
            return

        F = self.displayed_fft
        magnitude = float(np.abs(F[u, v]))
        phase = float(np.angle(F[u, v]))

        M, N = F.shape
        distance = float(np.sqrt((u - M / 2.0) ** 2 + (v - N / 2.0) ** 2))
        self.inspector_dock.update_info(u, v, magnitude, phase, distance)

    def on_spectrum_hover_exited(self):
        self.inspector_dock.clear_info()

    # =====================================================
    # Presets Menu Actions
    # =====================================================

    def preset_remove_dc(self):
        if self.engine.fft_shifted is None:
            return
        self.engine.remove_dc_component()
        self.refresh_from_engine()

    def preset_show_phase_only(self):
        if self.engine.fft_shifted is None:
            return
        img = self.engine.phase_only_reconstruction()
        self.last_output = img
        self.set_label_image(self.reconstructed_view, img, keep_aspect=True)

    def preset_show_magnitude_only(self):
        if self.engine.fft_shifted is None:
            return
        img = self.engine.magnitude_only_reconstruction()
        self.last_output = img
        self.set_label_image(self.reconstructed_view, img, keep_aspect=True)

    def preset_randomize_phase(self):
        if self.engine.fft_shifted is None:
            return
        amount = self.tools_panel.get_settings()["phase_random_amount"]
        self.engine.apply_global_phase_randomization(amount=amount)
        self.refresh_from_engine()

    def preset_swap_magnitude(self):
        if self.engine.fft_shifted is None:
            return
        if self.engine.secondary is None:
            self.load_secondary_image()
            if self.engine.secondary is None:
                return
        try:
            img = self.engine.swap_magnitude_with_secondary()
            self.refresh_from_engine()
            self.last_output = img
            self.set_label_image(self.reconstructed_view, img, keep_aspect=True)
        except Exception as exc:
            QMessageBox.warning(self, "Swap Magnitude", str(exc))

    def preset_apply_motion_blur(self):
        if self.engine.original is None:
            return
        img = self.engine.apply_motion_blur(a=0.08, b=0.08, T=1.0)
        self.refresh_from_engine()
        self.last_output = img
        self.set_label_image(self.reconstructed_view, img, keep_aspect=True)

    def preset_restore_wiener(self):
        if self.engine.original is None:
            return
        settings = self.preset_panel.get_settings()
        sigma = settings["sigma"] if self.engine.blur_transfer is None else None
        img = self.engine.apply_wiener_filter(
            K=settings["k"],
            sigma=sigma,
            kernel_size=settings["kernel_size"],
        )
        self.refresh_from_engine()
        self.last_output = img
        self.set_label_image(self.reconstructed_view, img, keep_aspect=True)

    # =====================================================
    # CLI
    # =====================================================

    def on_cli_command(self, command):
        self.cli_dock.append_output(f"> {command}")

        if self.engine.fft_shifted is None and command.strip().lower() not in {"help"}:
            self.cli_dock.append_output("No image loaded.")
            return

        cmd = command.strip().lower()
        if cmd == "help":
            self.cli_dock.append_output("Commands: undo, redo, reset, reconstruct, or filter commands.")
            return

        if cmd == "undo":
            self.undo_fft()
            return
        if cmd == "redo":
            self.redo_fft()
            return
        if cmd == "reset":
            self.reset_fft()
            return
        if cmd == "reconstruct":
            self.reconstruct_current_fft()
            return

        H = parse_command(command, self.engine.fft_shifted.shape)
        if H is None:
            self.cli_dock.append_output("Invalid command.")
            return

        self.engine.apply_filter(H, push_undo=True)
        self.refresh_from_engine()
        self.cli_dock.append_output("Filter applied.")

    # =====================================================
    # Help
    # =====================================================

    def show_help(self):
        QMessageBox.information(
            self,
            "Help",
            "Fourier Domain Image Analyzer\n\n"
            "1) Load an image from File menu.\n"
            "2) Use Preset Filters panel and optional Live Preview.\n"
            "3) Select a Fourier tool in toolbar and click spectrum.\n"
            "4) Use Presets menu for quick educational operations.\n"
            "5) Inspector panel shows (u,v), magnitude, phase, and D(u,v).\n"
            "6) CLI terminal is optional and can be toggled from View menu.",
        )


FourierGUI = MainWindow
