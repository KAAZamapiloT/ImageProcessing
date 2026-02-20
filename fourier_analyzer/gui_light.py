"""Light theme variant of the full-featured Fourier analyzer UI.

UI 2 reuses the same MainWindow architecture as UI 1 so feature parity is guaranteed.
Only theme and a few sizing defaults are changed for a bright preset.
"""

from gui import MainWindow


class LightMainWindow(MainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fourier Domain Image Analyzer (Light Preset)")

        # Slightly smaller preview minimums to keep all panels visible.
        self.original_view.setMinimumSize(240, 240)
        self.spectrum_view.setMinimumSize(240, 240)
        self.reconstructed_view.setMinimumSize(240, 240)

    def _apply_theme(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #f3f3f3;
                color: #1f1f1f;
            }
            QMenuBar, QMenu {
                background-color: #ffffff;
                color: #1f1f1f;
                border: 1px solid #d0d0d0;
            }
            QMenuBar::item:selected, QMenu::item:selected {
                background-color: #dceeff;
            }
            QLabel {
                background-color: #ffffff;
                border: 1px solid #cfcfcf;
                padding: 3px;
            }
            QLabel#StatusLabel {
                background-color: #ffffff;
                border: 1px solid #d6d6d6;
                color: #005f9f;
            }
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #c6c6c6;
                border-radius: 0px;
                padding: 5px 8px;
                color: #1f1f1f;
            }
            QPushButton:hover {
                border-color: #007acc;
            }
            QPushButton:checked, QPushButton:pressed {
                background-color: #007acc;
                color: #ffffff;
                border-color: #007acc;
            }
            QComboBox, QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {
                background-color: #ffffff;
                border: 1px solid #c6c6c6;
                border-radius: 0px;
                color: #1f1f1f;
                padding: 3px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #c6c6c6;
                height: 6px;
                background: #efefef;
            }
            QSlider::handle:horizontal {
                background: #007acc;
                border: 1px solid #0062a2;
                width: 12px;
                margin: -4px 0;
                border-radius: 0px;
            }
            QToolBar {
                background-color: #ffffff;
                border: 1px solid #d0d0d0;
                spacing: 4px;
            }
            QDockWidget {
                border: 1px solid #d0d0d0;
            }
            QDockWidget::title {
                background: #ffffff;
                border: 1px solid #d0d0d0;
                padding: 4px;
            }
            QPlainTextEdit#TerminalOutput {
                background-color: #111111;
                color: #cfcfcf;
                border: 1px solid #333333;
                font-family: Consolas, 'Courier New', monospace;
            }
            """
        )


FourierGUILight = LightMainWindow

