import argparse
import os
import sys


def _choose_ui_interactive():
    print("Select UI mode:")
    print("1) PySide6 UI (existing)")
    print("2) PySide6 Light Preset UI (feature-parity)")
    selected = input("Enter 1 or 2 [default: 1]: ").strip()
    return selected if selected in {"1", "2"} else "1"


def _run_pyside_ui():
    from PySide6.QtWidgets import QApplication

    from gui import FourierGUI

    app = QApplication(sys.argv)
    window = FourierGUI()
    window.show()
    return app.exec()


def _run_light_ui():
    from PySide6.QtWidgets import QApplication

    from gui_light import FourierGUILight

    app = QApplication(sys.argv)
    window = FourierGUILight()
    window.show()
    return app.exec()


def main():
    parser = argparse.ArgumentParser(description="Fourier Analyzer UI launcher")
    parser.add_argument("--ui", choices=["1", "2"], help="UI mode: 1=PySide6 Dark, 2=PySide6 Light")
    args = parser.parse_args()

    ui_choice = args.ui or os.environ.get("FOURIER_UI")
    if ui_choice not in {"1", "2"}:
        ui_choice = _choose_ui_interactive()

    if ui_choice == "2":
        return _run_light_ui()
    return _run_pyside_ui()


if __name__ == "__main__":
    sys.exit(main())
