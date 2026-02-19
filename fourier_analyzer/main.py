import sys

from gui import FourierGUI
from PySide6.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FourierGUI()
    window.show()
    sys.exit(app.exec())
