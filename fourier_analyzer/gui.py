import cv2
from cli_parser import parse_command
from fft_engine import FFTEngine
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from utils import ensure_grayscale, numpy_to_qimage


class FourierGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fourier Domain Analyzer")
        self.resize(1200, 800)

        self.engine = FFTEngine()
        self.last_output = None

        # Image labels
        self.label_original = QLabel("Original")
        self.label_magnitude = QLabel("Magnitude")
        self.label_recon = QLabel("Reconstructed")

        # Buttons
        btn_load = QPushButton("Load Image")
        btn_apply = QPushButton("Apply CLI Filter")
        btn_save = QPushButton("Save Output")
        btn_reset = QPushButton("Reset FFT")

        btn_load.clicked.connect(self.load_image)
        btn_apply.clicked.connect(self.apply_filter)
        btn_save.clicked.connect(self.save_output)
        btn_reset.clicked.connect(self.reset_fft)

        # CLI
        self.cli_input = QTextEdit()
        self.cli_input.setPlaceholderText(
            "gaussian 40\n"
            "butterworth 60 2\n"
            "highpass 50\n"
            "bandpass 20 80\n"
            "notch 120 80 10"
        )

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(200)
        self.slider.setValue(40)
        self.slider.valueChanged.connect(self.live_preview)

        # Layout
        img_layout = QHBoxLayout()
        img_layout.addWidget(self.label_original)
        img_layout.addWidget(self.label_magnitude)
        img_layout.addWidget(self.label_recon)

        control_layout = QVBoxLayout()
        control_layout.addWidget(btn_load)
        control_layout.addWidget(btn_apply)
        control_layout.addWidget(btn_reset)
        control_layout.addWidget(btn_save)
        control_layout.addWidget(self.slider)
        control_layout.addWidget(self.cli_input)

        main_layout = QVBoxLayout()
        main_layout.addLayout(img_layout)
        main_layout.addLayout(control_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def display_image(self, label, img):
        qimg = numpy_to_qimage(img)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(350, 350, Qt.KeepAspectRatio))

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName()
        if path:
            img = cv2.imread(path)
            img = ensure_grayscale(img)

            self.engine.load_image(img)
            self.engine.compute_fft()

            self.display_image(self.label_original, img)
            self.display_image(self.label_magnitude, self.engine.magnitude())

    def apply_filter(self):
        command = self.cli_input.toPlainText()
        shape = self.engine.original.shape

        H = parse_command(command, shape)
        if H is None:
            return

        self.engine.reset()
        self.engine.apply_filter(H)

        recon = self.engine.reconstruct()
        self.last_output = recon

        self.display_image(self.label_magnitude, self.engine.magnitude())
        self.display_image(self.label_recon, recon)

    def reset_fft(self):
        self.engine.reset()
        self.display_image(self.label_magnitude, self.engine.magnitude())

    def live_preview(self):
        if self.engine.original is None:
            return

        D0 = self.slider.value()
        shape = self.engine.original.shape

        from filters import gaussian_lowpass

        H = gaussian_lowpass(shape, D0)

        self.engine.reset()
        self.engine.apply_filter(H)

        recon = self.engine.reconstruct()
        self.display_image(self.label_recon, recon)

    def save_output(self):
        if self.last_output is None:
            return

        path, _ = QFileDialog.getSaveFileName()
        if path:
            cv2.imwrite(path, self.last_output)
