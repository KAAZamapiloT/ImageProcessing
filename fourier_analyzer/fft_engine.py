"""******************************************************************************
* File: fft_engine.py
* Description:
* Core FFT Engine.
*
* Handles:
*  - Image loading
*  - Forward FFT
*  - Magnitude / Phase extraction
*  - Filter application
*  - Inverse FFT reconstruction
*
* Based on Chapter 4:
* Filtering in the Frequency Domain
******************************************************************************"""

import numpy as np


class FFTEngine:
    def __init__(self):
        self.original = None
        self.fft_shifted = None

    def load_image(self, img):
        self.original = img.astype(np.float32)

    def compute_fft(self):
        f = np.fft.fft2(self.original)
        self.fft_shifted = np.fft.fftshift(f)

    def magnitude(self):
        return np.log(1 + np.abs(self.fft_shifted))

    def phase(self):
        return np.angle(self.fft_shifted)

    def apply_filter(self, H):
        self.fft_shifted = self.fft_shifted * H

    def reconstruct(self):
        f_ishift = np.fft.ifftshift(self.fft_shifted)
        img_back = np.fft.ifft2(f_ishift)
        return np.abs(img_back)

    def reset(self):
        self.compute_fft()
