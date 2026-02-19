"""******************************************************************************
 * File: utils.py
 * Author: Fourier Analyzer Project
 * Description:
 * Utility functions for:
 *  - Image normalization
 *  - Grayscale enforcement
 *  - Numpy to Qt conversion
 *
 * This file contains helper functions used across the application.
 ******************************************************************************"""

import cv2
import numpy as np
from PySide6.QtGui import QImage


def ensure_grayscale(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def normalize_image(img):
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img_norm.astype(np.uint8)


def numpy_to_qimage(img):
    img = normalize_image(img)
    h, w = img.shape
    return QImage(img.data, w, h, w, QImage.Format_Grayscale8)
