"""
ImageEnhancement.py
====================
Grayscale image processing  +  Color image processing  +  Pseudocolor
Dependencies: numpy, tifffile, pillow, scipy (optional – for fast convolution)
  pip install numpy tifffile pillow
"""

import copy
import math
import os
from enum import Enum

import numpy as np
import tifffile as tiff

# ============================================================
#  SECTION 1 – GRAYSCALE IMAGE OBJECT  (original, unchanged)
# ============================================================


class ImageObject:
    """Loads and stores a single-channel (grayscale) TIFF."""

    def __init__(self, path: str):
        self.path = path
        self.data = tiff.imread(path)

        if self.data.ndim != 2:
            raise RuntimeError(
                f"ImageObject only supports grayscale TIFF. "
                f"Got shape {self.data.shape}. Use ColorImageObject for RGB."
            )

        self.height, self.width = self.data.shape
        self.bits_per_sample = self.data.dtype.itemsize * 8
        self.levels = 1 << self.bits_per_sample

        print(f"TIFF loaded: {self.width}x{self.height}, {self.bits_per_sample}-bit")
        print(f"Actual range: [{self.data.min()}, {self.data.max()}]")

    def save_tiff(self, path: str):
        tiff.imwrite(path, self.data)
        print(f"Saved TIFF: {path}")

    def save_tiff_8bit(self, path: str):
        mn, mx = self.data.min(), self.data.max()
        if mn == mx:
            out = np.full(self.data.shape, 128, dtype=np.uint8)
        else:
            out = ((self.data - mn) * 255.0 / (mx - mn)).clip(0, 255).astype(np.uint8)
        tiff.imwrite(path, out)
        print(f"Saved 8-bit TIFF: {path}")

    def compute_histogram(self):
        hist = [0] * self.levels
        for v in self.data.flat:
            if v < self.levels:
                hist[int(v)] += 1
        return hist


# ============================================================
#  SECTION 2 – GRAYSCALE EVENTS  (original, unchanged)
# ============================================================


class InvertImageEvent:
    def __init__(self, inp, out):
        self.inp = inp
        self.out = out

    def execute(self):
        img = ImageObject(self.inp)
        img.data = (img.levels - 1) - img.data
        img.save_tiff_8bit(self.out)


class LogTransformEvent:
    def __init__(self, inp, out):
        self.inp = inp
        self.out = out

    def execute(self):
        img = ImageObject(self.inp)
        c = (img.levels - 1) / math.log(img.levels)
        img.data = (c * np.log(img.data.astype(np.float64) + 1)).astype(img.data.dtype)
        img.save_tiff_8bit(self.out)


class GammaTransformEvent:
    def __init__(self, inp, out, gamma):
        self.inp = inp
        self.out = out
        self.gamma = gamma

    def execute(self):
        img = ImageObject(self.inp)
        max_val = img.data.max()
        norm = img.data.astype(np.float64) / max_val
        img.data = np.round((norm**self.gamma) * max_val).astype(img.data.dtype)
        img.save_tiff_8bit(self.out)


class PieceWiseContrastEvent:
    def __init__(self, inp, out, r1, s1, r2, s2):
        self.inp = inp
        self.out = out
        self.r1, self.s1, self.r2, self.s2 = r1, s1, r2, s2

    def execute(self):
        img = ImageObject(self.inp)
        L = img.levels - 1
        lut = np.zeros(img.levels, dtype=np.uint16)
        for r in range(img.levels):
            if r <= self.r1:
                s = (self.s1 / self.r1) * r
            elif r <= self.r2:
                s = ((self.s2 - self.s1) / (self.r2 - self.r1)) * (
                    r - self.r1
                ) + self.s1
            else:
                s = ((L - self.s2) / (L - self.r2)) * (r - self.r2) + self.s2
            lut[r] = int(np.clip(s, 0, L))
        img.data = lut[img.data]
        img.save_tiff_8bit(self.out)


class IntensityRampEvent:
    def __init__(self, inp, out, start, end):
        self.inp = inp
        self.out = out
        self.start = start
        self.end = end

    def execute(self):
        img = ImageObject(self.inp)
        L = img.levels - 1
        lut = np.zeros(img.levels, dtype=np.uint16)
        slope = L / (self.end - self.start)
        for r in range(img.levels):
            if r < self.start:
                lut[r] = 0
            elif r > self.end:
                lut[r] = L
            else:
                lut[r] = int(slope * (r - self.start))
        img.data = lut[img.data]
        img.save_tiff(self.out)


class SliceMode(Enum):
    WITHOUT_BG = 0
    WITH_BG = 1


class IntensityLevelSlicingEvent:
    def __init__(self, inp, out, r1, r2, k, mode):
        self.inp = inp
        self.out = out
        self.r1, self.r2, self.k = r1, r2, k
        self.mode = SliceMode.WITH_BG if mode == "bg" else SliceMode.WITHOUT_BG

    def execute(self):
        img = ImageObject(self.inp)
        lut = np.arange(img.levels, dtype=np.uint16)  # default: identity (WITH_BG)
        if self.mode == SliceMode.WITHOUT_BG:
            lut[:] = 0
        lut[self.r1 : self.r2 + 1] = self.k
        img.data = lut[img.data]
        img.save_tiff(self.out)


class BitPlaneSliceEvent:
    def __init__(self, inp, out, bit, mode):
        self.inp = inp
        self.out = out
        self.bit = bit
        self.with_bg = mode == "bg"

    def execute(self):
        img = ImageObject(self.inp)
        max_val = img.levels - 1
        mask = (img.data >> self.bit) & 1
        img.data = np.where(mask, max_val, img.data if self.with_bg else 0).astype(
            img.data.dtype
        )
        img.save_tiff_8bit(self.out)


class HistogramEqualizationEvent:
    def __init__(self, inp, out):
        self.inp = inp
        self.out = out

    def execute(self):
        img = ImageObject(self.inp)
        hist, _ = np.histogram(
            img.data.flatten(), bins=img.levels, range=(0, img.levels)
        )
        cdf = hist.cumsum() / hist.sum()
        lut = np.round(cdf * (img.levels - 1)).astype(np.uint16)
        img.data = lut[img.data]
        img.save_tiff_8bit(self.out)


class HistogramStatsEvent:
    def __init__(self, inp):
        self.inp = inp

    def execute(self):
        img = ImageObject(self.inp)
        self._analyze(img)

    def _analyze(self, img):
        L = img.levels
        N = img.width * img.height
        hist = img.compute_histogram()
        min_level = next((i for i, h in enumerate(hist) if h > 0), L)
        max_level = next((i for i in range(L - 1, -1, -1) if hist[i] > 0), 0)
        mean = sum(i * hist[i] for i in range(L)) / N
        variance = sum((i - mean) ** 2 * hist[i] for i in range(L)) / N
        entropy = -sum(
            (hist[i] / N) * math.log2(hist[i] / N) for i in range(L) if hist[i] > 0
        )
        print(f"\n--- Histogram Statistics ---")
        print(f"Size    : {img.width} x {img.height}")
        print(f"Bits    : {img.bits_per_sample}")
        print(f"Levels  : {L}")
        print(f"Range   : [{min_level}, {max_level}]")
        print(f"Mean    : {mean:.4f}")
        print(f"Variance: {variance:.4f}")
        print(f"Entropy : {entropy:.4f} bits")
        self._print_compact_histogram(hist, N)

    def _print_compact_histogram(self, hist, N):
        bins, L = 16, len(hist)
        step = L // bins
        print("\nHistogram (compressed):")
        for b in range(bins):
            count = sum(hist[b * step : (b + 1) * step])
            pct = 100.0 * count / N
            print(
                f"[{b * step}-{(b + 1) * step - 1}] : {'#' * int(pct / 2)} ({pct:.2f}%)"
            )


class HistogramMatchingEvent:
    def __init__(self, src, ref, out):
        self.src = src
        self.ref = ref
        self.out = out

    def execute(self):
        src_img = ImageObject(self.src)
        ref_img = ImageObject(self.ref)
        if src_img.levels != ref_img.levels:
            raise RuntimeError("Source and reference must have same bit depth")
        self._match(src_img, ref_img)
        src_img.save_tiff_8bit(self.out)

    def _match(self, src, ref):
        L = src.levels
        Ns = src.width * src.height
        Nr = ref.width * ref.height
        histS = src.compute_histogram()
        histR = ref.compute_histogram()
        cdfS = np.cumsum([h / Ns for h in histS])
        cdfR = np.cumsum([h / Nr for h in histR])
        lut = np.zeros(L, dtype=np.uint16)
        r = 0
        for s in range(L):
            while r < L - 1 and cdfR[r] < cdfS[s]:
                r += 1
            lut[s] = r
        src.data = lut[src.data]


class LocalHistogramEnhancementEvent:
    def __init__(self, inp, out, window_size):
        if window_size < 3 or window_size % 2 == 0:
            raise RuntimeError("Window must be odd and >= 3")
        self.inp = inp
        self.out = out
        self.window = window_size

    def execute(self):
        img = ImageObject(self.inp)
        self._enhance(img)
        img.save_tiff_8bit(self.out)

    def _enhance(self, img):
        from numpy.lib.stride_tricks import sliding_window_view

        L, H, W, r = img.levels, img.height, img.width, self.window // 2
        # pad with edge values
        pad = np.pad(img.data, r, mode="edge")
        out = np.zeros((H, W), dtype=np.uint16)
        for y in range(H):
            for x in range(W):
                window = pad[y : y + self.window, x : x + self.window].flatten()
                hist = np.bincount(window.astype(np.int64), minlength=L)
                center = img.data[y, x]
                cdf = hist[: center + 1].sum()
                out[y, x] = int(round(cdf * (L - 1) / len(window)))
        img.data = out


class BoxSmoothingEvent:
    def __init__(self, inp, out, kernel):
        if kernel < 3 or kernel % 2 == 0:
            raise RuntimeError("Kernel must be odd and >= 3")
        self.inp = inp
        self.out = out
        self.kernel = kernel

    def execute(self):
        img = ImageObject(self.inp)
        self._smooth(img)
        img.save_tiff_8bit(self.out)

    def _smooth(self, img):
        from scipy.ndimage import uniform_filter

        # scipy uniform_filter is equivalent to box average with edge padding
        img.data = uniform_filter(img.data.astype(np.float64), size=self.kernel).astype(
            img.data.dtype
        )


class GaussianLowPassEvent:
    def __init__(self, inp, out, kernel_size, sigma):
        if kernel_size >= 3 and kernel_size % 2 == 0:
            raise RuntimeError("Kernel must be odd and >= 3")
        if sigma <= 0.0:
            raise RuntimeError("Sigma must be > 0")
        self.inp = inp
        self.out = out
        self.kernel_size = kernel_size
        self.sigma = sigma

    def execute(self):
        img = ImageObject(self.inp)
        self._apply_gaussian(img)
        img.save_tiff_8bit(self.out)

    def _build_kernel_1d(self):
        r = self.kernel_size // 2
        k = np.array(
            [math.exp(-(i * i) / (2 * self.sigma**2)) for i in range(-r, r + 1)],
            dtype=np.float64,
        )
        return k / k.sum()

    def _apply_gaussian(self, img):
        """Separable Gaussian: horizontal then vertical pass."""
        k = self._build_kernel_1d()
        r = self.kernel_size // 2
        H, W, L = img.height, img.width, img.levels
        pad_h = np.pad(img.data.astype(np.float64), ((0, 0), (r, r)), mode="edge")
        temp = np.zeros((H, W), dtype=np.float64)
        for x in range(W):
            temp[:, x] = (pad_h[:, x : x + self.kernel_size] * k).sum(axis=1)
        pad_v = np.pad(temp, ((r, r), (0, 0)), mode="edge")
        out = np.zeros((H, W), dtype=np.float64)
        for y in range(H):
            out[y, :] = (pad_v[y : y + self.kernel_size, :] * k[:, None]).sum(axis=0)
        img.data = np.clip(out, 0, L - 1).astype(img.data.dtype)


class HighPassSharpenEvent:
    def __init__(self, inp, out, strength=1.0):
        if strength <= 0:
            raise RuntimeError("Strength must be > 0")
        self.inp = inp
        self.out = out
        self.strength = strength

    def execute(self):
        img = ImageObject(self.inp)
        self._sharpen(img)
        img.save_tiff_8bit(self.out)

    def _sharpen(self, img):
        from scipy.ndimage import convolve

        K = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
        lap = convolve(img.data.astype(np.float64), K, mode="nearest")
        img.data = np.clip(
            img.data.astype(np.float64) + self.strength * lap, 0, img.levels - 1
        ).astype(img.data.dtype)


class UnsharpHighboostEvent:
    def __init__(self, inp, out, A):
        if A < 1.0:
            raise RuntimeError("A must be >= 1.0")
        self.inp = inp
        self.out = out
        self.A = A

    def execute(self):
        img = ImageObject(self.inp)
        self._apply(img)
        img.save_tiff_8bit(self.out)

    def _apply(self, img):
        from scipy.ndimage import convolve

        G = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 16.0
        blurred = convolve(img.data.astype(np.float64), G, mode="nearest")
        img.data = np.clip(
            self.A * img.data.astype(np.float64) - blurred, 0, img.levels - 1
        ).astype(img.data.dtype)


class GradientEdgeEnhancementEvent:
    def __init__(self, inp, out, k):
        if k <= 0:
            raise RuntimeError("k must be > 0")
        self.inp = inp
        self.out = out
        self.k = k

    def execute(self):
        img = ImageObject(self.inp)
        self._enhance(img)
        img.save_tiff_8bit(self.out)

    def _enhance(self, img):
        from scipy.ndimage import convolve

        Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        Sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
        d = img.data.astype(np.float64)
        gx = convolve(d, Sx, mode="nearest")
        gy = convolve(d, Sy, mode="nearest")
        grad = np.abs(gx) + np.abs(gy)
        img.data = np.clip(d + self.k * grad, 0, img.levels - 1).astype(img.data.dtype)


class LaplacianSobelSharpenEvent:
    def __init__(self, inp, lap_out, sharp_out, sobel_out):
        self.inp = inp
        self.lap_out = lap_out
        self.sharp_out = sharp_out
        self.sobel_out = sobel_out

    def execute(self):
        img = ImageObject(self.inp)
        lap = copy.deepcopy(img)
        sharp = copy.deepcopy(img)
        sobel = copy.deepcopy(img)
        self._apply_laplacian(img, lap)
        self._apply_sharpen(img, lap, sharp)
        self._apply_sobel(img, sobel)
        lap.save_tiff_8bit(self.lap_out)
        sharp.save_tiff_8bit(self.sharp_out)
        sobel.save_tiff_8bit(self.sobel_out)

    def _apply_laplacian(self, src, dst):
        from scipy.ndimage import convolve

        K = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
        dst.data = np.clip(
            convolve(src.data.astype(np.float64), K, mode="nearest"), 0, src.levels - 1
        ).astype(src.data.dtype)

    def _apply_sharpen(self, orig, lap, dst):
        dst.data = np.clip(
            orig.data.astype(np.int32) + lap.data.astype(np.int32), 0, orig.levels - 1
        ).astype(orig.data.dtype)

    def _apply_sobel(self, src, dst):
        from scipy.ndimage import convolve

        Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        Sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
        d = src.data.astype(np.float64)
        mag = np.abs(convolve(d, Sx, mode="nearest")) + np.abs(
            convolve(d, Sy, mode="nearest")
        )
        dst.data = np.clip(mag, 0, src.levels - 1).astype(src.data.dtype)


class MedianFilterEvent:
    def __init__(self, inp, out, window_size):
        if window_size < 3 or window_size % 2 == 0:
            raise RuntimeError("Window must be odd and >= 3")
        self.inp = inp
        self.out = out
        self.window = window_size

    def execute(self):
        img = ImageObject(self.inp)
        self._apply(img)
        img.save_tiff_8bit(self.out)

    def _apply(self, img):
        from scipy.ndimage import median_filter

        img.data = median_filter(img.data, size=self.window).astype(img.data.dtype)


class RobertsEdgeEvent:
    def __init__(self, inp, out):
        self.inp = inp
        self.out = out

    def execute(self):
        img = ImageObject(self.inp)
        self._apply(img)
        img.save_tiff_8bit(self.out)

    def _apply(self, img):
        d = img.data.astype(np.int32)
        gx = d[:-1, :-1] - d[1:, 1:]
        gy = d[:-1, 1:] - d[1:, :-1]
        mag = np.abs(gx) + np.abs(gy)
        out = np.zeros_like(img.data, dtype=np.int32)
        out[:-1, :-1] = mag
        img.data = np.clip(out, 0, img.levels - 1).astype(img.data.dtype)


class PrewittEdgeEvent:
    def __init__(self, inp, out):
        self.inp = inp
        self.out = out

    def execute(self):
        img = ImageObject(self.inp)
        self._apply(img)
        img.save_tiff_8bit(self.out)

    def _apply(self, img):
        from scipy.ndimage import convolve

        Gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
        Gy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
        d = img.data.astype(np.float64)
        mag = np.abs(convolve(d, Gx, mode="nearest")) + np.abs(
            convolve(d, Gy, mode="nearest")
        )
        img.data = np.clip(mag, 0, img.levels - 1).astype(img.data.dtype)


class SobelEdgeEvent:
    def __init__(self, inp, out, threshold=0):
        self.inp = inp
        self.out = out
        self.threshold = threshold

    def execute(self):
        img = ImageObject(self.inp)
        self._apply(img)
        img.save_tiff_8bit(self.out)

    def _apply(self, img):
        from scipy.ndimage import convolve

        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
        d = img.data.astype(np.float64)
        mag = np.abs(convolve(d, Gx, mode="nearest")) + np.abs(
            convolve(d, Gy, mode="nearest")
        )
        mx = mag.max()
        norm = np.clip(
            (mag * (img.levels - 1) / mx) if mx > 0 else mag, 0, img.levels - 1
        )
        if self.threshold > 0:
            img.data = np.where(norm >= self.threshold, img.levels - 1, 0).astype(
                img.data.dtype
            )
        else:
            img.data = norm.astype(img.data.dtype)


class LaplacianMode(Enum):
    FOUR = 4
    EIGHT = 8


class LaplacianSharpenEvent:
    def __init__(self, inp, lap_out, sharp_out, mode: LaplacianMode):
        self.inp = inp
        self.lap_out = lap_out
        self.sharp_out = sharp_out
        self.mode = mode

    def execute(self):
        img = ImageObject(self.inp)
        lap = copy.deepcopy(img)
        sharp = copy.deepcopy(img)
        self._apply_laplacian(img, lap)
        self._apply_sharpen(img, lap, sharp)
        lap.save_tiff_8bit(self.lap_out)
        sharp.save_tiff_8bit(self.sharp_out)

    def _apply_laplacian(self, src, dst):
        from scipy.ndimage import convolve

        K4 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
        K8 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float64)
        K = K4 if self.mode == LaplacianMode.FOUR else K8
        dst.data = np.clip(
            convolve(src.data.astype(np.float64), K, mode="nearest"), 0, src.levels - 1
        ).astype(src.data.dtype)

    def _apply_sharpen(self, orig, lap, dst):
        dst.data = np.clip(
            orig.data.astype(np.int32) + lap.data.astype(np.int32), 0, orig.levels - 1
        ).astype(orig.data.dtype)


class BandMode(Enum):
    BANDPASS = 1
    BANDREJECT = 2


class BandFilterEvent:
    def __init__(self, inp, out, k1, s1, k2, s2, mode: BandMode):
        self.inp = inp
        self.out = out
        self.k1, self.s1, self.k2, self.s2 = k1, s1, k2, s2
        self.mode = mode

    def execute(self):
        img = ImageObject(self.inp)
        lp1 = copy.deepcopy(img)
        lp2 = copy.deepcopy(img)
        GaussianLowPassEvent("", "", self.k1, self.s1)._apply_gaussian(lp1)
        GaussianLowPassEvent("", "", self.k2, self.s2)._apply_gaussian(lp2)
        self._apply(img, lp1, lp2)
        img.save_tiff_8bit(self.out)

    def _apply(self, img, lp1, lp2):
        L = img.levels - 1
        if self.mode == BandMode.BANDPASS:
            val = lp2.data.astype(np.float64) - lp1.data.astype(np.float64)
        else:
            val = lp1.data.astype(np.float64) + (
                img.data.astype(np.float64) - lp2.data.astype(np.float64)
            )
        img.data = np.clip(val, 0, L).astype(img.data.dtype)


class WeightedAveragingEvent:
    def __init__(self, inp, out):
        self.inp = inp
        self.out = out

    def execute(self):
        img = ImageObject(self.inp)
        self._apply(img)
        img.save_tiff_8bit(self.out)

    def _apply(self, img):
        from scipy.ndimage import convolve

        K = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 16.0
        img.data = np.clip(
            convolve(img.data.astype(np.float64), K, mode="nearest"), 0, img.levels - 1
        ).astype(img.data.dtype)


class GradientSharpenEvent:
    def __init__(self, inp, out, k):
        if k <= 0:
            raise RuntimeError("k must be > 0")
        self.inp = inp
        self.out = out
        self.k = k

    def execute(self):
        img = ImageObject(self.inp)
        self._sharpen(img)
        img.save_tiff_8bit(self.out)

    def _sharpen(self, img):
        from scipy.ndimage import convolve

        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
        d = img.data.astype(np.float64)
        grad = np.abs(convolve(d, Gx, mode="nearest")) + np.abs(
            convolve(d, Gy, mode="nearest")
        )
        img.data = np.clip(d + self.k * grad, 0, img.levels - 1).astype(img.data.dtype)


# ============================================================
#  SECTION 3 – COLOR IMAGE OBJECT
# ============================================================


class ColorImageObject:
    """
    Loads 1-channel (grayscale) or 3-channel (RGB) TIFF files.

    Internally stores data as a (H, W, 3) float32 array normalised to [0, 1].
    This makes all arithmetic clean and avoids dtype headaches.
    Use save_tiff_8bit() to write the result as an 8-bit RGB TIFF.
    """

    def __init__(self, path: str):
        self.path = path
        raw = tiff.imread(path).astype(np.float32)

        if raw.ndim == 2:  # grayscale → replicate to RGB
            self.is_grayscale = True
            norm = raw / float(
                (1 << (raw.dtype.itemsize * 8 if raw.dtype != np.float32 else 8)) - 1
            )
            # recompute: use actual max to normalise
            mx = raw.max()
            norm = raw / mx if mx > 0 else raw
            self.data = np.stack([norm, norm, norm], axis=-1)  # (H, W, 3)
        elif raw.ndim == 3 and raw.shape[2] in (3, 4):
            self.is_grayscale = False
            ch = raw[:, :, :3]
            mx = ch.max()
            self.data = (ch / mx) if mx > 0 else ch  # normalise to [0, 1]
        else:
            raise RuntimeError(f"Unsupported TIFF shape: {raw.shape}")

        self.height, self.width = self.data.shape[:2]
        print(
            f"ColorTIFF loaded: {self.width}x{self.height} "
            f"({'grayscale→RGB' if self.is_grayscale else 'RGB'})"
        )

    # ---- channel accessors (return (H,W) float32 views) ----
    @property
    def R(self) -> np.ndarray:
        return self.data[:, :, 0]

    @property
    def G(self) -> np.ndarray:
        return self.data[:, :, 1]

    @property
    def B(self) -> np.ndarray:
        return self.data[:, :, 2]

    @R.setter
    def R(self, v):
        self.data[:, :, 0] = v

    @G.setter
    def G(self, v):
        self.data[:, :, 1] = v

    @B.setter
    def B(self, v):
        self.data[:, :, 2] = v

    # ---- HSI conversion (vectorised) ----
    def to_hsi(self):
        """Returns (H, W, 3) array with channels H (radians), S, I all in [0,1]."""
        R, G, B = self.R, self.G, self.B
        I = (R + G + B) / 3.0
        mn = np.minimum(np.minimum(R, G), B)
        S = np.where(I > 0, 1.0 - mn / np.clip(I * 3.0, 1e-10, None), 0.0)

        # Hue
        num = 0.5 * ((R - G) + (R - B))
        den = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-10
        theta = np.arccos(np.clip(num / den, -1.0, 1.0))
        H = np.where(B <= G, theta, 2 * np.pi - theta)
        H = np.where(S < 1e-10, 0.0, H)  # hue undefined for achromatic

        hsi = np.stack([H, S, I], axis=-1).astype(np.float32)
        return hsi

    def from_hsi(self, hsi: np.ndarray):
        """Set data from (H, W, 3) HSI array (H in radians)."""
        H, S, I = hsi[:, :, 0], hsi[:, :, 1], hsi[:, :, 2]
        P2PI3 = 2 * np.pi / 3
        P4PI3 = 4 * np.pi / 3

        R = np.zeros_like(H)
        G = np.zeros_like(H)
        B = np.zeros_like(H)

        # Sector 1: 0 <= H < 2π/3
        m1 = (H >= 0) & (H < P2PI3)
        if m1.any():
            h = H[m1]
            i, s = I[m1], S[m1]
            b = i * (1 - s)
            r = i * (1 + s * np.cos(h) / (np.cos(np.pi / 3 - h) + 1e-10))
            g = 3 * i - (r + b)
            R[m1], G[m1], B[m1] = r, g, b

        # Sector 2: 2π/3 <= H < 4π/3
        m2 = (H >= P2PI3) & (H < P4PI3)
        if m2.any():
            h = H[m2] - P2PI3
            i, s = I[m2], S[m2]
            r = i * (1 - s)
            g = i * (1 + s * np.cos(h) / (np.cos(np.pi / 3 - h) + 1e-10))
            b = 3 * i - (r + g)
            R[m2], G[m2], B[m2] = r, g, b

        # Sector 3: 4π/3 <= H < 2π
        m3 = H >= P4PI3
        if m3.any():
            h = H[m3] - P4PI3
            i, s = I[m3], S[m3]
            g = i * (1 - s)
            b = i * (1 + s * np.cos(h) / (np.cos(np.pi / 3 - h) + 1e-10))
            r = 3 * i - (g + b)
            R[m3], G[m3], B[m3] = r, g, b

        self.data = np.stack(
            [
                np.clip(R, 0, 1),
                np.clip(G, 0, 1),
                np.clip(B, 0, 1),
            ],
            axis=-1,
        ).astype(np.float32)

    # ---- luminance (perceptual) ----
    def luminance(self) -> np.ndarray:
        return 0.299 * self.R + 0.587 * self.G + 0.114 * self.B

    # ---- save ----
    def save_tiff_8bit(self, path: str):
        out = np.clip(self.data * 255.0, 0, 255).astype(np.uint8)
        tiff.imwrite(path, out)
        print(f"Saved color TIFF: {path}")

    def save_channel_tiffs(self, r_path: str, g_path: str, b_path: str):
        for ch, p in zip([self.R, self.G, self.B], [r_path, g_path, b_path]):
            tiff.imwrite(p, np.clip(ch * 255, 0, 255).astype(np.uint8))
            print(f"Saved channel: {p}")


# ============================================================
#  SECTION 4 – COLORMAP HELPERS
# ============================================================


def _cmap_jet(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0, 1)
    R = np.clip(
        np.where(
            t < 0.375,
            0,
            np.where(
                t < 0.625,
                (t - 0.375) * 4,
                np.where(t < 0.875, 1.0, 1 - (t - 0.875) * 4),
            ),
        ),
        0,
        1,
    )
    G = np.clip(
        np.where(
            t < 0.125,
            0,
            np.where(
                t < 0.375,
                (t - 0.125) * 4,
                np.where(t < 0.625, 1.0, np.where(t < 0.875, 1 - (t - 0.625) * 4, 0)),
            ),
        ),
        0,
        1,
    )
    B = np.clip(
        np.where(
            t < 0.125,
            0.5 + t * 4,
            np.where(t < 0.375, 1.0, np.where(t < 0.625, 1 - (t - 0.375) * 4, 0)),
        ),
        0,
        1,
    )
    return np.stack([R, G, B], axis=-1)


def _cmap_hot(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0, 1)
    return np.stack(
        [
            np.clip(t * 3.0, 0, 1),
            np.clip(t * 3.0 - 1.0, 0, 1),
            np.clip(t * 3.0 - 2.0, 0, 1),
        ],
        axis=-1,
    )


def _cmap_cool(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0, 1)
    return np.stack([1 - t, t, np.ones_like(t)], axis=-1)


def _cmap_bone(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0, 1)
    return np.stack(
        [t * 0.875, t * 0.875, np.clip(t * 0.875 + t * 0.125, 0, 1)], axis=-1
    )


def _cmap_spring(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0, 1)
    return np.stack([np.ones_like(t), t, 1 - t], axis=-1)


def _cmap_gray(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0, 1)
    return np.stack([t, t, t], axis=-1)


_COLORMAPS = {
    "jet": _cmap_jet,
    "hot": _cmap_hot,
    "cool": _cmap_cool,
    "bone": _cmap_bone,
    "spring": _cmap_spring,
    "gray": _cmap_gray,
}


# ============================================================
#  SECTION 5 – PSEUDOCOLOR EVENTS  (grayscale → color)
# ============================================================


class PseudoColorEvent:
    """
    Map a grayscale TIFF through a named colormap and write an RGB TIFF.
    Available colormaps: jet, hot, cool, bone, spring, gray
    Usage:  pseudo_color <input> <output> <colormap>
    """

    def __init__(self, inp: str, out: str, colormap: str = "jet"):
        if colormap not in _COLORMAPS:
            raise RuntimeError(
                f"Unknown colormap '{colormap}'. Choose from: {list(_COLORMAPS.keys())}"
            )
        self.inp = inp
        self.out = out
        self.colormap = colormap

    def execute(self):
        raw = tiff.imread(self.inp)
        if raw.ndim != 2:
            raise RuntimeError("pseudo_color expects a grayscale input image")
        mn, mx = raw.min(), raw.max()
        t = (
            (raw.astype(np.float64) - mn) / (mx - mn)
            if mx > mn
            else np.zeros_like(raw, dtype=np.float64)
        )
        rgb = (_COLORMAPS[self.colormap](t.astype(np.float32)) * 255).astype(np.uint8)
        tiff.imwrite(self.out, rgb)
        print(f"Pseudocolor [{self.colormap}] done → {self.out}")


class DensitySliceEvent:
    """
    Assign a specific RGB color to each intensity band.
    Usage:  density_slice <input> <output> <N> <bg|nobg>
            Then provide N lines:  lo hi R G B   (R/G/B in 0-255)
    Example (N=2, bg):
        0  100  255  0  0
        101 200    0  255  0
    """

    def __init__(self, inp: str, out: str, bands: list, keep_background: bool = False):
        self.inp = inp
        self.out = out
        self.bands = bands  # list of (lo, hi, r, g, b)
        self.keep_bg = keep_background

    def execute(self):
        raw = tiff.imread(self.inp)
        if raw.ndim != 2:
            raise RuntimeError("density_slice expects a grayscale input image")
        H, W = raw.shape
        mn, mx = raw.min(), raw.max()
        norm = (raw.astype(np.float64) - mn) / (mx - mn + 1e-10)  # [0,1]
        # scale to 0-255 for band matching
        scaled = (norm * 255).astype(np.uint8)

        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        if self.keep_bg:
            gray = (norm * 255).astype(np.uint8)
            rgb[:, :, 0] = gray
            rgb[:, :, 1] = gray
            rgb[:, :, 2] = gray

        for lo, hi, r, g, b in self.bands:
            mask = (scaled >= lo) & (scaled <= hi)
            rgb[mask] = [r, g, b]

        tiff.imwrite(self.out, rgb)
        print(f"Density slicing done ({len(self.bands)} bands) → {self.out}")


# ============================================================
#  SECTION 6 – COLOR ENHANCEMENT EVENTS
# ============================================================


class ColorInvertEvent:
    """Invert each channel: output = 1 - input."""

    def __init__(self, inp: str, out: str):
        self.inp, self.out = inp, out

    def execute(self):
        img = ColorImageObject(self.inp)
        img.data = 1.0 - img.data
        img.save_tiff_8bit(self.out)
        print("Color invert done.")


class ColorGammaEvent:
    """Apply gamma correction to all channels: out = in^gamma."""

    def __init__(self, inp: str, out: str, gamma: float):
        self.inp, self.out, self.gamma = inp, out, gamma

    def execute(self):
        img = ColorImageObject(self.inp)
        img.data = np.clip(np.power(img.data, self.gamma), 0, 1).astype(np.float32)
        img.save_tiff_8bit(self.out)
        print(f"Color gamma({self.gamma}) done.")


class ColorLogEvent:
    """Apply log transform to all channels: out = log(1 + in) / log(2)."""

    def __init__(self, inp: str, out: str):
        self.inp, self.out = inp, out

    def execute(self):
        img = ColorImageObject(self.inp)
        img.data = np.clip(np.log1p(img.data) / np.log(2.0), 0, 1).astype(np.float32)
        img.save_tiff_8bit(self.out)
        print("Color log transform done.")


class ColorBalanceEvent:
    """
    Independently scale R, G, B channels.
    Scales: 1.0 = unchanged, > 1 = boost, < 1 = reduce.
    Usage: color_balance <in> <out> <r_scale> <g_scale> <b_scale>
    """

    def __init__(
        self, inp: str, out: str, r_scale: float, g_scale: float, b_scale: float
    ):
        self.inp, self.out = inp, out
        self.rs, self.gs, self.bs = r_scale, g_scale, b_scale

    def execute(self):
        img = ColorImageObject(self.inp)
        img.R = np.clip(img.R * self.rs, 0, 1)
        img.G = np.clip(img.G * self.gs, 0, 1)
        img.B = np.clip(img.B * self.bs, 0, 1)
        img.save_tiff_8bit(self.out)
        print(f"Color balance R*{self.rs} G*{self.gs} B*{self.bs} done.")


class ColorContrastEvent:
    """
    Piecewise linear contrast stretch applied identically to all channels.
    Usage: color_contrast <in> <out> <r1> <s1> <r2> <s2>   (values in 0-255)
    """

    def __init__(self, inp: str, out: str, r1: int, s1: int, r2: int, s2: int):
        self.inp, self.out = inp, out
        self.r1, self.s1, self.r2, self.s2 = r1, s1, r2, s2

    def execute(self):
        img = ColorImageObject(self.inp)
        # Build LUT in [0,255] space, then normalise
        lut = np.zeros(256, dtype=np.float32)
        L = 255
        for r in range(256):
            if r <= self.r1:
                s = (self.s1 / max(self.r1, 1)) * r
            elif r <= self.r2:
                s = ((self.s2 - self.s1) / max(self.r2 - self.r1, 1)) * (
                    r - self.r1
                ) + self.s1
            else:
                s = ((L - self.s2) / max(L - self.r2, 1)) * (r - self.r2) + self.s2
            lut[r] = np.clip(s, 0, L) / 255.0
        idx = np.clip((img.data * 255).astype(np.int32), 0, 255)
        img.data = lut[idx].astype(np.float32)
        img.save_tiff_8bit(self.out)
        print("Color contrast stretch done.")


class ColorHistEqChannelEvent:
    """
    Histogram equalisation applied independently to each R, G, B channel.
    Note: can shift hues. Use color_hist_eq_hsi to avoid hue drift.
    """

    def __init__(self, inp: str, out: str):
        self.inp, self.out = inp, out

    def execute(self):
        img = ColorImageObject(self.inp)
        for c in range(3):
            ch = (img.data[:, :, c] * 255).astype(np.uint8)
            hist, _ = np.histogram(ch.flatten(), bins=256, range=(0, 256))
            cdf = hist.cumsum() / hist.sum()
            lut = (cdf * 255).astype(np.uint8)
            img.data[:, :, c] = lut[ch].astype(np.float32) / 255.0
        img.save_tiff_8bit(self.out)
        print("Color histogram equalization (per-channel) done.")


class ColorHistEqHSIEvent:
    """
    Histogram equalisation in HSI space – only the Intensity channel is
    equalised. Hue and Saturation are untouched, so colours are preserved.
    This is the recommended method for colour images.
    """

    def __init__(self, inp: str, out: str):
        self.inp, self.out = inp, out

    def execute(self):
        img = ColorImageObject(self.inp)
        hsi = img.to_hsi()
        I_q = np.clip((hsi[:, :, 2] * 255), 0, 255).astype(np.uint8)
        hist, _ = np.histogram(I_q.flatten(), bins=256, range=(0, 256))
        cdf = hist.cumsum() / hist.sum()
        lut = (cdf * 255).astype(np.uint8)
        hsi[:, :, 2] = lut[I_q].astype(np.float32) / 255.0
        img.from_hsi(hsi)
        img.save_tiff_8bit(self.out)
        print("Color histogram equalization (HSI) done.")


class HSISaturationEvent:
    """
    Scale the Saturation channel in HSI space.
    scale=0 → grayscale, scale=1 → unchanged, scale>1 → vivid colours.
    Usage: hsi_saturate <in> <out> <scale>
    """

    def __init__(self, inp: str, out: str, scale: float):
        if scale < 0:
            raise RuntimeError("Scale must be >= 0")
        self.inp, self.out, self.scale = inp, out, scale

    def execute(self):
        img = ColorImageObject(self.inp)
        hsi = img.to_hsi()
        hsi[:, :, 1] = np.clip(hsi[:, :, 1] * self.scale, 0, 1)
        img.from_hsi(hsi)
        img.save_tiff_8bit(self.out)
        print(f"HSI saturation scale={self.scale} done.")


class HSIHueRotateEvent:
    """
    Rotate all hue values by a fixed number of degrees.
    Usage: hsi_hue_rotate <in> <out> <degrees>
    """

    def __init__(self, inp: str, out: str, degrees: float):
        self.inp, self.out = inp, out
        self.shift = degrees * np.pi / 180.0

    def execute(self):
        img = ColorImageObject(self.inp)
        hsi = img.to_hsi()
        hsi[:, :, 0] = np.mod(hsi[:, :, 0] + self.shift, 2 * np.pi)
        img.from_hsi(hsi)
        img.save_tiff_8bit(self.out)
        print(f"HSI hue rotation done.")


# ============================================================
#  SECTION 7 – COLOR SPATIAL FILTERING
# ============================================================


class ColorSmoothBoxEvent:
    """Box (mean) filter applied to each channel independently."""

    def __init__(self, inp: str, out: str, kernel: int):
        if kernel < 3 or kernel % 2 == 0:
            raise RuntimeError("Kernel must be odd and >= 3")
        self.inp, self.out, self.k = inp, out, kernel

    def execute(self):
        from scipy.ndimage import uniform_filter

        img = ColorImageObject(self.inp)
        for c in range(3):
            img.data[:, :, c] = np.clip(
                uniform_filter(img.data[:, :, c].astype(np.float64), size=self.k), 0, 1
            )
        img.save_tiff_8bit(self.out)
        print(f"Color box smooth (k={self.k}) done.")


class ColorGaussianEvent:
    """Gaussian smoothing applied to each channel independently."""

    def __init__(self, inp: str, out: str, kernel_size: int, sigma: float):
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise RuntimeError("Kernel must be odd and >= 3")
        self.inp, self.out = inp, out
        self.ks, self.sigma = kernel_size, sigma

    def _build_kernel(self):
        r = self.ks // 2
        k = np.array(
            [math.exp(-(i * i) / (2 * self.sigma**2)) for i in range(-r, r + 1)]
        )
        return k / k.sum()

    def execute(self):
        img = ColorImageObject(self.inp)
        k = self._build_kernel()
        r = self.ks // 2
        for c in range(3):
            ch = img.data[:, :, c].astype(np.float64)
            pad_h = np.pad(ch, ((0, 0), (r, r)), mode="edge")
            tmp = np.zeros_like(ch)
            for x in range(img.width):
                tmp[:, x] = (pad_h[:, x : x + self.ks] * k).sum(axis=1)
            pad_v = np.pad(tmp, ((r, r), (0, 0)), mode="edge")
            out = np.zeros_like(ch)
            for y in range(img.height):
                out[y, :] = (pad_v[y : y + self.ks, :] * k[:, None]).sum(axis=0)
            img.data[:, :, c] = np.clip(out, 0, 1)
        img.save_tiff_8bit(self.out)
        print(f"Color Gaussian (k={self.ks}, σ={self.sigma}) done.")


class ColorSharpenEvent:
    """Laplacian high-pass sharpening applied to each channel independently."""

    def __init__(self, inp: str, out: str, strength: float = 1.0):
        self.inp, self.out, self.strength = inp, out, strength

    def execute(self):
        from scipy.ndimage import convolve

        img = ColorImageObject(self.inp)
        K = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
        for c in range(3):
            ch = img.data[:, :, c].astype(np.float64)
            lap = convolve(ch, K, mode="nearest")
            img.data[:, :, c] = np.clip(ch + self.strength * lap, 0, 1)
        img.save_tiff_8bit(self.out)
        print("Color sharpen done.")


class ColorMedianEvent:
    """Median filter applied to each channel independently."""

    def __init__(self, inp: str, out: str, window: int):
        if window < 3 or window % 2 == 0:
            raise RuntimeError("Window must be odd and >= 3")
        self.inp, self.out, self.window = inp, out, window

    def execute(self):
        from scipy.ndimage import median_filter

        img = ColorImageObject(self.inp)
        for c in range(3):
            img.data[:, :, c] = np.clip(
                median_filter(img.data[:, :, c], size=self.window), 0, 1
            )
        img.save_tiff_8bit(self.out)
        print(f"Color median (w={self.window}) done.")


class ColorUnsharpEvent:
    """Unsharp masking / highboost applied to each channel independently."""

    def __init__(self, inp: str, out: str, A: float):
        if A < 1.0:
            raise RuntimeError("A must be >= 1.0")
        self.inp, self.out, self.A = inp, out, A

    def execute(self):
        from scipy.ndimage import convolve

        img = ColorImageObject(self.inp)
        G = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 16.0
        for c in range(3):
            ch = img.data[:, :, c].astype(np.float64)
            blurred = convolve(ch, G, mode="nearest")
            img.data[:, :, c] = np.clip(self.A * ch - blurred, 0, 1)
        img.save_tiff_8bit(self.out)
        print(f"Color unsharp/highboost (A={self.A}) done.")


# ============================================================
#  SECTION 8 – COLOR EDGE DETECTION
# ============================================================


class ColorEdgeSobelEvent:
    """
    Sobel edge detection on the perceptual luminance channel
    (0.299R + 0.587G + 0.114B). Outputs an 8-bit grayscale TIFF.
    Usage: color_edge <in> <out> [threshold]
    """

    def __init__(self, inp: str, out: str, threshold: int = 0):
        self.inp, self.out, self.threshold = inp, out, threshold

    def execute(self):
        from scipy.ndimage import convolve

        img = ColorImageObject(self.inp)
        lum = img.luminance().astype(np.float64)
        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
        mag = np.abs(convolve(lum, Gx, mode="nearest")) + np.abs(
            convolve(lum, Gy, mode="nearest")
        )
        mx = mag.max()
        norm = (mag / mx * 255).astype(np.uint8) if mx > 0 else mag.astype(np.uint8)
        if self.threshold > 0:
            norm = np.where(norm >= self.threshold, 255, 0).astype(np.uint8)
        tiff.imwrite(self.out, norm)
        print(f"Color Sobel edge (luminance) done → {self.out}")


class ColorEdgePerChannelEvent:
    """
    Sobel edge detection on each R, G, B channel separately.
    Outputs a colour image where edges are highlighted in the channel's colour.
    Usage: color_edge_rgb <in> <out>
    """

    def __init__(self, inp: str, out: str):
        self.inp, self.out = inp, out

    def execute(self):
        from scipy.ndimage import convolve

        img = ColorImageObject(self.inp)
        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
        for c in range(3):
            ch = img.data[:, :, c].astype(np.float64)
            mag = np.abs(convolve(ch, Gx, mode="nearest")) + np.abs(
                convolve(ch, Gy, mode="nearest")
            )
            mx = mag.max()
            img.data[:, :, c] = np.clip(mag / mx if mx > 0 else mag, 0, 1)
        img.save_tiff_8bit(self.out)
        print(f"Color per-channel edge detection done → {self.out}")


# ============================================================
#  SECTION 9 – CHANNEL SPLIT / MERGE
# ============================================================


class SplitChannelsEvent:
    """
    Split an RGB TIFF into three separate grayscale TIFFs.
    Usage: split <in> <r_out> <g_out> <b_out>
    """

    def __init__(self, inp: str, r_out: str, g_out: str, b_out: str):
        self.inp = inp
        self.r_out, self.g_out, self.b_out = r_out, g_out, b_out

    def execute(self):
        img = ColorImageObject(self.inp)
        img.save_channel_tiffs(self.r_out, self.g_out, self.b_out)
        print("Channels split done.")


class MergeChannelsEvent:
    """
    Merge three grayscale TIFFs into a single RGB TIFF.
    Usage: merge <r_in> <g_in> <b_in> <out>
    """

    def __init__(self, r_in: str, g_in: str, b_in: str, out: str):
        self.r_in, self.g_in, self.b_in, self.out = r_in, g_in, b_in, out

    def execute(self):
        r = tiff.imread(self.r_in).astype(np.float32)
        g = tiff.imread(self.g_in).astype(np.float32)
        b = tiff.imread(self.b_in).astype(np.float32)

        if not (r.shape == g.shape == b.shape):
            raise RuntimeError("Channel images must have the same dimensions")
        if r.ndim != 2:
            raise RuntimeError("Channel images must be grayscale")

        # Normalise each channel independently
        def _norm(ch):
            mn, mx = ch.min(), ch.max()
            return (ch - mn) / (mx - mn) if mx > mn else np.zeros_like(ch)

        rgb = np.stack([_norm(r), _norm(g), _norm(b)], axis=-1)
        merged_obj = ColorImageObject.__new__(ColorImageObject)
        merged_obj.data = rgb.astype(np.float32)
        merged_obj.height, merged_obj.width = rgb.shape[:2]
        merged_obj.is_grayscale = False
        merged_obj.save_tiff_8bit(self.out)
        print(f"Channels merged → {self.out}")


# ============================================================
#  SECTION 10 – INPUT HANDLER
# ============================================================


class InputHandler:
    @staticmethod
    def run():
        while True:
            InputHandler._print_menu()
            parts = input("> ").strip().split()
            if not parts:
                continue
            cmd = parts[0]
            try:
                # ---- Grayscale commands (original) ----
                if cmd == "invert":
                    _, inp, out = parts
                    InvertImageEvent(inp, out).execute()

                elif cmd == "log":
                    _, inp, out = parts
                    LogTransformEvent(inp, out).execute()

                elif cmd == "gamma":
                    _, inp, out, g = parts
                    GammaTransformEvent(inp, out, float(g)).execute()

                elif cmd == "contrast":
                    _, inp, out, r1, s1, r2, s2 = parts
                    PieceWiseContrastEvent(
                        inp, out, int(r1), int(s1), int(r2), int(s2)
                    ).execute()

                elif cmd == "ramp":
                    _, inp, out, start, end = parts
                    IntensityRampEvent(inp, out, int(start), int(end)).execute()

                elif cmd == "slice":
                    _, inp, out, r1, r2, k, mode = parts
                    IntensityLevelSlicingEvent(
                        inp, out, int(r1), int(r2), int(k), mode
                    ).execute()

                elif cmd == "bit_slice":
                    _, inp, out, bit, mode = parts
                    BitPlaneSliceEvent(inp, out, int(bit), mode).execute()

                elif cmd == "hist_eq":
                    _, inp, out = parts
                    HistogramEqualizationEvent(inp, out).execute()

                elif cmd == "hist_stats":
                    _, inp = parts
                    HistogramStatsEvent(inp).execute()

                elif cmd == "hist_match":
                    _, src, ref, out = parts
                    HistogramMatchingEvent(src, ref, out).execute()

                elif cmd == "local_hist":
                    _, inp, out, w = parts
                    LocalHistogramEnhancementEvent(inp, out, int(w)).execute()

                elif cmd == "smooth_box":
                    _, inp, out, k = parts
                    BoxSmoothingEvent(inp, out, int(k)).execute()

                elif cmd == "gaussian":
                    _, inp, out, k, sigma = parts
                    GaussianLowPassEvent(inp, out, int(k), float(sigma)).execute()

                elif cmd == "sharpen":
                    _, inp, out, strength = parts
                    HighPassSharpenEvent(inp, out, float(strength)).execute()

                elif cmd == "unsharp":
                    _, inp, out, A = parts
                    UnsharpHighboostEvent(inp, out, float(A)).execute()

                elif cmd == "grad_edge":
                    _, inp, out, k = parts
                    GradientEdgeEnhancementEvent(inp, out, float(k)).execute()

                elif cmd == "lap_sobel":
                    _, inp, lap, sharp, sobel = parts
                    LaplacianSobelSharpenEvent(inp, lap, sharp, sobel).execute()

                elif cmd == "median":
                    _, inp, out, w = parts
                    MedianFilterEvent(inp, out, int(w)).execute()

                elif cmd == "roberts":
                    _, inp, out = parts
                    RobertsEdgeEvent(inp, out).execute()

                elif cmd == "prewitt":
                    _, inp, out = parts
                    PrewittEdgeEvent(inp, out).execute()

                elif cmd == "sobel":
                    if len(parts) == 4:
                        _, inp, out, t = parts
                        SobelEdgeEvent(inp, out, int(t)).execute()
                    else:
                        _, inp, out = parts
                        SobelEdgeEvent(inp, out).execute()

                elif cmd == "laplacian":
                    _, inp, lap, sharp, mode = parts
                    m = LaplacianMode.EIGHT if mode == "8" else LaplacianMode.FOUR
                    LaplacianSharpenEvent(inp, lap, sharp, m).execute()

                elif cmd == "bandpass":
                    _, inp, out, k1, s1, k2, s2 = parts
                    BandFilterEvent(
                        inp,
                        out,
                        int(k1),
                        float(s1),
                        int(k2),
                        float(s2),
                        BandMode.BANDPASS,
                    ).execute()

                elif cmd == "bandreject":
                    _, inp, out, k1, s1, k2, s2 = parts
                    BandFilterEvent(
                        inp,
                        out,
                        int(k1),
                        float(s1),
                        int(k2),
                        float(s2),
                        BandMode.BANDREJECT,
                    ).execute()

                elif cmd == "weighted_avg":
                    _, inp, out = parts
                    WeightedAveragingEvent(inp, out).execute()

                elif cmd == "grad_sharpen":
                    _, inp, out, k = parts
                    GradientSharpenEvent(inp, out, float(k)).execute()

                # ---- Pseudocolor commands ----
                elif cmd == "pseudo_color":
                    # pseudo_color <in> <out> <colormap>
                    _, inp, out, cmap = parts
                    PseudoColorEvent(inp, out, cmap).execute()

                elif cmd == "density_slice":
                    # density_slice <in> <out> <N> <bg|nobg>
                    # then N lines: lo hi R G B
                    _, inp, out, N_str, bg_mode = parts
                    N = int(N_str)
                    bands = []
                    for _ in range(N):
                        line = input("  band> ").strip().split()
                        lo, hi, r, g, b = (
                            int(line[0]),
                            int(line[1]),
                            int(line[2]),
                            int(line[3]),
                            int(line[4]),
                        )
                        bands.append((lo, hi, r, g, b))
                    DensitySliceEvent(inp, out, bands, bg_mode == "bg").execute()

                # ---- Color enhancement commands ----
                elif cmd == "color_invert":
                    _, inp, out = parts
                    ColorInvertEvent(inp, out).execute()

                elif cmd == "color_gamma":
                    _, inp, out, g = parts
                    ColorGammaEvent(inp, out, float(g)).execute()

                elif cmd == "color_log":
                    _, inp, out = parts
                    ColorLogEvent(inp, out).execute()

                elif cmd == "color_balance":
                    _, inp, out, rs, gs, bs = parts
                    ColorBalanceEvent(
                        inp, out, float(rs), float(gs), float(bs)
                    ).execute()

                elif cmd == "color_contrast":
                    _, inp, out, r1, s1, r2, s2 = parts
                    ColorContrastEvent(
                        inp, out, int(r1), int(s1), int(r2), int(s2)
                    ).execute()

                elif cmd == "color_hist_eq":
                    _, inp, out = parts
                    ColorHistEqChannelEvent(inp, out).execute()

                elif cmd == "color_hist_eq_hsi":
                    _, inp, out = parts
                    ColorHistEqHSIEvent(inp, out).execute()

                elif cmd == "hsi_saturate":
                    _, inp, out, scale = parts
                    HSISaturationEvent(inp, out, float(scale)).execute()

                elif cmd == "hsi_hue_rotate":
                    _, inp, out, deg = parts
                    HSIHueRotateEvent(inp, out, float(deg)).execute()

                # ---- Color filtering commands ----
                elif cmd == "color_smooth_box":
                    _, inp, out, k = parts
                    ColorSmoothBoxEvent(inp, out, int(k)).execute()

                elif cmd == "color_gaussian":
                    _, inp, out, k, sigma = parts
                    ColorGaussianEvent(inp, out, int(k), float(sigma)).execute()

                elif cmd == "color_sharpen":
                    _, inp, out, st = parts
                    ColorSharpenEvent(inp, out, float(st)).execute()

                elif cmd == "color_median":
                    _, inp, out, w = parts
                    ColorMedianEvent(inp, out, int(w)).execute()

                elif cmd == "color_unsharp":
                    _, inp, out, A = parts
                    ColorUnsharpEvent(inp, out, float(A)).execute()

                # ---- Color edge detection ----
                elif cmd == "color_edge":
                    if len(parts) == 4:
                        _, inp, out, t = parts
                        ColorEdgeSobelEvent(inp, out, int(t)).execute()
                    else:
                        _, inp, out = parts
                        ColorEdgeSobelEvent(inp, out).execute()

                elif cmd == "color_edge_rgb":
                    _, inp, out = parts
                    ColorEdgePerChannelEvent(inp, out).execute()

                # ---- Channel split / merge ----
                elif cmd == "split":
                    _, inp, r_out, g_out, b_out = parts
                    SplitChannelsEvent(inp, r_out, g_out, b_out).execute()

                elif cmd == "merge":
                    _, r_in, g_in, b_in, out = parts
                    MergeChannelsEvent(r_in, g_in, b_in, out).execute()

                elif cmd == "quit":
                    print("Exiting.")
                    break

                else:
                    print(f"Unknown command: {cmd}")

            except Exception as e:
                print(f"Error: {e}")

    @staticmethod
    def _print_menu():
        print("""
╔══════════════════════════════════════════════════════════════╗
║              GRAYSCALE COMMANDS (original)                   ║
╠══════════════════════════════════════════════════════════════╣
  invert <in> <out>
  log <in> <out>
  gamma <in> <out> <gamma>
  contrast <in> <out> <r1> <s1> <r2> <s2>
  ramp <in> <out> <start> <end>
  slice <in> <out> <lo> <hi> <val> <bg|nobg>
  bit_slice <in> <out> <bit_idx> <bg|nobg>
  hist_eq <in> <out>
  hist_stats <in>
  hist_match <src> <ref> <out>
  local_hist <in> <out> <window>
  smooth_box <in> <out> <k>
  gaussian <in> <out> <k> <sigma>
  sharpen <in> <out> <strength>
  unsharp <in> <out> <A>
  grad_edge <in> <out> <k>
  lap_sobel <in> <lap_out> <sharp_out> <sobel_out>
  median <in> <out> <window>
  roberts <in> <out>
  prewitt <in> <out>
  sobel <in> <out> [threshold]
  laplacian <in> <lap_out> <sharp_out> <4|8>
  bandpass <in> <out> <k1> <s1> <k2> <s2>
  bandreject <in> <out> <k1> <s1> <k2> <s2>
  weighted_avg <in> <out>
  grad_sharpen <in> <out> <k>

╠══════════════════════════════════════════════════════════════╣
║        PSEUDOCOLOR  (grayscale → color)                      ║
╠══════════════════════════════════════════════════════════════╣
  pseudo_color <in> <out> <jet|hot|cool|bone|spring|gray>
  density_slice <in> <out> <N> <bg|nobg>
    → then N lines:  lo hi R G B    (intensities & colours in 0-255)

╠══════════════════════════════════════════════════════════════╣
║             COLOR ENHANCEMENT                                ║
╠══════════════════════════════════════════════════════════════╣
  color_invert <in> <out>
  color_gamma <in> <out> <gamma>
  color_log <in> <out>
  color_balance <in> <out> <r_scale> <g_scale> <b_scale>
  color_contrast <in> <out> <r1> <s1> <r2> <s2>
  color_hist_eq <in> <out>          ← per-channel (may shift hues)
  color_hist_eq_hsi <in> <out>      ← HSI-space (recommended)
  hsi_saturate <in> <out> <scale>   ← 0=gray, 1=unchanged, >1=vivid
  hsi_hue_rotate <in> <out> <deg>

╠══════════════════════════════════════════════════════════════╣
║             COLOR SPATIAL FILTERS                            ║
╠══════════════════════════════════════════════════════════════╣
  color_smooth_box <in> <out> <k>
  color_gaussian <in> <out> <k> <sigma>
  color_sharpen <in> <out> <strength>
  color_median <in> <out> <window>
  color_unsharp <in> <out> <A>

╠══════════════════════════════════════════════════════════════╣
║             COLOR EDGE DETECTION                             ║
╠══════════════════════════════════════════════════════════════╣
  color_edge <in> <out> [threshold]   ← Sobel on luminance
  color_edge_rgb <in> <out>           ← Sobel per channel

╠══════════════════════════════════════════════════════════════╣
║             CHANNEL SPLIT / MERGE                            ║
╠══════════════════════════════════════════════════════════════╣
  split <in> <r_out> <g_out> <b_out>
  merge <r_in> <g_in> <b_in> <out>

  quit
╚══════════════════════════════════════════════════════════════╝""")


if __name__ == "__main__":
    InputHandler.run()
