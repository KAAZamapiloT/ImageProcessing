"""
ImageEnhancementStudio.py
==========================
Combined single-file application:
  • Processing engine  – grayscale, colour, pseudocolor, spatial filters, edge detection
  • Dark-themed Tkinter GUI  – live before/after preview, history, undo, zoom
  • CLI InputHandler  – available via InputHandler.run() for terminal use

Dependencies:
  pip install numpy tifffile pillow scipy

Run GUI (default):
  python ImageEnhancementStudio.py

Run CLI:
  python -c "from ImageEnhancementStudio import InputHandler; InputHandler.run()"
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
#  SECTION 9b – MISSING COLOR EQUIVALENTS  (Assignment 6)
# ============================================================


class ColorRampEvent:
    """
    Intensity ramp applied per R, G, B channel independently.
    Pixels below start → 0, above end → 1, in between → linear ramp.
    start / end are in 0-255 scale (same convention as the grayscale version).
    Usage: color_ramp <in> <out> <start> <end>
    """

    def __init__(self, inp: str, out: str, start: int, end: int):
        if start >= end:
            raise RuntimeError("start must be < end")
        self.inp, self.out = inp, out
        self.start = start / 255.0
        self.end = end / 255.0

    def execute(self):
        img = ColorImageObject(self.inp)
        span = self.end - self.start
        for c in range(3):
            ch = img.data[:, :, c].astype(np.float64)
            ch = np.where(
                ch < self.start,
                0.0,
                np.where(ch > self.end, 1.0, (ch - self.start) / span),
            )
            img.data[:, :, c] = ch.astype(np.float32)
        img.save_tiff_8bit(self.out)
        print(
            f"Color ramp (start={int(self.start * 255)}, end={int(self.end * 255)}) done."
        )


class ColorLevelSlicingEvent:
    """
    Highlight pixels whose intensity (per channel) falls in [lo, hi] by setting
    them to val; all others are preserved (bg mode) or zeroed (nobg mode).
    lo / hi / val are in 0-255 scale.
    Usage: color_level_slice <in> <out> <lo> <hi> <val> <bg|nobg>
    """

    def __init__(self, inp: str, out: str, lo: int, hi: int, val: int, mode: str):
        self.inp, self.out = inp, out
        self.lo = lo / 255.0
        self.hi = hi / 255.0
        self.val = val / 255.0
        self.keep_bg = mode == "bg"

    def execute(self):
        img = ColorImageObject(self.inp)
        for c in range(3):
            ch = img.data[:, :, c].astype(np.float64)
            mask = (ch >= self.lo) & (ch <= self.hi)
            bg = ch if self.keep_bg else np.zeros_like(ch)
            img.data[:, :, c] = np.where(mask, self.val, bg).astype(np.float32)
        img.save_tiff_8bit(self.out)
        print(
            f"Color level slice [lo={int(self.lo * 255)}, hi={int(self.hi * 255)}] done."
        )


class ColorBitPlaneSliceEvent:
    """
    Extract a single bit plane from each R, G, B channel independently.
    Pixels where that bit is 1 are set to 1.0; others are preserved (bg mode)
    or zeroed (nobg mode).
    Usage: color_bit_slice <in> <out> <bit_idx> <bg|nobg>
    """

    def __init__(self, inp: str, out: str, bit: int, mode: str):
        if not (0 <= bit <= 7):
            raise RuntimeError("bit index must be 0-7")
        self.inp, self.out = inp, out
        self.bit = bit
        self.with_bg = mode == "bg"

    def execute(self):
        img = ColorImageObject(self.inp)
        for c in range(3):
            ch_u8 = np.clip(img.data[:, :, c] * 255, 0, 255).astype(np.uint8)
            mask = (ch_u8 >> self.bit) & 1  # 0 or 1
            bg = img.data[:, :, c] if self.with_bg else np.zeros_like(img.data[:, :, c])
            img.data[:, :, c] = np.where(mask, 1.0, bg).astype(np.float32)
        img.save_tiff_8bit(self.out)
        print(f"Color bit-plane slice (bit={self.bit}) done.")


class ColorHistMatchEvent:
    """
    Histogram matching applied per R, G, B channel independently.
    The reference image may be grayscale or RGB; if grayscale its single channel
    is used as the reference for all three output channels.
    Usage: color_hist_match <src> <ref> <out>
    """

    def __init__(self, src: str, ref: str, out: str):
        self.src, self.ref, self.out = src, ref, out

    @staticmethod
    def _match_channel(src_ch: np.ndarray, ref_ch: np.ndarray) -> np.ndarray:
        """CDF-based histogram matching for two float32 [0,1] arrays."""
        src_u8 = np.clip(src_ch * 255, 0, 255).astype(np.uint8)
        ref_u8 = np.clip(ref_ch * 255, 0, 255).astype(np.uint8)
        hist_s, _ = np.histogram(src_u8.flatten(), bins=256, range=(0, 256))
        hist_r, _ = np.histogram(ref_u8.flatten(), bins=256, range=(0, 256))
        cdf_s = hist_s.cumsum() / hist_s.sum()
        cdf_r = hist_r.cumsum() / hist_r.sum()
        lut = np.zeros(256, dtype=np.uint8)
        r_ptr = 0
        for s in range(256):
            while r_ptr < 255 and cdf_r[r_ptr] < cdf_s[s]:
                r_ptr += 1
            lut[s] = r_ptr
        return (lut[src_u8].astype(np.float32)) / 255.0

    def execute(self):
        src = ColorImageObject(self.src)
        ref = ColorImageObject(self.ref)
        for c in range(3):
            src.data[:, :, c] = self._match_channel(
                src.data[:, :, c], ref.data[:, :, c]
            )
        src.save_tiff_8bit(self.out)
        print("Color histogram matching done.")


class ColorLocalHistEqEvent:
    """
    Local histogram equalization applied per R, G, B channel independently.
    Each pixel is mapped using the CDF of its local neighbourhood window.
    Usage: color_local_hist <in> <out> <window>
    """

    def __init__(self, inp: str, out: str, window: int):
        if window < 3 or window % 2 == 0:
            raise RuntimeError("Window must be odd and >= 3")
        self.inp, self.out, self.window = inp, out, window

    def execute(self):
        from scipy.ndimage import uniform_filter

        img = ColorImageObject(self.inp)
        r = self.window // 2
        for c in range(3):
            ch_f = img.data[:, :, c].astype(np.float64)
            ch_u8 = np.clip(ch_f * 255, 0, 255).astype(np.uint8)
            H, W = ch_u8.shape
            pad = np.pad(ch_u8, r, mode="edge")
            out = np.zeros((H, W), dtype=np.float32)
            win2 = self.window * self.window
            for y in range(H):
                for x in range(W):
                    region = pad[y : y + self.window, x : x + self.window].flatten()
                    hist = np.bincount(region, minlength=256)
                    center = ch_u8[y, x]
                    cdf = int(hist[: center + 1].sum())
                    out[y, x] = (cdf * 255.0 / win2) / 255.0
            img.data[:, :, c] = np.clip(out, 0, 1)
        img.save_tiff_8bit(self.out)
        print(f"Color local histogram EQ (window={self.window}) done.")


class ColorWeightedAveragingEvent:
    """
    Weighted 3×3 average filter applied per R, G, B channel independently.
    Kernel: [[1,2,1],[2,4,2],[1,2,1]] / 16  (same as grayscale version).
    Usage: color_weighted_avg <in> <out>
    """

    def __init__(self, inp: str, out: str):
        self.inp, self.out = inp, out

    def execute(self):
        from scipy.ndimage import convolve

        img = ColorImageObject(self.inp)
        K = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 16.0
        for c in range(3):
            ch = img.data[:, :, c].astype(np.float64)
            img.data[:, :, c] = np.clip(convolve(ch, K, mode="nearest"), 0, 1).astype(
                np.float32
            )
        img.save_tiff_8bit(self.out)
        print("Color weighted averaging done.")


class ColorGradientEdgeEvent:
    """
    Sobel gradient edge enhancement applied per R, G, B channel independently.
    out[c] = channel[c] + k * |gradient[c]|
    Usage: color_grad_edge <in> <out> <k>
    """

    def __init__(self, inp: str, out: str, k: float):
        if k <= 0:
            raise RuntimeError("k must be > 0")
        self.inp, self.out, self.k = inp, out, k

    def execute(self):
        from scipy.ndimage import convolve

        img = ColorImageObject(self.inp)
        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
        for c in range(3):
            ch = img.data[:, :, c].astype(np.float64)
            grad = np.abs(convolve(ch, Gx, mode="nearest")) + np.abs(
                convolve(ch, Gy, mode="nearest")
            )
            img.data[:, :, c] = np.clip(ch + self.k * grad, 0, 1).astype(np.float32)
        img.save_tiff_8bit(self.out)
        print(f"Color gradient edge enhance (k={self.k}) done.")


class ColorRobertsEdgeEvent:
    """
    Roberts cross-gradient edge detection applied per R, G, B channel independently.
    Usage: color_roberts <in> <out>
    """

    def __init__(self, inp: str, out: str):
        self.inp, self.out = inp, out

    def execute(self):
        img = ColorImageObject(self.inp)
        for c in range(3):
            ch = img.data[:, :, c].astype(np.float64)
            gx = ch[:-1, :-1] - ch[1:, 1:]
            gy = ch[:-1, 1:] - ch[1:, :-1]
            mag = np.abs(gx) + np.abs(gy)
            out = np.zeros_like(ch)
            out[:-1, :-1] = mag
            # normalise so the channel stays in [0, 1]
            mx = out.max()
            img.data[:, :, c] = np.clip(out / mx if mx > 0 else out, 0, 1).astype(
                np.float32
            )
        img.save_tiff_8bit(self.out)
        print("Color Roberts edge done.")


class ColorPrewittEdgeEvent:
    """
    Prewitt gradient edge detection applied per R, G, B channel independently.
    Usage: color_prewitt <in> <out>
    """

    def __init__(self, inp: str, out: str):
        self.inp, self.out = inp, out

    def execute(self):
        from scipy.ndimage import convolve

        img = ColorImageObject(self.inp)
        Gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
        Gy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
        for c in range(3):
            ch = img.data[:, :, c].astype(np.float64)
            mag = np.abs(convolve(ch, Gx, mode="nearest")) + np.abs(
                convolve(ch, Gy, mode="nearest")
            )
            mx = mag.max()
            img.data[:, :, c] = np.clip(mag / mx if mx > 0 else mag, 0, 1).astype(
                np.float32
            )
        img.save_tiff_8bit(self.out)
        print("Color Prewitt edge done.")


class ColorLaplacianSharpenEvent:
    """
    Laplacian sharpening applied per R, G, B channel independently.
    mode: "4" → 4-connected kernel, "8" → 8-connected kernel.
    Outputs only the sharpened image (lap intermediate is discarded).
    Usage: color_laplacian <in> <out> <4|8>
    """

    def __init__(self, inp: str, out: str, mode: str = "4"):
        self.inp, self.out = inp, out
        self.mode = mode

    def execute(self):
        from scipy.ndimage import convolve

        img = ColorImageObject(self.inp)
        K4 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
        K8 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float64)
        K = K4 if self.mode == "4" else K8
        for c in range(3):
            ch = img.data[:, :, c].astype(np.float64)
            lap = convolve(ch, K, mode="nearest")
            img.data[:, :, c] = np.clip(ch + lap, 0, 1).astype(np.float32)
        img.save_tiff_8bit(self.out)
        print(f"Color Laplacian sharpen ({self.mode}-connected) done.")


class ColorLaplacianSobelEvent:
    """
    Combined Laplacian + Sobel pipeline applied per R, G, B channel.
    Step 1 – Laplacian sharpening (4-connected).
    Step 2 – Sobel edge magnitude is used to further refine the sharpened result.
    Outputs the final combined sharpened color image.
    Usage: color_lap_sobel <in> <out>
    """

    def __init__(self, inp: str, out: str):
        self.inp, self.out = inp, out

    def execute(self):
        from scipy.ndimage import convolve

        img = ColorImageObject(self.inp)
        K_lap = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
        for c in range(3):
            ch = img.data[:, :, c].astype(np.float64)
            sharp = np.clip(ch + convolve(ch, K_lap, mode="nearest"), 0, 1)
            sobel = np.abs(convolve(ch, Gx, mode="nearest")) + np.abs(
                convolve(ch, Gy, mode="nearest")
            )
            mx = sobel.max()
            sobel_n = sobel / mx if mx > 0 else sobel
            # product mask: keep sharpening where edges are strong
            img.data[:, :, c] = np.clip(sharp * (1.0 + sobel_n) / 2.0, 0, 1).astype(
                np.float32
            )
        img.save_tiff_8bit(self.out)
        print("Color Laplacian-Sobel sharpen done.")


class ColorBandFilterEvent:
    """
    Band-pass or band-reject filter applied per R, G, B channel independently.
    Uses two Gaussian low-pass filters (k1/s1 = narrow, k2/s2 = wide).
      Band-pass  : wide_LP  − narrow_LP  (keeps mid frequencies)
      Band-reject: narrow_LP + (original − wide_LP)  (removes mid frequencies)
    Usage: color_bandpass  <in> <out> <k1> <s1> <k2> <s2>
           color_bandreject <in> <out> <k1> <s1> <k2> <s2>
    """

    def __init__(
        self, inp: str, out: str, k1: int, s1: float, k2: int, s2: float, mode: BandMode
    ):
        self.inp, self.out = inp, out
        self.k1, self.s1 = k1, s1
        self.k2, self.s2 = k2, s2
        self.mode = mode

    @staticmethod
    def _gaussian_blur_channel(ch: np.ndarray, ks: int, sigma: float) -> np.ndarray:
        """Separable Gaussian blur on a single float64 channel."""
        r = ks // 2
        k = np.array([math.exp(-(i * i) / (2 * sigma**2)) for i in range(-r, r + 1)])
        k /= k.sum()
        H, W = ch.shape
        pad_h = np.pad(ch, ((0, 0), (r, r)), mode="edge")
        tmp = np.zeros_like(ch)
        for x in range(W):
            tmp[:, x] = (pad_h[:, x : x + ks] * k).sum(axis=1)
        pad_v = np.pad(tmp, ((r, r), (0, 0)), mode="edge")
        out = np.zeros_like(ch)
        for y in range(H):
            out[y, :] = (pad_v[y : y + ks, :] * k[:, None]).sum(axis=0)
        return out

    def execute(self):
        img = ColorImageObject(self.inp)
        for c in range(3):
            ch = img.data[:, :, c].astype(np.float64)
            lp1 = self._gaussian_blur_channel(ch, self.k1, self.s1)
            lp2 = self._gaussian_blur_channel(ch, self.k2, self.s2)
            if self.mode == BandMode.BANDPASS:
                val = lp2 - lp1
            else:
                val = lp1 + (ch - lp2)
            img.data[:, :, c] = np.clip(val, 0, 1).astype(np.float32)
        img.save_tiff_8bit(self.out)
        print(
            f"Color band {'pass' if self.mode == BandMode.BANDPASS else 'reject'} done."
        )


class ColorGradientSharpenEvent:
    """
    Gradient-based sharpening applied per R, G, B channel independently.
    out[c] = channel[c] + k * |Sobel_gradient[c]|
    (Adds the gradient magnitude back into the image to accentuate edges.)
    Usage: color_grad_sharpen <in> <out> <k>
    """

    def __init__(self, inp: str, out: str, k: float):
        if k <= 0:
            raise RuntimeError("k must be > 0")
        self.inp, self.out, self.k = inp, out, k

    def execute(self):
        from scipy.ndimage import convolve

        img = ColorImageObject(self.inp)
        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
        for c in range(3):
            ch = img.data[:, :, c].astype(np.float64)
            grad = np.abs(convolve(ch, Gx, mode="nearest")) + np.abs(
                convolve(ch, Gy, mode="nearest")
            )
            img.data[:, :, c] = np.clip(ch + self.k * grad, 0, 1).astype(np.float32)
        img.save_tiff_8bit(self.out)
        print(f"Color gradient sharpen (k={self.k}) done.")


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

                # ---- Color missing equivalents (Assignment 6) ----
                elif cmd == "color_ramp":
                    _, inp, out, start, end = parts
                    ColorRampEvent(inp, out, int(start), int(end)).execute()

                elif cmd == "color_level_slice":
                    _, inp, out, lo, hi, val, mode = parts
                    ColorLevelSlicingEvent(
                        inp, out, int(lo), int(hi), int(val), mode
                    ).execute()

                elif cmd == "color_bit_slice":
                    _, inp, out, bit, mode = parts
                    ColorBitPlaneSliceEvent(inp, out, int(bit), mode).execute()

                elif cmd == "color_hist_match":
                    _, src, ref, out = parts
                    ColorHistMatchEvent(src, ref, out).execute()

                elif cmd == "color_local_hist":
                    _, inp, out, w = parts
                    ColorLocalHistEqEvent(inp, out, int(w)).execute()

                elif cmd == "color_weighted_avg":
                    _, inp, out = parts
                    ColorWeightedAveragingEvent(inp, out).execute()

                elif cmd == "color_grad_edge":
                    _, inp, out, k = parts
                    ColorGradientEdgeEvent(inp, out, float(k)).execute()

                elif cmd == "color_roberts":
                    _, inp, out = parts
                    ColorRobertsEdgeEvent(inp, out).execute()

                elif cmd == "color_prewitt":
                    _, inp, out = parts
                    ColorPrewittEdgeEvent(inp, out).execute()

                elif cmd == "color_laplacian":
                    _, inp, out, mode = parts
                    ColorLaplacianSharpenEvent(inp, out, mode).execute()

                elif cmd == "color_lap_sobel":
                    _, inp, out = parts
                    ColorLaplacianSobelEvent(inp, out).execute()

                elif cmd == "color_bandpass":
                    _, inp, out, k1, s1, k2, s2 = parts
                    ColorBandFilterEvent(
                        inp,
                        out,
                        int(k1),
                        float(s1),
                        int(k2),
                        float(s2),
                        BandMode.BANDPASS,
                    ).execute()

                elif cmd == "color_bandreject":
                    _, inp, out, k1, s1, k2, s2 = parts
                    ColorBandFilterEvent(
                        inp,
                        out,
                        int(k1),
                        float(s1),
                        int(k2),
                        float(s2),
                        BandMode.BANDREJECT,
                    ).execute()

                elif cmd == "color_grad_sharpen":
                    _, inp, out, k = parts
                    ColorGradientSharpenEvent(inp, out, float(k)).execute()

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

╠══════════════════════════════════════════════════════════════╣
║        COLOR MISSING EQUIVALENTS  (Assignment 6)             ║
╠══════════════════════════════════════════════════════════════╣
  color_ramp <in> <out> <start> <end>
  color_level_slice <in> <out> <lo> <hi> <val> <bg|nobg>
  color_bit_slice <in> <out> <bit_idx> <bg|nobg>
  color_hist_match <src> <ref> <out>
  color_local_hist <in> <out> <window>
  color_weighted_avg <in> <out>
  color_grad_edge <in> <out> <k>
  color_roberts <in> <out>
  color_prewitt <in> <out>
  color_laplacian <in> <out> <4|8>
  color_lap_sobel <in> <out>
  color_bandpass <in> <out> <k1> <s1> <k2> <s2>
  color_bandreject <in> <out> <k1> <s1> <k2> <s2>
  color_grad_sharpen <in> <out> <k>

  quit
╚══════════════════════════════════════════════════════════════╝""")


# ============================================================
#  SECTION 11 – GUI  (Tkinter dark-themed desktop interface)
# ============================================================

import io
import sys
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

# ══════════════════════════════════════════════════════════════════════════════
#  THEME
# ══════════════════════════════════════════════════════════════════════════════
DARK = "#0f1117"
PANEL = "#1a1d27"
CARD = "#22263a"
BORDER = "#2e3452"
ACCENT = "#5c7cfa"
ACCENT2 = "#f06595"
TEXT = "#e8eaf6"
MUTED = "#6b7280"
GREEN = "#51cf66"
YELLOW = "#fcc419"
RED = "#ff6b6b"

FONT_MONO = ("Courier New", 10)
FONT_UI = ("Segoe UI", 10) if sys.platform == "win32" else ("SF Pro Text", 10)
FONT_TITLE = (
    ("Segoe UI Semibold", 11) if sys.platform == "win32" else ("SF Pro Text", 11)
)
FONT_SMALL = ("Segoe UI", 9) if sys.platform == "win32" else ("SF Pro Text", 9)

# ══════════════════════════════════════════════════════════════════════════════
#  OPERATION REGISTRY
#  Each entry: { label, category, params: [(name, type, default, *opts)], fn }
#  type = "float" | "int" | "choice" | "path"
# ══════════════════════════════════════════════════════════════════════════════
OPERATIONS = [
    # ── GRAYSCALE ─────────────────────────────────────────────────────────────
    dict(
        label="Invert",
        cat="Grayscale › Point",
        params=[],
        fn=lambda i, o, p: InvertImageEvent(i, o).execute(),
    ),
    dict(
        label="Log Transform",
        cat="Grayscale › Point",
        params=[],
        fn=lambda i, o, p: LogTransformEvent(i, o).execute(),
    ),
    dict(
        label="Gamma",
        cat="Grayscale › Point",
        params=[("gamma", "float", 1.0)],
        fn=lambda i, o, p: GammaTransformEvent(i, o, p["gamma"]).execute(),
    ),
    dict(
        label="Piecewise Contrast",
        cat="Grayscale › Point",
        params=[
            ("r1", "int", 64),
            ("s1", "int", 0),
            ("r2", "int", 192),
            ("s2", "int", 255),
        ],
        fn=lambda i, o, p: PieceWiseContrastEvent(
            i, o, p["r1"], p["s1"], p["r2"], p["s2"]
        ).execute(),
    ),
    dict(
        label="Intensity Ramp",
        cat="Grayscale › Point",
        params=[("start", "int", 64), ("end", "int", 192)],
        fn=lambda i, o, p: IntensityRampEvent(i, o, p["start"], p["end"]).execute(),
    ),
    dict(
        label="Level Slice",
        cat="Grayscale › Point",
        params=[
            ("lo", "int", 100),
            ("hi", "int", 200),
            ("val", "int", 255),
            ("mode", "choice", "nobg", ["bg", "nobg"]),
        ],
        fn=lambda i, o, p: IntensityLevelSlicingEvent(
            i, o, p["lo"], p["hi"], p["val"], p["mode"]
        ).execute(),
    ),
    dict(
        label="Bit Plane Slice",
        cat="Grayscale › Point",
        params=[("bit", "int", 7), ("mode", "choice", "nobg", ["bg", "nobg"])],
        fn=lambda i, o, p: BitPlaneSliceEvent(i, o, p["bit"], p["mode"]).execute(),
    ),
    dict(
        label="Histogram EQ",
        cat="Grayscale › Histogram",
        params=[],
        fn=lambda i, o, p: HistogramEqualizationEvent(i, o).execute(),
    ),
    dict(
        label="Local Hist EQ",
        cat="Grayscale › Histogram",
        params=[("window", "int", 15)],
        fn=lambda i, o, p: LocalHistogramEnhancementEvent(i, o, p["window"]).execute(),
    ),
    dict(
        label="Box Smooth",
        cat="Grayscale › Spatial",
        params=[("kernel", "int", 5)],
        fn=lambda i, o, p: BoxSmoothingEvent(i, o, p["kernel"]).execute(),
    ),
    dict(
        label="Gaussian Blur",
        cat="Grayscale › Spatial",
        params=[("kernel", "int", 5), ("sigma", "float", 1.0)],
        fn=lambda i, o, p: GaussianLowPassEvent(
            i, o, p["kernel"], p["sigma"]
        ).execute(),
    ),
    dict(
        label="Weighted Average",
        cat="Grayscale › Spatial",
        params=[],
        fn=lambda i, o, p: WeightedAveragingEvent(i, o).execute(),
    ),
    dict(
        label="Median Filter",
        cat="Grayscale › Spatial",
        params=[("window", "int", 5)],
        fn=lambda i, o, p: MedianFilterEvent(i, o, p["window"]).execute(),
    ),
    dict(
        label="Laplacian Sharpen",
        cat="Grayscale › Sharpen",
        params=[("mode", "choice", "4", ["4", "8"])],
        fn=lambda i, o, p: _laplacian_single(i, o, p),
    ),
    dict(
        label="High-Pass Sharpen",
        cat="Grayscale › Sharpen",
        params=[("strength", "float", 1.0)],
        fn=lambda i, o, p: HighPassSharpenEvent(i, o, p["strength"]).execute(),
    ),
    dict(
        label="Unsharp / Highboost",
        cat="Grayscale › Sharpen",
        params=[("A", "float", 1.5)],
        fn=lambda i, o, p: UnsharpHighboostEvent(i, o, p["A"]).execute(),
    ),
    dict(
        label="Gradient Sharpen",
        cat="Grayscale › Sharpen",
        params=[("k", "float", 0.2)],
        fn=lambda i, o, p: GradientSharpenEvent(i, o, p["k"]).execute(),
    ),
    dict(
        label="Roberts Edge",
        cat="Grayscale › Edge",
        params=[],
        fn=lambda i, o, p: RobertsEdgeEvent(i, o).execute(),
    ),
    dict(
        label="Prewitt Edge",
        cat="Grayscale › Edge",
        params=[],
        fn=lambda i, o, p: PrewittEdgeEvent(i, o).execute(),
    ),
    dict(
        label="Sobel Edge",
        cat="Grayscale › Edge",
        params=[("threshold", "int", 0)],
        fn=lambda i, o, p: SobelEdgeEvent(i, o, p["threshold"]).execute(),
    ),
    dict(
        label="Gradient Edge Enhance",
        cat="Grayscale › Edge",
        params=[("k", "float", 0.5)],
        fn=lambda i, o, p: GradientEdgeEnhancementEvent(i, o, p["k"]).execute(),
    ),
    dict(
        label="Band Pass",
        cat="Grayscale › Band Filter",
        params=[
            ("k1", "int", 3),
            ("s1", "float", 0.5),
            ("k2", "int", 9),
            ("s2", "float", 2.0),
        ],
        fn=lambda i, o, p: BandFilterEvent(
            i, o, p["k1"], p["s1"], p["k2"], p["s2"], BandMode.BANDPASS
        ).execute(),
    ),
    dict(
        label="Band Reject",
        cat="Grayscale › Band Filter",
        params=[
            ("k1", "int", 3),
            ("s1", "float", 0.5),
            ("k2", "int", 9),
            ("s2", "float", 2.0),
        ],
        fn=lambda i, o, p: BandFilterEvent(
            i, o, p["k1"], p["s1"], p["k2"], p["s2"], BandMode.BANDREJECT
        ).execute(),
    ),
    # ── PSEUDOCOLOR ────────────────────────────────────────────────────────────
    dict(
        label="Pseudocolor Map",
        cat="Pseudocolor",
        params=[
            (
                "colormap",
                "choice",
                "jet",
                ["jet", "hot", "cool", "bone", "spring", "gray"],
            )
        ],
        fn=lambda i, o, p: PseudoColorEvent(i, o, p["colormap"]).execute(),
    ),
    # ── COLOR ENHANCEMENT ──────────────────────────────────────────────────────
    dict(
        label="Color Invert",
        cat="Color › Point",
        params=[],
        fn=lambda i, o, p: ColorInvertEvent(i, o).execute(),
    ),
    dict(
        label="Color Gamma",
        cat="Color › Point",
        params=[("gamma", "float", 1.0)],
        fn=lambda i, o, p: ColorGammaEvent(i, o, p["gamma"]).execute(),
    ),
    dict(
        label="Color Log",
        cat="Color › Point",
        params=[],
        fn=lambda i, o, p: ColorLogEvent(i, o).execute(),
    ),
    dict(
        label="Color Balance",
        cat="Color › Point",
        params=[
            ("r_scale", "float", 1.0),
            ("g_scale", "float", 1.0),
            ("b_scale", "float", 1.0),
        ],
        fn=lambda i, o, p: ColorBalanceEvent(
            i, o, p["r_scale"], p["g_scale"], p["b_scale"]
        ).execute(),
    ),
    dict(
        label="Color Contrast",
        cat="Color › Point",
        params=[
            ("r1", "int", 64),
            ("s1", "int", 0),
            ("r2", "int", 192),
            ("s2", "int", 255),
        ],
        fn=lambda i, o, p: ColorContrastEvent(
            i, o, p["r1"], p["s1"], p["r2"], p["s2"]
        ).execute(),
    ),
    dict(
        label="Color Hist EQ (per-channel)",
        cat="Color › Histogram",
        params=[],
        fn=lambda i, o, p: ColorHistEqChannelEvent(i, o).execute(),
    ),
    dict(
        label="Color Hist EQ (HSI)",
        cat="Color › Histogram",
        params=[],
        fn=lambda i, o, p: ColorHistEqHSIEvent(i, o).execute(),
    ),
    dict(
        label="HSI Saturation",
        cat="Color › HSI",
        params=[("scale", "float", 1.5)],
        fn=lambda i, o, p: HSISaturationEvent(i, o, p["scale"]).execute(),
    ),
    dict(
        label="HSI Hue Rotate",
        cat="Color › HSI",
        params=[("degrees", "float", 90.0)],
        fn=lambda i, o, p: HSIHueRotateEvent(i, o, p["degrees"]).execute(),
    ),
    dict(
        label="Color Box Smooth",
        cat="Color › Spatial",
        params=[("kernel", "int", 5)],
        fn=lambda i, o, p: ColorSmoothBoxEvent(i, o, p["kernel"]).execute(),
    ),
    dict(
        label="Color Gaussian",
        cat="Color › Spatial",
        params=[("kernel", "int", 5), ("sigma", "float", 1.0)],
        fn=lambda i, o, p: ColorGaussianEvent(i, o, p["kernel"], p["sigma"]).execute(),
    ),
    dict(
        label="Color Sharpen",
        cat="Color › Spatial",
        params=[("strength", "float", 1.0)],
        fn=lambda i, o, p: ColorSharpenEvent(i, o, p["strength"]).execute(),
    ),
    dict(
        label="Color Median",
        cat="Color › Spatial",
        params=[("window", "int", 5)],
        fn=lambda i, o, p: ColorMedianEvent(i, o, p["window"]).execute(),
    ),
    dict(
        label="Color Unsharp",
        cat="Color › Spatial",
        params=[("A", "float", 1.5)],
        fn=lambda i, o, p: ColorUnsharpEvent(i, o, p["A"]).execute(),
    ),
    dict(
        label="Color Edge (Luminance)",
        cat="Color › Edge",
        params=[("threshold", "int", 0)],
        fn=lambda i, o, p: ColorEdgeSobelEvent(i, o, p["threshold"]).execute(),
    ),
    dict(
        label="Color Edge (per-channel)",
        cat="Color › Edge",
        params=[],
        fn=lambda i, o, p: ColorEdgePerChannelEvent(i, o).execute(),
    ),
    # ── COLOR MISSING EQUIVALENTS  (Assignment 6) ─────────────────────────────
    dict(
        label="Color Intensity Ramp",
        cat="Color › Point",
        params=[("start", "int", 64), ("end", "int", 192)],
        fn=lambda i, o, p: ColorRampEvent(i, o, p["start"], p["end"]).execute(),
    ),
    dict(
        label="Color Level Slice",
        cat="Color › Point",
        params=[
            ("lo", "int", 100),
            ("hi", "int", 200),
            ("val", "int", 255),
            ("mode", "choice", "nobg", ["bg", "nobg"]),
        ],
        fn=lambda i, o, p: ColorLevelSlicingEvent(
            i, o, p["lo"], p["hi"], p["val"], p["mode"]
        ).execute(),
    ),
    dict(
        label="Color Bit Plane Slice",
        cat="Color › Point",
        params=[
            ("bit", "int", 7),
            ("mode", "choice", "nobg", ["bg", "nobg"]),
        ],
        fn=lambda i, o, p: ColorBitPlaneSliceEvent(i, o, p["bit"], p["mode"]).execute(),
    ),
    dict(
        label="Color Hist Match",
        cat="Color › Histogram",
        params=[("ref", "path", "")],
        fn=lambda i, o, p: ColorHistMatchEvent(i, p["ref"], o).execute(),
    ),
    dict(
        label="Color Local Hist EQ",
        cat="Color › Histogram",
        params=[("window", "int", 15)],
        fn=lambda i, o, p: ColorLocalHistEqEvent(i, o, p["window"]).execute(),
    ),
    dict(
        label="Color Weighted Average",
        cat="Color › Spatial",
        params=[],
        fn=lambda i, o, p: ColorWeightedAveragingEvent(i, o).execute(),
    ),
    dict(
        label="Color Gradient Edge Enhance",
        cat="Color › Edge",
        params=[("k", "float", 0.5)],
        fn=lambda i, o, p: ColorGradientEdgeEvent(i, o, p["k"]).execute(),
    ),
    dict(
        label="Color Roberts Edge",
        cat="Color › Edge",
        params=[],
        fn=lambda i, o, p: ColorRobertsEdgeEvent(i, o).execute(),
    ),
    dict(
        label="Color Prewitt Edge",
        cat="Color › Edge",
        params=[],
        fn=lambda i, o, p: ColorPrewittEdgeEvent(i, o).execute(),
    ),
    dict(
        label="Color Laplacian Sharpen",
        cat="Color › Sharpen",
        params=[("mode", "choice", "4", ["4", "8"])],
        fn=lambda i, o, p: ColorLaplacianSharpenEvent(i, o, p["mode"]).execute(),
    ),
    dict(
        label="Color Laplacian-Sobel",
        cat="Color › Sharpen",
        params=[],
        fn=lambda i, o, p: ColorLaplacianSobelEvent(i, o).execute(),
    ),
    dict(
        label="Color Band Pass",
        cat="Color › Band Filter",
        params=[
            ("k1", "int", 3),
            ("s1", "float", 0.5),
            ("k2", "int", 9),
            ("s2", "float", 2.0),
        ],
        fn=lambda i, o, p: ColorBandFilterEvent(
            i, o, p["k1"], p["s1"], p["k2"], p["s2"], BandMode.BANDPASS
        ).execute(),
    ),
    dict(
        label="Color Band Reject",
        cat="Color › Band Filter",
        params=[
            ("k1", "int", 3),
            ("s1", "float", 0.5),
            ("k2", "int", 9),
            ("s2", "float", 2.0),
        ],
        fn=lambda i, o, p: ColorBandFilterEvent(
            i, o, p["k1"], p["s1"], p["k2"], p["s2"], BandMode.BANDREJECT
        ).execute(),
    ),
    dict(
        label="Color Gradient Sharpen",
        cat="Color › Sharpen",
        params=[("k", "float", 0.2)],
        fn=lambda i, o, p: ColorGradientSharpenEvent(i, o, p["k"]).execute(),
    ),
]


def _laplacian_single(inp, out, p):
    """Laplacian sharpen – save only the sharp output (discard lap_out)."""
    tmp = out.replace(".tif", "_lap_tmp.tif")
    m = LaplacianMode.EIGHT if p["mode"] == "8" else LaplacianMode.FOUR
    LaplacianSharpenEvent(inp, tmp, out, m).execute()
    if os.path.exists(tmp):
        os.remove(tmp)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def tiff_to_pil(path: str) -> Image.Image:
    """Load any TIFF (grayscale or RGB) and return a PIL Image for display."""
    arr = tiff.imread(path)
    if arr.ndim == 2:
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            arr8 = np.full(arr.shape, 128, dtype=np.uint8)
        else:
            arr8 = ((arr - mn) * 255.0 / (mx - mn)).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr8, mode="L").convert("RGB")
    elif arr.ndim == 3:
        if arr.dtype != np.uint8:
            mn, mx = arr.min(), arr.max()
            arr = ((arr - mn) * 255.0 / (mx - mn)).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr[:, :, :3].astype(np.uint8), mode="RGB")
    raise RuntimeError(f"Unsupported array shape: {arr.shape}")


def fit_image(pil_img: Image.Image, max_w: int, max_h: int) -> ImageTk.PhotoImage:
    img = pil_img.copy()
    img.thumbnail((max_w, max_h), Image.LANCZOS)
    return ImageTk.PhotoImage(img)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Enhancement Studio")
        self.configure(bg=DARK)
        self.minsize(1200, 720)
        self.geometry("1400x860")

        self._src_path = None  # original loaded file
        self._work_path = None  # current working file (may be output of last op)
        self._out_path = None  # last applied output
        self._pil_before = None
        self._pil_after = None
        self._param_vars = {}  # tkinter vars for current op's params
        self._history = []  # list of (label, path) tuples

        self._style()
        self._build()
        self._populate_op_tree()

    # ── ttk style ────────────────────────────────────────────────────────────
    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TFrame", background=DARK)
        s.configure("Panel.TFrame", background=PANEL)
        s.configure("Card.TFrame", background=CARD)
        s.configure("TLabel", background=DARK, foreground=TEXT, font=FONT_UI)
        s.configure("Panel.TLabel", background=PANEL, foreground=TEXT, font=FONT_UI)
        s.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=FONT_SMALL)
        s.configure("Title.TLabel", background=PANEL, foreground=TEXT, font=FONT_TITLE)
        s.configure("Card.TLabel", background=CARD, foreground=TEXT, font=FONT_UI)
        s.configure(
            "Accent.TLabel", background=PANEL, foreground=ACCENT, font=FONT_TITLE
        )

        s.configure(
            "TEntry",
            fieldbackground=CARD,
            foreground=TEXT,
            insertcolor=TEXT,
            bordercolor=BORDER,
            relief="flat",
            padding=(6, 4),
        )
        s.map("TEntry", bordercolor=[("focus", ACCENT)])

        s.configure(
            "TCombobox",
            fieldbackground=CARD,
            foreground=TEXT,
            selectbackground=ACCENT,
            selectforeground=TEXT,
            bordercolor=BORDER,
            relief="flat",
        )
        s.map("TCombobox", fieldbackground=[("readonly", CARD)])

        s.configure(
            "Treeview",
            background=PANEL,
            fieldbackground=PANEL,
            foreground=TEXT,
            font=FONT_UI,
            rowheight=26,
            bordercolor=BORDER,
            relief="flat",
        )
        s.configure(
            "Treeview.Heading",
            background=CARD,
            foreground=MUTED,
            font=FONT_SMALL,
            relief="flat",
        )
        s.map(
            "Treeview",
            background=[("selected", ACCENT)],
            foreground=[("selected", "#fff")],
        )

        s.configure(
            "TScrollbar",
            troughcolor=PANEL,
            background=BORDER,
            bordercolor=PANEL,
            arrowcolor=MUTED,
            relief="flat",
        )
        s.configure("Horizontal.TScrollbar", troughcolor=DARK, background=BORDER)

    # ── layout ───────────────────────────────────────────────────────────────
    def _build(self):
        # ── top bar ──
        topbar = tk.Frame(self, bg=PANEL, height=52)
        topbar.pack(fill="x", side="top")
        topbar.pack_propagate(False)

        tk.Label(
            topbar,
            text="◈  IMAGE ENHANCEMENT STUDIO",
            bg=PANEL,
            fg=ACCENT,
            font=("Courier New", 13, "bold"),
        ).pack(side="left", padx=20, pady=14)

        btn_kw = dict(
            bg=CARD,
            fg=TEXT,
            font=FONT_UI,
            relief="flat",
            activebackground=BORDER,
            activeforeground=TEXT,
            cursor="hand2",
            padx=14,
            pady=6,
        )

        tk.Button(topbar, text="⟵  Undo", **btn_kw, command=self._undo).pack(
            side="right", padx=6, pady=10
        )
        tk.Button(topbar, text="Reset", **btn_kw, command=self._reset).pack(
            side="right", padx=0, pady=10
        )
        tk.Button(
            topbar,
            text="💾  Save As",
            **{**btn_kw, "bg": ACCENT, "fg": "#fff", "activebackground": "#7b96fc"},
            command=self._save_as,
        ).pack(side="right", padx=6, pady=10)
        tk.Button(
            topbar,
            text="📂  Open",
            **{**btn_kw, "bg": "#2a3a5c", "fg": TEXT},
            command=self._open_file,
        ).pack(side="right", padx=6, pady=10)

        # ── main area ──
        main = tk.Frame(self, bg=DARK)
        main.pack(fill="both", expand=True)

        # left sidebar
        self._sidebar = tk.Frame(main, bg=PANEL, width=240)
        self._sidebar.pack(side="left", fill="y")
        self._sidebar.pack_propagate(False)
        self._build_sidebar()

        # center preview
        center = tk.Frame(main, bg=DARK)
        center.pack(side="left", fill="both", expand=True)
        self._build_preview(center)

        # right param panel
        self._param_panel = tk.Frame(main, bg=PANEL, width=270)
        self._param_panel.pack(side="right", fill="y")
        self._param_panel.pack_propagate(False)
        self._build_param_panel()

        # ── status bar ──
        self._status_var = tk.StringVar(value="Open a TIFF file to get started.")
        statusbar = tk.Frame(self, bg=CARD, height=28)
        statusbar.pack(fill="x", side="bottom")
        statusbar.pack_propagate(False)
        tk.Label(
            statusbar,
            textvariable=self._status_var,
            bg=CARD,
            fg=MUTED,
            font=FONT_SMALL,
            anchor="w",
        ).pack(side="left", padx=12, pady=5)

    def _build_sidebar(self):
        tk.Label(
            self._sidebar,
            text="OPERATIONS",
            bg=PANEL,
            fg=MUTED,
            font=("Courier New", 9, "bold"),
        ).pack(anchor="w", padx=12, pady=(14, 6))

        # search bar
        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", self._on_search)
        sf = tk.Frame(self._sidebar, bg=CARD)
        sf.pack(fill="x", padx=10, pady=(0, 8))
        tk.Label(sf, text="🔍", bg=CARD, fg=MUTED, font=FONT_UI).pack(
            side="left", padx=6
        )
        tk.Entry(
            sf,
            textvariable=self._search_var,
            bg=CARD,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            font=FONT_UI,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=ACCENT,
        ).pack(side="left", fill="x", expand=True, pady=4)

        tree_frame = tk.Frame(self._sidebar, bg=PANEL)
        tree_frame.pack(fill="both", expand=True, padx=6)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        self._tree = ttk.Treeview(
            tree_frame, show="tree", selectmode="browse", yscrollcommand=vsb.set
        )
        vsb.config(command=self._tree.yview)
        vsb.pack(side="right", fill="y")
        self._tree.pack(fill="both", expand=True)
        self._tree.bind("<<TreeviewSelect>>", self._on_op_select)
        self._tree.bind("<Double-1>", lambda e: self._apply_op())

        # history section
        tk.Frame(self._sidebar, bg=BORDER, height=1).pack(fill="x", padx=10, pady=8)
        tk.Label(
            self._sidebar,
            text="HISTORY",
            bg=PANEL,
            fg=MUTED,
            font=("Courier New", 9, "bold"),
        ).pack(anchor="w", padx=12, pady=(0, 4))
        hf = tk.Frame(self._sidebar, bg=PANEL)
        hf.pack(fill="x", padx=6, pady=(0, 8))
        vsb2 = ttk.Scrollbar(hf, orient="vertical")
        self._hist_box = tk.Listbox(
            hf,
            bg=CARD,
            fg=TEXT,
            selectbackground=ACCENT,
            font=FONT_SMALL,
            relief="flat",
            borderwidth=0,
            height=5,
            yscrollcommand=vsb2.set,
            highlightthickness=0,
        )
        vsb2.config(command=self._hist_box.yview)
        vsb2.pack(side="right", fill="y")
        self._hist_box.pack(fill="x", expand=True)
        self._hist_box.bind("<Double-1>", self._on_history_jump)

    def _build_preview(self, parent):
        header = tk.Frame(parent, bg=DARK)
        header.pack(fill="x", padx=16, pady=(12, 0))

        # image info label
        self._info_label = tk.Label(
            header, text="No image loaded", bg=DARK, fg=MUTED, font=FONT_SMALL
        )
        self._info_label.pack(side="left")

        # zoom controls
        zf = tk.Frame(header, bg=DARK)
        zf.pack(side="right")
        for lbl, cmd in [
            ("−", self._zoom_out),
            ("Fit", self._zoom_fit),
            ("+", self._zoom_in),
        ]:
            tk.Button(
                zf,
                text=lbl,
                bg=CARD,
                fg=TEXT,
                font=FONT_SMALL,
                relief="flat",
                padx=8,
                pady=2,
                activebackground=BORDER,
                cursor="hand2",
                command=cmd,
            ).pack(side="left", padx=2)

        # split view
        pane_frame = tk.Frame(parent, bg=DARK)
        pane_frame.pack(fill="both", expand=True, padx=12, pady=10)

        self._before_frame = self._make_preview_card(pane_frame, "BEFORE")
        self._before_frame.pack(side="left", fill="both", expand=True, padx=(0, 6))

        self._after_frame = self._make_preview_card(pane_frame, "AFTER")
        self._after_frame.pack(side="right", fill="both", expand=True, padx=(6, 0))

        # slider overlay label (shows which panel is shown when sliding)
        self._zoom_level = 1.0

    def _make_preview_card(self, parent, title: str):
        card = tk.Frame(
            parent,
            bg=CARD,
            relief="flat",
            highlightbackground=BORDER,
            highlightthickness=1,
        )

        header = tk.Frame(card, bg=CARD)
        header.pack(fill="x", padx=10, pady=(8, 4))
        tk.Label(
            header, text=title, bg=CARD, fg=MUTED, font=("Courier New", 9, "bold")
        ).pack(side="left")

        # pixel info label
        pix_lbl = tk.Label(header, text="", bg=CARD, fg=MUTED, font=FONT_SMALL)
        pix_lbl.pack(side="right")

        canvas = tk.Canvas(card, bg="#0a0c14", highlightthickness=0, cursor="crosshair")
        canvas.pack(fill="both", expand=True, padx=2, pady=(0, 2))

        card._canvas = canvas
        card._pix_lbl = pix_lbl
        card._photo = None
        card._offset_x = 0
        card._offset_y = 0

        canvas.bind("<Motion>", lambda e, c=card: self._on_hover(e, c))
        canvas.bind("<Leave>", lambda e, c=card: c._pix_lbl.config(text=""))
        return card

    def _build_param_panel(self):
        pp = self._param_panel
        tk.Label(
            pp, text="PARAMETERS", bg=PANEL, fg=MUTED, font=("Courier New", 9, "bold")
        ).pack(anchor="w", padx=14, pady=(14, 4))

        self._op_title_lbl = tk.Label(
            pp,
            text="Select an operation →",
            bg=PANEL,
            fg=TEXT,
            font=FONT_TITLE,
            wraplength=240,
            justify="left",
        )
        self._op_title_lbl.pack(anchor="w", padx=14, pady=(0, 6))

        tk.Frame(pp, bg=BORDER, height=1).pack(fill="x", padx=10, pady=4)

        # scrollable param area
        self._param_scroll_frame = tk.Frame(pp, bg=PANEL)
        self._param_scroll_frame.pack(fill="both", expand=True, padx=10)

        tk.Frame(pp, bg=BORDER, height=1).pack(fill="x", padx=10, pady=8)

        # chain mode
        chain_frame = tk.Frame(pp, bg=PANEL)
        chain_frame.pack(fill="x", padx=14, pady=(0, 4))
        self._chain_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            chain_frame,
            text="Chain (use last output as input)",
            variable=self._chain_var,
            bg=PANEL,
            fg=MUTED,
            selectcolor=CARD,
            activebackground=PANEL,
            font=FONT_SMALL,
            cursor="hand2",
        ).pack(anchor="w")

        # apply button
        self._apply_btn = tk.Button(
            pp,
            text="▶  Apply Operation",
            bg=ACCENT,
            fg="#fff",
            font=("Courier New", 11, "bold"),
            relief="flat",
            cursor="hand2",
            activebackground="#7b96fc",
            command=self._apply_op,
            state="disabled",
            pady=12,
        )
        self._apply_btn.pack(fill="x", padx=14, pady=(0, 8))

        # description area
        self._desc_lbl = tk.Label(
            pp,
            text="",
            bg=PANEL,
            fg=MUTED,
            font=FONT_SMALL,
            wraplength=240,
            justify="left",
        )
        self._desc_lbl.pack(anchor="w", padx=14, pady=4)

    # ── op tree population ────────────────────────────────────────────────────
    def _populate_op_tree(self, filter_text=""):
        self._tree.delete(*self._tree.get_children())
        cats = {}
        ft = filter_text.lower()
        for idx, op in enumerate(OPERATIONS):
            if ft and ft not in op["label"].lower() and ft not in op["cat"].lower():
                continue
            cat = op["cat"]
            if cat not in cats:
                node = self._tree.insert(
                    "",
                    "end",
                    text=f"  {cat}",
                    values=("",),
                    open=bool(ft),
                    tags=("cat",),
                )
                cats[cat] = node
                self._tree.tag_configure("cat", foreground=MUTED)
            self._tree.insert(
                cats[cat],
                "end",
                text=f"    {op['label']}",
                values=(str(idx),),
                tags=("op",),
            )
            self._tree.tag_configure("op", foreground=TEXT)
        if not filter_text:
            # expand first category
            children = self._tree.get_children()
            if children:
                self._tree.item(children[0], open=True)

    def _on_search(self, *_):
        self._populate_op_tree(self._search_var.get())

    def _on_op_select(self, _event=None):
        sel = self._tree.selection()
        if not sel:
            return
        vals = self._tree.item(sel[0], "values")
        if not vals or vals[0] == "":
            return
        idx = int(vals[0])
        self._load_op_params(idx)

    def _load_op_params(self, idx: int):
        op = OPERATIONS[idx]
        self._current_op_idx = idx

        self._op_title_lbl.config(text=op["label"])

        # clear old param widgets
        for w in self._param_scroll_frame.winfo_children():
            w.destroy()
        self._param_vars.clear()

        for param in op["params"]:
            name = param[0]
            ptype = param[1]
            default = param[2]

            row = tk.Frame(self._param_scroll_frame, bg=PANEL)
            row.pack(fill="x", pady=4)

            tk.Label(
                row,
                text=name,
                bg=PANEL,
                fg=MUTED,
                font=FONT_SMALL,
                width=12,
                anchor="w",
            ).pack(side="left")

            if ptype == "choice":
                choices = param[3]
                var = tk.StringVar(value=str(default))
                cb = ttk.Combobox(
                    row,
                    textvariable=var,
                    values=choices,
                    state="readonly",
                    width=14,
                    font=FONT_UI,
                )
                cb.pack(side="right", fill="x", expand=True)
                self._param_vars[name] = var

            elif ptype in ("float", "int"):
                var = tk.StringVar(value=str(default))
                e = tk.Entry(
                    row,
                    textvariable=var,
                    bg=CARD,
                    fg=TEXT,
                    insertbackground=TEXT,
                    relief="flat",
                    width=10,
                    font=FONT_MONO,
                    highlightthickness=1,
                    highlightbackground=BORDER,
                    highlightcolor=ACCENT,
                )
                e.pack(side="right", fill="x", expand=True, ipady=4)
                self._param_vars[name] = var

            elif ptype == "path":
                var = tk.StringVar(value=str(default))
                pf = tk.Frame(row, bg=PANEL)
                pf.pack(side="right", fill="x", expand=True)
                tk.Entry(
                    pf,
                    textvariable=var,
                    bg=CARD,
                    fg=TEXT,
                    insertbackground=TEXT,
                    relief="flat",
                    font=FONT_MONO,
                    highlightthickness=1,
                    highlightbackground=BORDER,
                    highlightcolor=ACCENT,
                ).pack(side="left", fill="x", expand=True, ipady=4)
                tk.Button(
                    pf,
                    text="…",
                    bg=BORDER,
                    fg=TEXT,
                    font=FONT_SMALL,
                    relief="flat",
                    cursor="hand2",
                    command=lambda v=var: v.set(
                        filedialog.askopenfilename(filetypes=[("TIFF", "*.tif *.tiff")])
                    ),
                ).pack(side="right", padx=2)
                self._param_vars[name] = var

        if not op["params"]:
            tk.Label(
                self._param_scroll_frame,
                text="No parameters required.",
                bg=PANEL,
                fg=MUTED,
                font=FONT_SMALL,
            ).pack(anchor="w", pady=8)

        has_src = self._src_path is not None
        self._apply_btn.config(state="normal" if has_src else "disabled")

    # ── file ops ──────────────────────────────────────────────────────────────
    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open TIFF Image",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")],
        )
        if not path:
            return
        self._src_path = path
        self._work_path = path
        self._out_path = None
        self._history.clear()
        self._hist_box.delete(0, "end")
        self._pil_before = tiff_to_pil(path)
        self._pil_after = None
        self._render_previews()
        self._update_info(path)
        self._status("Loaded: " + os.path.basename(path))
        if hasattr(self, "_current_op_idx"):
            self._apply_btn.config(state="normal")

    def _save_as(self):
        if not self._out_path and not self._work_path:
            messagebox.showwarning("Nothing to save", "Apply an operation first.")
            return
        src = self._out_path or self._work_path
        dest = filedialog.asksaveasfilename(
            defaultextension=".tif",
            filetypes=[("TIFF files", "*.tif *.tiff")],
            initialfile="enhanced.tif",
        )
        if dest:
            import shutil

            shutil.copy2(src, dest)
            self._status(f"Saved → {os.path.basename(dest)}")

    def _reset(self):
        if not self._src_path:
            return
        self._work_path = self._src_path
        self._out_path = None
        self._pil_before = tiff_to_pil(self._src_path)
        self._pil_after = None
        self._history.clear()
        self._hist_box.delete(0, "end")
        self._render_previews()
        self._status("Reset to original.")

    def _undo(self):
        if len(self._history) < 2:
            self._status("Nothing to undo.")
            return
        self._history.pop()
        self._hist_box.delete("end")
        label, path = self._history[-1]
        self._work_path = path
        self._out_path = path
        self._pil_after = tiff_to_pil(path)
        if len(self._history) >= 2:
            self._pil_before = tiff_to_pil(self._history[-2][1])
        else:
            self._pil_before = tiff_to_pil(self._src_path)
        self._render_previews()
        self._status(f"Undo → {label}")

    def _on_history_jump(self, _event):
        sel = self._hist_box.curselection()
        if not sel:
            return
        idx = sel[0]
        label, path = self._history[idx]
        self._work_path = path
        self._out_path = path
        self._pil_after = tiff_to_pil(path)
        if idx > 0:
            self._pil_before = tiff_to_pil(self._history[idx - 1][1])
        else:
            self._pil_before = tiff_to_pil(self._src_path)
        self._render_previews()
        self._status(f"Jumped to: {label}")

    # ── apply ──────────────────────────────────────────────────────────────────
    def _apply_op(self):
        if not hasattr(self, "_current_op_idx"):
            self._status("Select an operation first.")
            return
        if not self._src_path:
            self._status("Open an image first.")
            return

        op = OPERATIONS[self._current_op_idx]
        inp = self._work_path if self._chain_var.get() else self._src_path

        # collect params
        params = {}
        for pdef in op["params"]:
            name, ptype = pdef[0], pdef[1]
            raw = self._param_vars[name].get()
            try:
                if ptype == "float":
                    params[name] = float(raw)
                elif ptype == "int":
                    params[name] = int(raw)
                else:
                    params[name] = raw
            except ValueError:
                messagebox.showerror(
                    "Invalid parameter", f"'{name}' must be a {ptype}. Got: '{raw}'"
                )
                return

        # build temp output path
        suffix = op["label"].lower().replace(" ", "_").replace("/", "_")
        out_fd, out_path = tempfile.mkstemp(suffix=f"_{suffix}.tif")
        os.close(out_fd)

        # run in-process (show busy cursor)
        self.config(cursor="watch")
        self.update()
        self._apply_btn.config(text="⏳  Processing…", state="disabled")
        self.update()

        try:
            op["fn"](inp, out_path, params)
        except Exception as e:
            self.config(cursor="")
            self._apply_btn.config(text="▶  Apply Operation", state="normal")
            messagebox.showerror("Operation failed", str(e))
            self._status(f"Error: {e}")
            return

        self.config(cursor="")
        self._apply_btn.config(text="▶  Apply Operation", state="normal")

        # update state
        self._out_path = out_path
        self._work_path = out_path
        prev_src = inp
        self._pil_before = tiff_to_pil(prev_src)
        self._pil_after = tiff_to_pil(out_path)

        self._history.append((op["label"], out_path))
        self._hist_box.insert("end", f"  {len(self._history):02d}  {op['label']}")
        self._hist_box.see("end")

        self._render_previews()
        self._status(f"Applied: {op['label']}  ✓")

    # ── previews ──────────────────────────────────────────────────────────────
    def _render_previews(self):
        self._render_one(self._before_frame, self._pil_before, "BEFORE")
        self._render_one(self._after_frame, self._pil_after, "AFTER")

    def _render_one(self, card, pil_img, title: str):
        canvas = card._canvas
        canvas.update_idletasks()
        cw = canvas.winfo_width() or 560
        ch = canvas.winfo_height() or 460

        if pil_img is None:
            canvas.delete("all")
            canvas.create_text(
                cw // 2,
                ch // 2,
                text=f"No {title.lower()} image",
                fill=MUTED,
                font=FONT_UI,
            )
            card._photo = None
            return

        photo = fit_image(
            pil_img, int(cw * self._zoom_level), int(ch * self._zoom_level)
        )
        card._photo = photo  # keep reference

        pw, ph = photo.width(), photo.height()
        x = (cw - pw) // 2
        y = (ch - ph) // 2
        card._offset_x = x
        card._offset_y = y

        canvas.delete("all")
        canvas.create_image(x, y, anchor="nw", image=photo)

    def _on_hover(self, event, card):
        if card._photo is None or (
            (card is self._before_frame and self._pil_before is None)
            or (card is self._after_frame and self._pil_after is None)
        ):
            return
        pil_img = self._pil_before if card is self._before_frame else self._pil_after
        if pil_img is None:
            return
        # map canvas coords → image coords
        iw, ih = pil_img.size
        canvas = card._canvas
        cw = canvas.winfo_width() or 560
        ch = canvas.winfo_height() or 460
        pw = card._photo.width()
        ph = card._photo.height()
        ox = card._offset_x
        oy = card._offset_y
        # pixel in displayed image
        px_img = event.x - ox
        py_img = event.y - oy
        if 0 <= px_img < pw and 0 <= py_img < ph:
            # map to original image coords
            sx = int(px_img * iw / pw)
            sy = int(py_img * ih / ph)
            if 0 <= sx < iw and 0 <= sy < ih:
                pixel = pil_img.getpixel((sx, sy))
                if isinstance(pixel, int):
                    pixel = (pixel, pixel, pixel)
                card._pix_lbl.config(text=f"({sx}, {sy})  RGB {pixel[:3]}")

    def _update_info(self, path):
        try:
            arr = tiff.imread(path)
            shape = arr.shape
            dtype = arr.dtype
            size = os.path.getsize(path) // 1024
            info = f"{os.path.basename(path)}   {shape[1]}×{shape[0]}"
            if arr.ndim == 3:
                info += f"×{shape[2]}"
            info += f"  {dtype}  {size} KB"
            self._info_label.config(text=info, fg=TEXT)
        except Exception:
            pass

    def _zoom_in(self):
        self._zoom_level = min(self._zoom_level * 1.25, 4.0)
        self._render_previews()

    def _zoom_out(self):
        self._zoom_level = max(self._zoom_level / 1.25, 0.2)
        self._render_previews()

    def _zoom_fit(self):
        self._zoom_level = 1.0
        self._render_previews()

    def _status(self, msg: str):
        self._status_var.set(f"  {msg}")

    # ── resize debounce ───────────────────────────────────────────────────────
    def _on_resize(self, _event=None):
        if hasattr(self, "_resize_job"):
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(120, self._render_previews)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.bind("<Configure>", app._on_resize)
    app.mainloop()
