import math
from enum import Enum

import numpy as np
import tifffile as tiff
from typing_extensions import NoDefault


# -------------------------
# Image Object
# -------------------------
class ImageObject:
    def __init__(self, path):
        self.path = path
        self.data = tiff.imread(path)

        if self.data.ndim != 2:
            raise RuntimeError("Only grayscale images supported")

        self.height, self.width = self.data.shape
        self.bits_per_sample = self.data.dtype.itemsize * 8
        self.levels = 1 << self.bits_per_sample

        print(f"TIFF loaded: {self.width}x{self.height}, {self.bits_per_sample}-bit")
        print(f"Actual range: [{self.data.min()}, {self.data.max()}]")

    def save_tiff(self, path):
        tiff.imwrite(path, self.data)
        print(f"Saved TIFF: {path}")

    def save_tiff_8bit(self, path):
        min_val = self.data.min()
        max_val = self.data.max()

        if min_val == max_val:
            out = np.full(self.data.shape, 128, dtype=np.uint8)
        else:
            out = ((self.data - min_val) * 255.0 / (max_val - min_val)).clip(0, 255)

        tiff.imwrite(path, out.astype(np.uint8))
        print(f"Saved 8-bit TIFF: {path}")

    def compute_histogram(self):
        hist = [0] * self.levels

        for value in self.data.flat:  # flat = 1D view, like m_Data
            if value < self.levels:
                hist[int(value)] += 1

        return hist


# -------------------------
# Events
# -------------------------


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
        img.data = (c * np.log(img.data + 1)).astype(img.data.dtype)
        img.save_tiff_8bit(self.out)


class GammaTransformEvent:
    def __init__(self, inp, out, gamma):
        self.inp = inp
        self.out = out
        self.gamma = gamma

    def execute(self):
        img = ImageObject(self.inp)

        min_val = img.data.min()
        max_val = img.data.max()

        norm = img.data / max_val
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
        self.r1 = r1
        self.r2 = r2
        self.k = k
        self.mode = SliceMode.WITH_BG if mode == "bg" else SliceMode.WITHOUT_BG

    def execute(self):
        img = ImageObject(self.inp)
        lut = np.zeros(img.levels, dtype=np.uint16)

        for r in range(img.levels):
            if self.r1 <= r <= self.r2:
                lut[r] = self.k
            else:
                lut[r] = r if self.mode == SliceMode.WITH_BG else 0

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
        if self.with_bg:
            img.data = np.where(mask, max_val, img.data)
        else:
            img.data = np.where(mask, max_val, 0)

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
        self.analyze(img)

    def analyze(self, img):
        L = img.levels
        N = img.width * img.height

        hist = img.compute_histogram()

        min_level = L
        max_level = 0
        mean = 0.0
        variance = 0.0
        entropy = 0.0

        for i in range(L):
            if hist[i] > 0:
                min_level = min(min_level, i)
                max_level = max(max_level, i)
            mean += i * hist[i]

        mean /= N

        for i in range(L):
            diff = i - mean
            variance += diff * diff * hist[i]

            if hist[i] > 0:
                p = hist[i] / N
                entropy -= p * math.log2(p)

        variance /= N

        print("\n--- Histogram Statistics ---")
        print(f"Image size      : {img.width} x {img.height}")
        print(f"Bit depth       : {img.bits_per_sample}")
        print(f"Levels          : {L}")
        print(f"Used range      : [{min_level}, {max_level}]")
        print(f"Mean intensity  : {mean}")
        print(f"Variance        : {variance}")
        print(f"Entropy (bits)  : {entropy}")

        self.print_compact_histogram(hist, N)

    def print_compact_histogram(self, hist, N):
        bins = 16
        L = len(hist)
        step = L // bins

        print("\nHistogram (compressed):")

        for b in range(bins):
            start = b * step
            end = (b + 1) * step
            count = sum(hist[start:end])
            percent = (100.0 * count) / N

            bars = int(percent / 2)
            print(f"[{start}-{end - 1}] : {'#' * bars} ({percent:.2f}%)")


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

        self.match(src_img, ref_img)
        src_img.save_tiff_8bit(self.out)

    def match(self, src, ref):
        L = src.levels
        Ns = src.width * src.height
        Nr = ref.width * ref.height

        histS = src.compute_histogram()
        histR = ref.compute_histogram()

        cdfS = np.zeros(L, dtype=np.float64)
        cdfR = np.zeros(L, dtype=np.float64)

        cdfS[0] = histS[0] / Ns
        cdfR[0] = histR[0] / Nr

        for i in range(1, L):
            cdfS[i] = cdfS[i - 1] + histS[i] / Ns
            cdfR[i] = cdfR[i - 1] + histR[i] / Nr

        lut = np.zeros(L, dtype=np.uint16)

        r = 0
        for s in range(L):
            val = cdfS[s]
            while r < L - 1 and cdfR[r] < val:
                r += 1
            lut[s] = r

        src.data = lut[src.data]


class LocalHistogramEnhancementEvent:
    def __init__(self, inp, out, window_size):
        if window_size < 3 or window_size % 2 == 0:
            raise RuntimeError("Window size must be odd and >= 3")

        self.inp = inp
        self.out = out
        self.window = window_size

    def execute(self):
        img = ImageObject(self.inp)
        self.enhance(img)
        img.save_tiff_8bit(self.out)

    def enhance(self, img):
        L = img.levels
        W = img.width
        H = img.height
        r = self.window // 2

        out = np.zeros((H, W), dtype=np.uint16)

        for y in range(H):
            for x in range(W):
                hist = [0] * L
                total = 0

                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        yy = min(max(y + dy, 0), H - 1)
                        xx = min(max(x + dx, 0), W - 1)

                        v = img.data[yy, xx]
                        hist[v] += 1
                        total += 1

                center = img.data[y, x]
                cdf = sum(hist[: center + 1])

                mapped = (cdf * (L - 1)) / total
                out[y, x] = int(round(mapped))

        img.data = out
