import copy
import math
import os

# =========================
# SIMPLE UI (Fancy CLI)
# =========================
import tkinter as tk
from enum import Enum
from tkinter import filedialog, messagebox

import numpy as np
import tifffile as tiff
from PIL import Image, ImageTk  # pip install pillow  # pip install pillow
from typing_extensions import NoDefault


class ImageToolUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Image Processing Tool")

        self.input_path = ""
        self.orig_img = None
        self.out_img = None

        self._build()

    def _build(self):
        # ---- command ----
        tk.Label(self.root, text="Command").grid(row=0, column=0, sticky="w")
        self.cmd_entry = tk.Entry(self.root, width=30)
        self.cmd_entry.grid(row=0, column=1, sticky="w")

        # ---- params ----
        tk.Label(self.root, text="Parameters (space separated)").grid(
            row=1, column=0, sticky="w"
        )
        self.param_entry = tk.Entry(self.root, width=30)
        self.param_entry.grid(row=1, column=1, sticky="w")

        # ---- output name ----
        tk.Label(self.root, text="Output name (no extension)").grid(
            row=2, column=0, sticky="w"
        )
        self.output_entry = tk.Entry(self.root, width=30)
        self.output_entry.insert(0, "output")
        self.output_entry.grid(row=2, column=1, sticky="w")

        # ---- input ----
        tk.Button(self.root, text="Select Input Image", command=self.pick_input).grid(
            row=3, column=0, columnspan=2, pady=5
        )

        self.input_label = tk.Label(self.root, text="No file selected")
        self.input_label.grid(row=4, column=0, columnspan=2)

        # ---- run ----
        tk.Button(self.root, text="Run", command=self.run).grid(
            row=5, column=0, columnspan=2, pady=5
        )

        # ---- preview labels ----
        tk.Label(self.root, text="Original").grid(row=6, column=0)
        tk.Label(self.root, text="Transformed").grid(row=6, column=1)

        # ---- previews ----
        self.left_preview = tk.Label(self.root)
        self.left_preview.grid(row=7, column=0, pady=5)

        self.right_preview = tk.Label(self.root)
        self.right_preview.grid(row=7, column=1, pady=5)

        # ---- help ----
        tk.Label(self.root, text="Help / Commands").grid(row=0, column=2, sticky="w")
        self.help_box = tk.Text(self.root, width=55, height=30)
        self.help_box.grid(row=1, column=2, rowspan=7, padx=10)
        self.help_box.insert("1.0", self._help_text())
        self.help_box.config(state="disabled")

    # -------------------------
    # Input Handling
    # -------------------------
    def pick_input(self):
        self.input_path = filedialog.askopenfilename(
            filetypes=[("TIFF Images", "*.tif *.tiff"), ("All files", "*.*")]
        )
        if not self.input_path:
            return

        self.input_label.config(text=os.path.basename(self.input_path))
        self._show_original(self.input_path)

    def run(self):
        if not self.input_path:
            messagebox.showerror("Error", "No input image selected")
            return

        cmd = self.cmd_entry.get().strip()
        if not cmd:
            messagebox.showerror("Error", "Command is empty")
            return

        params = self.param_entry.get().strip().split()
        out_name = self.output_entry.get().strip()

        if not out_name:
            messagebox.showerror("Error", "Output name is empty")
            return

        out_path = os.path.join(os.path.dirname(self.input_path), out_name + ".tiff")

        try:
            InputHandler.run_from_ui(cmd, self.input_path, out_path, params)
            self._show_output(out_path)

        except Exception as e:
            messagebox.showerror("Execution Error", str(e))

    # -------------------------
    # Preview Helpers
    # -------------------------
    def _show_original(self, path):
        try:
            img = Image.open(path).convert("L")
            img.thumbnail((256, 256))
            self.orig_img = ImageTk.PhotoImage(img)
            self.left_preview.config(image=self.orig_img)
        except:
            pass

    def _show_output(self, path):
        try:
            img = Image.open(path).convert("L")
            img.thumbnail((256, 256))
            self.out_img = ImageTk.PhotoImage(img)
            self.right_preview.config(image=self.out_img)
        except:
            pass

    # -------------------------
    # Help Text
    # -------------------------
    def _help_text(self):
        return (
            "Available Commands:\n\n"
            "invert\n"
            "log\n"
            "gamma <gamma>\n"
            "contrast <r1> <s1> <r2> <s2>\n"
            "ramp <start> <end>\n"
            "slice <r1> <r2> <k> <bg|nobg>\n"
            "bit_slice <bit> <bg|nobg>\n"
            "hist_eq\n"
            "hist_stats\n"
            "hist_match <ref_image>\n"
            "local_hist <window>\n"
            "smooth_box <kernel>\n"
            "gaussian <kernel> <sigma>\n"
            "sharpen <strength>\n"
            "unsharp <A>\n"
            "grad_edge <k>\n"
            "lap_sobel <lap_out> <sharp_out> <sobel_out>\n"
            "median <window>\n"
            "roberts\n"
            "prewitt\n"
            "sobel [threshold]\n"
            "laplacian <lap_out> <sharp_out> <4|8>\n"
            "bandpass <k1> <sigma1> <k2> <sigma2>\n"
            "bandreject <k1> <sigma1> <k2> <sigma2>\n"
            "weighted_avg\n"
            "grad_sharpen <k>\n\n"
            "Notes:\n"
            "- Output is saved next to input image\n"
            "- Parameters are space-separated (CLI style)\n"
        )

    def start(self):
        self.root.mainloop()


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
        if gamma <= 0:
            raise RuntimeError("Gamma must be > 0")

        self.inp = inp
        self.out = out
        self.gamma = gamma

    def execute(self):
        img = ImageObject(self.inp)

        L = img.levels - 1

        # normalize to [0,1]
        r = img.data.astype(np.float64) / L

        # gamma transform
        s = np.power(r, self.gamma)

        # scale back to [0, L-1]
        img.data = np.round(s * L).astype(img.data.dtype)

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
        L = img.levels - 1

        if not (0 <= self.k <= L):
            raise RuntimeError(f"k must be in [0, {L}]")

        lut = np.zeros(img.levels, dtype=np.uint16)

        for r in range(img.levels):
            if self.r1 <= r <= self.r2:
                lut[r] = self.k
            else:
                lut[r] = r if self.mode == SliceMode.WITH_BG else 0

        img.data = lut[img.data]
        img.save_tiff_8bit(self.out)


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


class BoxSmoothingEvent:
    def __init__(self, inp: str, out: str, kernel: int):
        if kernel < 3 or kernel % 2 == 0:
            raise RuntimeError("Kernel size must be odd and >= 3")

        self.m_InputPath = inp
        self.m_OutputPath = out
        self.m_Kernel = kernel

    def execute(self):
        img = ImageObject(self.m_InputPath)
        self._smooth(img)
        img.save_tiff_8bit(self.m_OutputPath)

    def _smooth(self, img):
        W = img.width
        H = img.height
        r = self.m_Kernel // 2

        out = [[0] * W for _ in range(H)]

        for y in range(H):
            for x in range(W):
                s = 0
                count = 0

                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        yy = min(max(y + dy, 0), H - 1)
                        xx = min(max(x + dx, 0), W - 1)
                        s += img.data[yy][xx]
                        count += 1

                out[y][x] = s // count  # integer average

        # write back
        for y in range(H):
            for x in range(W):
                img.data[y][x] = out[y][x]


class GaussianLowPassEvent:
    def __init__(self, inp: str, out: str, kernel_size: int, sigma: float):
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise RuntimeError("Gaussian kernel size must be odd and >= 3")
        if sigma <= 0.0:
            raise RuntimeError("Sigma must be > 0")

        self.m_InputPath = inp
        self.m_OutputPath = out
        self.m_KernelSize = kernel_size
        self.m_Sigma = sigma

    def execute(self):
        img = ImageObject(self.m_InputPath)
        self._apply_gaussian(img)
        img.save_tiff_8bit(self.m_OutputPath)

    def _build_gaussian_kernel_1d(self):
        r = self.m_KernelSize // 2
        kernel = [0.0] * self.m_KernelSize
        s = 0.0

        for i in range(-r, r + 1):
            v = math.exp(-(i * i) / (2.0 * self.m_Sigma * self.m_Sigma))
            kernel[i + r] = v
            s += v

        # normalize
        for i in range(self.m_KernelSize):
            kernel[i] /= s

        return kernel

    def _apply_gaussian(self, img):
        W = img.width
        H = img.height
        L = img.levels
        r = self.m_KernelSize // 2

        kernel = self._build_gaussian_kernel_1d()

        # ---- horizontal pass ----
        temp = [[0] * W for _ in range(H)]

        for y in range(H):
            for x in range(W):
                acc = 0.0
                for k in range(-r, r + 1):
                    xx = min(max(x + k, 0), W - 1)
                    acc += kernel[k + r] * img.data[y][xx]

                temp[y][x] = int(min(max(acc, 0.0), L - 1))

        # ---- vertical pass ----
        for y in range(H):
            for x in range(W):
                acc = 0.0
                for k in range(-r, r + 1):
                    yy = min(max(y + k, 0), H - 1)
                    acc += kernel[k + r] * temp[yy][x]

                img.data[y][x] = int(min(max(acc, 0.0), L - 1))


class HighPassSharpenEvent:
    def __init__(self, inp: str, out: str, strength: float = 1.0):
        if strength <= 0.0:
            raise RuntimeError("Sharpen strength must be > 0")

        self.m_InputPath = inp
        self.m_OutputPath = out
        self.m_Strength = strength

    def execute(self):
        img = ImageObject(self.m_InputPath)
        self._sharpen(img)
        img.save_tiff_8bit(self.m_OutputPath)

    def _sharpen(self, img):
        H = img.height
        W = img.width
        L = img.levels

        # 4-connected Laplacian kernel
        K = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]

        # high-pass buffer
        lap = [[0] * W for _ in range(H)]

        # ---- compute Laplacian ----
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                s = 0
                for ky in range(-1, 2):
                    for kx in range(-1, 2):
                        s += K[ky + 1][kx + 1] * img.data[y + ky][x + kx]
                lap[y][x] = s

        # ---- add scaled high-pass back ----
        for y in range(H):
            for x in range(W):
                sharpened = img.data[y][x] + self.m_Strength * lap[y][x]
                img.data[y][x] = int(min(max(sharpened, 0.0), L - 1))


class UnsharpHighboostEvent:
    def __init__(self, inp: str, out: str, A: float):
        if A < 1.0:
            raise RuntimeError("A must be >= 1.0")

        self.m_InputPath = inp
        self.m_OutputPath = out
        self.m_A = A

    def execute(self):
        img = ImageObject(self.m_InputPath)
        self._apply(img)
        img.save_tiff_8bit(self.m_OutputPath)

    def _apply(self, img):
        H = img.height
        W = img.width
        L = img.levels

        # 3x3 Gaussian kernel
        G = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        Gsum = 16

        blurred = [[0.0] * W for _ in range(H)]

        # ---- Gaussian blur ----
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                acc = 0
                for ky in range(-1, 2):
                    for kx in range(-1, 2):
                        acc += G[ky + 1][kx + 1] * img.data[y + ky][x + kx]
                blurred[y][x] = acc / Gsum

        # ---- Highboost ----
        for y in range(H):
            for x in range(W):
                original = img.data[y][x]
                result = self.m_A * original - blurred[y][x]

                img.data[y][x] = int(min(max(result, 0.0), L - 1))


class GradientEdgeEnhancementEvent:
    def __init__(self, inp: str, out: str, k: float):
        if k <= 0.0:
            raise RuntimeError("k must be > 0")

        self.m_InputPath = inp
        self.m_OutputPath = out
        self.m_K = k

    def execute(self):
        img = ImageObject(self.m_InputPath)
        self._enhance(img)
        img.save_tiff_8bit(self.m_OutputPath)

    def _enhance(self, img):
        H = img.height
        W = img.width
        L = img.levels

        # Sobel kernels
        Sx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        Sy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

        out = [[0] * W for _ in range(H)]

        for y in range(1, H - 1):
            for x in range(1, W - 1):
                gx = 0
                gy = 0

                for ky in range(-1, 2):
                    for kx in range(-1, 2):
                        p = img.data[y + ky][x + kx]
                        gx += Sx[ky + 1][kx + 1] * p
                        gy += Sy[ky + 1][kx + 1] * p

                grad = abs(gx) + abs(gy)
                enhanced = img.data[y][x] + self.m_K * grad

                out[y][x] = int(min(max(enhanced, 0.0), L - 1))

        # write back
        for y in range(H):
            for x in range(W):
                img.data[y][x] = out[y][x]


class LaplacianSobelSharpenEvent:
    def __init__(self, inp: str, lap_out: str, sharp_out: str, sobel_out: str):
        self.m_Input = inp
        self.m_LapOut = lap_out
        self.m_SharpOut = sharp_out
        self.m_SobelOut = sobel_out

    def execute(self):
        img = ImageObject(self.m_Input)

        lap = copy.deepcopy(img)
        sharp = copy.deepcopy(img)
        sobel = copy.deepcopy(img)

        self._apply_laplacian(img, lap)
        self._apply_sharpen(img, lap, sharp)
        self._apply_sobel(img, sobel)

        lap.save_tiff_8bit(self.m_LapOut)
        sharp.save_tiff_8bit(self.m_SharpOut)
        sobel.save_tiff_8bit(self.m_SobelOut)

    def _apply_laplacian(self, inp, out):
        K = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]

        H = inp.height
        W = inp.width
        L = inp.levels

        for y in range(1, H - 1):
            for x in range(1, W - 1):
                s = 0
                for j in range(-1, 2):
                    for i in range(-1, 2):
                        s += K[j + 1][i + 1] * inp.data[y + j][x + i]

                out.data[y][x] = int(min(max(s, 0), L - 1))

    def _apply_sharpen(self, orig, lap, out):
        H = orig.height
        W = orig.width
        L = orig.levels

        for y in range(H):
            for x in range(W):
                v = orig.data[y][x] + lap.data[y][x]
                out.data[y][x] = int(min(max(v, 0), L - 1))

    def _apply_sobel(self, inp, out):
        Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

        H = inp.height
        W = inp.width
        L = inp.levels

        for y in range(1, H - 1):
            for x in range(1, W - 1):
                sx = 0
                sy = 0

                for j in range(-1, 2):
                    for i in range(-1, 2):
                        p = inp.data[y + j][x + i]
                        sx += Gx[j + 1][i + 1] * p
                        sy += Gy[j + 1][i + 1] * p

                mag = abs(sx) + abs(sy)
                out.data[y][x] = int(min(max(mag, 0), L - 1))


class MedianFilterEvent:
    def __init__(self, inp: str, out: str, window_size: int):
        if window_size < 3 or window_size % 2 == 0:
            raise RuntimeError("Median window must be odd and >= 3")

        self.m_InputPath = inp
        self.m_OutputPath = out
        self.m_Window = window_size

    def execute(self):
        img = ImageObject(self.m_InputPath)
        self._apply(img)
        img.save_tiff_8bit(self.m_OutputPath)

    def _apply(self, img):
        W = img.width
        H = img.height
        r = self.m_Window // 2

        out = [[0] * W for _ in range(H)]

        for y in range(H):
            for x in range(W):
                neighborhood = []

                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        yy = min(max(y + dy, 0), H - 1)
                        xx = min(max(x + dx, 0), W - 1)
                        neighborhood.append(img.data[yy][xx])

                neighborhood.sort()
                out[y][x] = neighborhood[len(neighborhood) // 2]

        # write back
        for y in range(H):
            for x in range(W):
                img.data[y][x] = out[y][x]


class RobertsEdgeEvent:
    def __init__(self, inp: str, out: str):
        self.m_InputPath = inp
        self.m_OutputPath = out

    def execute(self):
        img = ImageObject(self.m_InputPath)
        self._apply(img)
        img.save_tiff_8bit(self.m_OutputPath)

    def _apply(self, img):
        W = img.width
        H = img.height
        L = img.levels

        out = [[0] * W for _ in range(H)]

        for y in range(H - 1):
            for x in range(W - 1):
                gx = img.data[y][x] - img.data[y + 1][x + 1]
                gy = img.data[y][x + 1] - img.data[y + 1][x]

                mag = abs(gx) + abs(gy)

                out[y][x] = int(min(max(mag, 0), L - 1))

        # write back
        for y in range(H):
            for x in range(W):
                img.data[y][x] = out[y][x]


class PrewittEdgeEvent:
    def __init__(self, inp: str, out: str):
        self.m_InputPath = inp
        self.m_OutputPath = out

    def execute(self):
        img = ImageObject(self.m_InputPath)
        self._apply(img)
        img.save_tiff_8bit(self.m_OutputPath)

    def _apply(self, img):
        W = img.width
        H = img.height
        L = img.levels

        Gx = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]

        Gy = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]

        out = [[0] * W for _ in range(H)]

        for y in range(1, H - 1):
            for x in range(1, W - 1):
                gx = 0
                gy = 0

                for ky in range(-1, 2):
                    for kx in range(-1, 2):
                        p = img.data[y + ky][x + kx]
                        gx += Gx[ky + 1][kx + 1] * p
                        gy += Gy[ky + 1][kx + 1] * p

                mag = abs(gx) + abs(gy)

                out[y][x] = int(min(max(mag, 0), L - 1))

        # write back
        for y in range(H):
            for x in range(W):
                img.data[y][x] = out[y][x]


class SobelEdgeEvent:
    def __init__(self, inp: str, out: str, threshold: int = 0):
        self.m_InputPath = inp
        self.m_OutputPath = out
        self.m_Threshold = threshold  # 0 = no threshold

    def execute(self):
        img = ImageObject(self.m_InputPath)
        self._apply(img)
        img.save_tiff_8bit(self.m_OutputPath)

    def _apply(self, img):
        W = img.width
        H = img.height
        L = img.levels

        Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

        Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

        mag = [[0] * W for _ in range(H)]
        max_mag = 0

        # ---- compute gradient magnitude ----
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                gx = 0
                gy = 0

                for ky in range(-1, 2):
                    for kx in range(-1, 2):
                        p = img.data[y + ky][x + kx]
                        gx += Gx[ky + 1][kx + 1] * p
                        gy += Gy[ky + 1][kx + 1] * p

                g = abs(gx) + abs(gy)
                mag[y][x] = g
                if g > max_mag:
                    max_mag = g

        # ---- normalize + threshold ----
        for y in range(H):
            for x in range(W):
                v = mag[y][x]

                norm = int((v * (L - 1)) / max_mag) if max_mag > 0 else 0

                if self.m_Threshold > 0:
                    img.data[y][x] = (L - 1) if norm >= self.m_Threshold else 0
                else:
                    img.data[y][x] = norm


class LaplacianMode(Enum):
    FOUR = 4
    EIGHT = 8


class LaplacianSharpenEvent:
    def __init__(self, inp: str, lap_out: str, sharp_out: str, mode: LaplacianMode):
        self.m_Input = inp
        self.m_LapOut = lap_out
        self.m_SharpOut = sharp_out
        self.m_Mode = mode

    def execute(self):
        img = ImageObject(self.m_Input)

        lap = copy.deepcopy(img)
        sharp = copy.deepcopy(img)

        self._apply_laplacian(img, lap)
        self._apply_sharpen(img, lap, sharp)

        lap.save_tiff_8bit(self.m_LapOut)
        sharp.save_tiff_8bit(self.m_SharpOut)

    def _apply_laplacian(self, inp, out):
        K4 = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]

        K8 = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]

        K = K4 if self.m_Mode == LaplacianMode.FOUR else K8

        H = inp.height
        W = inp.width
        L = inp.levels

        for y in range(1, H - 1):
            for x in range(1, W - 1):
                s = 0
                for j in range(-1, 2):
                    for i in range(-1, 2):
                        s += K[j + 1][i + 1] * inp.data[y + j][x + i]

                out.data[y][x] = int(min(max(s, 0), L - 1))

    def _apply_sharpen(self, orig, lap, out):
        H = orig.height
        W = orig.width
        L = orig.levels

        for y in range(H):
            for x in range(W):
                v = orig.data[y][x] + lap.data[y][x]
                out.data[y][x] = int(min(max(v, 0), L - 1))


class BandMode(Enum):
    BANDPASS = 1
    BANDREJECT = 2


class BandFilterEvent:
    def __init__(
        self, inp: str, out: str, k1: int, s1: float, k2: int, s2: float, mode: BandMode
    ):

        self.m_Input = inp
        self.m_Output = out
        self.m_K1 = k1
        self.m_S1 = s1
        self.m_K2 = k2
        self.m_S2 = s2
        self.m_Mode = mode

    def execute(self):
        img = ImageObject(self.m_Input)

        lp1 = copy.deepcopy(img)
        lp2 = copy.deepcopy(img)

        # reuse GaussianLowPassEvent logic
        GaussianLowPassEvent("", "", self.m_K1, self.m_S1)._apply_gaussian(lp1)
        GaussianLowPassEvent("", "", self.m_K2, self.m_S2)._apply_gaussian(lp2)

        self._apply(img, lp1, lp2)
        img.save_tiff_8bit(self.m_Output)

    def _apply(self, img, lp1, lp2):
        H = img.height
        W = img.width
        L = img.levels

        for y in range(H):
            for x in range(W):
                if self.m_Mode == BandMode.BANDPASS:
                    val = lp2.data[y][x] - lp1.data[y][x]
                else:  # BANDREJECT
                    val = lp1.data[y][x] + (img.data[y][x] - lp2.data[y][x])

                img.data[y][x] = int(min(max(val, 0.0), L - 1))


class WeightedAveragingEvent:
    def __init__(self, inp: str, out: str):
        self.m_InputPath = inp
        self.m_OutputPath = out

    def execute(self):
        img = ImageObject(self.m_InputPath)
        self._apply(img)
        img.save_tiff_8bit(self.m_OutputPath)

    def _apply(self, img):
        W = img.width
        H = img.height
        L = img.levels

        K = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        Ksum = 16

        out = [[0] * W for _ in range(H)]

        for y in range(1, H - 1):
            for x in range(1, W - 1):
                acc = 0
                for ky in range(-1, 2):
                    for kx in range(-1, 2):
                        acc += K[ky + 1][kx + 1] * img.data[y + ky][x + kx]

                out[y][x] = int(min(max(acc // Ksum, 0), L - 1))

        # write back
        for y in range(H):
            for x in range(W):
                img.data[y][x] = out[y][x]


class GradientSharpenEvent:
    def __init__(self, inp: str, out: str, k: float):
        if k <= 0.0:
            raise RuntimeError("Sharpen factor k must be > 0")

        self.m_InputPath = inp
        self.m_OutputPath = out
        self.m_K = k

    def execute(self):
        img = ImageObject(self.m_InputPath)
        self._sharpen(img)
        img.save_tiff_8bit(self.m_OutputPath)

    def _sharpen(self, img):
        W = img.width
        H = img.height
        L = img.levels

        Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

        out = [[0] * W for _ in range(H)]

        for y in range(1, H - 1):
            for x in range(1, W - 1):
                sx = 0
                sy = 0

                for ky in range(-1, 2):
                    for kx in range(-1, 2):
                        p = img.data[y + ky][x + kx]
                        sx += Gx[ky + 1][kx + 1] * p
                        sy += Gy[ky + 1][kx + 1] * p

                grad_mag = abs(sx) + abs(sy)
                sharpened = img.data[y][x] + self.m_K * grad_mag

                out[y][x] = int(min(max(sharpened, 0.0), L - 1))

        # write back
        for y in range(H):
            for x in range(W):
                img.data[y][x] = out[y][x]


class InputHandler:
    @staticmethod
    def run_from_ui(cmd, inp, out, params):
        """
        Called by GUI.
        cmd: str
        inp: input image path
        out: output image path
        params: list[str]
        """
        parts = [cmd, inp, out] + params
        InputHandler._dispatch(parts)

    @staticmethod
    def run():
        while True:
            print("\nCommands:")
            print("invert <input> <output>")
            print("log <input> <output>")
            print("gamma <input> <output> <gamma>")
            print("contrast <input> <output> <r1> <s1> <r2> <s2>")
            print("ramp <input> <output> <start> <end>")
            print("slice <input> <output> <r1> <r2> <k> <bg|nobg>")
            print("bit_slice <input> <output> <bit> <bg|nobg>")
            print("hist_eq <input> <output>")
            print("hist_stats <input>")
            print("hist_match <src> <ref> <output>")
            print("local_hist <input> <output> <window>")
            print("smooth_box <input> <output> <kernel>")
            print("gaussian <input> <output> <kernel> <sigma>")
            print("sharpen <input> <output> <strength>")
            print("unsharp <input> <output> <A>")
            print("grad_edge <input> <output> <k>")
            print("lap_sobel <input> <lap_out> <sharp_out> <sobel_out>")
            print("median <input> <output> <window>")
            print("roberts <input> <output>")
            print("prewitt <input> <output>")
            print("sobel <input> <output> [threshold]")
            print("laplacian <input> <lap_out> <sharp_out> <4|8>")
            print("bandpass <input> <output> <k1> <sigma1> <k2> <sigma2>")
            print("bandreject <input> <output> <k1> <sigma1> <k2> <sigma2>")
            print("weighted_avg <input> <output>")
            print("grad_sharpen <input> <output> <k>")
            print("quit")

            parts = input("> ").strip().split()
            if not parts:
                continue

            if parts[0] == "quit":
                print("Exiting.")
                break

            InputHandler._dispatch(parts)

    @staticmethod
    def _dispatch(parts):
        cmd = parts[0]

        try:
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
                lap_mode = LaplacianMode.EIGHT if mode == "8" else LaplacianMode.FOUR
                LaplacianSharpenEvent(inp, lap, sharp, lap_mode).execute()

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

            else:
                print("Unknown command")

        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    mode = input("Select mode (cli / ui): ").strip().lower()

    if mode == "ui":
        ImageToolUI().start()
    else:
        InputHandler.run()
