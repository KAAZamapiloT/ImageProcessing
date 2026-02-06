import cmath
import math
import tkinter as tk
from abc import ABC, abstractmethod
from tkinter import messagebox


# --------------------------------------------------
# MATRIX UTILITIES
# --------------------------------------------------
def zeros(rows, cols, complex_val=False):
    v = 0j if complex_val else 0.0
    return [[v for _ in range(cols)] for _ in range(rows)]


def matmul(A, B):
    r1, c1 = len(A), len(A[0])
    r2, c2 = len(B), len(B[0])
    if c1 != r2:
        raise ValueError("Matrix dimension mismatch")

    C = zeros(r1, c2, complex_val=True)
    for i in range(r1):
        for j in range(c2):
            s = 0j
            for k in range(c1):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


# --------------------------------------------------
# DATA MODEL
# --------------------------------------------------
class VisualComponent(ABC):
    def __init__(self, matrix, label):
        self._matrix = matrix
        self.label = label

    @property
    def matrix(self):
        return self._matrix

    @property
    def shape(self):
        return len(self._matrix), len(self._matrix[0])


class ImageSignal(VisualComponent):
    pass


# --------------------------------------------------
# SIGNAL PROCESSING (DFT)
# --------------------------------------------------
class SignalProcessor:
    def dft_matrix(self, N):
        W = zeros(N, N, complex_val=True)
        for u in range(N):
            for x in range(N):
                W[u][x] = cmath.exp(-2j * math.pi * u * x / N)
        return W

    def dft2(self, img: ImageSignal):
        f = img.matrix
        M, N = img.shape

        W_M = self.dft_matrix(M)
        W_N = self.dft_matrix(N)

        temp = matmul(W_M, f)
        F = matmul(temp, W_N)

        # Proper normalization
        for u in range(M):
            for v in range(N):
                F[u][v] /= M * N

        return F

    def center_image(self, img: ImageSignal):
        mat = img.matrix
        M, N = img.shape
        out = zeros(M, N)
        for x in range(M):
            for y in range(N):
                out[x][y] = mat[x][y] * ((-1) ** (x + y))
        return ImageSignal(out, "Centered Image")

    def log_magnitude(self, F, label):
        M, N = len(F), len(F[0])
        out = zeros(M, N)
        for i in range(M):
            for j in range(N):
                out[i][j] = math.log1p(abs(F[i][j]))
        return ImageSignal(out, label)


# --------------------------------------------------
# CONTENT GENERATION
# --------------------------------------------------
class ContentBuilder:
    @staticmethod
    def dft_basis_8x8():
        N = 8
        canvas = zeros(64, 64)

        for u in range(N):
            for v in range(N):
                for x in range(N):
                    for y in range(N):
                        val = cmath.exp(-2j * math.pi * (u * x + v * y) / N).real
                        canvas[u * N + x][v * N + y] = val

        return ImageSignal(canvas, "8×8 DFT Basis (64×64)")

    @staticmethod
    def rectangle_image(size, top_left, wh):
        img = zeros(size, size)
        y0, x0 = top_left
        w, h = wh
        for y in range(y0, min(y0 + h, size)):
            for x in range(x0, min(x0 + w, size)):
                img[y][x] = 1.0
        return ImageSignal(img, "Rectangle Image")


# --------------------------------------------------
# GUI
# --------------------------------------------------
class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Manual 2-D DFT (Correct Implementation)")
        self.proc = SignalProcessor()
        self.zoom = 4

        self._build_ui()

    def _build_ui(self):
        ctrl = tk.Frame(self.root)
        ctrl.pack(pady=5)

        def field(lbl, val, col):
            tk.Label(ctrl, text=lbl).grid(row=0, column=col * 2)
            e = tk.Entry(ctrl, width=5)
            e.insert(0, str(val))
            e.grid(row=0, column=col * 2 + 1)
            return e

        self.in_y = field("Y", 20, 0)
        self.in_x = field("X", 20, 1)
        self.in_w = field("W", 10, 2)
        self.in_h = field("H", 20, 3)

        tk.Button(ctrl, text="Run", command=self.run).grid(row=0, column=8)

        self.frames = []
        self.canvases = []

        view = tk.Frame(self.root)
        view.pack()

        for i in range(4):
            f = tk.Frame(view)
            f.grid(row=i // 2, column=i % 2)
            lbl = tk.Label(f)
            lbl.pack()
            c = tk.Canvas(f, width=256, height=256, bg="black")
            c.pack()
            self.frames.append(lbl)
            self.canvases.append(c)

    def draw(self, idx, img: ImageSignal):
        mat = img.matrix
        H, W = len(mat), len(mat[0])
        flat = [v for row in mat for v in row]
        mn, mx = min(flat), max(flat)
        d = mx - mn if mx != mn else 1

        cvs = self.canvases[idx]
        cvs.delete("all")
        self.frames[idx].config(text=img.label)

        for y in range(H):
            for x in range(W):
                g = int((mat[y][x] - mn) / d * 255)
                col = f"#{g:02x}{g:02x}{g:02x}"
                cvs.create_rectangle(
                    x * self.zoom,
                    y * self.zoom,
                    (x + 1) * self.zoom,
                    (y + 1) * self.zoom,
                    fill=col,
                    width=0,
                )

    def run(self):
        try:
            tl = (int(self.in_y.get()), int(self.in_x.get()))
            wh = (int(self.in_w.get()), int(self.in_h.get()))

            basis = ContentBuilder.dft_basis_8x8()
            self.draw(0, basis)

            rect = ContentBuilder.rectangle_image(64, tl, wh)
            self.draw(1, rect)

            F = self.proc.dft2(rect)
            mag = self.proc.log_magnitude(F, "DFT Magnitude")
            self.draw(2, mag)

            centered = self.proc.center_image(rect)
            Fc = self.proc.dft2(centered)
            magc = self.proc.log_magnitude(Fc, "Centered DFT")
            self.draw(3, magc)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def launch(self):
        self.root.mainloop()


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    App().launch()
