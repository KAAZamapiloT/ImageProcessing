"""Core Fourier engine with vectorized editing, caching, and history."""

import numpy as np

from filters import (
    gaussian_blur_transfer,
    ideal_highpass,
    ideal_lowpass,
    motion_blur_transfer,
)


class FFTEngine:
    def __init__(self, max_history=120):
        self.original = None
        self.secondary = None

        self.original_fft_shifted = None
        self.fft_shifted = None

        self.degraded_image = None
        self.degraded_fft_shifted = None
        self.blur_transfer = None

        self.max_history = max_history
        self.undo_stack = []
        self.redo_stack = []

    # =====================================================
    # Image Handling / Caching
    # =====================================================

    def load_image(self, img):
        if img is None:
            raise ValueError("No image loaded.")
        self.original = img.astype(np.float32)
        self.original_fft_shifted = self.compute_fft(self.original)
        self.fft_shifted = self.original_fft_shifted.copy()

        self.secondary = None
        self.degraded_image = None
        self.degraded_fft_shifted = None
        self.blur_transfer = None
        self.clear_history()

    def load_secondary_image(self, img):
        if img is None:
            raise ValueError("No secondary image loaded.")
        self.secondary = img.astype(np.float32)

    def compute_fft(self, image=None):
        source = self.original if image is None else image
        if source is None:
            raise ValueError("No image loaded.")
        return np.fft.fftshift(np.fft.fft2(source))

    def reset(self):
        if self.original_fft_shifted is None:
            raise ValueError("No image loaded.")
        self.fft_shifted = self.original_fft_shifted.copy()
        self.degraded_image = None
        self.degraded_fft_shifted = None
        self.blur_transfer = None
        self.clear_history()

    # =====================================================
    # History
    # =====================================================

    def clear_history(self):
        self.undo_stack.clear()
        self.redo_stack.clear()

    def _push_undo(self):
        if self.fft_shifted is None:
            return
        self.undo_stack.append(self.fft_shifted.copy())
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack or self.fft_shifted is None:
            return False
        self.redo_stack.append(self.fft_shifted.copy())
        self.fft_shifted = self.undo_stack.pop()
        return True

    def redo(self):
        if not self.redo_stack or self.fft_shifted is None:
            return False
        self.undo_stack.append(self.fft_shifted.copy())
        self.fft_shifted = self.redo_stack.pop()
        return True

    # =====================================================
    # Spectrum / Reconstruction
    # =====================================================

    def magnitude(self, source_fft=None, log_scale=True):
        F = self.fft_shifted if source_fft is None else source_fft
        if F is None:
            raise ValueError("FFT not available.")
        mag = np.abs(F)
        return np.log1p(mag) if log_scale else mag

    def phase(self, source_fft=None):
        F = self.fft_shifted if source_fft is None else source_fft
        if F is None:
            raise ValueError("FFT not available.")
        return np.angle(F)

    def reconstruct(self, source_fft=None):
        F = self.fft_shifted if source_fft is None else source_fft
        if F is None:
            raise ValueError("FFT not available.")
        out = np.fft.ifft2(np.fft.ifftshift(F))
        return np.real(out).astype(np.float32)

    def reconstruct_from(self, F):
        return self.reconstruct(F)

    # =====================================================
    # Core FFT Mutations
    # =====================================================

    def apply_filter(self, H, push_undo=True):
        if self.fft_shifted is None:
            raise ValueError("FFT not available.")
        if H.shape != self.fft_shifted.shape:
            raise ValueError("Filter shape mismatch.")
        if push_undo:
            self._push_undo()
        self.fft_shifted = self.fft_shifted * H

    def set_fft(self, new_fft, push_undo=False):
        if push_undo and self.fft_shifted is not None:
            self._push_undo()
        self.fft_shifted = np.asarray(new_fft, dtype=np.complex128).copy()

    def get_fft_copy(self):
        if self.fft_shifted is None:
            raise ValueError("FFT not available.")
        return self.fft_shifted.copy()

    def _ensure_fft(self):
        if self.fft_shifted is None:
            raise ValueError("FFT not available.")

    def _symmetric_mask(self, mask):
        M, N = mask.shape
        sym_rows = (M - np.arange(M)) % M
        sym_cols = (N - np.arange(N)) % N
        return mask[np.ix_(sym_rows, sym_cols)]

    def _full_symmetric_mask(self, mask):
        return np.logical_or(mask, self._symmetric_mask(mask))

    def enforce_conjugate_symmetry(self):
        self._ensure_fft()
        F = self.fft_shifted
        M, N = F.shape

        sym_rows = (M - np.arange(M)) % M
        sym_cols = (N - np.arange(N)) % N
        F_sym = np.conj(F[np.ix_(sym_rows, sym_cols)])
        F_new = 0.5 * (F + F_sym)

        # Force self-conjugate points to be real.
        self_rows = np.where((2 * np.arange(M)) % M == 0)[0]
        self_cols = np.where((2 * np.arange(N)) % N == 0)[0]
        if len(self_rows) and len(self_cols):
            rr, cc = np.meshgrid(self_rows, self_cols, indexing="ij")
            F_new[rr, cc] = np.real(F_new[rr, cc]) + 0j

        self.fft_shifted = F_new

    def _apply_mask_edit(self, mask, mode="zero", boost_factor=5.0, phase_amount=1.0):
        self._ensure_fft()
        if not np.any(mask):
            return False

        full_mask = self._full_symmetric_mask(mask)
        self._push_undo()
        F = self.fft_shifted.copy()

        values = F[full_mask]
        mag = np.abs(values)
        phase = np.angle(values)

        if mode == "zero":
            F[full_mask] = 0.0 + 0.0j
        elif mode == "boost":
            new_mag = mag * float(boost_factor)
            F[full_mask] = new_mag * np.exp(1j * phase)
        elif mode == "phase_randomize":
            amount = float(np.clip(phase_amount, 0.0, 1.0))
            random_phase = phase + (np.random.uniform(-np.pi, np.pi, size=phase.shape) * amount)
            F[full_mask] = mag * np.exp(1j * random_phase)
        else:
            raise ValueError(f"Unknown mask edit mode: {mode}")

        self.fft_shifted = F
        self.enforce_conjugate_symmetry()
        return True

    # =====================================================
    # Vectorized Tool Operations
    # =====================================================

    def circle_mask(self, u, v, radius):
        self._ensure_fft()
        M, N = self.fft_shifted.shape
        Y, X = np.ogrid[:M, :N]
        return (X - int(v)) ** 2 + (Y - int(u)) ** 2 <= int(radius) ** 2

    def rectangle_mask(self, u, v, half_height, half_width):
        self._ensure_fft()
        M, N = self.fft_shifted.shape
        top = max(0, int(u) - int(half_height))
        bottom = min(M, int(u) + int(half_height) + 1)
        left = max(0, int(v) - int(half_width))
        right = min(N, int(v) + int(half_width) + 1)
        mask = np.zeros((M, N), dtype=bool)
        mask[top:bottom, left:right] = True
        return mask

    def line_mask(self, u, v, length, thickness, orientation="horizontal"):
        self._ensure_fft()
        M, N = self.fft_shifted.shape
        Y, X = np.ogrid[:M, :N]

        u = int(u)
        v = int(v)
        length = max(int(length), 1)
        thickness = max(int(thickness), 1)

        half_len = length // 2
        half_t = thickness // 2
        orientation = orientation.lower()

        if orientation == "vertical":
            return (np.abs(X - v) <= half_t) & (np.abs(Y - u) <= half_len)
        if orientation == "diag_down":
            return (np.abs((Y - u) - (X - v)) <= half_t) & (np.abs(Y - u) <= half_len)
        if orientation == "diag_up":
            return (np.abs((Y - u) + (X - v)) <= half_t) & (np.abs(Y - u) <= half_len)
        return (np.abs(Y - u) <= half_t) & (np.abs(X - v) <= half_len)

    def apply_circle_zero(self, u, v, radius):
        return self._apply_mask_edit(self.circle_mask(u, v, radius), mode="zero")

    def apply_circle_boost(self, u, v, radius, boost_factor=5.0):
        return self._apply_mask_edit(
            self.circle_mask(u, v, radius),
            mode="boost",
            boost_factor=boost_factor,
        )

    def apply_rectangular_mask(self, u, v, half_height, half_width):
        return self._apply_mask_edit(
            self.rectangle_mask(u, v, half_height, half_width),
            mode="zero",
        )

    def apply_line_suppression(self, u, v, length=120, thickness=2, orientation="horizontal"):
        return self._apply_mask_edit(
            self.line_mask(u, v, length, thickness, orientation=orientation),
            mode="zero",
        )

    def apply_phase_randomizer(self, u, v, radius=12, amount=1.0):
        return self._apply_mask_edit(
            self.circle_mask(u, v, radius),
            mode="phase_randomize",
            phase_amount=amount,
        )

    def apply_global_phase_randomization(self, amount=1.0):
        self._ensure_fft()
        self._push_undo()
        F = self.fft_shifted.copy()
        mag = np.abs(F)
        phase = np.angle(F)
        phase = phase + (np.random.uniform(-np.pi, np.pi, size=phase.shape) * float(np.clip(amount, 0.0, 1.0)))
        self.fft_shifted = mag * np.exp(1j * phase)
        self.enforce_conjugate_symmetry()

    # =====================================================
    # Educational Operations
    # =====================================================

    def remove_dc_component(self):
        self._ensure_fft()
        self._push_undo()
        M, N = self.fft_shifted.shape
        self.fft_shifted[M // 2, N // 2] = 0.0 + 0.0j

    def remove_high_frequencies(self, cutoff):
        self._ensure_fft()
        H = ideal_lowpass(self.fft_shifted.shape, float(cutoff))
        self.apply_filter(H)

    def remove_low_frequencies(self, cutoff):
        self._ensure_fft()
        H = ideal_highpass(self.fft_shifted.shape, float(cutoff))
        self.apply_filter(H)

    def magnitude_only_reconstruction(self):
        self._ensure_fft()
        mag = np.abs(self.fft_shifted)
        return self.reconstruct_from(mag.astype(np.complex128))

    def phase_only_reconstruction(self):
        self._ensure_fft()
        phase = np.angle(self.fft_shifted)
        return self.reconstruct_from(np.exp(1j * phase))

    def swap_magnitude_with_secondary(self):
        self._ensure_fft()
        if self.secondary is None:
            raise ValueError("Secondary image not loaded.")

        secondary_fft = self.compute_fft(self.secondary)
        if secondary_fft.shape != self.fft_shifted.shape:
            raise ValueError("Secondary image shape mismatch.")

        self._push_undo()
        mag = np.abs(secondary_fft)
        phase = np.angle(self.fft_shifted)
        self.fft_shifted = mag * np.exp(1j * phase)
        self.enforce_conjugate_symmetry()
        return self.reconstruct()

    # =====================================================
    # Degradation / Restoration
    # =====================================================

    def simulate_gaussian_blur(self, sigma=2.0, kernel_size=15):
        if self.original is None or self.original_fft_shifted is None:
            raise ValueError("No image loaded.")

        H = gaussian_blur_transfer(self.original.shape, sigma=sigma, kernel_size=kernel_size)
        G = H * self.original_fft_shifted
        degraded = np.real(np.fft.ifft2(np.fft.ifftshift(G))).astype(np.float32)

        self.blur_transfer = H
        self.degraded_fft_shifted = G
        self.degraded_image = degraded

        self._push_undo()
        self.fft_shifted = G.copy()
        return degraded

    def apply_motion_blur(self, a=0.08, b=0.08, T=1.0):
        if self.original is None or self.original_fft_shifted is None:
            raise ValueError("No image loaded.")

        H = motion_blur_transfer(self.original.shape, a=a, b=b, T=T)
        G = H * self.original_fft_shifted
        degraded = np.real(np.fft.ifft2(np.fft.ifftshift(G))).astype(np.float32)

        self.blur_transfer = H
        self.degraded_fft_shifted = G
        self.degraded_image = degraded

        self._push_undo()
        self.fft_shifted = G.copy()
        return degraded

    def apply_wiener_filter(self, K=0.005, sigma=None, kernel_size=15):
        if self.original is None:
            raise ValueError("No image loaded.")

        if sigma is not None:
            self.blur_transfer = gaussian_blur_transfer(
                self.original.shape,
                sigma=float(sigma),
                kernel_size=int(kernel_size),
            )

        if self.blur_transfer is None:
            self.blur_transfer = gaussian_blur_transfer(
                self.original.shape,
                sigma=2.0,
                kernel_size=int(kernel_size),
            )

        G = self.degraded_fft_shifted if self.degraded_fft_shifted is not None else self.fft_shifted
        if G is None:
            raise ValueError("No degraded spectrum available.")

        H = self.blur_transfer
        K = max(float(K), 0.0)
        denom = np.maximum(np.abs(H) ** 2 + K, 1e-12)
        F_restored = (np.conj(H) / denom) * G

        self._push_undo()
        self.fft_shifted = F_restored
        self.enforce_conjugate_symmetry()
        return self.reconstruct()
