import numpy as np

EPS = 1e-12


def centered_mesh(shape):
    M, N = shape
    U, V = np.meshgrid(np.arange(M), np.arange(N), indexing="ij")
    uc = M / 2.0
    vc = N / 2.0
    return U, V, uc, vc


def centered_frequency_grid(shape):
    M, N = shape
    u = (np.arange(M) - M / 2.0) / max(M, 1)
    v = (np.arange(N) - N / 2.0) / max(N, 1)
    U, V = np.meshgrid(u, v, indexing="ij")
    return U, V


def distance_matrix(shape):
    U, V, uc, vc = centered_mesh(shape)
    return np.sqrt((U - uc) ** 2 + (V - vc) ** 2)


# ============================================================
# Lowpass / Highpass (Ideal, Butterworth, Gaussian)
# ============================================================


def ideal_lowpass(shape, D0):
    D = distance_matrix(shape)
    return (D <= float(D0)).astype(np.float32)


def ideal_highpass(shape, D0):
    return 1.0 - ideal_lowpass(shape, D0)


def butterworth_lowpass(shape, D0, n=2):
    D = distance_matrix(shape)
    D0 = max(float(D0), EPS)
    n = max(int(n), 1)
    return (1.0 / (1.0 + (D / D0) ** (2 * n))).astype(np.float32)


def butterworth_highpass(shape, D0, n=2):
    return 1.0 - butterworth_lowpass(shape, D0, n)


def gaussian_lowpass(shape, D0):
    D = distance_matrix(shape)
    D0 = max(float(D0), EPS)
    return np.exp(-(D**2) / (2.0 * (D0**2))).astype(np.float32)


def gaussian_highpass(shape, D0):
    return 1.0 - gaussian_lowpass(shape, D0)


# ============================================================
# Bandreject / Bandpass (Ideal, Butterworth, Gaussian)
# ============================================================


def ideal_bandreject(shape, D0, W):
    D = distance_matrix(shape)
    D0 = float(D0)
    W = max(float(W), EPS)
    half_w = W / 2.0
    return np.logical_or(D <= (D0 - half_w), D >= (D0 + half_w)).astype(np.float32)


def ideal_bandpass(shape, D0, W):
    return 1.0 - ideal_bandreject(shape, D0, W)


def butterworth_bandreject(shape, D0, W, n=2):
    D = distance_matrix(shape)
    D0 = max(float(D0), EPS)
    W = max(float(W), EPS)
    n = max(int(n), 1)

    numerator = D * W
    denominator = np.maximum(np.abs(D**2 - D0**2), EPS)
    return (1.0 / (1.0 + (numerator / denominator) ** (2 * n))).astype(np.float32)


def butterworth_bandpass(shape, D0, W, n=2):
    return 1.0 - butterworth_bandreject(shape, D0, W, n)


def gaussian_bandreject(shape, D0, W):
    D = distance_matrix(shape)
    D0 = max(float(D0), EPS)
    W = max(float(W), EPS)

    numerator = D**2 - D0**2
    denominator = np.maximum(D * W, EPS)
    return (1.0 - np.exp(-((numerator / denominator) ** 2))).astype(np.float32)


def gaussian_bandpass(shape, D0, W):
    return 1.0 - gaussian_bandreject(shape, D0, W)


# ============================================================
# Notch Filters (Ideal, Butterworth)
# u0, v0 are offsets from centered spectrum origin.
# ============================================================


def _notch_distances(shape, u0, v0):
    U, V, uc, vc = centered_mesh(shape)
    u0 = float(u0)
    v0 = float(v0)
    D1 = np.sqrt((U - (uc + u0)) ** 2 + (V - (vc + v0)) ** 2)
    D2 = np.sqrt((U - (uc - u0)) ** 2 + (V - (vc - v0)) ** 2)
    return D1, D2


def notch_reject_ideal(shape, u0, v0, D0):
    D1, D2 = _notch_distances(shape, u0, v0)
    D0 = max(float(D0), EPS)
    return np.logical_and(D1 > D0, D2 > D0).astype(np.float32)


def notch_pass_ideal(shape, u0, v0, D0):
    return 1.0 - notch_reject_ideal(shape, u0, v0, D0)


def notch_reject_butterworth(shape, u0, v0, D0, n=2):
    D1, D2 = _notch_distances(shape, u0, v0)
    D0 = max(float(D0), EPS)
    n = max(int(n), 1)

    product = np.maximum(D1 * D2, EPS)
    return (1.0 / (1.0 + (D0**2 / product) ** n)).astype(np.float32)


def notch_pass_butterworth(shape, u0, v0, D0, n=2):
    return 1.0 - notch_reject_butterworth(shape, u0, v0, D0, n)


# ============================================================
# Laplacian / High-Frequency Emphasis / Homomorphic
# ============================================================


def laplacian_filter(shape):
    U, V = centered_frequency_grid(shape)
    return (-4.0 * (np.pi**2) * (U**2 + V**2)).astype(np.float32)


def high_frequency_emphasis(shape, D0, alpha=0.5, beta=2.0, family="gaussian", n=2):
    alpha = float(alpha)
    beta = float(beta)
    family = family.lower()

    if family == "ideal":
        hp = ideal_highpass(shape, D0)
    elif family == "butterworth":
        hp = butterworth_highpass(shape, D0, n=n)
    else:
        hp = gaussian_highpass(shape, D0)

    return (alpha + beta * hp).astype(np.float32)


def homomorphic_filter(shape, D0, gammaH=2.0, gammaL=0.5, c=1.0):
    D = distance_matrix(shape)
    D0 = max(float(D0), EPS)
    gammaH = float(gammaH)
    gammaL = float(gammaL)
    c = float(c)
    return (
        (gammaH - gammaL) * (1.0 - np.exp(-c * (D**2) / (D0**2))) + gammaL
    ).astype(np.float32)


# ============================================================
# Restoration / Degradation
# ============================================================


def gaussian_kernel(kernel_size, sigma):
    kernel_size = max(int(kernel_size), 3)
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = max(float(sigma), EPS)

    ax = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax, indexing="ij")
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma * sigma))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def gaussian_blur_transfer(shape, sigma, kernel_size=15):
    kernel = gaussian_kernel(kernel_size, sigma)
    M, N = shape
    H = np.zeros((M, N), dtype=np.float32)

    kh, kw = kernel.shape
    y0 = (M - kh) // 2
    x0 = (N - kw) // 2
    H[y0 : y0 + kh, x0 : x0 + kw] = kernel

    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(H)))


def motion_blur_transfer(shape, a=0.08, b=0.08, T=1.0):
    U, V = centered_frequency_grid(shape)
    a = float(a)
    b = float(b)
    T = float(T)

    phase = np.pi * (U * a + V * b)
    H = np.empty_like(phase, dtype=np.complex128)

    small = np.abs(phase) < EPS
    H[small] = T
    H[~small] = T * np.sin(phase[~small]) * np.exp(-1j * phase[~small]) / phase[~small]
    return H


def wiener_filter(F, H, K):
    H_conj = np.conj(H)
    H_abs2 = np.abs(H) ** 2
    return (H_conj / (H_abs2 + max(float(K), 0.0))) * F


# ============================================================
# Backward-compatible names for existing code paths
# ============================================================


def bandpass(shape, D0_low, D0_high):
    D0 = (float(D0_low) + float(D0_high)) / 2.0
    W = max(float(D0_high) - float(D0_low), EPS)
    return gaussian_bandpass(shape, D0, W)


def bandreject(shape, D0_low, D0_high):
    D0 = (float(D0_low) + float(D0_high)) / 2.0
    W = max(float(D0_high) - float(D0_low), EPS)
    return gaussian_bandreject(shape, D0, W)


def notch_filter(shape, u0, v0, D0):
    return notch_reject_ideal(shape, u0, v0, D0)
