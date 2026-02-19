import numpy as np


def distance_matrix(shape):
    M, N = shape
    U, V = np.meshgrid(np.arange(M), np.arange(N), indexing="ij")
    D = np.sqrt((U - M / 2) ** 2 + (V - N / 2) ** 2)
    return D


def ideal_lowpass(shape, D0):
    D = distance_matrix(shape)
    return (D <= D0).astype(float)


def ideal_highpass(shape, D0):
    return 1 - ideal_lowpass(shape, D0)


def gaussian_lowpass(shape, D0):
    D = distance_matrix(shape)
    return np.exp(-(D**2) / (2 * (D0**2)))


def gaussian_highpass(shape, D0):
    return 1 - gaussian_lowpass(shape, D0)


def butterworth_lowpass(shape, D0, n):
    D = distance_matrix(shape)
    return 1 / (1 + (D / D0) ** (2 * n))


def butterworth_highpass(shape, D0, n):
    return 1 - butterworth_lowpass(shape, D0, n)


def bandpass(shape, D0_low, D0_high):
    low = gaussian_lowpass(shape, D0_high)
    high = gaussian_lowpass(shape, D0_low)
    return low - high


def bandreject(shape, D0_low, D0_high):
    return 1 - bandpass(shape, D0_low, D0_high)


def notch_filter(shape, u0, v0, D0):
    M, N = shape
    U, V = np.meshgrid(np.arange(M), np.arange(N), indexing="ij")

    D1 = np.sqrt((U - u0) ** 2 + (V - v0) ** 2)
    D2 = np.sqrt((U - (M - u0)) ** 2 + (V - (N - v0)) ** 2)

    return (D1 > D0) * (D2 > D0)


def wiener_filter(F, H, K):

    H_conj = np.conj(H)
    H_abs2 = np.abs(H) ** 2
    return (H_conj / (H_abs2 + K)) * F


def homomorphic_filter(shape, D0, gammaH=2.0, gammaL=0.5, c=1.0):
    D = distance_matrix(shape)
    return (gammaH - gammaL) * (1 - np.exp(-c * (D**2) / (D0**2))) + gammaL
