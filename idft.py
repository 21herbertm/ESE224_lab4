
# Standard libraries
import numpy as np
import matplotlib.pyplot as plt
import cmath
from scipy.io.wavfile import write
import sounddevice as sd




class idft():
    """
    idft Inverse Discrete Fourier transform.
    """

    def __init__(self, X, fs, N, K=None):
        """
        :param X: Input DFT X
        :param fs: Input integer fs contains the sample frequency
        :param N: The number of total signal samples N
        :param K: Input positive integer that determines the number of coeffients
        used to calculate the iDFT.
        """
        self.X = X
        self.fs = fs
        self.N = N
        self.K = K
        if self.K == None:
            self.K = int(len(X) / 2) - 1

    def solve_K(self):
        """
        \\\\\ METHOD: Compute the iDFT with truncated K coefficients (Due to computation complexity, we will not use it in Section 2)
        :return iDFT x of duration N from partial DFT X, i.e., X[0], ..., X[K] with K < N/2
        :Treal the realt time vector of size N
        """
        x = np.zeros(self.N)

        for n in range(self.N):
            x[n] = 1 / np.sqrt(self.N) * self.X[0] * np.exp(1j * 2 * cmath.pi * 0 * n / self.N)
            for k in range(1, self.K + 1):
                x[n] = x[n] + 1 / np.sqrt(self.N) * self.X[k] * np.exp(1j * 2 * cmath.pi * k * n / self.N)
                x[n] = x[n] + 1 / np.sqrt(self.N) * np.conj(self.X[k]) * np.exp(-1j * 2 * cmath.pi * k * n / self.N)

        Ts = 1 / self.fs
        Treal = np.arange(self.N) * Ts

        return x, Treal

    def solve_ifft(self):
        """
        \\\\\ METHOD: Compute the iDFT with provided function np.fft.ifft (Computationally efficient)
        :Treal the realt time vector of size N
        """
        x = np.fft.ifft(self.X, self.N) * np.sqrt(self.N)

        Ts = 1 / self.fs
        Treal = np.arange(self.N) * Ts

        return x, Treal
