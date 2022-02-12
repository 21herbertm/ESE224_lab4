
# Standard libraries
import numpy as np
import matplotlib.pyplot as plt
import cmath
from scipy.io.wavfile import write
import sounddevice as sd

'''
CLASS FOR INVERSE DFT CALCULATIONS
    ATTRIBUTES:  x: INPUT VECTOR DFT X 
                 fs: INPUT INTEGER FS CONTAINS THE SAMPLE FREQUENCY
                 N: THE NUMBER OF TOTAL SIGNAL SAMPLES N
                 K: INPUT POSITIVE INTEGER THAT DETERMINES THE NUMBER OF COEFFIENTS USED TO CALCULATE THE DFT. 
                    If K IS NOT PROVIDED, K=length(x). 
                f: ARE THE FREQUENCIES STARTING AT F=0 AND X ARE THE CORRESPONDING FREQUENCY COMPONENTS. 
                f_c: VECTOR CONTAINING THE FREQUENCIES SUCH THAT f_c=0 is at the center
                X_c: CONTAINS THE FREQUENCY COMPONENTS CORRESPONDING TO f_c.

'''


class idft():
    def __init__(self, X, fs, N, K=None):
        self.X = X
        self.fs = fs
        self.N = N
        self.K = K
        if self.K == None:
            self.K = int(len(X) / 2) - 1

    '''
    METHOD TO COMPUTE THE iDFT THE HARD WAY  WITH TRUNCATED K COEEFFICIENTS 
        RETURNS:    x: iDFT X OF THE DURATION N 
                    Treal: THE REAL TIME VECOR OF SIZE N '''

    def solve_K(self):

        # CREATE A VECTOR FILLED WITH ZEROS OF SIZE N
        x = np.zeros(self.N)

        # LOOP OVER THE LENGTH OF N AND FILL IN THE iDFT ONE BY ONE
        for n in range(self.N):
            x[n] = 1 / np.sqrt(self.N) * self.X[0] * np.exp(1j * 2 * cmath.pi * 0 * n / self.N)
            for k in range(1, self.K + 1):
                # GENERAL PYTHON: NUMPY CONJ OR CONJUGATE IS THE COMPLEX CONJUGATE OF A COMPLEX NUMBER IS OBTAINED
                # BY CHANGING THE SIGN OF ITS IMAGINARY PART.
                x[n] = x[n] + 1 / np.sqrt(self.N) * self.X[k] * np.exp(1j * 2 * cmath.pi * k * n / self.N)
                x[n] = x[n] + 1 / np.sqrt(self.N) * np.conj(self.X[k]) * np.exp(-1j * 2 * cmath.pi * k * n / self.N)

        Ts = 1 / self.fs

        # CALC THE TS
        Treal = np.arange(self.N) * Ts

        return x, Treal

    '''
      METHOD TO COMPUTE THE iDFT WITH EASY NUMPY FUNCTION ifft
          RETURNS:    x: iDFT X OF THE DURATION N 
                      Treal: THE REAL TIME VECOR OF SIZE N '''

    def solve_ifft(self):
        x = np.fft.ifft(self.X, self.N) * np.sqrt(self.N)

        Ts = 1 / self.fs
        Treal = np.arange(self.N) * Ts

        return x, Treal
