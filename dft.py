#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 12:38:57 2021

@author: zhiyangwang
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
import warnings
from scipy.io.wavfile import write
from scipy.io.wavfile import read

from scipy import signal
import sounddevice as sd

'''
CLASS FOR DFT CALCULATIONS
    ATTRIBUTES:  x: INPUT VECTOR X OF THE DISCRETE SIGNAL
                 fs: INPUT INTEGER FS CONTAINS THE SAMPLE FREQUENCY
                 K: INPUT POSITIVE INTEGER THAT DETERMINES THE NUMBER OF COEFFIENTS USED TO CALCULATE THE DFT. 
                    If K IS NOT PROVIDED, K=length(x). 
                f: ARE THE FREQUENCIES STARTING AT F=0 AND X ARE THE CORRESPONDING FREQUENCY COMPONENTS. 
                f_c: VECTOR CONTAINING THE FREQUENCIES SUCH THAT f_c=0 is at the center
                X_c: CONTAINS THE FREQUENCY COMPONENTS CORRESPONDING TO f_c.

'''


class dft():

    def __init__(self, x, fs, K=None):
        """
        :param x: Input vector x contains the discrete signal
        :param fs: Input integer fs contains the sample frequency
        :param K: Input positive integer that determines the number of coeffients
        used to calculate the DFT. If K is not provided, K=length(x).
        """
        # START: SANITY CHECK OF INPUTS.
        if (type(fs) != int) or (fs <= 0):
            raise NameError('The frequency fs should be a positive integer.')
        if not isinstance(x, np.ndarray):
            raise NameError('The input signal x must be a numpy array.')
        if isinstance(x, np.ndarray):
            if x.ndim != 1:
                raise NameError('The input signal x must be a numpy vector array.')
        self.x = x
        self.fs = fs
        self.N = len(x)
        if K == None:
            K = len(self.x)
        # START: SANITY CHECK OF INPUTS.
        if (type(K) != int) or (K <= 0) or (K < 0):
            raise NameError('K should be a positive integer.')
        self.K = K

        '''
            GENERAL PYTHON: NUMPY ARRANGE: THE ARANGE() FUNCTION IS USED TO GET EVENLY SPACED VALUES WITHIN A 
            GIVEN INTERVAL.numpy.arange([start, ]stop, [step, ]dtype=None)
            >>> np.arange(5.0)
            >>> array([ 0.,  1.,  2.,  3.,  4.])
            SEE https://www.w3resource.com/numpy/array-creation/arange.php 
            '''

        '''CREATE FRQUENCY VECTOR FROM 0 TO K USING A STEP OF 1'''
        self.f = np.arange(self.N) * self.fs / self.N

        ''' NEED TO HANDLE CONDITIONS WHERE VECTOR CONTAINING THE FREQUENCIES SUCH THAT f_c=0 is at the center'''
        ''' FLOOR ROUNDS DOWN, CEILING ROUNDS UP'''
        self.f_c = np.arange(-np.ceil(self.N / 2) + 1, np.floor(self.N / 2) + 1) * self.fs / self.N

        # This accounts for the frequencies
        # centered at zero. I want to be guaranteed that k=0 is always a
        # possible k. Then, I also have to account for both even and odd choices
        # of K, and that's why the floor() function appears to round down the
        # numbers.

    def changeK(self, K):
        """
        :param K: Input positive integer that determines the number of coeffients
        used to calculate the DFT. This function changes the attribute K of the class.
        """
        if (type(K) != int) or (K <= 0) or (K < 0):
            raise NameError('K should be a positive integer.')
        old_K = self.K
        self.K = K
        self.f = np.arange(self.K) * self.fs / self.K  # (0:K-1) just creates a vector from 0 to K by steps of 1.
        self.f_c = np.arange(-np.ceil(K / 2) + 1, np.floor(self.K / 2) + 1) * self.fs / self.K
        # This accounts for the frequencies
        # centered at zero. I want to be guaranteed that k=0 is always a
        # possible k. Then, I also have to account for both even and odd choices
        # of K, and that's why the floor() function appears to round down the
        # numbers.
        print('The value of K was succefully change from %d to %d' % (old_K, self.K))
        pass

    def solve_using_numpy_fft(self):

        X = np.fft.fft(self.x, self.N) / np.sqrt(self.N);
        # \\\\\ CENTER FFT.
        X_c = np.roll(X, np.int(np.ceil(self.N / 2 - 1)))  # Circularly shift X to get it centered in f_c==0
        return [self.f, X, self.f_c, X_c]
