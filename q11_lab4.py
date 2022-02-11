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



# Local files
import cexp
import dft
import idft
 

class dft():
    """
    dft Discrete Fourier transform.
    solve1,solve2,solve3 are the discrete Fourier transform (DFT) of
    vector x. There are three different ways of obtaining the DFT.
    The most efficient way of obtaining the dft is method 3, but 1
    and 2 are included for educational purposes.
    X is a vector that contains the DFT coefficients and f is a
    vector that contains the real frequencies determined by sampling
    frequency fs. f are the frequencies starting at f=0 and X are the
    corresponding frequency components. f_c is a vector containing the
    frequencies such that f_c=0 is at the center, and X_c contains the
    frequency components corresponding to f_c. In essence, f_c and X_c
    are the centered counterparts of f and X. Frequency 0 is always
    present.
    If a parameter K is given, solve1,solve2, solve3 computes the DFT with
    only K coefficients. Recall that this periodizes signal x with period K.
    If the length of the signal x is less than K, then the signal will be
    padded with zeros. If the length of the signal x is greater than K,
    then there will be aliasing occured from periodizing singal x with
    period K. f_c and X_c are the centered counterparts of f and X
    """
    def __init__(self, x, fs, K=None):
        """
        :param x: Input vector x contains the discrete signal
        :param fs: Input integer fs contains the sample frequency
        :param K: Input positive integer that determines the number of coeffients
        used to calculate the DFT. If K is not provided, K=length(x).
        """
    # START: SANITY CHECK OF INPUTS.
        if (type(fs) != int) or (fs<=0):
            raise NameError('The frequency fs should be a positive integer.')
        if not isinstance(x, np. ndarray):
            raise NameError('The input signal x must be a numpy array.')
        if isinstance(x, np. ndarray):
            if x.ndim!=1:
                raise NameError('The input signal x must be a numpy vector array.')
        self.x=x
        self.fs=fs
        self.N=len(x)
        if K == None:
            K = len(self.x)
        # START: SANITY CHECK OF INPUTS.
        if (type(K) != int) or (K <= 0) or (K < 0):
            raise NameError('K should be a positive integer.')
        self.K=K
        self.f=np.arange(self.K)*self.fs/self.K # (0:K-1) just creates a vector from 0 to K by steps of 1.
        self.f_c=np.arange(-np.ceil(K/2)+1,np.floor(self.K/2)+1)*self.fs/self.K
        # This accounts for the frequencies
        # centered at zero. I want to be guaranteed that k=0 is always a
        # possible k. Then, I also have to account for both even and odd choices
        # of K, and that's why the floor() function appears to round down the
        # numbers.
    def changeK(self,K):
        """
        :param K: Input positive integer that determines the number of coeffients
        used to calculate the DFT. This function changes the attribute K of the class.
        """
        if (type(K) != int) or (K <= 0) or (K <  0):
            raise NameError('K should be a positive integer.')
        old_K=self.K
        self.K=K
        self.f=np.arange(self.K)*self.fs/self.K # (0:K-1) just creates a vector from 0 to K by steps of 1.
        self.f_c=np.arange(-np.ceil(K/2)+1,np.floor(self.K/2)+1)*self.fs/self.K
        # This accounts for the frequencies
        # centered at zero. I want to be guaranteed that k=0 is always a
        # possible k. Then, I also have to account for both even and odd choices
        # of K, and that's why the floor() function appears to round down the
        # numbers.
        print('The value of K was succefully change from %d to %d'%(old_K,self.K))
        pass

    def solve1(self):
        """
        \\\\\ METHOD 1: For loops
        By definition of DFT (eq. 1, lab 2) we have that
        X(k) = 1/sqrt(N) * sum_{n=0 to N-1} x(n) exp(-j 2 pi k n / N)
        where n is the discrete time index, k is the discrete time frequency and N is
        the length of the time signal x. Observe that this gives the DFT coefficient
        for a single coefficient k. We are trying to look for all coefficients
        k=0,1,...,K.
        This means that, for each value of k=0,1,...,K we will need to compute N
        multiplications x(n)*exp(-j 2 pi k n / N), for n=0,1,...,N-1, and sum the
        result.
        First thing we need is to create the variable X that we will output. This will
        be a vector of length K.
        """

        X=np.zeros(self.K,dtype=np.complex_)
        for k in range (self.K) :# For each time index k=0,1,...,K;
            for n in range (self.N):  # For each frequency n=0,1,...,N-1:
                X[k]=X[k]+1/np.sqrt(self.N)*self.x[n]*np.exp(-1j*2*cmath.pi*k*n/self.K)

        # Obs: in the case we have K different from N, then the
   		# signal will be periodized with period K. That is why the
        # exponential is divided by K instead of N.
        X_c=np.roll(X,np.int(np.ceil(self.K/2)-1)) # Circularly shift X to get it centered in f_c==0
        return [self.f,X,self.f_c,X_c]

    def solve2(self):
        """
        \\\\ METHOD 2: Matrix form
        Using for loops in python is rather expensive. Numpy is already optimized to
        work with vectors and matrices, so it's a good idea to take advantage of this,
        together with the built-in elementwise operators.
        Observe that we want a vector X=[X(0) X(1) ... X(K-1)]'=[X(k), k=0,...,K-1]'
        and that each element of the vector is computed as before:
        X(k) = 1/sqrt(N) * sum_{n=0 to N-1} x(n) exp(-j 2 pi k n / N)
        But, as mentioned in (eq. 2, lab. 2) this is nothing more than the inner
        product of x=[x(0) ... x(N-1)]'=[x(n), n=0,...,N-1]' with the complex
        exponential of frequency K and length N ekN=[ekN(0) ... ekN(N-1)]'=[ekN(n),
        n=0,...,N-1]', so that X(0)=<x,e0N>=e0N'*x, X(1)=<x,e1N>=e1N*x', ...,
        X(K-1)=<x,e(K-1)N>=e(K-1)N'*x. So we see that we have a bunch of vectors
        {ekN, k=0,...,K-1} that always multiply the same vector x. We can achieve
        this by creating a matrix where each of the ekN is a row, and then get
        the full vector X by multiplying this matrix by x. We will denote the
        matrix as WKN for future reference:
        WKN=[e0N'; e1N'; ... ; e(K-1)N'];
        Finally, observe that as we move from row to row, k grows from 0 to K-1.
        And as we move from column to column, it is n that grows from 0 to N-1.
        Also, observe that ekN is a function of (k,n) given by exp(-j 2 pi k n /K).
        So if we create a matrices of indices (k,n) then we can directly apply
        the exponential by making use of the elementwise nature of this operation
        in Numpy.
        """
        matrix_k=np.transpose(np.tile(np.arange(self.K),(self.N,1)))
        matrix_n=np.tile(np.transpose(np.arange(self.N)),(self.K,1))
        indices=np.multiply(matrix_k,matrix_n)
        WKN=1/np.sqrt(self.N)*np.exp(-1j*2*np.pi*indices/self.K)
        X=WKN@self.x

        X_c=np.roll(X,np.int(np.ceil(self.K/2)-1)) # Circularly shift X to get it centered in f_c==0
        return [self.f,X,self.f_c,X_c]

    def solve3(self):
        """
        \\\\\ METHOD 3: Built-in fft() function
        Even though the matrix form is fast, it is still not fast enough for large
        signals x. For that, it is better to use the built in fft() function which is
        the optimal way to compute a dft. Besides, it is really easy to code.
        """
        X=np.fft.fft(self.x,self.K)/np.sqrt(self.N);
        # \\\\\ CENTER FFT.
        X_c=np.roll(X,np.int(np.ceil(self.K/2-1))) # Circularly shift X to get it centered in f_c==0
        return [self.f,X,self.f_c,X_c]

###############################################################################
############################ Q U E S T I O N 1.1 ##############################
###############################################################################
class gaussian_pulse(object):
    """
    Generates a gaussian pulse signal
    Arguments:
        mu, sigma: mean and variance of the Gaussian pulse
        T: duration of the signal (int)
        fs: sampling frequncy (int)
    """

    def __init__(self, mu, sigma, T, fs):
        self.N = np.int(np.floor( 2 * T * fs))

        # Create the active part
        # self.sig = signal.gaussian(self.N, std=sigma)
        self.t = np.arange(-T, T, 1 / fs)
        self.sig = np.exp(-(self.t-mu)**2 / (2 * sigma**2))

        # Create the time array
        self.t = np.arange(-T, T, 1 / fs)
        
def gau_ft(mu, sigma, f):
    
    gauss_ft = np.sqrt(2 * math.pi) * sigma * np.exp(- 2* (math.pi**2) *(f**2) *(sigma**2) + 1j * 2* math.pi * f * mu)
        
    return gauss_ft  

def q_11(mu, sigma_list, T, fs):

    for sigma in sigma_list:
        # Generate Gaussian signals
        gau_pulse = gaussian_pulse(mu, sigma, T, fs)
        # Calculate the dft
        gaupulse_dft = dft(gau_pulse.sig, fs)
        [freqs, X, f_c, X_c] = gaupulse_dft.solve3()

        # Calculate the ft

        # gau_ft = gaussian_pulse(mu, 1 / (2 * math.pi * sigma), T, fs) 
        # gau_ft_sig = gau_ft.sig
        # gau_ft_sig = gau_ft.sig  * np.sqrt(2*math.pi) * sigma * fs/ np.sqrt( np.int(np.floor(T * fs)) ) 
    
        gau_ft_sig = gau_ft(mu, sigma, f_c) * fs/ np.sqrt( np.int(np.floor(2 * T * fs)) )

        # Plot
        fig, axs = plt.subplots(3)
        axs[0].grid()
        axs[1].grid()
        axs[2].grid()
        fig.suptitle('Gaussian Pulse of Sigma %3.2f ' % (sigma))
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
        axs[0].plot(gau_pulse.t, gau_pulse.sig)
        axs[0].set_xlabel('Time (s)')
        axs[0].set_xlim((-10,10))

        axs[0].set_ylabel('Signal')
        axs[1].plot(f_c, abs(X_c))
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylabel('DFT')
        axs[2].plot(f_c, abs(gau_ft_sig))
        axs[2].set_xlabel('Frequency (Hz)')
        # axs[2].set_xlim((-5,5))
        axs[2].set_ylabel('Scaled FT')
        plt.show()
        
if __name__ == '__main__':
    
    sigmalist = [1, 2, 4]
    duration_of_signal = 50
    sampling_frequency = 10
    q_11(2, sigmalist, duration_of_signal, sampling_frequency)