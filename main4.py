# ESE 224 --- Signal and Information Processing
#
# Standard libraries
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
#import cexp
import dft
import idft

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
        self.N = np.int(np.floor(T * fs))

        # Create the active part
        # self.sig = signal.gaussian(self.N, std=sigma)
        self.t = np.arange(-T, T, 1 / fs)
        self.sig = np.exp(-(self.t-mu)**2 / (2 * sigma**2))

        # Create the time array
        self.t = np.arange(-T, T, 1 / fs)
        
def gau_ft(mu, sigma, f):
    """
    :param mu, sigma: parameters of Gaussian signal
    :param f: the real frequency sequence
    :return: gauss_ft: the calculated Fourier transform of Gaussian signal
    """
    
    gauss_ft = np.sqrt(2 * math.pi) * sigma * np.exp(- 2* (math.pi**2) *(f**2) *(sigma**2) + 1j * 2* math.pi * f * mu)
        
    return gauss_ft    

class recordsound():
    """
    recordsound Record your voice for T time sampled at a frequency fs
    solve() generates a sampled signal of your voice and save it into a wav file
    """
    def __init__(self, T, fs):
        """
        :param T: the duration time
        :param fs: the sampling frequency
        """
        self.T = T
        self.fs = fs
                
    def solve(self):
        """
        :return: [x,N]
        :x: the triangular pulse vector x
        :N: The sample duration
        """
        print('start recording')
        voicerecording = sd.rec(int(self.T * self.fs), self.fs, 1)
        sd.wait()  # Wait until recording is finished
        print('end recording')
        write('myvoice.wav', self.fs, voicerecording)  # Save as WAV file 
        
        return voicerecording
    
    
    

def q_11(mu, sigma_list, T, fs):
    """
    :param mu, sigma_list: the parameters of Gaussian signal
    :param T: the time duration
    :param fs: the sampling frequency
    """

    for sigma in sigma_list:
        # Generate Gaussian signals
        gau_pulse = gaussian_pulse(mu, sigma, T, fs)
        
        # Calculate the dft
        gaupulse_dft = dft.dft(gau_pulse.sig, fs)
        [freqs, X, f_c, X_c] = gaupulse_dft.solve3()

        # Calculate the ft and scale with fs/sqrt(N)
        gau_ft_sig = gau_ft(mu, sigma, f_c) * fs/ np.sqrt( np.int(np.floor(T * fs))*2 )

        # Plot
        fig, axs = plt.subplots(3)
        axs[0].grid()
        axs[1].grid()
        axs[2].grid()
        fig.suptitle('Gaussian Pulse of Sigma %3.2fs ' % (sigma))
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
        axs[0].plot(gau_pulse.t, gau_pulse.sig)
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Signal')
        axs[1].plot(f_c, abs(X_c))
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylabel('DFT')
        axs[2].plot(f_c, gau_ft_sig)
        axs[2].set_xlabel('Frequency (Hz)')
        axs[2].set_ylabel('FT')
        plt.show()
        
        
def bandlimit(x, fs, fmax):
    """
    :param x: original signal
    :param fs: the sampling frequency
    :param fmax: the maximum frequency
    """    
    N = len(x)
    # calculate the dft 
    DFT = dft.dft(x, fs)
    [freqs, X, f_c, X_c] = DFT.solve_using_numpy_fft()
    # calculate the minimum and maximum index for the frequencies under the frequency threshold
    index_min = np.min( np.where(f_c >= -fmax)[0])
    index_max = np.max( np.where(f_c <= fmax)[0])
    # pad zeros to make the truncated frequency signals keep N duration
    X_band = np.concatenate(( np.zeros(index_min+1), X_c[index_min: index_max] ,  np.zeros( N - index_max-1)))
    # roll back the frequency signals to calculate ifft
    X_band_n = np.roll( X_band, np.int(np.ceil( N / 2 )))
      
    iDFT = idft.idft(X_band_n, fs, N)
    x_bandlim, Treal = iDFT.solve_ifft()
    
    return x_bandlim.real, Treal,  X_c, f_c, X_band

"""
QUESTION 2.1 Voice as a bandlimited signal. Record 3 seconds of the voice of one of group as members
at a sampling rate of 40kHz. Feel free to use the 2.1 your Voice a bandlimited signal
provided python class recordsound() to do this. Take the DFT of your
voice and observe that coefficients with frequencies f > 4kHz are close
to null. Set these coefficients to zero to create a bandlimited signal. Play
your voice back and observe that the removed frequencies don’t affect the
quality of your voice.
"""
        
def q_21(x, x_bl, Treal, X_c, f_c, X_band):
    """
    Question 2.1_create bandlimited signal, plot the original and limited signal, 
    save the bandlimited voice signal    
    """
    
    # Plot
    fig, axs = plt.subplots(2)
    axs[0].grid()
    axs[1].grid()
    fig.suptitle('original signal and spectrum' )
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
    axs[0].plot(Treal, x)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Signal')
    axs[1].plot(f_c, X_c)
    axs[1].set_xlabel('Frequency')
    axs[1].set_ylabel('DFT')

    plt.show() 
    
    fig, axs = plt.subplots(2)
    axs[0].grid()
    axs[1].grid()
    fig.suptitle('bandlimited spectrum and signal' )
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
    axs[1].plot(Treal, x_bl)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Signal')
    axs[0].plot(f_c, X_band)
    axs[0].set_xlabel('Frequency')
    axs[0].set_ylabel('DFT')
    plt.show() 
    
    write('bandlimited_myvoice.wav', fs, x_bl.astype(np.float32))

    
'''
QUESTION 2.2 Voice modulation. Take the bandlimited signal you created in Part 2.1
and modulate it with center frequency g1 = 5kHz.'''
def q_22(x, g, fs):
    """
    Question 2.2_exponential modulation
    
    """
    N = len(x)
    n = np.arange(N)
    discrete_exponential = np.exp(2 * math.pi*1j* g * n / fs)
    x_mod = x * discrete_exponential
    
    return x_mod
    
def cos_modu(x, g, fs):
    """
    Question 2.3_cosine modulation
    
    """    
    N = len(x)
    n = np.arange(N)
    discrete_cosine = np.cos(2 * math.pi * g * n / fs)
    x_mod = x * discrete_cosine
    
    return x_mod
    
'''
QUESTION 2.3 Modulation with a cosine. The problem with modulating with a
complex exponential as we did in Part 2.2 is that complex exponentials
are signals with imaginary parts that, therefore, can’t be generated in a
real system. In a real system we have to modulate using a cosine or a
sine. Redefine then the modulated signal as
xg(t) = cos(2πgt)x(t), (12)
and let Xg = F(xg) be the respective Fourier transform. Write down an
expression for Xg in terms of X (it might be useful to recall how to write
the cos(x) function as a combination of complex exponentials). Take the
bandlimited signal you created in Part 2.1 and modulate it with a cosine
with frequency g1 = 5kHz. Verify that expression you derived is correct.
'''
def q_23(x_band, x_mod, fs, Treal):
    """
    Question 2.3_cosine modulation, plot the signals in time and frequency domain
    
    """       
    
    DFT = dft.dft(x_band, fs)
    [freqs, X, f_c, X_c] = DFT.solve_using_numpy_fft()
    
    DFT_mod = dft.dft(x_mod, fs)
    [freqs, X, f_mod, X_mod] = DFT_mod.solve_using_numpy_fft()
    
    
    fig, axs = plt.subplots(2)
    axs[0].grid()
    axs[1].grid()
    fig.suptitle('original signal and modulated signal' )
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
    axs[0].plot(Treal, x_band)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Signal')
    axs[1].plot(Treal, x_mod)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Signal')
    plt.show() 
    
    
    fig, axs = plt.subplots(2)
    axs[0].grid()
    axs[1].grid()
    fig.suptitle('original signal and modulated signal spectrum' )
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
    axs[0].plot(f_c, X_c)
    axs[0].set_xlabel('Frequency')
    axs[0].set_ylabel('DFT')
    axs[1].plot(f_mod, X_mod)
    axs[1].set_xlabel('Frequency')
    axs[1].set_ylabel('DFT')
    plt.show() 

'''
QUESTION 2.5 Recover individual voices. Explain how to recover your voice and
the voice of your partner from the mixed signal z. Implement the recovery
and play back the individual voice pieces.
'''
    
def q_25(z, g, fs, fmax):
    """
    Question 2.5_demodulation
    
    """      
    N= len(z)
    
    DFT = dft.dft(z, fs)
    [freqs, X, f_Zc, Z_c] = DFT.solve_using_numpy_fft()
    
    y_demod_inter = cos_modu(z, g, fs)
    DFT = dft.dft(y_demod_inter, fs)
    [freqs, Y, f_c, Y_demod_inter] = DFT.solve_using_numpy_fft()
    
    index_min = np.min( np.where(f_c >= -fmax)[0])
    index_max = np.max( np.where(f_c <= fmax)[0])
    Y_demod = np.concatenate(( np.zeros(index_min+1), Y_demod_inter[index_min: index_max]*2 ,  np.zeros( N - index_max-1)))
    Y_demod_n = np.roll( Y_demod, np.int(np.ceil( N / 2 )) )
    iDFT = idft.idft(Y_demod_n, fs, N)
    y_demod, Treal = iDFT.solve_ifft()
    
    fig, axs = plt.subplots(2)
    axs[0].grid()
    axs[1].grid()
    fig.suptitle('demodulated signal spectrum' )
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    axs[0].plot(f_c, Y_demod_inter)
    axs[0].set_xlabel('Frequency')
    axs[0].set_ylabel('DFT')
    axs[1].plot(f_c, Y_demod)
    axs[1].set_xlabel('Frequency')
    axs[1].set_ylabel('DFT')
    plt.show() 
    
    
    fig, axs = plt.subplots(2)
    axs[0].grid()
    axs[1].grid()
    fig.suptitle('original signal and modulated signal spectrum' )
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    axs[0].plot(f_Zc, Z_c)
    axs[0].set_xlabel('Frequency')
    axs[0].set_ylabel('DFT')
    axs[1].plot(f_c, Y_demod)
    axs[1].set_xlabel('Frequency')
    axs[1].set_ylabel('DFT')
    plt.show() 
    
    fig, axs = plt.subplots(2)
    axs[0].grid()
    axs[1].grid()
    fig.suptitle('mixed signal and demodulated signal' )
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    axs[0].plot(Treal, z)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Signal')
    axs[1].plot(Treal, y_demod.real)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Signal')
    plt.show() 
    
    write('demodulated_myvoice'+str(g)+'.wav', fs, y_demod.real.astype(np.float32))
    
    return y_demod.real

    
    
    
    




if __name__ == '__main__':


    """
    2.1 Voice as a bandlimited signal. Record 3 seconds of the voice of one
    of your group members at a sampling rate of 40kHz. Feel free to use the
    provided python class recordsound() to do this"""

    # sigmalist = [1, 2, 4]
    # duration_of_signal = 50
    # sampling_frequency = 10
    # q_11(0, sigmalist, duration_of_signal, sampling_frequency)
    
    fs = 40000
    T = 3
    fmax = 4000
    g1 = 5000
    g2 = 13000



    myvoice = recordsound(T, fs)      
    x1 = myvoice.solve().reshape(T * fs)
    
    myvoice = recordsound(T, fs)      
    x2 = myvoice.solve().reshape(T * fs)
    # fs, x = read("myvoice.wav")



    N = len(x1)
    print(N)
    x_band, Treal, X_c, f_c, X_band = bandlimit(x1, fs, fmax)
    q_21(x1, x_band, Treal, X_c, f_c, X_band)
    
    x_mod = cos_modu(x_band, g1, fs)
    x_mod = q_22(x_band, g1, fs)
    q_23(x_band, x_mod, fs, Treal)
    
    
    # fs, x1 = read("myvoice1.wav")
    x_band1, Treal, X_c, f_c, X_band = bandlimit(x1, fs, fmax)
    x_mod1 = cos_modu(x_band1, g1, fs)
    q_23(x_band1, x_mod1, fs, Treal)

    # fs, x2 = read("myvoice.wav")
    x_band2, Treal,X_c, f_c, X_band = bandlimit(x2, fs, fmax)
    x_mod2 = cos_modu(x_band2, g2, fs)
    q_23(x_band2, x_mod2, fs, Treal)
    
    z = x_mod1 + x_mod2
    
    fig, axs = plt.subplots(2)
    axs[0].grid()
    axs[1].grid()
    fig.suptitle('two original signals' )
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    axs[0].plot(Treal, x1)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('x(t)')
    axs[1].plot(Treal, x2)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('y(t)')
    plt.show() 
    
    x_recons1 = q_25(z, g1, fs, fmax)
    x_recons2 = q_25(z, g2, fs, fmax)
    
    fig, axs = plt.subplots(2)
    axs[0].grid()
    axs[1].grid()
    fig.suptitle('demodulated signals' )
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    axs[0].plot(Treal, x_recons1)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('x(t)')
    axs[1].plot(Treal, x_recons2)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('y(t)')
    plt.show() 

    
    
    
    
    
    
    
    
    
    

