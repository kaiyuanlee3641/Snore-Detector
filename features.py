# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy.signal import lfilter
from scipy.signal import butter, freqz, filtfilt, firwin, iirnotch, lfilter, find_peaks
from audiolazy.lazy_lpc import lpc
from python_speech_features import mfcc
denominator_val = -1
import warnings
warnings.filterwarnings("ignore")
class FeatureExtractor():
    def __init__(self, debug=True):
        self.debug = debug
    def _compute_mean_features(self, window):
        """
        Computes the mean x, y and z acceleration over the given window. 
        """
        return np.mean(window, axis=0)

    def _compute_peak_features(self, window):
        # Filter requirements.
        order = 4
        fs = 50.0
        cutoff = 5
        # Create the filter.
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        # Frequency response graph
        w, h = freqz(b, a, worN=8000)
        filtered_signal = []
        if len(window) > 15:
            filtered_signal = filtfilt(b, a, window)
        total_steps = 0 
        differentiate=np.diff(filtered_signal)
        signs = np.sign(differentiate)
        count = 0
        last = 0
        interval = int(len(filtered_signal) / 2)
        mean = np.mean(filtered_signal) * 1.2
        val = []
        for i in range(0, len(signs)):
            if last == 1.0 and signs[i] == -1.0 and filtered_signal[i] >= mean and i > interval:
                if filtered_signal[i] == np.amax(filtered_signal[i-interval:i+interval]):
                    total_steps += 1
                    # stepvals[i] = filtered_signal[i]
            last = signs[i]
        # print(len(peaks))
        return total_steps

    def _compute_formants(self, audio_buffer):
        """
        Computes the frequencies of formants of the window of audio data, along with their bandwidths.
    
        A formant is a frequency band over which there is a concentration of energy. 
        They correspond to tones produced by the vocal tract and are therefore often 
        used to characterize vowels, which have distinct frequencies. In the task of 
        speaker identification, it can be used to characterize a person's speech 
        patterns.
        
        This implementation is based on the Matlab tutorial on Estimating Formants using 
        LPC (Linear Predictive Coding) Coefficients: 
        http://www.mathworks.com/help/signal/ug/formant-estimation-with-lpc-coefficients.html.
        
        """
        
        # Get Hamming window. More on window functions can be found at https://en.wikipedia.org/wiki/Window_function
        # The idea of the Hamming window is to smooth out discontinuities at the edges of the window. 
        # Simply multiply to apply the window.
        N = len(audio_buffer)
        Fs = 8000 # sampling frequency
        hamming_window = np.hamming(N)
        window = audio_buffer * hamming_window
    
        # Apply a pre-emphasis filter; this amplifies high-frequency components and attenuates low-frequency components.
        # The purpose in voice processing is to remove noise.
        filtered_buffer = lfilter([1], [1., 0.63], window)
        
        # Speech can be broken down into (1) The raw sound emitted by the larynx and (2) Filtering that occurs when transmitted from the larynx, defined by, for instance, mouth shape and tongue position.
        # The larynx emits a periodic function defined by its amplitude and frequency.
        # The transmission is more complex to model but is in the form 1/(1-sum(a_k * z^-k)), where the coefficients 
        # a_k sufficiently encode the function (because we know it's of that form).
        # Linear Predictive Coding is a method for estimating these coefficients given a pre-filtered audio signal.
        # These value are called the roots, because the are the points at which the difference 
        # from the actual signal and the reconstructed signal (using that transmission function) is closest to 0.
        # See http://dsp.stackexchange.com/questions/2482/speech-compression-in-lpc-how-does-the-linear-predictive-filter-work-on-a-gene.
    
        # Get the roots using linear predictive coding.
        # As a rule of thumb, the order of the LPC should be 2 more than the sampling frequency (in kHz).
        ncoeff = 2 + Fs / 1000
        A = lpc(filtered_buffer, int(ncoeff))
        # print(list(A)[0])
        if len(list(A)[0]) != 11:
            "       list(A)[0] length != 11"
            return 0
        A = np.array([list(A)[0][i] for i in range(0,10)])

        roots = np.roots(A)
        roots = [r for r in roots if np.imag(r) >= 0]
    
        # Get angles from the roots. Each root represents a complex number. The angle in the 
        # complex coordinate system (where x is the real part and y is the imaginary part) 
        # corresponds to the "frequency" of the formant (in rad/s, however, so we need to convert them).
        # Note it really is a frequency band, not a single frequency, but this is a simplification that is acceptable.
        angz = np.arctan2(np.imag(roots), np.real(roots))
    
        # Convert the angular frequencies from rad/sample to Hz; then calculate the 
        # bandwidths of the formants. The distance of the roots from the unit circle 
        # gives the bandwidths of the formants (*Extra credit* if you can explain this!).
        unsorted_freqs = angz * (Fs / (2 * math.pi))
        
        # Let's sort the frequencies so that when we later compare them, we don't overestimate
        # the difference due to ordering choices.
        freqs = sorted(unsorted_freqs)
        
        # also get the indices so that we can get the bandwidths in the same order
        indices = np.argsort(unsorted_freqs)
        sorted_roots = np.asarray(roots)[indices]
        
        #compute the bandwidths of each formant
        bandwidths = -1/2. * (Fs/(2*math.pi))*np.log(np.abs(sorted_roots))

        if self.debug:
            print("Identified {} formants.".format(len(freqs)))

        return (freqs, bandwidths)
        
    def _compute_fft_features(self, window):
        fourier = np.fft.rfft(window).astype(float)
        freq = np.fft.fftfreq(len(window))
        return freq[np.argmax(np.abs(fourier))]
        
    def _compute_entropy_features(self, window):
        (hist, b) = np.histogram(window)
        # print(hist)
        return np.sum(hist*np.where(hist != 0, np.log(hist), 0))

    def _compute_formant_features(self, window):
        """
        Computes the distribution of the frequencies of formants over the given window. 
        Call _compute_formants to get the formats; it will return (frequencies,bandwidths). 
        You should compute the distribution of the frequencies in fixed bins.
        This will give you a feature vector of length len(bins).
        
        """
        
        """ADD your code here"""
        #
        #
        #
        #
        (frequencies, bandwidths) = self._compute_formants(window)
        (hist, bin_edges) = np.histogram(frequencies, range=(0, 5500))

        return hist #Dummy value

    
    def _compute_mfcc(self, window):
        """
        Computes the MFCCs of the audio data. MFCCs are not computed over 
        the entire 1-second window but instead over frames of between 20-40 ms. 
        This is large enough to capture the power spectrum of the audio 
        but small enough to be informative, e.g. capture vowels.
        
        The number of frames depends on the frame size and step size.
        By default, we choose the frame size to be 25ms and frames to overlap 
        by 50% (i.e. step size is half the frame size, so 12.5 ms). Then the 
        number of frames will be the number of samples (8000) divided by the 
        step size (12.5) minus one because the frame size is too large for 
        the last frame. Therefore, we expect to get 79 frames using the 
        default parameters.
        
        The return value should be an array of size n_frames X n_coeffs, where
        n_coeff=13 by default.
        
        See http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/ for implementation details.
        """
        mfccs = mfcc(window, 8000, winstep=.0125)
        if self.debug:
            print("{} MFCCs were computed over {} frames.".format(mfccs.shape[1], mfccs.shape[0]))
        return mfccs


    def _compute_delta_coefficients(self, window, N=2):
        """
        Computes the delta of the MFCC coefficients. See the equation in the assignment details.
        
        This method should return a feature vector of size n_frames - 2 * n, 
        or of size n_frames, depending on how you handle edge cases.
        The running-time is O(n_frames * n), so we generally want relatively small n. Default is n=2.
        
        See section "Deltas and Delta-Deltas" at http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/.
        
        """
        """ADD your code here"""
        #
        #
        #
        #

        mfccs = self._compute_mfcc(window)
        d = []
        denominator = (2*sum(i**2 for i in range(1, N)))
        n_frames = 79
        for t in range(N, n_frames - N):
            d.append(sum(i*(mfccs[t+i,:]-mfccs[t-i,:]) for i in range(1, N))/denominator)

        # print(len(d))

        a = np.array(d).flatten()

        # print(len(a))

        return a
        
    def extract_features(self, window, debug=True):
        """
        Here is where you will extract your features from the data in 
        the given window.
        
        Make sure that x is a vector of length d matrix, where d is the number of features.
        
        """
        
        x = []
        # x.append(self._compute_formant_features(window))
        # x.append(self._compute_delta_coefficients(window))

        # x = np.append(x, self._compute_formant_features(window))
        # x = np.append(x, self._compute_delta_coefficients(window))

        x = np.append(x, self._compute_mean_features(window))
        x = np.append(x, self._compute_peak_features(window))
        x = np.append(x, self._compute_fft_features(window))
        x = np.append(x, self._compute_entropy_features(window))

        # print(len(x))
        
        return np.array(x)


if __name__ == "__main__":
    pass