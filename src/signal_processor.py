import numpy as np
import pywt
from scipy.ndimage import gaussian_filter
from wfdb.processing import find_local_peaks
from scipy.signal import butter, filtfilt, sosfilt


class SignalProcessor:
    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate

    def filter_signal(self, signal):
        """
        Filtracja sygnału w zakresie 1–30 Hz z opcjonalnym wygładzaniem.
        """
        nyquist = 0.5 * self.sampling_rate
        low = 1 / nyquist
        high = 30 / nyquist

        sos = butter(4, [low, high], btype='band', output='sos')
        filtered_signal = sosfilt(sos, signal)

        smoothed_signal = gaussian_filter(filtered_signal, sigma=20)
        return smoothed_signal

    def wavelet_transform(self, signal, wavelet='db4', level=2):
        """
        Transformacja falkowa i rekonstrukcja sygnału.
        """
        coeffs = pywt.wavedec(signal, wavelet, level=level)

        for i in range(1, 3):
            coeffs[i] = np.zeros_like(coeffs[i])
        reconstructed_signal = pywt.waverec(coeffs, wavelet)
        reconstructed_signal = np.abs(reconstructed_signal) / np.max(np.abs(reconstructed_signal))
        return reconstructed_signal

    def detect_r_peaks(self, signal, min_distance=500):
        """
        Detekcja pików R z dynamicznym progiem.
        """
        mean_signal = np.mean(signal)
        std_signal = np.std(signal)
        dynamic_threshold = mean_signal + 1.5 * std_signal

        r_peaks = find_local_peaks(signal, radius=min_distance)
        r_peaks_filtered = [idx for idx in r_peaks if signal[idx] > dynamic_threshold]
        return np.array(r_peaks_filtered)

    def calculate_rr_intervals(self, r_peak_indices):
        """
        Obliczanie odstępów R-R w milisekundach.
        """
        rr_intervals = np.diff(r_peak_indices) * 1000 / self.sampling_rate
        return rr_intervals