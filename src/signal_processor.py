import numpy as np
import pywt
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, sosfilt, find_peaks, hilbert
from scipy.signal import argrelextrema


def smooth_signal(signal, sigma):
    """
    Wygładzanie sygnału za pomocą filtru Gaussa.
    """
    return gaussian_filter(signal, sigma=sigma)


class SignalProcessor:
    """
    Klasa do przetwarzania sygnałów EKG i PCG z zaawansowaną filtracją i detekcją pików.
    """
    DEFAULT_ECG_FILTER_LOW = 1  # Dolna częstotliwość filtra dla EKG (Hz)
    DEFAULT_ECG_FILTER_HIGH = 30  # Górna częstotliwość filtra dla EKG (Hz)
    DEFAULT_PCG_FILTER_LOW = 20  # Dolna częstotliwość filtra dla PCG (Hz)
    DEFAULT_PCG_FILTER_HIGH = 400  # Górna częstotliwość filtra dla PCG (Hz)
    DEFAULT_WAVELET = 'db4'  # Domyślna nazwa falki
    DEFAULT_WAVELET_LEVEL = 2  # Domyślny poziom dekompozycji
    GAUSSIAN_SIGMA_ECG = 30  # Parametr sigma dla wygładzania EKG
    GAUSSIAN_SIGMA_PCG = 5  # Parametr sigma dla wygładzania PCG
    R_PEAK_THRESHOLD_FACTOR = 2.0  # Współczynnik progu dla wykrywania pików R
    S1_S2_THRESHOLD_FACTOR = 2.5  # Współczynnik progu dla wykrywania tonów S1/S2
    MIN_DISTANCE_R_PEAKS = 800  # Minimalna odległość między pikami R (w próbkach)
    MIN_DISTANCE_S1_S2 = 200  # Minimalna odległość między tonami S1/S2 (w próbkach)

    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate

    def filter_signal(self, signal, low_cutoff, high_cutoff):
        """
        Filtracja sygnału w zadanym zakresie częstotliwości.
        """
        nyquist = 0.5 * self.sampling_rate
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        sos = butter(4, [low, high], btype='band', output='sos')
        filtered_signal = sosfilt(sos, signal)
        return filtered_signal

    def wavelet_transform(self, signal):
        """
        Transformacja falkowa z uwypukleniem cech charakterystycznych sygnału.
        """
        coeffs = pywt.wavedec(signal, self.DEFAULT_WAVELET, level=self.DEFAULT_WAVELET_LEVEL)
        for i in range(1, len(coeffs)):
            coeffs[i] = np.zeros_like(coeffs[i])  # Usunięcie szczegółowych współczynników
        reconstructed_signal = pywt.waverec(coeffs, self.DEFAULT_WAVELET)
        reconstructed_signal = np.abs(reconstructed_signal)
        return reconstructed_signal / np.max(reconstructed_signal)

    def detect_r_peaks(self, signal):
        """
        Detekcja pików R na podstawie dynamicznego progu i minimalnej odległości.
        """
        mean_signal = np.mean(signal)
        std_signal = np.std(signal)
        dynamic_threshold = mean_signal + self.R_PEAK_THRESHOLD_FACTOR * std_signal

        peaks, _ = find_peaks(signal, height=dynamic_threshold, distance=self.MIN_DISTANCE_R_PEAKS)
        return peaks

    def hilbert_envelope(self, signal):
        """
        Obwiednia sygnału PCG wyznaczona za pomocą transformacji Hilberta.
        """
        analytic_signal = hilbert(signal)
        envelope = np.abs(analytic_signal)
        return envelope

    def detect_s1_s2(self, signal):
        signal = signal / np.max(np.abs(signal))  # Normalizacja
        smoothed_signal = smooth_signal(signal, sigma=1)

        # Wstępna detekcja peaków
        threshold = 0.18 * np.max(smoothed_signal)
        candidate_peaks, _ = find_peaks(
            smoothed_signal,
            height=threshold,
            distance=int(0.2 * self.sampling_rate)  # minimum 200ms odstępu
        )

        s1_peaks = []
        s2_peaks = []

        i = 0
        while i < len(candidate_peaks):
            s1 = candidate_peaks[i]
            s1_peaks.append(s1)


            s2_window_start = s1 + int(0.1 * self.sampling_rate)  # 100 ms
            s2_window_end = s1 + int(0.5 * self.sampling_rate)  # 500 ms

            s2_candidates = [p for p in candidate_peaks if s2_window_start <= p <= s2_window_end]

            if s2_candidates:

                s2 = max(s2_candidates, key=lambda idx: smoothed_signal[idx])
                s2_peaks.append(s2)
                i = np.where(candidate_peaks == s2)[0][0] + 1
            else:
                i += 1

        intervals = [(s2 - s1) / self.sampling_rate * 1000 for s1, s2 in zip(s1_peaks, s2_peaks)]

        return s1_peaks, s2_peaks, intervals








