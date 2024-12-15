import numpy as np
import pywt
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, sosfilt, find_peaks, hilbert
from scipy.signal import argrelextrema

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

    def smooth_signal(self, signal, sigma):
        """
        Wygładzanie sygnału za pomocą filtru Gaussa.
        """
        return gaussian_filter(signal, sigma=sigma)

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
        """
        Detekcja tonów S1 i S2 oraz obliczenie odstępów między nimi.

        Args:
            signal (numpy.ndarray): Obwiednia sygnału PCG (wygładzona).

        Returns:
            tuple: Listy detekcji (S1, S2) oraz lista odstępów między S1 i S2 (w ms).
        """
        # Normalizacja sygnału
        signal = signal / np.max(np.abs(signal))

        # Wygładzenie sygnału Hilberta
        smoothed_signal = self.smooth_signal(signal, sigma=5)

        # Detekcja lokalnych maksimów
        peaks, _ = find_peaks(smoothed_signal, height=0.1, distance=150)

        # Grupowanie pików w pary S1-S2
        s1_peaks = []
        s2_peaks = []
        intervals = []

        for i in range(len(peaks) - 1):
            current_peak = peaks[i]
            next_peak = peaks[i + 1]
            interval = (next_peak - current_peak) / self.sampling_rate * 1000  # odstęp w ms

            if 50 <= interval <= 200:
                s1_peaks.append(current_peak)
                s2_peaks.append(next_peak)
                intervals.append(interval)

        # Debugowanie
        print(f"Debug: Detected Peaks: {peaks}")
        print(f"Debug: S1 Peaks: {s1_peaks}")
        print(f"Debug: S2 Peaks: {s2_peaks}")
        print(f"Debug: Intervals: {intervals} ms")

        return s1_peaks, s2_peaks, intervals