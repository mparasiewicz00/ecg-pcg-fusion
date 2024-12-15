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

    def detect_s1_s2(self, signal, r_peaks=None):
        """
        Detekcja tonów S1 i S2 z dopasowaniem do pików R z sygnału EKG.

        Args:
            signal (numpy.ndarray): Obwiednia sygnału PCG.
            r_peaks (list or None): Lista pików R z sygnału EKG.

        Returns:
            tuple: Listy detekcji (S1, S2) oraz lista odstępów między S1 i S2 (w ms).
        """
        # Normalizacja i wygładzenie sygnału
        signal = signal / np.max(np.abs(signal))
        smoothed_signal = self.smooth_signal(signal, sigma=1)

        s1_peaks = []
        s2_peaks = []
        intervals = []

        # Dopasowanie do pików R (jeśli istnieją)
        if r_peaks is not None and len(r_peaks) > 0:
            for r_peak in r_peaks:
                # Szukanie S1: 50 próbek przed R i 100 próbek po R
                s1_window_start = max(0, r_peak - 50)
                s1_window_end = r_peak + 100
                s1_window = smoothed_signal[s1_window_start:s1_window_end]

                if len(s1_window) > 0:
                    s1_local_peak = np.argmax(s1_window) + s1_window_start
                    s1_peaks.append(s1_local_peak)

                    # Szukanie S2: 150-700 próbek po S1
                    s2_window_start = s1_local_peak + 150
                    s2_window_end = s1_local_peak + 700
                    s2_window = smoothed_signal[s2_window_start:s2_window_end]

                    if len(s2_window) > 0:
                        s2_local_peak = np.argmax(s2_window) + s2_window_start

                        # Sprawdzenie minimalnego odstępu czasowego między S1 a S2
                        if (s2_local_peak - s1_local_peak) >= 200:
                            s2_peaks.append(s2_local_peak)
                            interval = (s2_local_peak - s1_local_peak) / self.sampling_rate * 1000
                            intervals.append(interval)

            # Usunięcie duplikatów lub fałszywych detekcji S2
            s2_peaks = [peak for peak in s2_peaks if not any(abs(peak - s) < 50 for s in s1_peaks)]

        print(f"Debug: Detected S1 Peaks: {s1_peaks}")
        print(f"Debug: Detected S2 Peaks: {s2_peaks}")
        print(f"Debug: Intervals (ms): {intervals}")

        return s1_peaks, s2_peaks, intervals








