import numpy as np
import pywt
from scipy.ndimage import gaussian_filter
from wfdb.processing import find_local_peaks
from scipy.signal import butter, sosfilt


class SignalProcessor:
    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate

    """
    ECG Processing
    """

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

    """
    PCG Processing
    """

    def filter_pcg_signal(self, signal):
        """
        Filtracja sygnału PCG w zakresie 20–500 Hz.
        """
        nyquist = 0.5 * self.sampling_rate
        low = 20 / nyquist
        high = 500 / nyquist

        sos = butter(4, [low, high], btype='band', output='sos')
        filtered_signal = sosfilt(sos, signal)

        # Opcjonalne wygładzanie Gaussian Filter
        smoothed_signal = gaussian_filter(filtered_signal, sigma=5)
        return smoothed_signal

    def detect_s1_s2_peaks(self, signal, ecg_r_peaks, fs_pcg, search_window=0.1, dynamic_threshold_factor=2.0,
                           min_distance=0.06):
        """
        Detekcja tonów S1/S2 w sygnale PCG, zsynchronizowana z pikami R z EKG.

        Args:
            signal (np.ndarray): Sygnał PCG.
            ecg_r_peaks (np.ndarray): Indeksy pików R w EKG.
            fs_pcg (float): Częstotliwość próbkowania PCG.
            search_window (float): Czas (w sekundach) naokoło każdego piku R, w którym szukamy tonów S1/S2.
            dynamic_threshold_factor (float): Współczynnik do ustalenia dynamicznego progu.
            min_distance (float): Minimalna odległość między wykrytymi tonami S1/S2 w sekundach.

        Returns:
            np.ndarray: Indeksy wykrytych tonów S1/S2.
        """
        s1_s2_peaks = []

        # Dynamiczny próg na podstawie sygnału
        mean_signal = np.mean(signal)
        std_signal = np.std(signal)
        dynamic_threshold = mean_signal + dynamic_threshold_factor * std_signal

        # Minimalna odległość między pikami w próbkach
        min_distance_samples = int(min_distance * fs_pcg)

        # Przeszukiwanie w oknie wokół pików R
        for r_peak in ecg_r_peaks:
            start_idx = max(0, int(r_peak - search_window * fs_pcg))
            end_idx = min(len(signal), int(r_peak + search_window * fs_pcg))
            local_signal = signal[start_idx:end_idx]

            if len(local_signal) == 0:
                continue

            # Znajdź lokalne peaki w oknie
            local_peaks = find_local_peaks(local_signal, radius=int(fs_pcg * 0.02))
            if len(local_peaks) == 0:
                continue

            # Przeskaluj indeksy lokalne do globalnych
            local_peaks_global = [start_idx + idx for idx in local_peaks]

            # Filtrowanie na podstawie dynamicznego progu
            valid_peaks = [idx for idx in local_peaks_global if signal[idx] > dynamic_threshold]

            # Usuwanie peaków zbyt blisko siebie
            filtered_peaks = []
            for peak in valid_peaks:
                if len(filtered_peaks) == 0 or (peak - filtered_peaks[-1]) > min_distance_samples:
                    filtered_peaks.append(peak)

            s1_s2_peaks.extend(filtered_peaks)

        return np.array(s1_s2_peaks)