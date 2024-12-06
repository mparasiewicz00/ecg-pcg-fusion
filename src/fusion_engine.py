import matplotlib.pyplot as plt
import numpy as np

class FusionEngine:
    def __init__(self, ecg_signal, pcg_signal, fs_ecg, fs_pcg):
        """
        Klasa do fuzji sygnałów EKG i PCG.

        Args:
            ecg_signal (np.ndarray): Przefiltrowany sygnał EKG.
            pcg_signal (np.ndarray): Przefiltrowany sygnał PCG.
            fs_ecg (float): Częstotliwość próbkowania sygnału EKG.
            fs_pcg (float): Częstotliwość próbkowania sygnału PCG.
        """
        self.ecg_signal = ecg_signal
        self.pcg_signal = pcg_signal
        self.fs_ecg = fs_ecg
        self.fs_pcg = fs_pcg

    @staticmethod
    def normalize_signal(signal):
        """
        Normalizacja sygnału do przedziału [-1, 1].
        """
        return 2 * (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) - 1

    def synchronize_signals(self):
        """
        Synchronizacja czasowa i normalizacja sygnałów PCG i EKG.
        """
        time_ecg = np.arange(len(self.ecg_signal)) / self.fs_ecg

        # Skalowanie PCG do czasu EKG
        pcg_resampled = np.interp(
            time_ecg,
            np.arange(len(self.pcg_signal)) / self.fs_pcg,
            self.pcg_signal
        )

        # Normalizacja sygnałów
        ecg_normalized = self.normalize_signal(self.ecg_signal)
        pcg_normalized = self.normalize_signal(pcg_resampled)

        return ecg_normalized, pcg_normalized, time_ecg

    def visualize_fusion(self):
        """
        Wizualizacja fuzji sygnałów EKG i PCG na jednej osi czasu.
        """
        ecg_signal, pcg_signal, time = self.synchronize_signals()

        plt.figure(figsize=(12, 6))
        plt.plot(time, ecg_signal, label="Normalized ECG Signal")
        plt.plot(time, pcg_signal, label="Normalized PCG Signal", alpha=0.7)
        plt.title("Fusion of ECG and PCG Signals (Normalized)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()
