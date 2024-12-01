import matplotlib.pyplot as plt

from data_loader import DataLoader
from signal_processor import SignalProcessor

hea_path = "../data/a0011.hea"
dat_path = "../data/a0011.dat"
wav_path = "../data/a0011.wav"

loader = DataLoader(hea_path, dat_path, wav_path)
ecg_signal, fs_ecg = loader.load_ecg_signal()

processor = SignalProcessor(sampling_rate=fs_ecg)

# Surowy sygnał
plt.figure(figsize=(12, 4))
plt.plot(ecg_signal, label="Raw ECG Signal")
plt.title("Raw ECG Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# Filtracja sygnału
filtered_ecg = processor.filter_signal(ecg_signal)

plt.figure(figsize=(12, 4))
plt.plot(filtered_ecg, label="Filtered ECG Signal")
plt.title("Filtered ECG Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# Transformacja falkowa
transformed_signal = processor.wavelet_transform(filtered_ecg)

plt.figure(figsize=(12, 4))
plt.plot(transformed_signal, label="Wavelet Transformed Signal")
plt.title("Wavelet Transformed Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# Detekcja pików R
r_peaks = processor.detect_r_peaks(transformed_signal, min_distance=int(fs_ecg * 0.6))

# Obliczanie odstępów R-R
rr_intervals = processor.calculate_rr_intervals(r_peaks)

# Wyświetlenie wyników detekcji
plt.figure(figsize=(12, 6))
plt.plot(transformed_signal, label="Wavelet Transformed Signal")
plt.plot(r_peaks, transformed_signal[r_peaks], "rx", label="Detected R Peaks")
plt.title("Wavelet Transformed Signal with Detected R Peaks")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

print(f"Detected R Peaks (samples): {r_peaks}")
print(f"R-R Intervals (ms): {rr_intervals}")