import matplotlib.pyplot as plt
from data_loader import DataLoader
from signal_processor import SignalProcessor

# Ścieżki do plików danych
hea_path = "../data/a0011.hea"
dat_path = "../data/a0011.dat"
wav_path = "../data/a0011.wav"

# Inicjalizacja DataLoader
loader = DataLoader(hea_path, dat_path, wav_path)

# Wczytanie sygnału EKG
ecg_signal, fs_ecg = loader.load_ecg_signal(sample_rate=10000)
ecg_processor = SignalProcessor(sampling_rate=fs_ecg)

# Wczytanie sygnału PCG
pcg_signal, fs_pcg = loader.load_pcg_signal(sample_rate=10000)
pcg_processor = SignalProcessor(sampling_rate=fs_pcg)

# Wyświetlenie surowego sygnału EKG
plt.figure(figsize=(12, 4))
plt.plot(ecg_signal, label="Raw ECG Signal")
plt.title("Raw ECG Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# Filtracja sygnału EKG
filtered_ecg = ecg_processor.filter_signal(ecg_signal)

plt.figure(figsize=(12, 4))
plt.plot(filtered_ecg, label="Filtered ECG Signal")
plt.title("Filtered ECG Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# Wyświetlenie surowego sygnału PCG
plt.figure(figsize=(12, 4))
plt.plot(pcg_signal, label="Raw PCG Signal")
plt.title("Raw PCG Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# Filtracja sygnału PCG
filtered_pcg = pcg_processor.filter_pcg_signal(pcg_signal)

plt.figure(figsize=(12, 4))
plt.plot(pcg_signal, label="Raw PCG Signal")
plt.plot(filtered_pcg, label="Filtered PCG Signal", linestyle="--")
plt.title("Raw vs Filtered PCG Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# Transformacja falkowa EKG
transformed_ecg = ecg_processor.wavelet_transform(filtered_ecg)

plt.figure(figsize=(12, 4))
plt.plot(transformed_ecg, label="Wavelet Transformed ECG Signal")
plt.title("Wavelet Transformed ECG Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# Detekcja pików R
r_peaks = ecg_processor.detect_r_peaks(transformed_ecg, min_distance=int(fs_ecg * 0.6))

# Obliczanie odstępów R-R
rr_intervals = ecg_processor.calculate_rr_intervals(r_peaks)

plt.figure(figsize=(12, 6))
plt.plot(transformed_ecg, label="Wavelet Transformed ECG Signal")
plt.plot(r_peaks, transformed_ecg[r_peaks], "rx", label="Detected R Peaks")
plt.title("Wavelet Transformed ECG Signal with Detected R Peaks")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

print(f"Detected R Peaks (samples): {r_peaks}")
print(f"R-R Intervals (ms): {rr_intervals}")

# Detekcja tonów S1/S2 w synchronizacji z EKG
s1_s2_peaks = pcg_processor.detect_s1_s2_peaks(
    signal=filtered_pcg,
    ecg_r_peaks=r_peaks,
    fs_pcg=fs_pcg
)

# Wyświetlenie wyników detekcji na sygnale PCG
plt.figure(figsize=(12, 6))
plt.plot(filtered_pcg, label="Filtered PCG Signal")
plt.plot(s1_s2_peaks, filtered_pcg[s1_s2_peaks], "rx", label="Detected S1/S2 Peaks")
plt.title("Filtered PCG Signal with Detected S1/S2 Peaks")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

print(f"Detected S1/S2 Peaks (samples): {s1_s2_peaks}")