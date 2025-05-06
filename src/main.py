import matplotlib.pyplot as plt
import numpy as np

from data_loader import DataLoader
from signal_processor import SignalProcessor
from fusion_engine import FusionEngine
from scipy.signal import resample

def main():
    # Ścieżki do plików
    hea_path = "../data/a0011.hea"
    dat_path = "../data/a0011.dat"
    wav_path = "../data/a0011.wav"

    # Liczba próbek do analizy
    num_samples = 50000
    # Wczytanie danych
    loader = DataLoader(hea_path, dat_path, wav_path)

    # Wczytywanie sygnałów
    ecg_signal, fs_ecg = loader.load_ecg_signal(sample_rate=num_samples)
    pcg_signal, fs_pcg = loader.load_pcg_signal(sample_rate=num_samples)

    if fs_pcg != fs_ecg:
        num_samples_pcg = int(len(pcg_signal) * fs_ecg / fs_pcg)
        pcg_signal = resample(pcg_signal, num_samples_pcg)
        fs_pcg = fs_ecg

    # Przetwarzanie sygnałów
    ecg_processor = SignalProcessor(fs_ecg)
    pcg_processor = SignalProcessor(fs_pcg)

    # Filtracja sygnałów
    filtered_ecg = ecg_processor.filter_signal(ecg_signal, 1, 30)
    filtered_pcg = pcg_processor.filter_signal(pcg_signal, 20, 400)

    # --- Przefiltrowany sygnał EKG ---
    plt.figure(figsize=(12, 4))
    plt.plot(filtered_ecg, label="Filtered ECG Signal")
    plt.title("Filtered ECG Signal")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.show()

    # --- Przefiltrowany sygnał PCG ---
    plt.figure(figsize=(12, 4))
    plt.plot(filtered_pcg, label="Filtered PCG Signal")
    plt.title("Filtered PCG Signal")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.show()

    # Analiza falkowa EKG
    transformed_ecg = ecg_processor.wavelet_transform(filtered_ecg)
    r_peaks = ecg_processor.detect_r_peaks(transformed_ecg)

    # --- Sygnał EKG po analizie falkowej ---
    plt.figure(figsize=(12, 6))
    plt.plot(transformed_ecg, label="Wavelet Transformed ECG Signal")
    plt.plot(r_peaks, transformed_ecg[r_peaks], "rx", label="Detected R Peaks")
    plt.title("Wavelet Transformed ECG Signal with Detected R Peaks")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.show()

    # Analiza Hilberta dla PCG
    envelope_pcg = pcg_processor.hilbert_envelope(filtered_pcg)
    s1_peaks, s2_peaks, s1_s2_intervals = pcg_processor.detect_s1_s2(envelope_pcg)


    # --- Wykres z wynikami detekcji ---
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_pcg, label="Filtered PCG Signal")
    plt.plot(envelope_pcg, label="Hilbert Envelope", alpha=0.8)
    plt.plot(s1_peaks, envelope_pcg[s1_peaks], "rx", label="S1 Peaks")
    plt.plot(s2_peaks, envelope_pcg[s2_peaks], "gx", label="S2 Peaks")
    plt.title("PCG Signal with Hilbert Envelope and Detected S1/S2 Peaks")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.show()

    print(f"Detected S1 Peaks: {len(s1_peaks)}")
    print(f"Detected S2 Peaks: {len(s2_peaks)}")
    print(f"Average S1-S2 Interval (ms): {np.mean(s1_s2_intervals) if s1_s2_intervals else 'N/A'}")

    # Fuzja danych
    fusion_engine = FusionEngine(
        ecg_results={'r_peaks': r_peaks, 'transformed_signal': transformed_ecg},
        pcg_results={
            's1_s2_peaks': list(s1_peaks) + list(s2_peaks),
            's1_peaks': s1_peaks,
            's2_peaks': s2_peaks,
            'sampling_rate': fs_pcg
        }
    )

    diagnostic_params = FusionEngine.calculate_fusion_parameters(
        r_peaks=r_peaks,
        s1_peaks=s1_peaks,
        s2_peaks=s2_peaks,
        fs_ecg=fs_ecg,
        fs_pcg=fs_pcg
    )
    print("\n--- Diagnostic Parameters ---")
    for param, value in diagnostic_params.items():
        if value is None:
            print(f"{param}: N/A")
        else:
            suffix = " %" if param == "CVRR (%)" else ""
            print(f"{param}: {value:.2f}{suffix}")


    time_axis = np.arange(len(transformed_ecg)) / fs_ecg * 1000

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, transformed_ecg, label="Wavelet Transformed ECG Signal")
    plt.plot(time_axis[r_peaks], transformed_ecg[r_peaks], "rx", label="Detected R Peaks")

    # Linie S1
    for s1 in s1_peaks:
        plt.axvline(time_axis[s1], color='blue', linestyle='--', alpha=0.5,
                    label="Detected S1 (PCG)" if s1 == s1_peaks[0] else "")

    # Linie R
    for peak in r_peaks:
        plt.axvline(time_axis[peak], color='r', linestyle='-', alpha=0.5,
                    label="Detected R Peak" if peak == r_peaks[0] else "")

    plt.title("Enhanced ECG Signal with PCG Peaks Overlay")
    plt.xlabel("Time [ms]")
    plt.ylabel("Amplitude")
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.show()

    # --- Wykres EKG i PCG razem ---
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Sygnał EKG
    ax[0].plot(filtered_ecg, color='green', label="Filtered ECG Signal")
    ax[0].set_title("Filtered ECG Signal")
    ax[0].set_ylabel("Amplitude")
    ax[0].grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax[0].legend()

    # Sygnał PCG
    ax[1].plot(filtered_pcg, color='gray', label="Filtered PCG Signal")
    ax[1].plot(envelope_pcg, color='orange', alpha=0.8, label="Hilbert Envelope")
    ax[1].plot(s1_peaks, envelope_pcg[s1_peaks], "rx", label="S1 Peaks")
    ax[1].plot(s2_peaks, envelope_pcg[s2_peaks], "gx", label="S2 Peaks")
    ax[1].set_title("Filtered PCG Signal with Hilbert Envelope and Detected Peaks")
    ax[1].set_xlabel("Samples")
    ax[1].set_ylabel("Amplitude")
    ax[1].grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    # Raport końcowy
    report = fusion_engine.generate_report()
    print(report)

if __name__ == "__main__":
    main()
