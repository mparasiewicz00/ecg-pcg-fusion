import matplotlib.pyplot as plt
import numpy as np

from data_loader import DataLoader
from signal_processor import SignalProcessor
from fusion_engine import FusionEngine

def main():
    # Ścieżki do plików
    hea_path = "../data/a0011.hea"
    dat_path = "../data/a0011.dat"
    wav_path = "../data/a0011.wav"

    # Liczba próbek do analizy
    num_samples = 10000
    # Wczytanie danych
    loader = DataLoader(hea_path, dat_path, wav_path)

    # Wczytywanie sygnałów
    ecg_signal, fs_ecg = loader.load_ecg_signal(sample_rate=num_samples)
    pcg_signal, fs_pcg = loader.load_pcg_signal(sample_rate=num_samples)

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
    s1_peaks, s2_peaks, s1_s2_intervals = pcg_processor.detect_s1_s2(envelope_pcg, r_peaks=r_peaks)


    # --- Wykres z wynikami detekcji ---
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_pcg, label="Filtered PCG Signal")
    plt.plot(envelope_pcg, label="Hilbert Envelope", alpha=0.8)
    plt.plot(s1_peaks, envelope_pcg[s1_peaks], "rx", label="S1 Peaks")
    plt.plot(s2_peaks, envelope_pcg[s2_peaks], "gx", label="S2 Peaks")
    plt.title("PCG Signal with Hilbert Envelope and Detected S1/S2 Peaks")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

    print(f"Detected S1 Peaks: {len(s1_peaks)}")
    print(f"Detected S2 Peaks: {len(s2_peaks)}")
    print(f"Average S1-S2 Interval (ms): {np.mean(s1_s2_intervals) if s1_s2_intervals else 'N/A'}")

    # Fuzja danych
    fusion_engine = FusionEngine(
        ecg_results={'r_peaks': r_peaks, 'transformed_signal': transformed_ecg},
        pcg_results={'s1_s2_peaks': s1_peaks + s2_peaks, 's1_peaks': s1_peaks, 's2_peaks': s2_peaks}
    )

    # Agregacja wyników PCG i EKG
    fusion_results = fusion_engine.aggregate_results()

    # --- Wykres EKG z informacjami z PCG ---
    plt.figure(figsize=(12, 6))
    plt.plot(transformed_ecg, label="Wavelet Transformed ECG Signal")
    plt.plot(r_peaks, transformed_ecg[r_peaks], "rx", label="Detected R Peaks")
    for peak in fusion_results['enhanced_r_peaks']:
        plt.axvline(peak, color='g', linestyle='--', alpha=0.7,
                    label="PCG-Enhanced R Peak" if peak == fusion_results['enhanced_r_peaks'][0] else "")
    plt.title("Enhanced ECG Signal with PCG Peaks Overlay")
    plt.xlabel("Samples")
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
