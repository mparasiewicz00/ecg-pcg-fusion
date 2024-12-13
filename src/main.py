import matplotlib.pyplot as plt
from data_loader import DataLoader
from signal_processor import SignalProcessor
from fusion_engine import FusionEngine

def main():
    # Ścieżki do plików
    hea_path = "../data/a0011.hea"
    dat_path = "../data/a0011.dat"
    wav_path = "../data/a0011.wav"

    # Liczba próbek do analizy
    num_samples = 5000

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
    s1_s2_peaks = pcg_processor.detect_pcg_peaks(envelope_pcg)

    # --- Sygnał PCG po transformacji Hilberta ---
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_pcg, label="Filtered PCG Signal")
    plt.plot(envelope_pcg, label="Hilbert Envelope", alpha=0.8)
    plt.plot(s1_s2_peaks, envelope_pcg[s1_s2_peaks], "rx", label="Detected Peaks (S1/S2)")
    plt.title("PCG Signal with Hilbert Envelope and Detected Peaks")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.show()

    # Fuzja danych
    fusion_engine = FusionEngine(
        ecg_results={'r_peaks': r_peaks, 'transformed_signal': transformed_ecg},
        pcg_results={'s1_s2_peaks': s1_s2_peaks}
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

    # Raport końcowy
    report = fusion_engine.generate_report()
    print(report)


if __name__ == "__main__":
    main()