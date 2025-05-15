import os

import matplotlib.pyplot as plt
import numpy as np
import pywt
import scipy.signal as sig
import wfdb
from scipy.signal import hilbert, find_peaks

# ----- I/O -----
def load_record(record_path: str, record_name: str):
    full_path = os.path.join(record_path, record_name)
    return wfdb.rdrecord(full_path)

# ----- PREPROCESSING -----
def preprocess_signals(record, ecg_chan: int = 0, pcg_chan: int = 1):
    """Notch 50 Hz + bandpass ECG 10–40 Hz, PCG 10–100 Hz"""
    fs = record.fs
    raw = record.p_signal
    ecg_raw = raw[:, ecg_chan]
    pcg_raw = raw[:, pcg_chan]

    # SOS notch filter
    b, a = sig.iirnotch(50/(fs/2), Q=30)
    sos_notch = sig.tf2sos(b, a)
    ecg_nn = sig.sosfiltfilt(sos_notch, ecg_raw)
    pcg_nn = sig.sosfiltfilt(sos_notch, pcg_raw)

    # SOS bandpass
    sos_ecg = sig.butter(4, [10/(fs/2), 40/(fs/2)], btype='band', output='sos')
    sos_pcg = sig.butter(4, [10/(fs/2),100/(fs/2)], btype='band', output='sos')
    ecg_bp = sig.sosfiltfilt(sos_ecg, ecg_nn)
    pcg_bp = sig.sosfiltfilt(sos_pcg, pcg_nn)

    print(f"⇒ ECG raw min/max: {ecg_raw.min():.3f}/{ecg_raw.max():.3f}")
    print(f"⇒ ECG filt min/max: {ecg_bp.min():.3f}/{ecg_bp.max():.3f}")

    return ecg_bp, pcg_bp

# ----- DETEKCJA R-PEAKÓW -----
def wavelet_feature(ecg: np.ndarray,
                    wavelet: str = 'db4',
                    level: int = 4) -> np.ndarray:

    coeffs = pywt.wavedec(ecg, wavelet, level=level)
    for i in range(1, len(coeffs)):
        coeffs[i] = np.zeros_like(coeffs[i])
    rec = pywt.waverec(coeffs, wavelet)
    rec = np.abs(rec)
    return rec / np.max(rec)

# ----- DETEKCJA R-PEAKS PO TRANSFORMACJI FALKOWEJ -----
def detect_r_peaks_wavelet(ecg: np.ndarray,
                           fs: float,
                           wavelet: str = 'db4',
                           level: int = 4,
                           threshold_factor: float = 1.0,
                           min_dist_s: float = 0.2) -> np.ndarray:
    feat = wavelet_feature(ecg, wavelet, level)
    mu, sigma = feat.mean(), feat.std()
    thresh = mu + threshold_factor * sigma
    min_dist = int(min_dist_s * fs)
    peaks, _ = find_peaks(feat, height=thresh, distance=min_dist)
    return peaks


# ----- OBWIEDNIA HILBERTA PCG -----
def compute_hilbert_envelope(pcg: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(pcg))

def detect_s1_s2_envelope(envelope: np.ndarray,
                          fs: float,
                          threshold_factor: float = 1.0,
                          min_dist_s: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    μ, σ = envelope.mean(), envelope.std()
    thresh = μ + threshold_factor * σ
    min_dist = int(min_dist_s * fs)
    peaks, props = find_peaks(envelope,
                              height=thresh,
                              distance=min_dist,
                              prominence=σ*0.5)

    S1 = peaks[::2].copy()
    S2 = peaks[1::2].copy()
    return S1, S2

# ----- RYSOWANIE POJEDYNCZYCH Sygnałów -----
def plot_ecg_raw(record, t_max: float = 30.0):
    fs   = record.fs
    sigs = record.p_signal[:, 0]
    n    = int(min(t_max * fs, sigs.size))
    t    = np.arange(n) / fs

    plt.figure(figsize=(10, 3))
    plt.plot(t, sigs[:n], linewidth=0.8)
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda [mV]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_pcg_raw(record, t_max: float = 30.0):
    fs   = record.fs
    sigs = record.p_signal[:, 1]
    n    = int(min(t_max * fs, sigs.size))
    t    = np.arange(n) / fs

    plt.figure(figsize=(10, 3))
    plt.plot(t, sigs[:n], color='tab:orange', linewidth=0.8)
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda [mV]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_ecg_filtered(ecg_bp, fs, t_max: float = 30.0):
    n = int(min(t_max * fs, ecg_bp.size))
    t = np.arange(n) / fs

    plt.figure(figsize=(10, 3))
    plt.plot(t, ecg_bp[:n], linewidth=0.8)
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda [mV]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_pcg_filtered(pcg_bp, fs, t_max: float = 30.0):
    n = int(min(t_max * fs, pcg_bp.size))
    t = np.arange(n) / fs

    plt.figure(figsize=(10, 3))
    plt.plot(t, pcg_bp[:n], linewidth=0.8, color='tab:orange')
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda [mV]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ----- GŁÓWNY BLOK -----

if __name__ == '__main__':
    data_folder = '../data'
    record_name = 'ECGPCG0013'
    rec = load_record(data_folder, record_name)
    print(f"Kanały: {rec.sig_name}  fs={rec.fs} Hz")

    ecg_bp, pcg_bp = preprocess_signals(rec)

    # wykres surowe + filtrowane
    plot_ecg_raw(rec)
    plot_pcg_raw(rec)
    plot_ecg_filtered(ecg_bp, rec.fs)
    plot_pcg_filtered(pcg_bp, rec.fs)


    fs = rec.fs
    t_max = 30.0
    n_max = int(t_max * fs)
    t = np.arange(n_max) / fs
    ecg_seg = ecg_bp[:n_max]
    pcg_seg = pcg_bp[:n_max]

    feat = wavelet_feature(ecg_seg, wavelet='db4', level=4)

    r_peaks = detect_r_peaks_wavelet(ecg_seg, fs,
                                     wavelet='db4',
                                     level=4,
                                     threshold_factor=1.2,
                                     min_dist_s=0.3)
    print(f"Znaleziono {len(r_peaks)} R-peaks (wavelet)")

    # ECG po transformacji falkowej i detekcja R-peaks
    plt.figure(figsize=(12, 4))
    plt.plot(t, feat[:n_max], label='Wavelet feature')
    plt.plot(r_peaks / fs, feat[r_peaks], 'ro', label='R-peaks')
    plt.xlabel('Czas [s]')
    plt.ylabel('Znormalizowana wartość')
    plt.legend()
    plt.grid()
    plt.show()

    env = compute_hilbert_envelope(pcg_seg)
    S1, S2 = detect_s1_s2_envelope(
        envelope=env,
        fs=fs,
        threshold_factor=1.0,
        min_dist_s=0.3
    )
    print(f"S1_env: {len(S1)}, S2_env: {len(S2)}")

    # wykres detekcji
    plt.figure(figsize=(12,4))
    plt.plot(t, ecg_seg, label='ECG filtrowane')
    if r_peaks.size>0:
        plt.plot(r_peaks/fs, ecg_seg[r_peaks], 'ro', label='R-peaks')

    plt.xlabel('Czas [s]')
    plt.ylabel('Znormalizowana wartość')
    plt.legend();
    plt.grid();
    plt.show()

    # Detekcja S1/S2 z obwiedni
    t = np.arange(len(env)) / fs
    plt.figure(figsize=(12, 4))
    plt.plot(t, env, label='PCG envelope')
    plt.plot(S1 / fs, env[S1], 'gx', label='S1_env')
    plt.plot(S2 / fs, env[S2], 'kx', label='S2_env')
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda')
    plt.legend()
    plt.grid()
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 4))

    # 1) ECG filtrowane + R-peaki
    ax.plot(t, ecg_seg[:n_max], label='ECG filtrowane')
    ax.plot(r_peaks / fs, ecg_seg[r_peaks], 'ro', label='R-peaks')

    # 2) pionowe linie w miejscach S1
    for i, s in enumerate(S1):
        if s < n_max:
            ax.axvline(x=s / fs,
                       color='green',
                       linestyle='--',
                       label='S1' if i == 0 else None)

    ax.set_xlim(0, t_max)
    ax.set_xlabel('Czas [s]')
    ax.set_ylabel('Amplituda [mV]')
    ax.legend()
    ax.grid(True)

    plt.show()

    # WYLICZANIE PARAMETRÓW

    # 1) RR-intervals, HR, HRV
    RR = np.diff(r_peaks) / fs
    HR_inst = 60.0 / RR
    HR_mean = HR_inst.mean()
    SDNN = RR.std()
    RMSSD = np.sqrt(np.mean(np.diff(RR) ** 2))

    RR = np.diff(r_peaks) / fs
    ET = (S2[:-1] - S1[:-1]) / fs
    DT = RR - ET

    # 2) IVCT, IVRT
    IVCT = (S1[:-1] - r_peaks[:-1]) / fs
    IVRT = (S1[1:]  - S2[:-1]) / fs

    # 3) Tei Index (MPI) – dla pierwszych N-1 cykli
    MPI = (IVCT + IVRT) / ET

    print(f"Średnie HR: {HR_mean:.1f} BPM")
    print(f"SDNN (std RR): {SDNN * 1000:.1f} ms")
    print(f"RMSSD: {RMSSD * 1000:.1f} ms")
    print(f"IVCT: {IVCT.mean() * 1000:.1f} ms")
    print(f"IVRT: {IVRT.mean() * 1000:.1f} ms")
    print(f"MPI: {MPI.mean():.1f} ms")

    # 2) Electromechanical delay (EMD): czas od R-peak do najbliższego S1
    EMD = []
    for r in r_peaks:
        # znajdź pierwszy S1 po R
        s1_after = S1[S1 > r]
        if s1_after.size:
            EMD.append((s1_after[0] - r) / fs)
    EMD = np.array(EMD)
    EMD_mean = EMD.mean()
    print(f"EMD: {EMD_mean * 1000:.1f} ms")

    # 3) S1–S2 intervals i czas skurczu serca (HS)
    S1S2 = []
    for s1 in S1:
        s2_after = S2[S2 > s1]
        if s2_after.size:
            S1S2.append((s2_after[0] - s1) / fs)
    S1S2 = np.array(S1S2)
    SYST = S1S2.mean()
    print(f"Średni czas skurczu (S1–S2): {SYST * 1000:.1f} ms")

    # Po obliczeniu wszystkich tablic: RR, HR_inst, EMD, ET, DT, IVCT, IVRT, MPI, S1S2

    # 1) Histogram odstępów RR [ms]
    plt.figure(figsize=(6, 4))
    plt.hist(RR * 1000, bins=10)
    plt.xlabel('Interwał RR [ms]')
    plt.ylabel('Liczba')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2) Histogram electromechanical delay EMD [ms]
    plt.figure(figsize=(6, 4))
    plt.hist(EMD * 1000, bins=10)
    plt.xlabel('Opóźnienie elektromechaniczne EMD [ms]')
    plt.ylabel('Liczba')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3) Histogram czasu wyrzutu ET [ms]
    plt.figure(figsize=(6, 4))
    plt.hist(ET * 1000, bins=10)
    plt.xlabel('Czas skurczu ET [ms]')
    plt.ylabel('Liczba')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 4) Histogram czasu rozkurczu DT [ms]
    plt.figure(figsize=(6, 4))
    plt.hist(DT * 1000, bins=10)
    plt.xlabel('Czas rozkurczu DT [ms]')
    plt.ylabel('Liczba')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 5) Histogram IVCT [ms]
    plt.figure(figsize=(6, 4))
    plt.hist(IVCT * 1000, bins=10)
    plt.xlabel('IVCT [ms]')
    plt.ylabel('Liczba')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 6) Histogram IVRT [ms]
    plt.figure(figsize=(6, 4))
    plt.hist(IVRT * 1000, bins=10)
    plt.xlabel('IVRT [ms]')
    plt.ylabel('Liczba')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 7) Histogram indeksu Tei (MPI) [−]
    plt.figure(figsize=(6, 4))
    plt.hist(MPI, bins=10)
    plt.xlabel('MPI (Indeks Tei) [–]')
    plt.ylabel('Liczba')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 8) Histogram interwału S1–S2 [ms]
    plt.figure(figsize=(6, 4))
    plt.hist(S1S2 * 1000, bins=10)
    plt.xlabel('Interwał S1–S2 [ms]')
    plt.ylabel('Liczba')
    plt.grid(True)
    plt.tight_layout()
    plt.show()





