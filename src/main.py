import os

import matplotlib.pyplot as plt
import numpy as np
import pywt
import scipy.signal as sig
import wfdb
from scipy.signal import hilbert, find_peaks, firwin, filtfilt, iirnotch, sosfiltfilt, butter


# ----- I/O -----
def load_record(record_path: str, record_name: str):
    full_path = os.path.join(record_path, record_name)
    return wfdb.rdrecord(full_path)

# ----- PREPROCESSING -----
def preprocess_signals(record, ecg_chan=0, pcg_chan=1):
    fs = record.fs
    raw = record.p_signal
    ecg_raw = raw[:, ecg_chan]
    pcg_raw = raw[:, pcg_chan]

    # --- ECG z filtrami FIR ---
    # 1) FIR-notch 50 Hz, szer. tłumienia ~2 Hz
    taps_notch = firwin(401, [49, 51], pass_zero=True, fs=fs)
    ecg_nn = filtfilt(taps_notch, [1.0], ecg_raw)
    # 2) FIR-bandpass 0.5–40 Hz
    taps_bp = firwin(801, [0.5, 40], pass_zero=False, fs=fs)
    ecg_bp = filtfilt(taps_bp, [1.0], ecg_nn)

    # --- PCG z filtrami IIR ---
    # IIR-notch 50 Hz (Q=30)
    bN, aN = iirnotch(50/(fs/2), Q=30)
    sosN = sig.tf2sos(bN, aN)
    pcg_nn = sosfiltfilt(sosN, pcg_raw)
    # IIR-bandpass 10–100 Hz, 4. rzędu
    sosP = butter(4, [10/(fs/2), 100/(fs/2)], btype='band', output='sos')
    pcg_bp = sosfiltfilt(sosP, pcg_nn)

    print(f"ECG filtrowane: {ecg_bp.min():.3f}…{ecg_bp.max():.3f} mV")
    print(f"PCG filtrowane: {pcg_bp.min():.3f}…{pcg_bp.max():.3f} mV")

    return ecg_bp, pcg_bp

#TRANSFORMACJA FALKOWA
def wavelet_feature(ecg: np.ndarray,
                    wavelet: str = 'db4',
                    level: int = 4) -> np.ndarray:

    coeffs = pywt.wavedec(ecg, wavelet, level=level)
    for i in range(1, len(coeffs)):
        coeffs[i] = np.zeros_like(coeffs[i])
    rec = pywt.waverec(coeffs, wavelet)
    rec = np.abs(rec)
    return rec / np.max(rec)

#DETEKCJA R PEAKS
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

#OBWIEDNIA HILBERTA
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

def plot_ecg_filtered(ecg_bp, fs, t_max: float = 3.0):
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
    t_max = 2.5
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

    # ECG filtrowane + R-peaki
    ax.plot(t, ecg_seg[:n_max], label='ECG filtrowane')
    ax.plot(r_peaks / fs, ecg_seg[r_peaks], 'ro', label='R-peaks')

    # pionowe linie w miejscach S1
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


    # czasy i wartości parametrów:
    # __________________________________

    # HR/RR
    t_R = r_peaks[1:] / fs
    RR = np.diff(r_peaks) / fs
    HR_inst = 60.0 / RR
    HR_mean = HR_inst.mean()

    # SDNN / RMSSD
    SDNN = RR.std()
    RMSSD = np.sqrt(np.mean(np.diff(RR) ** 2))

    # ET / DT
    t_S1 = S1[:-1] / fs
    ET = (S2[:-1] - S1[:-1]) / fs
    DT = RR - ET

    # IVCT / IVRT / MPI

    t_IVCT = S1[:-1] / fs
    IVCT = (S1[:-1] - r_peaks[:-1]) / fs

    t_IVRT = S1[1:] / fs
    IVRT = (S1[1:] - S2[:-1]) / fs

    t_MPI = S1[1:] / fs
    MPI = (IVCT + IVRT) / ET

    # S1 / S2
    S1_indices = []
    S1S2 = []
    for s1 in S1:
        s2_after = S2[S2 > s1]
        if s2_after.size:
            S1_indices.append(s1)
            S1S2.append((s2_after[0] - s1) / fs)
    S1S2 = np.array(S1S2)
    t_S1S2 = np.array(S1_indices) / fs

    # EMD
    EMD = []
    EMD_t = []
    for r in r_peaks:
        s1_after = S1[S1 > r]
        if s1_after.size:
            EMD.append((s1_after[0] - r) / fs)
            EMD_t.append(r / fs)

    EMD = np.array(EMD)
    EMD_t = np.array(EMD_t)
    EMD_mean = EMD.mean() * 1000


    print(f"Średnie HR: {HR_mean:.1f} BPM")
    print(f"Średnie EMD: {EMD_mean:.1f} ms")
    print(f"SDNN (std RR): {SDNN * 1000:.1f} ms")
    print(f"RMSSD: {RMSSD * 1000:.1f} ms")
    print(f"IVCT: {IVCT.mean() * 1000:.1f} ms")
    print(f"IVRT: {IVRT.mean() * 1000:.1f} ms")
    print(f"MPI: {MPI.mean():.1f} ms")

    def plot_time_series(t, values, ylabel, ylim=None):
        plt.figure(figsize=(8, 3))
        scale = 1000 if '[ms]' in ylabel else 1
        plt.plot(t, values * scale, '-o', markersize=4, linewidth=1)
        plt.xlabel('Czas [s]')
        plt.ylabel(ylabel)
        if ylim: plt.ylim(ylim)
        plt.grid(linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    plot_time_series(t_R, RR, 'Interwał RR [s]',  ylim=(0.7, 1.2))
    plot_time_series(t_R, HR_inst, 'Chwilowy HR [BPM]',  ylim=(50, 85))
    plot_time_series(t_S1, ET, 'Czas wyrzutu ET [s]', ylim=(0.27, 0.35))
    plot_time_series(t_S1, DT, 'Czas rozkurczu DT [s]',ylim=(0.4, 0.9))
    plot_time_series(EMD_t,EMD,'EMD [s]',ylim=(0.02, 0.045))
    plot_time_series(t_IVCT, IVCT, 'IVCT [s]', ylim=(0.02, 0.045))
    plot_time_series(t_IVRT, IVRT, 'IVRT [s]', ylim=(0.45, 0.75))
    plot_time_series(t_MPI, MPI, 'MPI (Indeks Tei) [–]',  ylim=(1.6, 2.5))
    plot_time_series(t_S1S2, S1S2, 'Interwał S1–S2 [s]',  ylim=(0.28, 0.35))









