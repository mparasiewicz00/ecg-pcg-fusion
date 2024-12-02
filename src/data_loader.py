import wfdb
import soundfile as sf

class DataLoader:
    """
    Klasa do wczytywania danych z plików .hea, .dat i .wav.
    """
    def __init__(self, hea_path, dat_path, wav_path):
        self.hea_path = hea_path
        self.dat_path = dat_path
        self.wav_path = wav_path

    def load_metadata(self):
        with open(self.hea_path, 'r') as f:
            metadata = f.readlines()
        return metadata

    def load_ecg_signal(self, sample_rate=10000):
        record = wfdb.rdrecord(self.dat_path.replace('.dat', ''), sampto=sample_rate)
        return record.p_signal[:, 0], record.fs

    def load_pcg_signal(self, sample_rate=None, start=0, end=None):
        pcg_signal, fs_pcg = sf.read(self.wav_path, start=start, stop=end)
        if sample_rate and len(pcg_signal) > sample_rate:
            pcg_signal = pcg_signal[:sample_rate]
        return pcg_signal, fs_pcg