import numpy as np


class FusionEngine:
    """
    Klasa do analizy różnic i fuzji wyników z EKG i PCG.
    """

    def __init__(self, ecg_results, pcg_results):
        self.ecg_results = ecg_results
        self.pcg_results = pcg_results

    def aggregate_results(self):

        ecg_peaks = self.ecg_results['r_peaks']
        return {
            'r_peaks': sorted(ecg_peaks),
            'enhanced_r_peaks': sorted(ecg_peaks)  # teraz identyczne z oryginałem
        }

    def compare_results(self):
        ecg_cycles = len(self.ecg_results['r_peaks'])/1
        pcg_cycles = len(self.pcg_results['s1_s2_peaks'])/2
        difference = pcg_cycles - ecg_cycles
        return {
            'ecg_cycles': ecg_cycles,
            'pcg_cycles': pcg_cycles,
            'difference': difference
        }

    def generate_report(self):
        comparison = self.compare_results()
        added_peaks = len(self.aggregate_results()['enhanced_r_peaks']) - len(self.ecg_results['r_peaks'])
        report = f"""
        EKG Cycles (Detected R Peaks): {comparison['ecg_cycles']}
        PCG Cycles (Detected S1/S2): {comparison['pcg_cycles']}
        Difference (PCG - EKG): {comparison['difference']}
        Estimated Additional R Peaks from PCG: {added_peaks}
        """
        return report

    @staticmethod
    def calculate_fusion_parameters(r_peaks, s1_peaks, s2_peaks, fs_ecg, fs_pcg):
        # 1) RR intervals [ms]
        rr = np.diff(r_peaks) / fs_ecg * 1000
        mean_rr = np.mean(rr) if rr.size else None

        # 2) Heart rate [bpm]
        hr = 60000 / mean_rr if mean_rr else None

        # 3) HRV time-domain
        sdnn = np.std(rr) if rr.size else None
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr)))) if rr.size > 1 else None
        cvrr = (sdnn / mean_rr * 100) if mean_rr else None

        # 4) Systolic (ET) = S1→S2
        et = [(s2 - s1) / fs_pcg * 1000 for s1, s2 in zip(s1_peaks, s2_peaks) if s2 > s1]
        mean_et = np.mean(et) if et else None

        # 5) Diastolic/IRT = S2→następne S1
        irt = []
        for i in range(1, len(s1_peaks)):
            prev_s2 = [s for s in s2_peaks if s < s1_peaks[i]]
            if prev_s2:
                irt.append((s1_peaks[i] - prev_s2[-1]) / fs_pcg * 1000)
        mean_irt = np.mean(irt) if irt else None

        # 6) IVCT ≈ RR – ET – IRT (uśrednione)
        ivct = None
        if mean_rr and mean_et is not None and mean_irt is not None:
            ivct = mean_rr - mean_et - mean_irt

        # 7) Tei index (MPI) = (IVCT + IRT) / ET
        tei = None
        if ivct is not None and mean_irt is not None and mean_et:
            tei = (ivct + mean_irt) / mean_et

        return {
            "Mean RR (ms)": mean_rr,
            "Heart Rate (bpm)": hr,
            "SDNN (ms)": sdnn,
            "RMSSD (ms)": rmssd,
            "CVRR (%)": cvrr,
            "Mean Systolic (ET, ms)": mean_et,
            "Mean Diastolic (IRT, ms)": mean_irt,
            "Mean IVCT (ms)": ivct,
            "Tei Index (MPI)": tei
        }

