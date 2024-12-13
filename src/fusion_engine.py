class FusionEngine:
    """
    Klasa do analizy różnic i fuzji wyników z EKG i PCG.
    """

    def __init__(self, ecg_results, pcg_results):
        """
        Inicjalizacja wyników z EKG i PCG.

        Args:
            ecg_results (dict): Wyniki przetwarzania EKG (np. piki R).
            pcg_results (dict): Wyniki przetwarzania PCG (np. piki S1/S2).
        """
        self.ecg_results = ecg_results
        self.pcg_results = pcg_results

    def aggregate_results(self):
        """
        Fuzja wyników EKG i PCG. Dodanie informacji o pikach z PCG do wyników EKG.
        """
        ecg_peaks = self.ecg_results['r_peaks']
        pcg_peaks = self.pcg_results['s1_s2_peaks']

        # Dodanie pików PCG jako wsparcia do EKG
        enhanced_peaks = set(ecg_peaks)
        for peak in pcg_peaks:
            # Wyszukiwanie pików z PCG blisko pików z EKG
            if not any(abs(peak - ecg_peak) < 50 for ecg_peak in ecg_peaks):
                enhanced_peaks.add(peak)

        return {
            'r_peaks': ecg_peaks,
            'enhanced_r_peaks': sorted(enhanced_peaks)  # Sortowanie wyników
        }

    def compare_results(self):
        """
        Porównanie liczby cykli serca między EKG a PCG.
        """
        ecg_cycles = len(self.ecg_results['r_peaks'])
        pcg_cycles = len(self.pcg_results['s1_s2_peaks'])
        difference = pcg_cycles - ecg_cycles
        return {
            'ecg_cycles': ecg_cycles,
            'pcg_cycles': pcg_cycles,
            'difference': difference
        }

    def generate_report(self):
        """
        Generowanie raportu końcowego.
        """
        results = self.compare_results()
        report = f"""
        EKG Cycles: {results['ecg_cycles']}
        PCG Cycles: {results['pcg_cycles']}
        Difference: {results['difference']}
        """
        return report
