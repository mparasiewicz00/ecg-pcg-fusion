# Cardiac Diagnostics using ECG and PCG Signal Fusion

## Project Overview
This project focuses on the application of ECG (electrocardiography) and PCG (phonocardiography) signal fusion for advanced cardiac diagnostics. The goal is to create a robust Python-based console application capable of processing and analyzing physiological signals.

## Features
- **Input Data Handling**:
  - Parsing `.hea`, `.dat` (ECG), and `.wav` (PCG) files from PhysioNet datasets.
- **Signal Processing**:
  - ECG analysis: R-peak detection, R-R interval calculation, and frequency analysis.
  - PCG analysis: Heart sound (S1, S2) detection and time-frequency analysis.
- **Signal Fusion**:
  - Correlation and synchronization between ECG and PCG parameters.
- **Results**:
  - Visualization of signals and results (charts, tables).
  - Exporting results to `.csv` or `.png`.

## Setup
### Requirements
- Python 3.13
- [Anaconda](https://www.anaconda.com/) environment
