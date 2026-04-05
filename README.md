# RF Signal Detection using FFT and Energy Detection

## Overview
This project demonstrates RF signal detection using:
- FFT-based spectral analysis
- Power Spectral Density (PSD)
- Energy detection with thresholding
- ROC curve evaluation

## Features
- Interference detection via frequency-domain peak
- Statistical detection using chi-square threshold
- GUI visualization using Tkinter
- ROC performance analysis

## Technologies Used
- Python
- NumPy
- SciPy
- Matplotlib
- Tkinter

## How to Run

1. Install dependencies: pip install -r requirements.txt
2. Run: python main.py

## Theory

### Energy Detection
- H0: Noise only
- H1: Signal + Noise
- Decision based on energy threshold

### Threshold Selection
- Derived using Chi-square distribution
- Controlled using Probability of False Alarm (Pfa)

### FFT & PSD
- FFT converts signal to frequency domain
- Interference appears as sharp peak in PSD

### ROC Curve
- Shows trade-off between detection probability and false alarm

## Output
![Output](output.png)

## Future Work
- Real-time RF data acquisition using Raspberry Pi
- SDR (Software Defined Radio) integration
- Embedded implementation

## Author
K Sangavi
