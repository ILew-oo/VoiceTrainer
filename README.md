<img src="https://github.com/user-attachments/assets/7c81e333-421c-427e-a60f-d3c9fd24552e" alt="drawing" width="600"/>

# Transgender Voice Training Tool

Transgender Voice Training Tool being built in Python.

**Latest Download:** [https://github.com/Kavex/VocalTuner/releases/](https://github.com/Kavex/VocalTuner/releases/)

This Python-based tool provides real-time analysis of vocal pitch and resonance (through formant frequencies) for transgender voice training. The tool offers live visual feedback and summary metrics to help users monitor and adjust their voice toward their desired pitch and resonance targets.

## Features

- **Real-Time Audio Analysis:**  
  Captures audio from your selected microphone in real time using PyAudio.

- **Pitch Detection & Note Conversion:**  
  Uses an autocorrelation algorithm to estimate the fundamental frequency (pitch) and converts it to the nearest musical note (e.g. A3, D4).

- **Formant Analysis:**  
  Computes the first two formant frequencies (F1 and F2) using a basic Linear Predictive Coding (LPC) approach. These formants are key for evaluating vocal resonance and are critical in transgender voice training.

- **Quality Metrics:**  
  Estimates basic voice quality metrics such as jitter, shimmer, and Harmonic-to-Noise Ratio (HNR). These estimates and the formant analysis are not clinical measurements.

- **Live Charts:**  
  Displays a live, moving-window chart (last 100 samples) of pitch and formant data on the left side of the window.

- **Metrics Summary:**  
  Provides a fixed metrics summary box on the right side of the window that displays:
  - **Pitch Summary:** Minimum, maximum, and average pitch.
  - **Formant Summary:** Minimum, maximum, and average values for F1 and F2.
  - **Quality Metrics:** Jitter, shimmer, and HNR.
  
  The summary metrics are updated automatically at a configurable reset interval (default is 3 seconds) without affecting the live chart.

- **User Controls:**  
  - **Microphone Selection:** Choose from available audio devices.
  - **Microphone Level and Threshold:** The level meter shows the normalized RMS level on a logarithmic scale; the pink marker shows the threshold. Adjust the threshold in real time using the slider (higher values ignore quieter sounds). The microphone name and listening state are displayed above it. For guided calibration, press **Start**, then **Calibrate**, and stay quiet for three seconds. Review and accept or decline the suggested threshold. Calibration does not make a separate audio recording.
  - **Target Presets:** Enter a minimum and maximum pitch and press Enter, then use **Save** to name a personal preset. Select a saved preset to restore it, or use **Delete** to remove it. The target indicator shows whether the detected pitch is below, inside, or above the range.
  - **Reset Interval Field:** Set the interval (in seconds) at which summary metrics are recalculated and reset.
  - **Start/Stop Buttons:** Control the real-time audio processing.
  - **Record Audio:** Enable or disable recording during listening. Captured portions are saved together as a WAV file when you stop or close the application, even if recording was disabled before stopping. If saving fails, the application stays open and keeps the recording in memory so you can change the output directory and try again.
  - **Display Details:** Show or hide the current note, metrics summary, microphone settings, and target settings independently. Hiding target settings keeps the target band visible on the chart; the microphone status remains visible when its settings are hidden. These choices are saved between sessions.

The pitch detector rejects low-level and non-periodic sounds, but periodic background noise can still be detected as a pitch. Adjust the microphone threshold to suit your surroundings; setting it too high can also hide quiet speech.

- **Guidance Information:**  
  The tool displays pitch range guidelines for transgender voice training:
  - **Gender Neutral:** 150–185 Hz
  - **Vocal Feminization:** Aim ~180 Hz
  - **Vocal Masculinization:** Aim ~150 Hz
  - **Average Female Pitch:** ~225 Hz
  - **Average Male Pitch:** ~125 Hz
  
  *Note: These are general guidelines. Please consult a speech therapist for personalized advice.*

## Dependencies

- Python 3.x
- [NumPy](https://numpy.org/)
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/)
- [Matplotlib](https://matplotlib.org/)
- Tkinter (usually bundled with Python)

## Installation

To run from source code, open your terminal or command prompt in the project directory and install the required packages:

```python
pip install numpy pyaudio matplotlib
```

Le plus simple est d’utiliser PyInstaller sur Windows : il ne génère pas d’exécutable Windows directement depuis Linux.

Copie ton projet actuel sur Windows, sans le dossier venv Linux.
Installe Python 3.12 pour Windows, puis ouvre PowerShell dans le dossier du projet.
Lance :

```powershell
py -3.12 -m venv .venv-win
.\.venv-win\Scripts\python.exe -m pip install numpy pyaudio matplotlib pyinstaller
.\.venv-win\Scripts\python.exe -m PyInstaller --onefile --name VocalTuner VocalTuner.py
```

L’exécutable sera dans dist\VocalTuner.exe. Teste-le sur Windows avec ton micro : je n’ai pas pu vérifier cette compilation ici.

Il ouvrira aussi une console, utile pour voir les erreurs. Sans machine Windows, on peut également utiliser GitHub Actions pour le générer.