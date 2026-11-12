import os
import json
import wave
import datetime
import numpy as np
import pyaudio
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import math
import traceback
import queue
import threading
import time
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Audio configuration
CHUNK = 2048             # Frames per buffer
FORMAT = pyaudio.paInt16 # 16-bit int format
CHANNELS = 1             # Mono

# Frequency range limits for pitch detection (typical human voice)
MIN_FREQ = 80.0   # Hz
MAX_FREQ = 400.0  # Hz
MAX_FORMANT_BANDWIDTH = 400.0  # Hz

# Minimum normalized autocorrelation for a voiced sound.
MIN_PITCH_PERIODICITY = 0.55
CALIBRATION_SECONDS = 3

def freq_to_note_name(f):
    """Convert frequency (Hz) to the nearest musical note (A4=440Hz)."""
    try:
        if f is None or f <= 0 or math.isnan(f):
            return "0"
        n = round(69 + 12 * np.log2(f / 440.0))
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note_index = n % 12
        octave = n // 12 - 1
        return note_names[note_index] + str(octave)
    except Exception:
        traceback.print_exc()
        return "0"

def detect_pitch_and_hnr(signal, rate, sensitivity):
    """
    Uses autocorrelation to estimate pitch and HNR.
    Signal and RMS threshold use the normalized audio range [-1, 1].
    """
    try:
        rms = np.sqrt(np.mean(signal**2))
        if rms < sensitivity or rms == 0:
            return (None, np.nan)
        x = signal - np.mean(signal)
        corr = np.correlate(x, x, mode='full')[len(x)-1:]
        min_lag = int(rate / MAX_FREQ)
        max_lag = int(rate / MIN_FREQ)
        if max_lag > len(corr):
            max_lag = len(corr)
        if max_lag <= min_lag:
            return (None, np.nan)
        segment = corr[min_lag:max_lag]
        peak_index = np.argmax(segment) + min_lag
        overlap_energy = np.linalg.norm(x[:-peak_index]) * np.linalg.norm(x[peak_index:])
        if overlap_energy == 0:
            return (None, np.nan)
        periodicity = corr[peak_index] / overlap_energy
        if periodicity < MIN_PITCH_PERIODICITY:
            return (None, np.nan)
        pitch = rate / peak_index
        periodicity = min(float(periodicity), 1.0 - 1e-12)
        hnr = 10 * np.log10(periodicity / (1.0 - periodicity))
        return (pitch, hnr)
    except Exception:
        traceback.print_exc()
        return (None, np.nan)

def levinson_durbin(r, order):
    """Robust Levinson-Durbin algorithm for LPC estimation."""
    try:
        a = np.zeros(order+1)
        e = r[0]
        a[0] = 1.0
        min_error = np.finfo(float).eps * abs(r[0])
        for i in range(1, order+1):
            if e <= min_error:
                break
            acc = 0.0
            for j in range(1, i):
                acc += a[j] * r[i-j]
            k = (r[i] - acc) / e
            previous = a[1:i].copy()
            a[1:i] = previous - k * previous[::-1]
            a[i] = k
            e *= (1 - k * k)
        return a
    except Exception:
        traceback.print_exc()
        return np.zeros(order+1)

def compute_formants(signal, rate, lpc_order=None):
    """
    Compute up to three formant frequencies using LPC.
    Applies pre-emphasis and a Hamming window.
    Returns a list of three values (F1, F2, F3) in Hz; if fewer are found, pads with np.nan.
    Only stable resonances below 4000 Hz with bandwidth under 400 Hz are considered.
    """
    try:
        if lpc_order is None:
            lpc_order = min(64, max(12, round(rate / 1000) + 2))
        pre_emphasis = 0.97
        emphasized = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
        windowed = emphasized * np.hamming(len(emphasized))
        corr = np.correlate(windowed, windowed, mode='full')[len(windowed)-1:]
        if len(corr) < lpc_order + 1:
            return [np.nan, np.nan, np.nan]
        r = corr[:lpc_order+1]
        if r[0] <= np.finfo(float).tiny:
            return [np.nan, np.nan, np.nan]
        A = levinson_durbin(r, lpc_order)
        A[1:] *= -1
        roots = np.roots(A)
        roots = [root for root in roots if np.imag(root) > 0 and 0 < abs(root) < 1
                 and -rate / np.pi * np.log(abs(root)) < MAX_FORMANT_BANDWIDTH]
        angles = np.arctan2(np.imag(roots), np.real(roots))
        freqs = angles * (rate / (2 * np.pi))
        formants = sorted([f for f in freqs if 0 < f < 4000])
        while len(formants) < 3:
            formants.append(np.nan)
        return formants[:3]
    except Exception:
        traceback.print_exc()
        return [np.nan, np.nan, np.nan]

###############################################################################
# Audio capture using PyAudio callback mode with a result queue
###############################################################################

class AudioStream:
    def __init__(self, device_index, rate, record_audio, sensitivity, result_queue):
        self.device_index = device_index
        self.rate = rate
        self.record_audio = record_audio
        self.sensitivity = sensitivity
        self.result_queue = result_queue  # Queue for sending results to main thread
        self.recorded_frames = []
        self.calibration_lock = threading.Lock()
        self.calibration_samples = None
        self.calibration_deadline = None
        self.closed = False
        self.pa = pyaudio.PyAudio()
        try:
            self.stream = self.pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=self.rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=CHUNK,
                stream_callback=self.callback
            )
            self.stream.start_stream()
            self.sample_width = self.pa.get_sample_size(FORMAT)
            print(f"Audio stream opened on device {self.device_index} at {self.rate} Hz (callback mode).")
        except Exception as e:
            print("Error opening stream:", e)
            traceback.print_exc()
            self.stream = None

    def callback(self, in_data, frame_count, time_info, status_flags):
        try:
            if not in_data:
                return (in_data, pyaudio.paContinue)
            if self.record_audio:
                self.recorded_frames.append(in_data)
            audio_data = np.frombuffer(in_data, np.int16).astype(np.float32) / 32768.0
            rms = np.sqrt(np.mean(audio_data**2))
            amplitude = float(rms)
            with self.calibration_lock:
                if (self.calibration_samples is not None
                        and self.calibration_deadline is not None
                        and time.monotonic() <= self.calibration_deadline):
                    self.calibration_samples.append((amplitude, len(audio_data)))
            pitch_val, hnr_val = detect_pitch_and_hnr(audio_data, self.rate, self.sensitivity)
            if pitch_val is None:
                note = "0"
                formants_val = [np.nan, np.nan, np.nan]
            else:
                note = freq_to_note_name(pitch_val).upper()
                formants_val = compute_formants(audio_data, self.rate)
            self.result_queue.put((note, pitch_val, formants_val, hnr_val, amplitude))
        except Exception as e:
            print("Error in callback:")
            traceback.print_exc()
            self.result_queue.put(("N/A", 0, [np.nan, np.nan, np.nan], np.nan, 0.0))
        return (in_data, pyaudio.paContinue)

    def begin_calibration(self):
        """Collect RMS levels from the next three seconds of captured audio."""
        with self.calibration_lock:
            self.calibration_samples = []
            self.calibration_deadline = time.monotonic() + CALIBRATION_SECONDS

    def end_calibration(self):
        """Stop collecting and return the captured RMS levels and frame counts."""
        with self.calibration_lock:
            samples = self.calibration_samples
            self.calibration_samples = None
            self.calibration_deadline = None
        return samples

    def stop(self):
        try:
            if self.closed:
                return True
            if self.stream is not None:
                self.stream.stop_stream()
                self.stream.close()
            self.pa.terminate()
            self.closed = True
            print("Audio stream stopped.")
            return True
        except Exception as e:
            print("Error stopping audio stream:")
            traceback.print_exc()
            return False

###############################################################################
# UI and Application
###############################################################################

class App:
    def __init__(self, root):
        self.root = root
        root.title("Transgender Voice Training Tool")
        
        self.BG_PRIMARY = "#f7eedf"
        self.BG_SECONDARY = "#fff9f3"
        self.BG_TERTIARY = "#e5f2f7"
        self.TEXT_PRIMARY = "#382b39"
        self.TEXT_SECONDARY = "#62515d"
        self.ACCENT_PRIMARY = "#b9dfea"
        self.ACCENT_SECONDARY = "#f3c1d2"
        self.COLOR_PITCH = "#236990"
        self.COLOR_F1 = "#ad345e"
        self.COLOR_F2 = "#296b8e"
        self.COLOR_F3 = "#73529b"
        self.COLOR_TARGET_ZONE = "#ef9fbb"

        root.config(bg=self.BG_PRIMARY)
        
        # Setup default output directory
        # Use %APPDATA% on Windows, ~/.config on Linux/Mac
        appdata = os.getenv("APPDATA")
        if appdata:
            self.default_output = os.path.join(appdata, "VocalTuner")
        else:
            self.default_output = os.path.join(os.path.expanduser("~"), ".config", "VocalTuner")
        os.makedirs(self.default_output, exist_ok=True)
        self.output_dir = self.default_output
        self.config_path = os.path.join(self.output_dir, "config.json")
        self.metrics_log_path = os.path.join(self.output_dir, "metrics.log")
        # Load configuration
        self.selected_mic = tk.StringVar()
        self.recording = tk.BooleanVar(value=False)
        self.sensitivity = 0.005  # Normalized RMS threshold
        self.reset_interval_value = 3.0  # default reset interval (seconds)
        self.audio_stream = None
        self.active_mic_name = None
        self.current_pitch = None
        self.current_amplitude = 0.0
        self.sensitivity_save_id = None
        self.calibration_after_id = None
        self.calibration_stream = None
        self.target_presets = {}
        self.selected_preset = tk.StringVar(value="")

        
        # Display mode and target zone settings (create before load_config)
        self.display_mode = tk.StringVar(value="pitch_only")  # pitch_only, formants_only, both
        self.show_note = tk.BooleanVar(value=True)
        self.show_summary = tk.BooleanVar(value=True)
        self.show_mic_settings = tk.BooleanVar(value=True)
        self.show_target_settings = tk.BooleanVar(value=True)
        self.target_zone_min = tk.DoubleVar(value=200.0)
        self.target_zone_max = tk.DoubleVar(value=250.0)
        
        self.load_config()
        self.target_limits = (self.target_zone_min.get(), self.target_zone_max.get())
        
        # Create a thread-safe queue for audio results
        self.result_queue = queue.Queue()
        # Store last valid tone and pitch
        self.last_tone = "0"
        self.last_pitch = 0
        # Initialize formant smoothing variables (one for each formant)
        self.formant_smoothing_factor = 0.1
        self.last_formants = [0, 0, 0]
        
        # Track polling ID for cleanup
        self.poll_queue_id = None
        
        
        # Menus
        self.menubar = tk.Menu(root, bg=self.BG_SECONDARY, fg=self.TEXT_PRIMARY, activebackground=self.ACCENT_PRIMARY, activeforeground=self.TEXT_PRIMARY)
        root.config(menu=self.menubar)
        file_menu = tk.Menu(self.menubar, tearoff=0, bg=self.BG_SECONDARY, fg=self.TEXT_PRIMARY, activebackground=self.ACCENT_PRIMARY, activeforeground=self.TEXT_PRIMARY)
        file_menu.add_command(label="Mic Settings", command=self.open_mic_settings)
        file_menu.add_command(label="Set Output Directory", command=self.set_output_directory)
        file_menu.add_checkbutton(label="Record Audio", variable=self.recording,
                                  command=self.on_recording_changed)
        self.menubar.add_cascade(label="File", menu=file_menu)
        settings_menu = tk.Menu(self.menubar, tearoff=0, bg=self.BG_SECONDARY, fg=self.TEXT_PRIMARY, activebackground=self.ACCENT_PRIMARY, activeforeground=self.TEXT_PRIMARY)
        settings_menu.add_command(label="Set Mic Sensitivity", command=self.set_mic_sensitivity)
        settings_menu.add_command(label="Set Reset Interval", command=self.set_reset_interval)
        self.menubar.add_cascade(label="Settings", menu=settings_menu)
        ranges_menu = tk.Menu(self.menubar, tearoff=0, bg=self.BG_SECONDARY, fg=self.TEXT_PRIMARY, activebackground=self.ACCENT_PRIMARY, activeforeground=self.TEXT_PRIMARY)
        ranges_menu.add_command(label="View Transgender Pitch Ranges", command=self.show_ranges)
        self.menubar.add_cascade(label="Ranges", menu=ranges_menu)
        
        # Display controls frame with tabs and settings
        self.controls_frame = tk.Frame(root, bg=self.BG_SECONDARY)
        self.controls_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Display mode selection (tabs)
        mode_frame = tk.LabelFrame(self.controls_frame, text="Display Mode", padx=10, pady=5, 
                                    bg=self.BG_SECONDARY, fg=self.TEXT_PRIMARY, font=("Helvetica", 10, "bold"))
        mode_frame.pack(side=tk.LEFT, padx=5, fill=tk.X)
        
        tk.Radiobutton(mode_frame, text="Pitch Only", variable=self.display_mode, 
                       value="pitch_only", command=self.on_display_mode_changed,
                       bg=self.BG_SECONDARY, fg=self.TEXT_PRIMARY, selectcolor=self.BG_TERTIARY, 
                       activebackground=self.BG_SECONDARY, activeforeground=self.TEXT_PRIMARY).pack(anchor=tk.W)
        tk.Radiobutton(mode_frame, text="Formants Only", variable=self.display_mode, 
                       value="formants_only", command=self.on_display_mode_changed,
                       bg=self.BG_SECONDARY, fg=self.TEXT_PRIMARY, selectcolor=self.BG_TERTIARY,
                       activebackground=self.BG_SECONDARY, activeforeground=self.TEXT_PRIMARY).pack(anchor=tk.W)
        tk.Radiobutton(mode_frame, text="Pitch + Formants", variable=self.display_mode, 
                       value="both", command=self.on_display_mode_changed,
                       bg=self.BG_SECONDARY, fg=self.TEXT_PRIMARY, selectcolor=self.BG_TERTIARY,
                       activebackground=self.BG_SECONDARY, activeforeground=self.TEXT_PRIMARY).pack(anchor=tk.W)

        details_frame = tk.LabelFrame(self.controls_frame, text="Display Details", padx=10, pady=5,
                                      bg=self.BG_SECONDARY, fg=self.TEXT_PRIMARY, font=("Helvetica", 10, "bold"))
        details_frame.pack(side=tk.LEFT, padx=5, fill=tk.X)
        tk.Checkbutton(details_frame, text="Current Note", variable=self.show_note,
                       command=self.on_display_details_changed, bg=self.BG_SECONDARY,
                       fg=self.TEXT_PRIMARY, selectcolor=self.BG_TERTIARY,
                       activebackground=self.BG_SECONDARY, activeforeground=self.TEXT_PRIMARY).pack(anchor=tk.W)
        tk.Checkbutton(details_frame, text="Metrics Summary", variable=self.show_summary,
                       command=self.on_display_details_changed, bg=self.BG_SECONDARY,
                       fg=self.TEXT_PRIMARY, selectcolor=self.BG_TERTIARY,
                       activebackground=self.BG_SECONDARY, activeforeground=self.TEXT_PRIMARY).pack(anchor=tk.W)
        tk.Checkbutton(details_frame, text="Mic Settings", variable=self.show_mic_settings,
                       command=self.on_display_details_changed, bg=self.BG_SECONDARY,
                       fg=self.TEXT_PRIMARY, selectcolor=self.BG_TERTIARY,
                       activebackground=self.BG_SECONDARY, activeforeground=self.TEXT_PRIMARY).pack(anchor=tk.W)
        tk.Checkbutton(details_frame, text="Target Settings", variable=self.show_target_settings,
                       command=self.on_display_details_changed, bg=self.BG_SECONDARY,
                       fg=self.TEXT_PRIMARY, selectcolor=self.BG_TERTIARY,
                       activebackground=self.BG_SECONDARY, activeforeground=self.TEXT_PRIMARY).pack(anchor=tk.W)
        
        # Target zone settings
        self.target_frame = tk.LabelFrame(self.controls_frame, text="Target Zone (Hz)", padx=10, pady=5,
                                          bg=self.BG_SECONDARY, fg=self.TEXT_PRIMARY, font=("Helvetica", 10, "bold"))
        target_frame = self.target_frame
        target_frame.pack(side=tk.LEFT, padx=5, fill=tk.X)
        
        tk.Label(target_frame, text="Min:", bg=self.BG_SECONDARY, fg=self.TEXT_PRIMARY).pack(side=tk.LEFT, padx=2)
        self.target_min_entry = tk.Entry(target_frame, textvariable=self.target_zone_min, width=8,
                                         bg=self.BG_TERTIARY, fg=self.TEXT_PRIMARY, insertbackground=self.TEXT_PRIMARY)
        self.target_min_entry.pack(side=tk.LEFT, padx=2)
        self.target_min_entry.bind("<Return>", lambda e: self.update_target_zone())
        
        tk.Label(target_frame, text="Max:", bg=self.BG_SECONDARY, fg=self.TEXT_PRIMARY).pack(side=tk.LEFT, padx=2)
        self.target_max_entry = tk.Entry(target_frame, textvariable=self.target_zone_max, width=8,
                                         bg=self.BG_TERTIARY, fg=self.TEXT_PRIMARY, insertbackground=self.TEXT_PRIMARY)
        self.target_max_entry.pack(side=tk.LEFT, padx=2)
        self.target_max_entry.bind("<Return>", lambda e: self.update_target_zone())

        self.preset_frame = tk.LabelFrame(self.controls_frame, text="Target Presets", padx=10, pady=5,
                                          bg=self.BG_SECONDARY, fg=self.TEXT_PRIMARY, font=("Helvetica", 10, "bold"))
        preset_frame = self.preset_frame
        preset_frame.pack(side=tk.LEFT, padx=5, fill=tk.X)
        self.preset_menu = tk.OptionMenu(preset_frame, self.selected_preset, "")
        self.preset_menu.config(width=12, bg=self.BG_TERTIARY, fg=self.TEXT_PRIMARY,
                                activebackground=self.ACCENT_PRIMARY, activeforeground=self.TEXT_PRIMARY)
        self.preset_menu.pack(fill=tk.X)
        tk.Button(preset_frame, text="Save", command=self.save_target_preset,
                  bg=self.ACCENT_PRIMARY, fg=self.TEXT_PRIMARY).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(preset_frame, text="Delete", command=self.delete_target_preset,
                  bg=self.ACCENT_SECONDARY, fg=self.TEXT_PRIMARY).pack(side=tk.LEFT, padx=2, pady=2)
        self.refresh_target_presets()

        self.target_status_label = tk.Label(root, text="Target: waiting for voice",
                                            bg=self.BG_PRIMARY, fg=self.TEXT_SECONDARY)
        self.target_status_label.pack(padx=10, pady=(2, 0))

        mic_frame = tk.Frame(root, bg=self.BG_PRIMARY)
        mic_frame.pack(fill=tk.X, padx=10, pady=2)
        self.mic_status_label = tk.Label(mic_frame, bg=self.BG_PRIMARY, fg=self.TEXT_PRIMARY)
        self.mic_status_label.pack(anchor=tk.W)
        self.selected_mic.trace_add("write", self.update_mic_status)
        self.update_mic_status()
        self.level_frame = tk.Frame(mic_frame, bg=self.BG_PRIMARY)
        level_frame = self.level_frame
        level_frame.pack(fill=tk.X)
        tk.Label(level_frame, text="Mic level:", bg=self.BG_PRIMARY, fg=self.TEXT_PRIMARY).pack(side=tk.LEFT)
        self.level_meter = tk.Canvas(level_frame, width=160, height=14, bg=self.BG_TERTIARY,
                                     highlightthickness=1, highlightbackground=self.TEXT_SECONDARY)
        self.level_meter.pack(side=tk.LEFT, padx=5)
        self.level_bar = self.level_meter.create_rectangle(0, 0, 0, 14, fill=self.COLOR_PITCH, outline="")
        self.threshold_marker = self.level_meter.create_line(0, 0, 0, 14, fill=self.COLOR_F1, width=2)
        self.level_label = tk.Label(level_frame, text="0.0%", bg=self.BG_PRIMARY, fg=self.TEXT_PRIMARY)
        self.level_label.pack(side=tk.LEFT)
        tk.Label(level_frame, text="Threshold:", bg=self.BG_PRIMARY, fg=self.TEXT_PRIMARY).pack(side=tk.LEFT, padx=(10, 0))
        self.sensitivity_scale = tk.Scale(level_frame, from_=0, to=100, resolution=0.01,
                                          orient=tk.HORIZONTAL, length=200, label="RMS %",
                                          bg=self.BG_PRIMARY, fg=self.TEXT_PRIMARY,
                                          activebackground=self.ACCENT_SECONDARY,
                                          highlightthickness=0, command=self.on_sensitivity_changed)
        self.sensitivity_scale.set(self.sensitivity * 100)
        self.sensitivity_scale.pack(side=tk.LEFT)
        self.calibration_button = tk.Button(level_frame, text="Calibrate", command=self.start_calibration,
                                            bg=self.ACCENT_PRIMARY, fg=self.TEXT_PRIMARY)
        self.calibration_button.pack(side=tk.LEFT, padx=5)
        self.calibration_label = tk.Label(level_frame, text="", bg=self.BG_PRIMARY, fg=self.TEXT_SECONDARY)
        self.calibration_label.pack(side=tk.LEFT)
        
        # Start/Stop Buttons
        btn_frame = tk.Frame(root, bg=self.BG_PRIMARY)
        btn_frame.pack(padx=10, pady=5)
        self.start_button = tk.Button(btn_frame, text="Start", command=self.start_listening,
                                       bg=self.ACCENT_PRIMARY, fg=self.TEXT_PRIMARY,
                                       activebackground=self.ACCENT_SECONDARY, activeforeground=self.TEXT_PRIMARY,
                                       font=("Helvetica", 10, "bold"), padx=15, pady=5)
        self.start_button.pack(side=tk.LEFT, padx=10)
        self.stop_button = tk.Button(btn_frame, text="Stop", command=self.stop_listening, state=tk.DISABLED,
                                      bg=self.ACCENT_SECONDARY, fg=self.TEXT_PRIMARY,
                                      activebackground=self.ACCENT_PRIMARY, activeforeground=self.TEXT_PRIMARY,
                                      font=("Helvetica", 10, "bold"), padx=15, pady=5)
        self.stop_button.pack(side=tk.LEFT, padx=10)
        
        # Tone display area (smaller)
        self.tone_frame = tk.Frame(root, width=300, height=80, bd=2, relief=tk.SUNKEN, bg=self.BG_SECONDARY)
        self.tone_frame.pack_propagate(False)
        self.tone_label = tk.Label(self.tone_frame, text="0", font=("Helvetica", 48, "bold"), 
                                   bg=self.BG_SECONDARY, fg=self.COLOR_F1)
        self.tone_label.pack(expand=True)
        
        self.quality_label = tk.Label(root, text="Jitter: N/A | Shimmer: N/A | HNR: N/A", 
                                      font=("Helvetica", 10), bg=self.BG_PRIMARY, fg=self.TEXT_SECONDARY)
        self.metrics_frame = tk.Frame(root, bd=2, relief=tk.SUNKEN, bg=self.BG_SECONDARY)
        self.pitch_summary_label = tk.Label(self.metrics_frame, text="Pitch Summary: Low: N/A | High: N/A | Avg: N/A", font=("Helvetica", 10), bg=self.BG_SECONDARY, fg=self.TEXT_PRIMARY)
        self.pitch_summary_label.pack(padx=10, pady=2)
        self.formant_summary_label = tk.Label(self.metrics_frame, text="Formant Summary: F1: N/A | F2: N/A | F3: N/A", font=("Helvetica", 10), bg=self.BG_SECONDARY, fg=self.TEXT_PRIMARY)
        self.formant_summary_label.pack(padx=10, pady=2)
        self.pitch_history = []
        self.amplitude_history = []
        self.formant_histories = [[], [], []]  # For F1, F2, F3
        self.metrics_pitch_history = []
        self.metrics_formant_histories = [[], [], []]
        
        # Create a container for the chart that we'll rebuild dynamically
        self.chart_container = tk.Frame(root, bg=self.BG_PRIMARY)
        self.chart_container.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        self.update_display_details()
        
        # Initialize the figure and axes
        self.fig = None
        self.ax_pitch = None
        self.ax_formants = None
        self.pitch_line = None
        self.formant_lines = []
        self.pitch_target_patch = None
        self.canvas = None
        
        # Build initial chart based on display mode
        self.rebuild_chart()
        
        self.reset_timer = None
        self.schedule_reset()
        self.poll_queue()

    def load_config(self):
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                self.selected_mic.set(config.get("mic", ""))
                self.recording.set(config.get("record_audio", False))
                self.sensitivity = config.get("mic_threshold")
                if self.sensitivity is None:
                    legacy_sensitivity = config.get("sensitivity", 0.01)
                    self.sensitivity = (legacy_sensitivity * 0.5
                                        if isinstance(legacy_sensitivity, (int, float))
                                        else legacy_sensitivity)
                if (not isinstance(self.sensitivity, (int, float))
                        or not math.isfinite(self.sensitivity) or not 0 <= self.sensitivity <= 1):
                    print("Invalid microphone threshold in configuration; using 0.5%.")
                    self.sensitivity = 0.005
                self.reset_interval_value = config.get("reset_interval", 3.0)
                out_dir = config.get("output_dir", "")
                if out_dir:
                    self.output_dir = out_dir
                    os.makedirs(self.output_dir, exist_ok=True)
                    self.metrics_log_path = os.path.join(self.output_dir, "metrics.log")
                
                # Load display preferences
                self.display_mode.set(config.get("display_mode", "pitch_only"))
                self.show_note.set(config.get("show_note", True))
                self.show_summary.set(config.get("show_summary", True))
                self.show_mic_settings.set(config.get("show_mic_settings", True))
                self.show_target_settings.set(config.get("show_target_settings", True))
                self.target_zone_min.set(config.get("target_zone_min", 200.0))
                self.target_zone_max.set(config.get("target_zone_max", 250.0))
                self.target_presets = {}
                presets = config.get("target_presets", {})
                if isinstance(presets, dict):
                    for name, limits in presets.items():
                        if (isinstance(name, str) and name and isinstance(limits, list)
                                and len(limits) == 2 and all(isinstance(value, (int, float)) for value in limits)
                                and self.valid_target_range(*limits)):
                            self.target_presets[name] = limits
                        else:
                            print(f"Ignoring invalid target preset: {name!r}")
                else:
                    print("Invalid target presets in configuration.")
            
            # Auto-detect a microphone if none is configured
            if not self.selected_mic.get():
                devices = self.get_mic_devices()
                if devices:
                    best_mic_name = devices[0][1]
                    self.selected_mic.set(best_mic_name)
                    print(f"Auto-detected microphone: {best_mic_name}")
        except Exception:
            traceback.print_exc()

    def save_config(self):
        try:
            config = {
                "mic": self.selected_mic.get(),
                "record_audio": self.recording.get(),
                "sensitivity": self.sensitivity * 2,
                "mic_threshold": self.sensitivity,
                "reset_interval": self.reset_interval_value,
                "output_dir": self.output_dir,
                "display_mode": self.display_mode.get(),
                "show_note": self.show_note.get(),
                "show_summary": self.show_summary.get(),
                "show_mic_settings": self.show_mic_settings.get(),
                "show_target_settings": self.show_target_settings.get(),
                "target_zone_min": self.target_limits[0],
                "target_zone_max": self.target_limits[1],
                "target_presets": self.target_presets
            }
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f)
            print("Config saved.")
        except Exception:
            traceback.print_exc()

    def get_mic_devices(self):
        """Get list of available input devices with more intelligent filtering."""
        try:
            pa = pyaudio.PyAudio()
            devices = []
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0:
                    name = info.get("name", "Unknown")
                    devices.append((i, name))
            pa.terminate()
            
            # Prefer hardware devices over virtual ones
            # Prioritize: USB devices > Analog > others
            def device_priority(device):
                idx, name = device
                name_lower = name.lower()
                if "usb" in name_lower:
                    return 0  # Highest priority
                elif "analog" in name_lower or "adc" in name_lower:
                    return 1
                elif "default" in name_lower or "pipewire" in name_lower:
                    return 2
                else:
                    return 3  # Lowest priority (virtual devices like dmix, sysdefault)
            
            devices.sort(key=device_priority)
            return devices
        except Exception:
            traceback.print_exc()
            return []

    def rebuild_chart(self):
        """Rebuild the matplotlib figure based on current display mode."""
        # Clear old chart
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        if self.fig:
            plt.close(self.fig)
        self.pitch_line = None
        self.pitch_target_patch = None
        self.formant_lines = []
        
        mode = self.display_mode.get()
        
        # Create new figure with appropriate layout
        if mode == "pitch_only":
            self.fig, self.ax_pitch = plt.subplots(1, 1, figsize=(6, 4))
            self.ax_formants = None
        elif mode == "formants_only":
            self.fig, self.ax_formants = plt.subplots(1, 1, figsize=(6, 4))
            self.ax_pitch = None
        else:  # both
            self.fig, (self.ax_pitch, self.ax_formants) = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
        
        self.fig.patch.set_facecolor(self.BG_SECONDARY)
        for ax in [self.ax_pitch, self.ax_formants]:
            if ax:
                ax.set_facecolor(self.BG_SECONDARY)
                ax.tick_params(colors=self.TEXT_PRIMARY)
                ax.spines['bottom'].set_color(self.TEXT_SECONDARY)
                ax.spines['left'].set_color(self.TEXT_SECONDARY)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_color(self.TEXT_PRIMARY)
        
        # Setup pitch axis if visible
        if self.ax_pitch:
            self.ax_pitch.set_title("Live Pitch History", fontsize=12, fontweight='bold', color=self.TEXT_PRIMARY)
            self.ax_pitch.set_ylabel("Frequency (Hz)", color=self.TEXT_PRIMARY)
            self.ax_pitch.set_ylim(0, 400)  # Fixed range from 0 to 400 Hz
            self.ax_pitch.set_xlim(0, 100)  # Fixed x-axis range (100 frames max)
            self.pitch_line, = self.ax_pitch.plot(range(len(self.pitch_history)), self.pitch_history,
                                                  color=self.COLOR_PITCH, label="Pitch", linewidth=2)
            
            # Add target zone as a colored rectangle patch
            try:
                min_hz, max_hz = self.target_limits
                self.pitch_target_patch = plt.Rectangle((0, min_hz), 100, max_hz - min_hz, 
                                                         alpha=0.2, color=self.COLOR_TARGET_ZONE, label='Target Zone')
                self.ax_pitch.add_patch(self.pitch_target_patch)
            except Exception:
                pass
            
            legend = self.ax_pitch.legend(loc='upper right', facecolor=self.BG_SECONDARY, edgecolor=self.TEXT_SECONDARY, labelcolor=self.TEXT_PRIMARY)
            self.ax_pitch.grid(True, alpha=0.15, color=self.TEXT_SECONDARY)
            self.ax_pitch.set_xlabel("Time (frames)", color=self.TEXT_PRIMARY)
        
        # Setup formants axis if visible
        if self.ax_formants:
            self.ax_formants.set_title("Live Formant History (F1, F2, F3)", fontsize=12, fontweight='bold', color=self.TEXT_PRIMARY)
            self.ax_formants.set_xlabel("Time (frames)", color=self.TEXT_PRIMARY)
            self.ax_formants.set_ylabel("Frequency (Hz)", color=self.TEXT_PRIMARY)
            
            colors = [self.COLOR_F1, self.COLOR_F2, self.COLOR_F3]
            for i in range(3):
                history = self.formant_histories[i]
                line, = self.ax_formants.plot(range(len(history)), history,
                                               color=colors[i], label=f"F{i+1}", linewidth=2)
                self.formant_lines.append(line)
            self.ax_formants.relim()
            self.ax_formants.autoscale_view()
            
            legend = self.ax_formants.legend(loc='upper right', facecolor=self.BG_SECONDARY, edgecolor=self.TEXT_SECONDARY, labelcolor=self.TEXT_PRIMARY)
            self.ax_formants.grid(True, alpha=0.15, color=self.TEXT_SECONDARY)
        
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()

    def on_display_mode_changed(self):
        """Handle display mode change."""
        self.rebuild_chart()
        self.save_config()

    def update_display_details(self):
        """Show selected details before the chart."""
        self.target_frame.pack_forget()
        self.preset_frame.pack_forget()
        if self.show_target_settings.get():
            self.target_frame.pack(side=tk.LEFT, padx=5, fill=tk.X)
            self.preset_frame.pack(side=tk.LEFT, padx=5, fill=tk.X)
        self.level_frame.pack_forget()
        if self.show_mic_settings.get():
            self.level_frame.pack(fill=tk.X)
        self.tone_frame.pack_forget()
        self.quality_label.pack_forget()
        self.metrics_frame.pack_forget()
        if self.show_note.get():
            self.tone_frame.pack(before=self.chart_container, padx=10, pady=5)
        if self.show_summary.get():
            self.quality_label.pack(before=self.chart_container, padx=10, pady=5)
            self.metrics_frame.pack(before=self.chart_container, padx=10, pady=5, fill=tk.X)

    def on_display_details_changed(self):
        """Apply and save the selected details."""
        self.update_display_details()
        self.save_config()

    @staticmethod
    def valid_target_range(min_hz, max_hz):
        """Check that the range fits the pitch chart."""
        return math.isfinite(min_hz) and math.isfinite(max_hz) and 0 <= min_hz < max_hz <= 400

    def update_target_zone(self):
        """Update target zone on chart when entries change."""
        try:
            min_hz = self.target_zone_min.get()
            max_hz = self.target_zone_max.get()
            if not self.valid_target_range(min_hz, max_hz):
                raise ValueError("Invalid range")
            
            self.target_limits = (min_hz, max_hz)
            name = self.selected_preset.get()
            if name in self.target_presets and tuple(self.target_presets[name]) != self.target_limits:
                self.selected_preset.set("")
            if self.pitch_target_patch and self.ax_pitch:
                self.pitch_target_patch.set_height(max_hz - min_hz)
                self.pitch_target_patch.set_y(min_hz)
                self.canvas.draw()
            self.update_target_status()
            self.save_config()
            return True
        except (ValueError, tk.TclError):
            messagebox.showerror("Target Zone", "Enter a valid range between 0 and 400 Hz (min < max).")
            return False

    def refresh_target_presets(self):
        """Update the list of saved target ranges."""
        menu = self.preset_menu["menu"]
        menu.delete(0, tk.END)
        for name in sorted(self.target_presets):
            menu.add_command(label=name, command=lambda name=name: self.apply_target_preset(name))
        if self.selected_preset.get() not in self.target_presets:
            self.selected_preset.set("")

    def apply_target_preset(self, name):
        """Apply one saved target range."""
        min_hz, max_hz = self.target_presets[name]
        self.selected_preset.set(name)
        self.target_zone_min.set(min_hz)
        self.target_zone_max.set(max_hz)
        self.update_target_zone()

    def save_target_preset(self):
        """Save the entered range under a user-provided name."""
        if not self.update_target_zone():
            return
        name = simpledialog.askstring("Save Target Preset", "Name for this target range:", parent=self.root)
        if name is None:
            return
        name = name.strip()
        if not name:
            messagebox.showerror("Target Preset", "Enter a name for the preset.")
            return
        if name in self.target_presets and not messagebox.askyesno(
                "Target Preset", f"Replace the preset '{name}'?"):
            return
        self.target_presets[name] = list(self.target_limits)
        self.selected_preset.set(name)
        self.refresh_target_presets()
        self.save_config()

    def delete_target_preset(self):
        """Delete the selected target preset."""
        name = self.selected_preset.get()
        if name not in self.target_presets:
            messagebox.showinfo("Target Preset", "Select a preset to delete.")
            return
        if messagebox.askyesno("Target Preset", f"Delete the preset '{name}'?"):
            del self.target_presets[name]
            self.refresh_target_presets()
            self.save_config()

    def update_target_status(self):
        """Show whether the current detected pitch is in the target range."""
        if self.current_pitch is None:
            text = "Target: waiting for voice"
        elif self.current_pitch < self.target_limits[0]:
            text = "Target: below range"
        elif self.current_pitch > self.target_limits[1]:
            text = "Target: above range"
        else:
            text = "Target: in range"
        self.target_status_label.config(text=text)

    def update_mic_status(self, *_args):
        """Show the active microphone and listening state."""
        selected = self.selected_mic.get() or "Not selected"
        if self.active_mic_name:
            text = f"Listening: {self.active_mic_name}"
            if selected != self.active_mic_name:
                text += f" | Next start: {selected}"
        else:
            text = f"Stopped | Microphone: {selected}"
        self.mic_status_label.config(text=text)

    @staticmethod
    def meter_position(amplitude):
        """Map normalized RMS onto a -60 to 0 dB meter."""
        if amplitude <= 0:
            return 0
        return round(160 * max(0, min(1, (20 * math.log10(amplitude) + 60) / 60)))

    def update_mic_level(self, amplitude):
        """Update the microphone level meter and its threshold marker."""
        self.current_amplitude = amplitude
        self.level_meter.coords(self.level_bar, 0, 0, self.meter_position(amplitude), 14)
        marker = self.meter_position(self.sensitivity)
        self.level_meter.coords(self.threshold_marker, marker, 0, marker, 14)
        self.level_label.config(text=f"{amplitude * 100:.1f}%")

    def on_sensitivity_changed(self, value):
        """Apply the normalized RMS threshold to the running stream."""
        self.sensitivity = float(value) / 100
        if self.audio_stream is not None:
            self.audio_stream.sensitivity = self.sensitivity
        self.update_mic_level(self.current_amplitude)
        if self.sensitivity_save_id is not None:
            self.root.after_cancel(self.sensitivity_save_id)
        self.sensitivity_save_id = self.root.after(500, self.save_sensitivity)

    def on_recording_changed(self):
        """Apply the recording option to the active stream."""
        if self.audio_stream is not None:
            self.audio_stream.record_audio = self.recording.get()
        self.save_config()

    def save_sensitivity(self):
        """Persist a settled threshold adjustment."""
        self.sensitivity_save_id = None
        self.save_config()

    def start_calibration(self):
        """Measure background audio while the microphone is running."""
        if self.audio_stream is None or self.active_mic_name is None:
            messagebox.showinfo("Microphone Calibration", "Press Start before calibrating the microphone.")
            return
        if self.calibration_after_id is not None:
            return
        self.calibration_stream = self.audio_stream
        self.calibration_stream.begin_calibration()
        self.calibration_button.config(state=tk.DISABLED)
        self.calibration_label.config(text="Keep quiet for 3 seconds...")
        self.calibration_after_id = self.root.after(CALIBRATION_SECONDS * 1000, self.finish_calibration)

    def cancel_calibration(self):
        """Discard an unfinished calibration."""
        if self.calibration_after_id is not None:
            self.root.after_cancel(self.calibration_after_id)
            self.calibration_after_id = None
            self.calibration_stream.end_calibration()
            self.calibration_stream = None
            self.calibration_button.config(state=tk.NORMAL)
            self.calibration_label.config(text="Calibration cancelled.")

    def finish_calibration(self):
        """Propose an RMS threshold from the collected background audio."""
        self.calibration_after_id = None
        stream = self.calibration_stream
        samples = stream.end_calibration()
        self.calibration_stream = None
        self.calibration_button.config(state=tk.NORMAL)
        if not samples or sum(frames for _, frames in samples) < stream.rate * CALIBRATION_SECONDS * 0.6:
            self.calibration_label.config(text="Calibration failed.")
            messagebox.showwarning("Microphone Calibration", "Not enough audio was received. Please try again.")
            return
        background = float(np.percentile([level for level, _ in samples], 95))
        proposed = round(max(0.001, background * 1.5 + 0.001), 4)
        if proposed >= 1:
            self.calibration_label.config(text="Calibration failed.")
            messagebox.showwarning("Microphone Calibration", "Background noise is too loud to calibrate.")
            return
        self.calibration_label.config(text="Calibration complete.")
        if messagebox.askyesno(
                "Microphone Calibration",
                f"Background level: {background * 100:.2f}% RMS.\n"
                f"Suggested threshold: {proposed * 100:.2f}% RMS.\n\n"
                "Apply this threshold? Quiet speech may require a lower value."):
            if self.audio_stream is not stream or self.active_mic_name is None:
                messagebox.showwarning("Microphone Calibration", "The microphone changed; please calibrate again.")
                return
            self.sensitivity_scale.set(proposed * 100)
            self.on_sensitivity_changed(str(proposed * 100))
            self.calibration_label.config(text="Threshold applied.")
        else:
            self.calibration_label.config(text="Threshold unchanged.")

    def open_mic_settings(self):
        mic_win = tk.Toplevel(self.root)
        mic_win.title("Microphone Settings")
        mic_win.geometry("400x250")
        mic_win.config(bg=self.BG_PRIMARY)
        
        # Title and info
        tk.Label(mic_win, text="Select Microphone:", font=("Helvetica", 12, "bold"),
                 bg=self.BG_PRIMARY, fg=self.TEXT_PRIMARY).pack(padx=10, pady=10)
        
        devices = self.get_mic_devices()
        mic_names = [name for idx, name in devices]
        
        if not mic_names:
            tk.Label(mic_win, text="❌ No microphone found!", bg=self.BG_PRIMARY,
                     fg=self.COLOR_F1).pack(padx=10, pady=5)
            tk.Button(mic_win, text="Close", command=mic_win.destroy,
                      bg=self.ACCENT_SECONDARY, fg=self.TEXT_PRIMARY).pack(padx=10, pady=10)
            return
        
        sel = tk.StringVar()
        current = self.selected_mic.get()
        if current in mic_names:
            sel.set(current)
        else:
            sel.set(mic_names[0])
        
        # Dropdown menu
        dropdown = tk.OptionMenu(mic_win, sel, *mic_names)
        dropdown.config(bg=self.BG_TERTIARY, fg=self.TEXT_PRIMARY,
                        activebackground=self.ACCENT_PRIMARY, activeforeground=self.TEXT_PRIMARY)
        dropdown["menu"].config(bg=self.BG_SECONDARY, fg=self.TEXT_PRIMARY,
                                activebackground=self.ACCENT_PRIMARY, activeforeground=self.TEXT_PRIMARY)
        dropdown.pack(padx=10, pady=5, fill=tk.X)
        
        # Info frame showing device details
        info_frame = tk.Frame(mic_win, bd=1, relief=tk.SUNKEN, bg=self.BG_SECONDARY)
        info_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        info_label = tk.Label(info_frame, text="", justify=tk.LEFT, font=("Helvetica", 9),
                              bg=self.BG_SECONDARY, fg=self.TEXT_PRIMARY)
        info_label.pack(padx=5, pady=5, anchor='nw')
        
        def update_info(*args):
            selected_name = sel.get()
            pa = pyaudio.PyAudio()
            for idx, name in devices:
                if name == selected_name:
                    try:
                        info = pa.get_device_info_by_index(idx)
                        channels = info.get("maxInputChannels", 0)
                        sample_rate = int(info.get("defaultSampleRate", 44100))
                        info_text = f"Device: {selected_name}\n\n"
                        info_text += f"Input Channels: {channels}\n"
                        info_text += f"Sample Rate: {sample_rate} Hz\n"
                        info_text += f"Index: {idx}"
                        info_label.config(text=info_text)
                    except Exception:
                        info_label.config(text="Could not get device info")
                    break
            pa.terminate()
        
        sel.trace_add("write", update_info)
        update_info()
        
        # Buttons
        button_frame = tk.Frame(mic_win, bg=self.BG_PRIMARY)
        button_frame.pack(padx=10, pady=10, fill=tk.X)
        
        def save_mic():
            self.selected_mic.set(sel.get())
            self.save_config()
            print(f"Microphone set to: {sel.get()}")
            mic_win.destroy()
        
        tk.Button(button_frame, text="Save", command=save_mic, bg=self.ACCENT_PRIMARY,
                  fg=self.TEXT_PRIMARY).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=mic_win.destroy, bg=self.ACCENT_SECONDARY,
                  fg=self.TEXT_PRIMARY).pack(side=tk.LEFT, padx=5)

    def set_output_directory(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir = directory
            os.makedirs(self.output_dir, exist_ok=True)
            self.metrics_log_path = os.path.join(self.output_dir, "metrics.log")
            self.save_config()
            messagebox.showinfo("Output Directory", f"Files will be saved to:\n{self.output_dir}")

    def set_mic_sensitivity(self):
        new_val = simpledialog.askfloat("Mic Sensitivity", "Enter RMS threshold (0 to 1):",
                                        initialvalue=self.sensitivity, minvalue=0, maxvalue=1)
        if new_val is not None:
            self.sensitivity_scale.set(new_val * 100)
            self.on_sensitivity_changed(str(new_val * 100))

    def set_reset_interval(self):
        new_val = simpledialog.askfloat("Reset Interval", "Enter reset interval (sec):", initialvalue=self.reset_interval_value)
        if new_val is not None:
            self.reset_interval_value = new_val
            self.save_config()

    def show_ranges(self):
        ranges_text = (
            "Transgender Pitch Ranges:\n"
            "• Gender neutral: 150–185 Hz\n"
            "• Vocal feminization: Aim ~180 Hz\n"
            "• Vocal masculinization: Aim ~150 Hz\n"
            "• Average female pitch: ~225 Hz\n"
            "• Average male pitch: ~125 Hz\n"
            "Consult a speech therapist for personalized guidance."
        )
        messagebox.showinfo("Transgender Pitch Ranges", ranges_text)

    def update_ui(self, note, pitch, formants, hnr, amplitude):
        try:
            self.update_mic_level(amplitude)
            if pitch is None or not np.isfinite(pitch) or pitch <= 0 or amplitude < self.sensitivity:
                display_tone = "0"
                pitch_to_append = None
            else:
                display_tone = note.upper()
                self.last_tone = display_tone
                self.last_pitch = pitch
                pitch_to_append = pitch

            self.tone_label.config(text=display_tone)
            self.current_pitch = pitch_to_append
            self.update_target_status()
            
            # Always add to history (None during silence, actual pitch when sound detected)
            # This keeps the X-axis scrolling continuously
            self.pitch_history.append(pitch_to_append)
            if len(self.pitch_history) > 100:
                self.pitch_history = self.pitch_history[-100:]
            
            if self.pitch_line and self.ax_pitch:
                self.pitch_line.set_data(range(len(self.pitch_history)), self.pitch_history)
                # Keep y-axis fixed at 0-400 Hz
                self.ax_pitch.set_ylim(0, 400)

            self.amplitude_history.append(amplitude)
            if len(self.amplitude_history) > 100:
                self.amplitude_history = self.amplitude_history[-100:]

            
            # Smooth formant values with exponential smoothing (always compute, but only display if needed)
            if not hasattr(self, "last_formants"):
                self.last_formants = [0, 0, 0]
            smoothed_formants = []
            for i in range(3):
                new_val = formants[i] if len(formants) > i else np.nan
                if pitch_to_append is not None and np.isfinite(new_val):
                    # If last value is not finite, initialize it.
                    if not np.isfinite(self.last_formants[i]):
                        smoothed = new_val
                    else:
                        smoothed = self.formant_smoothing_factor * new_val + (1 - self.formant_smoothing_factor) * self.last_formants[i]
                    self.last_formants[i] = smoothed
                else:
                    smoothed = np.nan
                smoothed_formants.append(smoothed)
                self.formant_histories[i].append(smoothed)
                if len(self.formant_histories[i]) > 100:
                    self.formant_histories[i] = self.formant_histories[i][-100:]
                
                # Update formant chart only if formants are displayed
                if self.formant_lines and i < len(self.formant_lines):
                    self.formant_lines[i].set_data(range(len(self.formant_histories[i])), self.formant_histories[i])
            
            # Update formants axis if it's displayed
            if self.ax_formants:
                self.ax_formants.relim()
                self.ax_formants.autoscale_view()
            
            self.canvas.draw_idle()

            periods = [1.0 / p for p in self.pitch_history if p and p > 0 and not np.isnan(p)]
            jitter = np.mean(np.abs(np.diff(periods))) / np.mean(periods) * 100 if len(periods) >= 2 else np.nan
            valid_amps = [a for a in self.amplitude_history if a and not np.isnan(a)]
            shimmer = np.mean(np.abs(np.diff(valid_amps))) / np.mean(valid_amps) * 100 if len(valid_amps) >= 2 else np.nan
            self.quality_label.config(text=f"Jitter: {jitter:.2f}% | Shimmer: {shimmer:.2f}% | HNR: {hnr:.2f} dB")

            if pitch_to_append is not None:
                self.metrics_pitch_history.append(pitch_to_append)
            for i in range(3):
                if pitch_to_append is not None and len(formants) > i and formants[i] > 0:
                    self.metrics_formant_histories[i].append(formants[i])
                else:
                    self.metrics_formant_histories[i].append(np.nan)
        except Exception:
            traceback.print_exc()

    def poll_queue(self):
        try:
            while not self.result_queue.empty():
                result = self.result_queue.get_nowait()
                self.update_ui(*result)
        except Exception:
            traceback.print_exc()
        self.root.after(50, self.poll_queue)

    def start_listening(self):
        try:
            devices = self.get_mic_devices()
            device_index = None
            for idx, name in devices:
                if name == self.selected_mic.get():
                    device_index = idx
                    break
            if device_index is None:
                if devices:
                    device_index = devices[0][0]
                    self.selected_mic.set(devices[0][1])
                else:
                    print("No microphone found!")
                    return
            pa = pyaudio.PyAudio()
            try:
                info = pa.get_device_info_by_index(device_index)
                device_rate = int(info.get("defaultSampleRate", 44100))
            except Exception:
                device_rate = 44100
            pa.terminate()
            print(f"Starting audio on '{self.selected_mic.get()}' (device {device_index}) at {device_rate} Hz.")
            self.audio_stream = AudioStream(device_index, device_rate, self.recording.get(), self.sensitivity, self.result_queue)
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
        except Exception:
            traceback.print_exc()

    def stop_listening(self):
        try:
            if self.audio_stream:
                self.audio_stream.stop()
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.NORMAL)
            self.tone_label.config(text="N/A")
            if self.recording.get() and self.audio_stream and self.audio_stream.recorded_frames:
                self.save_recording()
            self.save_full_chart()
        except Exception:
            traceback.print_exc()

    def save_recording(self):
        try:
            if not (self.audio_stream and self.audio_stream.recorded_frames):
                print("No audio recorded.")
                return False
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(self.output_dir, f"recording_{timestamp}.wav")
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio_stream.sample_width)
                wf.setframerate(self.audio_stream.rate)
                wf.writeframes(b"".join(self.audio_stream.recorded_frames))
            print(f"Audio recording saved to {filename}")
            return True
        except Exception as error:
            traceback.print_exc()
            messagebox.showerror("Recording", f"Could not save the recording: {error}")
            return False

    def save_full_chart(self):
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"full_chart_{timestamp}.png")
            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            ax1.set_title("Full Session Pitch History")
            ax1.set_ylabel("Frequency (Hz)")
            ax1.plot(range(len(self.pitch_history)), self.pitch_history, color='blue', label="Pitch")
            ax1.legend(loc='upper right')
            ax1.grid(True)
            ax2.set_title("Full Session Formant History (F1, F2, F3)")
            ax2.set_xlabel("Time (frames)")
            ax2.set_ylabel("Frequency (Hz)")
            colors = ['red', 'green', 'orange']
            for i in range(3):
                ax2.plot(range(len(self.formant_histories[i])), self.formant_histories[i], color=colors[i], label=f"F{i+1}")
            ax2.legend(loc='upper right')
            ax2.grid(True)
            fig2.tight_layout()
            fig2.savefig(filename)
            plt.close(fig2)
            print(f"Full chart image saved to {filename}")
        except Exception:
            traceback.print_exc()

    def reset_interval_metrics(self):
        try:
            if self.metrics_pitch_history:
                arr = np.array(self.metrics_pitch_history)
                valid = arr[arr > 0]
                if valid.size > 0:
                    low = np.nanmin(valid)
                    high = np.nanmax(valid)
                    avg = np.nanmean(valid)
                    pitch_text = f"Low: {low:.1f} Hz | High: {high:.1f} Hz | Avg: {avg:.1f} Hz"
                else:
                    pitch_text = "Low: N/A | High: N/A | Avg: N/A"
            else:
                pitch_text = "Low: N/A | High: N/A | Avg: N/A"
            formant_text = ""
            for i in range(3):
                arr = np.array(self.metrics_formant_histories[i])
                valid = arr[arr > 0]
                if valid.size == 0:
                    formant_text += f"F{i+1}: Low: N/A, High: N/A, Avg: N/A   "
                else:
                    low = np.nanmin(valid)
                    high = np.nanmax(valid)
                    avg = np.nanmean(valid)
                    formant_text += f"F{i+1}: Low: {low:.1f} Hz, High: {high:.1f} Hz, Avg: {avg:.1f} Hz   "
            self.pitch_summary_label.config(text="Pitch Summary: " + pitch_text)
            self.formant_summary_label.config(text="Formant Summary: " + formant_text)
            log_line = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Pitch: {pitch_text} | Formants: {formant_text}\n"
            with open(self.metrics_log_path, "a") as f:
                f.write(log_line)
        except Exception:
            traceback.print_exc()

    def schedule_reset(self):
        try:
            self.reset_interval_metrics()
            self.reset_timer = self.root.after(int(self.reset_interval_value * 1000), self.schedule_reset)
        except Exception:
            traceback.print_exc()

    def on_close(self):
        try:
            if self.reset_timer is not None:
                self.root.after_cancel(self.reset_timer)
            if self.audio_stream:
                self.audio_stream.stop()
            self.root.destroy()
        except Exception:
            traceback.print_exc()

    def poll_queue(self):
        try:
            while not self.result_queue.empty():
                result = self.result_queue.get_nowait()
                self.update_ui(*result)
        except Exception:
            traceback.print_exc()
        self.root.after(50, self.poll_queue)

    def start_listening(self):
        try:
            devices = self.get_mic_devices()
            device_index = None
            for idx, name in devices:
                if name == self.selected_mic.get():
                    device_index = idx
                    break
            if device_index is None:
                if devices:
                    device_index = devices[0][0]
                    self.selected_mic.set(devices[0][1])
                else:
                    print("No microphone found!")
                    return
            pa = pyaudio.PyAudio()
            try:
                info = pa.get_device_info_by_index(device_index)
                device_rate = int(info.get("defaultSampleRate", 44100))
            except Exception:
                device_rate = 44100
            pa.terminate()
            print(f"Starting audio on '{self.selected_mic.get()}' (device {device_index}) at {device_rate} Hz.")
            self.audio_stream = AudioStream(device_index, device_rate, self.recording.get(), self.sensitivity, self.result_queue)
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
        except Exception:
            traceback.print_exc()

    def stop_listening(self):
        try:
            if self.audio_stream:
                self.audio_stream.stop()
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.NORMAL)
            self.tone_label.config(text="N/A")
            if self.recording.get() and self.audio_stream and self.audio_stream.recorded_frames:
                self.save_recording()
            self.save_full_chart()
        except Exception:
            traceback.print_exc()

    def save_recording(self):
        try:
            if not (self.audio_stream and self.audio_stream.recorded_frames):
                print("No audio recorded.")
                return False
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(self.output_dir, f"recording_{timestamp}.wav")
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio_stream.sample_width)
                wf.setframerate(self.audio_stream.rate)
                wf.writeframes(b"".join(self.audio_stream.recorded_frames))
            print(f"Audio recording saved to {filename}")
            return True
        except Exception as error:
            traceback.print_exc()
            messagebox.showerror("Recording", f"Could not save the recording: {error}")
            return False

    def save_full_chart(self):
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"full_chart_{timestamp}.png")
            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            ax1.set_title("Full Session Pitch History")
            ax1.set_ylabel("Frequency (Hz)")
            ax1.plot(range(len(self.pitch_history)), self.pitch_history, color='blue', label="Pitch")
            ax1.legend(loc='upper right')
            ax1.grid(True)
            ax2.set_title("Full Session Formant History (F1, F2, F3)")
            ax2.set_xlabel("Time (frames)")
            ax2.set_ylabel("Frequency (Hz)")
            colors = ['red', 'green', 'orange']
            for i in range(3):
                ax2.plot(range(len(self.formant_histories[i])), self.formant_histories[i], color=colors[i], label=f"F{i+1}")
            ax2.legend(loc='upper right')
            ax2.grid(True)
            fig2.tight_layout()
            fig2.savefig(filename)
            plt.close(fig2)
            print(f"Full chart image saved to {filename}")
        except Exception:
            traceback.print_exc()

    def reset_interval_metrics(self):
        try:
            if self.metrics_pitch_history:
                arr = np.array(self.metrics_pitch_history)
                valid = arr[arr > 0]
                if valid.size > 0:
                    low = np.nanmin(valid)
                    high = np.nanmax(valid)
                    avg = np.nanmean(valid)
                    pitch_text = f"Low: {low:.1f} Hz | High: {high:.1f} Hz | Avg: {avg:.1f} Hz"
                else:
                    pitch_text = "Low: N/A | High: N/A | Avg: N/A"
            else:
                pitch_text = "Low: N/A | High: N/A | Avg: N/A"
            formant_text = ""
            for i in range(3):
                arr = np.array(self.metrics_formant_histories[i])
                valid = arr[arr > 0]
                if valid.size == 0:
                    formant_text += f"F{i+1}: Low: N/A, High: N/A, Avg: N/A   "
                else:
                    low = np.nanmin(valid)
                    high = np.nanmax(valid)
                    avg = np.nanmean(valid)
                    formant_text += f"F{i+1}: Low: {low:.1f} Hz, High: {high:.1f} Hz, Avg: {avg:.1f} Hz   "
            self.pitch_summary_label.config(text="Pitch Summary: " + pitch_text)
            self.formant_summary_label.config(text="Formant Summary: " + formant_text)
            self.metrics_pitch_history.clear()
            for history in self.metrics_formant_histories:
                history.clear()
            log_line = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Pitch: {pitch_text} | Formants: {formant_text}\n"
            with open(self.metrics_log_path, "a", encoding="utf-8") as f:
                f.write(log_line)
        except Exception:
            traceback.print_exc()

    def schedule_reset(self):
        try:
            self.reset_interval_metrics()
            self.reset_timer = self.root.after(int(self.reset_interval_value * 1000), self.schedule_reset)
        except Exception:
            traceback.print_exc()

    def on_close(self):
        try:
            self.cancel_calibration()
            if self.audio_stream:
                if self.audio_stream.stop() is False:
                    messagebox.showerror("Microphone", "Could not stop the microphone. Check the console for details.")
                    return
                if self.audio_stream.recorded_frames and not self.save_recording():
                    self.active_mic_name = None
                    self.update_mic_status()
                    return
                self.audio_stream = None
            if self.sensitivity_save_id is not None:
                self.root.after_cancel(self.sensitivity_save_id)
                self.save_sensitivity()
            # Stop polling immediately
            if self.poll_queue_id is not None:
                self.root.after_cancel(self.poll_queue_id)
            
            # Cancel reset timer
            if self.reset_timer is not None:
                self.root.after_cancel(self.reset_timer)
            
            # Close matplotlib figure
            if self.fig:
                try:
                    plt.close(self.fig)
                except Exception:
                    pass
            
            # Destroy window
            self.root.destroy()
        except Exception:
            traceback.print_exc()
            import sys
            sys.exit(0)

    def poll_queue(self):
        try:
            while not self.result_queue.empty():
                result = self.result_queue.get_nowait()
                self.update_ui(*result)
        except Exception:
            traceback.print_exc()
        self.poll_queue_id = self.root.after(50, self.poll_queue)

    def start_listening(self):
        try:
            devices = self.get_mic_devices()
            device_index = None
            for idx, name in devices:
                if name == self.selected_mic.get():
                    device_index = idx
                    break
            if device_index is None:
                if devices:
                    device_index = devices[0][0]
                    self.selected_mic.set(devices[0][1])
                else:
                    messagebox.showerror("Microphone", "No microphone found.")
                    return
            pa = pyaudio.PyAudio()
            try:
                info = pa.get_device_info_by_index(device_index)
                device_rate = int(info.get("defaultSampleRate", 44100))
            except Exception:
                device_rate = 44100
            pa.terminate()
            print(f"Starting audio on '{self.selected_mic.get()}' (device {device_index}) at {device_rate} Hz.")
            self.audio_stream = AudioStream(device_index, device_rate, self.recording.get(), self.sensitivity, self.result_queue)
            if self.audio_stream.stream is None:
                self.audio_stream.stop()
                self.audio_stream = None
                messagebox.showerror("Microphone", "Could not open the microphone. Check the device and try again.")
                return
            self.active_mic_name = self.selected_mic.get()
            self.update_mic_status()
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
        except Exception:
            traceback.print_exc()
            messagebox.showerror("Microphone", "Could not start listening. Check the console for details.")

    def stop_listening(self):
        try:
            self.cancel_calibration()
            if self.audio_stream:
                if self.audio_stream.stop() is False:
                    messagebox.showerror("Microphone", "Could not stop the microphone. Check the console for details.")
                    return
                if self.audio_stream.recorded_frames and not self.save_recording():
                    self.active_mic_name = None
                    self.update_mic_status()
                    return
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.tone_label.config(text="N/A")
            self.audio_stream = None
            self.active_mic_name = None
            self.current_pitch = None
            self.update_mic_status()
            self.update_mic_level(0)
            self.update_target_status()
            while not self.result_queue.empty():
                self.result_queue.get_nowait()
            self.save_full_chart()
        except Exception:
            traceback.print_exc()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
