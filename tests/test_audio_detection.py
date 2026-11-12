import queue
import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from VocalTuner import (
    AudioStream, CHUNK, compute_formants, detect_pitch_and_hnr,
    freq_to_note_name, levinson_durbin,
)


class AudioDetectionTests(unittest.TestCase):
    def test_note_octaves_change_at_c(self):
        for frequency, note in ((261.625565, "C4"), (415.3047, "G#4"),
                                (440, "A4"), (523.251, "C5")):
            self.assertEqual(freq_to_note_name(frequency), note)

    def test_hnr_is_finite_and_decreases_with_added_noise(self):
        rng = np.random.default_rng(42)
        time = np.arange(CHUNK) / 44100
        clean = 0.1 * np.sin(2 * np.pi * 220 * time)
        noisy = clean + rng.normal(0, 0.02, CHUNK)
        _, clean_hnr = detect_pitch_and_hnr(clean, 44100, 0.005)
        _, noisy_hnr = detect_pitch_and_hnr(noisy, 44100, 0.005)
        self.assertTrue(np.isfinite(clean_hnr))
        self.assertTrue(np.isfinite(noisy_hnr))
        self.assertGreater(clean_hnr, noisy_hnr)
        expected_hnr = 10 * np.log10((0.1 ** 2 / 2) / 0.02 ** 2)
        self.assertAlmostEqual(noisy_hnr, expected_hnr, delta=1.5)

    def test_levinson_coefficients_match_autocorrelation_equations(self):
        rng = np.random.default_rng(42)
        time = np.arange(CHUNK) / 16000
        signal = np.sin(2 * np.pi * 220 * time) + rng.normal(0, 0.1, CHUNK)
        r = np.correlate(signal, signal, mode="full")[CHUNK - 1:CHUNK + 16]
        indices = np.abs(np.subtract.outer(np.arange(16), np.arange(16)))
        expected = np.linalg.solve(r[indices], r[1:17])
        np.testing.assert_allclose(levinson_durbin(r, 16)[1:], expected, atol=1e-8)
        np.testing.assert_allclose(levinson_durbin(r * 1e-10, 16)[1:], expected, atol=1e-8)

    def test_formant_roots_use_predictor_coefficient_sign(self):
        rate = 16000
        roots = [0.95 * np.exp(sign * 2j * np.pi * frequency / rate)
                 for frequency in (500, 1500) for sign in (-1, 1)]
        polynomial = np.poly(roots).real
        predictor = np.r_[1.0, -polynomial[1:]]
        signal = np.sin(2 * np.pi * 220 * np.arange(CHUNK) / rate)
        with patch("VocalTuner.levinson_durbin", return_value=predictor):
            formants = compute_formants(signal, rate, lpc_order=4)
        np.testing.assert_allclose(formants[:2], [500, 1500], atol=0.01)

    def test_formants_of_synthetic_voiced_signal(self):
        for rate in (16000, 44100, 48000):
            radius = np.exp(-np.pi * 200 / rate)
            roots = [radius * np.exp(sign * 2j * np.pi * frequency / rate)
                     for frequency in (500, 1500) for sign in (-1, 1)]
            polynomial = np.poly(roots).real
            source = np.zeros(8192 + CHUNK)
            source[::round(rate / 200)] = 1
            signal = np.zeros_like(source)
            excitation = 0.0
            for i in range(len(signal)):
                excitation = source[i] + 0.97 * excitation
                signal[i] = excitation
                for j in range(1, min(i, 4) + 1):
                    signal[i] -= polynomial[j] * signal[i - j]
            signal = signal[-CHUNK:]
            signal = signal / np.sqrt(np.mean(signal ** 2)) * 0.05
            for gain in (1, 0.1):
                formants = compute_formants(signal * gain, rate)
                np.testing.assert_allclose(formants[:2], [500, 1500], atol=100)

    def test_silence_and_noise_are_not_pitched(self):
        rng = np.random.default_rng(42)
        for rate in (16000, 44100, 48000):
            for signal in (np.zeros(CHUNK), rng.normal(0, 0.05, CHUNK)):
                pitch, _ = detect_pitch_and_hnr(signal, rate, 0.005)
                self.assertIsNone(pitch)

    def test_detects_voice_like_periodic_signal_with_moderate_noise(self):
        rng = np.random.default_rng(42)
        for rate in (16000, 44100, 48000):
            time = np.arange(CHUNK) / rate
            for frequency in (90, 150, 220, 350):
                signal = 0.05 * np.sin(2 * np.pi * frequency * time) + rng.normal(0, 0.01, CHUNK)
                pitch, _ = detect_pitch_and_hnr(signal, rate, 0.005)
                self.assertIsNotNone(pitch)
                self.assertAlmostEqual(pitch, frequency, delta=8)

    def test_callback_uses_normalized_threshold_and_updates_without_restart(self):
        stream = AudioStream.__new__(AudioStream)
        stream.rate = 44100
        stream.record_audio = False
        stream.sensitivity = 0.02
        stream.result_queue = queue.Queue()
        stream.calibration_lock = threading.Lock()
        stream.calibration_samples = None
        stream.calibration_deadline = None
        time = np.arange(CHUNK) / stream.rate
        signal = (0.015 * np.sin(2 * np.pi * 220 * time) * 32767).astype(np.int16).tobytes()

        stream.callback(signal, CHUNK, None, 0)
        _, pitch, _, _, level = stream.result_queue.get_nowait()
        self.assertIsNone(pitch)
        self.assertAlmostEqual(level, 0.015 / np.sqrt(2), delta=0.001)

        stream.sensitivity = 0.005
        stream.callback(signal, CHUNK, None, 0)
        _, pitch, _, _, _ = stream.result_queue.get_nowait()
        self.assertIsNotNone(pitch)
        self.assertAlmostEqual(pitch, 220, delta=5)

    def test_calibration_collects_audio_only_within_the_measurement_window(self):
        stream = AudioStream.__new__(AudioStream)
        stream.rate = 44100
        stream.record_audio = False
        stream.sensitivity = 0.02
        stream.result_queue = queue.Queue()
        stream.calibration_lock = threading.Lock()
        stream.calibration_samples = None
        stream.calibration_deadline = None
        signal = np.full(CHUNK, 328, dtype=np.int16).tobytes()
        with patch("VocalTuner.time.monotonic", side_effect=[10.0, 10.1, 13.1]):
            stream.begin_calibration()
            stream.callback(signal, CHUNK, None, 0)
            stream.callback(signal, CHUNK, None, 0)
            samples = stream.end_calibration()
        self.assertEqual(len(samples), 1)
        self.assertAlmostEqual(samples[0][0], 328 / 32768, places=5)
        self.assertEqual(samples[0][1], CHUNK)
        self.assertEqual(stream.result_queue.qsize(), 2)

    def test_recording_collects_only_enabled_nonempty_buffers(self):
        stream = AudioStream.__new__(AudioStream)
        stream.rate = 44100
        stream.record_audio = True
        stream.recorded_frames = []
        stream.sensitivity = 0.02
        stream.result_queue = queue.Queue()
        stream.calibration_lock = threading.Lock()
        stream.calibration_samples = None
        stream.calibration_deadline = None
        signal = np.zeros(CHUNK, dtype=np.int16).tobytes()
        stream.callback(None, 0, None, 0)
        stream.callback(signal, CHUNK, None, 0)
        stream.record_audio = False
        stream.callback(signal, CHUNK, None, 0)
        self.assertEqual(stream.recorded_frames, [signal])

    def test_stopping_stream_twice_releases_resources_once(self):
        stream = AudioStream.__new__(AudioStream)
        stream.closed = False
        stream.stream = MagicMock()
        stream.pa = MagicMock()
        stream.stop()
        stream.stop()
        stream.stream.stop_stream.assert_called_once()
        stream.stream.close.assert_called_once()
        stream.pa.terminate.assert_called_once()

    def test_failed_stream_stop_is_reported(self):
        stream = AudioStream.__new__(AudioStream)
        stream.closed = False
        stream.stream = MagicMock()
        stream.stream.stop_stream.side_effect = OSError("device error")
        stream.pa = MagicMock()
        with patch("VocalTuner.traceback.print_exc"):
            self.assertFalse(stream.stop())
        self.assertFalse(stream.closed)


if __name__ == "__main__":
    unittest.main()
