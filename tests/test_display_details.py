import json
import math
import os
import tempfile
import tkinter as tk
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt

from VocalTuner import App


class DisplayDetailsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.environment = patch.dict(os.environ, {"APPDATA": self.directory.name})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.devices = patch.object(App, "get_mic_devices", return_value=[])
        self.devices.start()
        self.addCleanup(self.devices.stop)
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        self.root.withdraw()
        self.app = App(self.root)
        self.addCleanup(self.close_app)

    def close_app(self):
        self.app.cancel_calibration()
        if self.app.sensitivity_save_id is not None:
            self.root.after_cancel(self.app.sensitivity_save_id)
        self.root.after_cancel(self.app.poll_queue_id)
        self.root.after_cancel(self.app.reset_timer)
        plt.close(self.app.fig)
        self.root.destroy()

    def test_details_toggle_independently_and_restore_order(self):
        app = self.app
        self.assertEqual(app.tone_frame.winfo_manager(), "pack")
        self.assertEqual(app.quality_label.winfo_manager(), "pack")
        self.assertEqual(app.metrics_frame.winfo_manager(), "pack")

        app.show_note.set(False)
        app.on_display_details_changed()
        self.assertEqual(app.tone_frame.winfo_manager(), "")
        self.assertEqual(app.metrics_frame.winfo_manager(), "pack")

        app.show_summary.set(False)
        app.on_display_details_changed()
        self.assertEqual(app.quality_label.winfo_manager(), "")
        self.assertEqual(app.metrics_frame.winfo_manager(), "")
        app.update_ui("A3", 220, [500, 1500, 2500], 10, 0.1)
        self.assertEqual(app.tone_label.cget("text"), "A3")
        self.assertEqual(app.pitch_history[-1], 220)

        app.show_summary.set(True)
        app.on_display_details_changed()
        self.assertEqual(app.tone_frame.winfo_manager(), "")
        app.show_note.set(True)
        app.on_display_details_changed()
        widgets = self.root.pack_slaves()
        self.assertLess(widgets.index(app.tone_frame), widgets.index(app.quality_label))
        self.assertLess(widgets.index(app.quality_label), widgets.index(app.metrics_frame))
        self.assertLess(widgets.index(app.metrics_frame), widgets.index(app.chart_container))

    def test_saved_choices_are_restored(self):
        self.app.show_note.set(False)
        self.app.show_summary.set(False)
        self.app.show_mic_settings.set(False)
        self.app.show_target_settings.set(False)
        self.app.on_display_details_changed()
        self.app.show_note.set(True)
        self.app.show_summary.set(True)
        self.app.show_mic_settings.set(True)
        self.app.show_target_settings.set(True)
        self.app.load_config()
        self.app.update_display_details()
        self.assertFalse(self.app.show_note.get())
        self.assertFalse(self.app.show_summary.get())
        self.assertEqual(self.app.tone_frame.winfo_manager(), "")
        self.assertEqual(self.app.quality_label.winfo_manager(), "")
        self.assertEqual(self.app.metrics_frame.winfo_manager(), "")
        self.assertEqual(self.app.level_frame.winfo_manager(), "")
        self.assertEqual(self.app.target_frame.winfo_manager(), "")
        self.assertEqual(self.app.preset_frame.winfo_manager(), "")
        self.assertTrue(self.app.pitch_target_patch.get_visible())
        self.app.show_target_settings.set(True)
        self.app.show_mic_settings.set(True)
        self.app.update_display_details()
        self.assertEqual(self.app.level_frame.winfo_manager(), "pack")
        self.assertEqual(self.app.target_frame.winfo_manager(), "pack")
        self.assertEqual(self.app.preset_frame.winfo_manager(), "pack")

    def test_calibration_requires_listening(self):
        with patch("VocalTuner.messagebox.showinfo") as showinfo:
            self.app.start_calibration()
        showinfo.assert_called_once()
        self.assertIsNone(self.app.calibration_after_id)

    def test_calibration_proposal_is_applied_only_after_confirmation(self):
        app = self.app
        stream = MagicMock()
        stream.rate = 44100
        stream.end_calibration.return_value = [(0.01, 2048)] * 65
        app.audio_stream = stream
        app.active_mic_name = "Microphone"
        original = app.sensitivity
        for accepted in (False, True):
            app.start_calibration()
            self.assertEqual(str(app.calibration_button.cget("state")), "disabled")
            self.root.after_cancel(app.calibration_after_id)
            with patch("VocalTuner.messagebox.askyesno", return_value=accepted) as confirm:
                app.finish_calibration()
            confirm.assert_called_once()
            self.assertEqual(str(app.calibration_button.cget("state")), "normal")
            self.assertAlmostEqual(app.sensitivity, 0.016 if accepted else original)
        self.assertAlmostEqual(stream.sensitivity, 0.016)
        app.audio_stream = None

    def test_calibration_rejects_missing_or_excessive_audio(self):
        app = self.app
        stream = MagicMock()
        stream.rate = 44100
        app.audio_stream = stream
        app.active_mic_name = "Microphone"
        original = app.sensitivity
        for samples in ([], [(0.8, 2048)] * 65):
            stream.end_calibration.return_value = samples
            app.start_calibration()
            self.root.after_cancel(app.calibration_after_id)
            with (patch("VocalTuner.messagebox.showwarning") as warning,
                  patch("VocalTuner.messagebox.askyesno") as confirm):
                app.finish_calibration()
            warning.assert_called_once()
            confirm.assert_not_called()
            self.assertEqual(app.sensitivity, original)
        app.audio_stream = None

    def test_stop_cancels_an_ongoing_calibration(self):
        app = self.app
        stream = MagicMock()
        stream.rate = 44100
        stream.recorded_frames = []
        app.audio_stream = stream
        app.active_mic_name = "Microphone"
        app.start_calibration()
        with patch.object(app, "save_full_chart"):
            app.stop_listening()
        stream.end_calibration.assert_called_once()
        self.assertIsNone(app.calibration_after_id)
        self.assertEqual(str(app.calibration_button.cget("state")), "normal")

    def test_threshold_and_target_status_update_live(self):
        app = self.app
        app.audio_stream = MagicMock()
        app.on_sensitivity_changed("2.0")
        self.assertEqual(app.audio_stream.sensitivity, 0.02)
        app.update_ui("A3", 220, [500, 1500, 2500], 10, 0.03)
        self.assertEqual(app.target_status_label.cget("text"), "Target: in range")
        self.assertEqual(app.level_label.cget("text"), "3.0%")
        app.update_ui("A3", 180, [500, 1500, 2500], 10, 0.03)
        self.assertEqual(app.target_status_label.cget("text"), "Target: below range")
        app.update_ui("A3", 270, [500, 1500, 2500], 10, 0.03)
        self.assertEqual(app.target_status_label.cget("text"), "Target: above range")
        app.update_ui("0", None, [float("nan")] * 3, float("nan"), 0.03)
        self.assertEqual(app.target_status_label.cget("text"), "Target: waiting for voice")
        self.assertIsNone(app.pitch_history[-1])
        self.assertTrue(all(math.isnan(history[-1]) for history in app.formant_histories))
        app.audio_stream = None

    def test_summary_covers_only_the_latest_interval(self):
        app = self.app
        app.update_ui("G2", 100, [500, 1500, 2500], 10, 0.1)
        app.reset_interval_metrics()
        self.assertIn("Avg: 100.0 Hz", app.pitch_summary_label.cget("text"))
        app.update_ui("D4", 300, [600, 1600, 2600], 10, 0.1)
        app.reset_interval_metrics()
        self.assertIn("Avg: 300.0 Hz", app.pitch_summary_label.cget("text"))
        self.assertEqual(app.metrics_pitch_history, [])
        self.assertTrue(all(not history for history in app.metrics_formant_histories))

    def test_switching_chart_modes_keeps_history_visible(self):
        app = self.app
        app.update_ui("A3", 220, [500, 1500, 2500], 10, 0.1)
        app.update_ui("0", None, [float("nan")] * 3, float("nan"), 0.0)
        for mode in ("both", "formants_only", "pitch_only"):
            app.display_mode.set(mode)
            app.rebuild_chart()
            if app.ax_pitch is not None:
                self.assertEqual(list(app.pitch_line.get_ydata()), app.pitch_history)
            if app.ax_formants is not None:
                for line, history in zip(app.formant_lines, app.formant_histories):
                    self.assertEqual(list(line.get_xdata()), list(range(len(history))))
                    self.assertEqual(line.get_ydata()[0], history[0])
                    self.assertTrue(math.isnan(line.get_ydata()[1]))

    def test_recording_menu_updates_the_running_stream(self):
        app = self.app
        stream = MagicMock()
        stream.record_audio = False
        app.audio_stream = stream
        menu = self.root.nametowidget(app.menubar.entrycget("File", "menu"))
        menu.invoke("Record Audio")
        self.assertTrue(stream.record_audio)
        menu.invoke("Record Audio")
        self.assertFalse(stream.record_audio)
        app.audio_stream = None

    def test_disabled_recording_is_saved_on_stop_and_close(self):
        app = self.app
        for close in (False, True):
            stream = MagicMock()
            stream.sample_width = 2
            stream.rate = 44100
            stream.recorded_frames = [b"\x01\x00\x02\x00"]
            app.audio_stream = stream
            app.recording.set(False)
            with (patch.object(app, "save_full_chart"),
                  patch.object(app.root, "destroy")):
                if close:
                    app.on_close()
                else:
                    app.stop_listening()
            paths = list(Path(app.output_dir).glob("recording_*.wav"))
            self.assertEqual(len(paths), 1)
            with wave.open(str(paths[0]), "rb") as file:
                self.assertEqual(file.readframes(2), b"\x01\x00\x02\x00")
            paths[0].unlink()
            stream.stop.assert_called_once()
            app.audio_stream = None

    def test_recording_is_kept_when_saving_fails(self):
        app = self.app
        stream = MagicMock()
        stream.recorded_frames = [b"\x01\x00"]
        stream.sample_width = 2
        stream.rate = 44100
        app.audio_stream = stream
        app.active_mic_name = "Microphone"
        with (patch("VocalTuner.wave.open", side_effect=OSError("disk full")),
              patch("VocalTuner.messagebox.showerror") as showerror,
              patch.object(app.root, "destroy") as destroy,
              patch.object(app, "save_full_chart")):
            app.stop_listening()
            self.assertIs(app.audio_stream, stream)
            app.on_close()
            self.assertIs(app.audio_stream, stream)
            self.assertEqual(showerror.call_count, 2)
            destroy.assert_not_called()
        with patch.object(app.root, "destroy"):
            app.on_close()
        paths = list(Path(app.output_dir).glob("recording_*.wav"))
        self.assertEqual(len(paths), 1)
        with wave.open(str(paths[0]), "rb") as file:
            self.assertEqual(file.readframes(1), b"\x01\x00")
        app.audio_stream = None

    def test_recording_is_kept_when_microphone_cannot_stop(self):
        app = self.app
        stream = MagicMock()
        stream.recorded_frames = [b"\x01\x00"]
        stream.stop.return_value = False
        app.audio_stream = stream
        with (patch("VocalTuner.messagebox.showerror") as showerror,
              patch.object(app, "save_recording") as save,
              patch.object(app.root, "destroy") as destroy):
            app.stop_listening()
            app.on_close()
            self.assertIs(app.audio_stream, stream)
            self.assertEqual(showerror.call_count, 2)
            save.assert_not_called()
            destroy.assert_not_called()
        app.audio_stream = None

    def test_legacy_threshold_is_preserved_and_migrated(self):
        app = self.app
        with open(app.config_path, "w", encoding="utf-8") as file:
            json.dump({"sensitivity": 0.04}, file)
        app.load_config()
        self.assertEqual(app.sensitivity, 0.02)
        app.save_config()
        with open(app.config_path, encoding="utf-8") as file:
            saved = json.load(file)
        self.assertEqual(saved["mic_threshold"], 0.02)

    def test_personal_presets_can_be_saved_loaded_and_deleted(self):
        app = self.app
        app.target_zone_min.set(150)
        app.target_zone_max.set(200)
        with patch("VocalTuner.simpledialog.askstring", return_value="Practice"):
            app.save_target_preset()
        app.target_zone_min.set(210)
        app.target_zone_max.set(250)
        self.assertTrue(app.update_target_zone())
        self.assertEqual(app.selected_preset.get(), "")
        app.apply_target_preset("Practice")
        self.assertEqual(app.target_limits, (150, 200))
        app.target_presets = {}
        app.load_config()
        self.assertEqual(app.target_presets["Practice"], [150, 200])
        with patch("VocalTuner.messagebox.askyesno", return_value=True):
            app.delete_target_preset()
        self.assertNotIn("Practice", app.target_presets)

    def test_invalid_target_range_is_not_saved(self):
        app = self.app
        original_limits = app.target_limits
        app.target_zone_min.set(300)
        app.target_zone_max.set(200)
        with patch("VocalTuner.messagebox.showerror") as showerror:
            self.assertFalse(app.update_target_zone())
        showerror.assert_called_once()
        self.assertEqual(app.target_limits, original_limits)

    def test_microphone_status_tracks_active_and_next_device(self):
        app = self.app
        audio = MagicMock()
        audio.stream = object()
        audio.recorded_frames = []
        with (patch.object(App, "get_mic_devices", return_value=[(1, "First"), (2, "Second")]),
              patch("VocalTuner.pyaudio.PyAudio") as pa,
              patch("VocalTuner.AudioStream", return_value=audio),
              patch.object(app, "save_full_chart")):
            pa.return_value.get_device_info_by_index.return_value = {"defaultSampleRate": 44100}
            app.selected_mic.set("First")
            app.start_listening()
            self.assertEqual(app.mic_status_label.cget("text"), "Listening: First")
            app.selected_mic.set("Second")
            self.assertEqual(app.mic_status_label.cget("text"), "Listening: First | Next start: Second")
            app.stop_listening()
            self.assertEqual(app.mic_status_label.cget("text"), "Stopped | Microphone: Second")
            self.assertEqual(str(app.stop_button.cget("state")), "disabled")

    def test_microphone_failure_does_not_show_listening(self):
        app = self.app
        audio = MagicMock()
        audio.stream = None
        with (patch.object(App, "get_mic_devices", return_value=[(1, "First")]),
              patch("VocalTuner.pyaudio.PyAudio") as pa,
              patch("VocalTuner.AudioStream", return_value=audio),
              patch("VocalTuner.messagebox.showerror") as showerror):
            pa.return_value.get_device_info_by_index.return_value = {"defaultSampleRate": 44100}
            app.selected_mic.set("First")
            app.start_listening()
            showerror.assert_called_once()
            audio.stop.assert_called_once()
            self.assertIsNone(app.audio_stream)
            self.assertEqual(app.mic_status_label.cget("text"), "Stopped | Microphone: First")
            self.assertEqual(str(app.start_button.cget("state")), "normal")


if __name__ == "__main__":
    unittest.main()
