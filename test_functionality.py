"""Functional checks without a GUI or audio device; failures return a nonzero exit code."""

import io
from pathlib import Path
import shutil
import tempfile
import unittest
import wave

import mido
import numpy as np

import midi_chiptune_player_wx_turntable_profiles_chipaccurate as player


class FunctionalityTests(unittest.TestCase):
    NOTES = [
        (0.0, 0.3, 60, 0, 100, False),
        (0.3, 0.6, 64, 1, 100, False),
        (0.6, 0.9, 67, 2, 100, False),
        (0.9, 1.2, 36, 9, 100, True),
    ]

    def assert_audio(self, audio, frames, channels=None):
        shape = (frames,) if channels is None else (frames, channels)
        self.assertEqual(audio.shape, shape)
        self.assertEqual(audio.dtype, np.float32)
        self.assertTrue(np.isfinite(audio).all())
        self.assertGreater(float(np.max(np.abs(audio))), 0.0)
        self.assertLessEqual(float(np.max(np.abs(audio))), 1.0)

    def test_available_profiles(self):
        # These profiles are implemented by get_profiles() and exposed by the GUI.
        self.assertEqual(set(player.PROFILES), {"Neutral", "NES-ish", "GB-ish", "C64-ish", "Beeper"})

    def test_synthesizer_waveforms(self):
        for synth in (player.pulse_tone, player.triangle_tone, player.square_tone,
                      player.saw_tone, player.wavetable_tone):
            with self.subTest(synth=synth.__name__):
                self.assert_audio(synth(440.0, 0.1, 0.5), 4410)
        self.assert_audio(player.noise_tone(0.1, 0.5), 4410)

    def test_wavetable(self):
        table = player._choose_wave_table("gb_default")
        self.assert_audio(table, 32)
        self.assertIs(table, player._choose_wave_table("gb_default"))

    def test_midi_frequencies(self):
        self.assertAlmostEqual(float(player.MIDI_TO_FREQ[69]), 440.0, places=2)
        self.assertAlmostEqual(float(player.MIDI_TO_FREQ[60]), 261.63, places=2)

    def test_render_all_profiles(self):
        for name, profile in player.PROFILES.items():
            with self.subTest(profile=name):
                audio = player.render_chiptune_float32(self.NOTES, profile_name=name)
                self.assert_audio(audio, int(1.4 * player.SAMPLE_RATE), 2)
                if (profile.get("chip") or {}).get("mono"):
                    np.testing.assert_array_equal(audio[:, 0], audio[:, 1])

    def test_channel_volume(self):
        notes = [(0.0, 0.3, 60, 0, 100, False)]
        normal = player.render_chiptune_float32(notes)
        half = player.render_chiptune_float32(notes, channel_volumes={0: 0.5})
        muted = player.render_chiptune_float32(notes, channel_volumes={0: 0.0})
        np.testing.assert_allclose(half, normal * 0.5, atol=1e-7)
        np.testing.assert_array_equal(muted, np.zeros_like(muted))

    def test_midi_file_roundtrip_and_tempo(self):
        mid = mido.MidiFile(ticks_per_beat=480)
        mid.tracks.append(mido.MidiTrack([
            mido.Message("note_on", note=60, velocity=100),
            mido.MetaMessage("set_tempo", tempo=1000000, time=480),
            mido.Message("note_off", note=60, time=480),
            mido.Message("note_on", note=36, channel=9, velocity=80),
            mido.Message("note_on", note=36, channel=9, velocity=0, time=240),
        ]))
        data = io.BytesIO()
        mid.save(file=data)
        data.seek(0)
        notes = player.collect_notes_faithful(mido.MidiFile(file=data))
        self.assertEqual(notes, [(0.0, 1.5, 60, 0, 100, False), (1.5, 2.0, 36, 9, 80, True)])
        self.assertEqual(player.channels_in_notes(notes), [0, 9])
        self.assert_audio(player.render_chiptune_float32(notes), int(2.2 * player.SAMPLE_RATE), 2)

    def test_turntable_callback_and_seek(self):
        audio = player.render_chiptune_float32(self.NOTES)
        turntable = player.TurntablePlayer(audio, loop=True, volume=0.5)
        output = np.empty((256, 2), dtype=np.float32)
        turntable._callback(output, 256, None, None)
        np.testing.assert_allclose(output, audio[:256] * 0.5, atol=1e-7)
        turntable.set_rate(1.5)
        turntable.set_fraction(0.5)
        position = turntable.idx
        turntable._callback(output, 256, None, None)
        self.assertAlmostEqual(turntable.idx, position + 256 * 1.5)
        self.assertTrue(np.isfinite(output).all())

    def test_wav_export(self):
        audio = player.render_chiptune_float32(self.NOTES)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "test.wav")
            player.export_wav_float32(path, audio)
            with wave.open(path, "rb") as wav:
                self.assertEqual(wav.getnchannels(), 2)
                self.assertEqual(wav.getsampwidth(), 2)
                self.assertEqual(wav.getframerate(), player.SAMPLE_RATE)
                self.assertEqual(wav.getnframes(), len(audio))
                samples = np.frombuffer(wav.readframes(len(audio)), dtype="<i2").reshape(-1, 2)
            np.testing.assert_array_equal(samples, (audio * 32767.0).astype(np.int16))

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is optional and is not installed")
    def test_mp3_export(self):
        audio = player.render_chiptune_float32(self.NOTES)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.mp3"
            player.export_mp3_via_ffmpeg(str(path), audio)
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
