"""Live MIDI logic tests without opening an audio stream or MIDI input."""

import unittest

import numpy as np
import midi_chiptune_player_wx_turntable_profiles_chipaccurate as player


class LiveMidiTests(unittest.TestCase):
    def test_voice_reset(self):
        voice = player.LiveVoice()
        self.assertTrue(not voice.active)
        self.assertTrue(voice.note == 0)
        self.assertTrue(voice.freq == 440.0)
        self.assertTrue(voice.env_phase == 0)
        voice.reset()
        self.assertTrue(not voice.active)
        self.assertTrue(voice.env_phase == 4)

    def test_waveform_samples(self):
        pulse = player._rt_pulse_sample(0.25, 0.5)
        self.assertTrue(pulse == 1.0, f'Expected 1.0, got {pulse}')
        pulse_neg = player._rt_pulse_sample(0.75, 0.5)
        self.assertTrue(pulse_neg == -1.0, f'Expected -1.0, got {pulse_neg}')
        tri = player._rt_triangle_sample(0.25)
        self.assertTrue(abs(tri - 0.0) < 0.01, f'Expected ~0.0, got {tri}')
        tri_peak = player._rt_triangle_sample(0.5)
        self.assertTrue(abs(tri_peak - -1.0) < 0.01, f'Expected ~-1.0, got {tri_peak}')
        sq = player._rt_square_sample(0.25)
        self.assertTrue(sq == 1.0)
        saw = player._rt_saw_sample(0.5)
        self.assertTrue(abs(saw - 0.0) < 0.01)

    def test_waveform_buffers(self):
        frames = 256
        samples, end_phase, end_vib = player._rt_generate_pulse_buffer(0.0, 440.0, 0.5, frames, 0.0, 0.0, 5.0, 0.0)
        self.assertTrue(samples.shape == (frames,), f'Expected ({frames},), got {samples.shape}')
        self.assertTrue(samples.dtype == np.float32)
        samples, end_phase = player._rt_generate_triangle_buffer(0.0, 440.0, frames, 0.0)
        self.assertTrue(samples.shape == (frames,))
        samples, end_phase = player._rt_generate_square_buffer(0.0, 440.0, frames, 0.0)
        self.assertTrue(samples.shape == (frames,))
        samples, end_phase = player._rt_generate_saw_buffer(0.0, 440.0, frames, 0.0)
        self.assertTrue(samples.shape == (frames,))
        samples_bent, _, _ = player._rt_generate_pulse_buffer(0.0, 440.0, 0.5, frames, 2.0, 0.0, 5.0, 0.0)
        samples_vib, _, _ = player._rt_generate_pulse_buffer(0.0, 440.0, 0.5, frames, 0.0, 0.5, 5.0, 0.0)
        for output in (samples_bent, samples_vib):
            self.assertEqual(output.shape, (frames,))
            self.assertTrue(np.isfinite(output).all())

    def test_player_creation(self):
        live_player = player.LiveMidiPlayer(num_voices=8, profile_name='Neutral')
        self.assertTrue(len(live_player.voices) == 8)
        self.assertTrue(live_player.profile_name == 'Neutral')
        self.assertTrue(not live_player.active)

    def test_voice_allocation(self):
        live_player = player.LiveMidiPlayer(num_voices=4, profile_name='Neutral')
        idx = live_player._find_free_voice()
        self.assertTrue(idx >= 0, 'Should find free voice')
        for v in live_player.voices:
            v.active = True
            v.start_time = 0.0
        idx = live_player._find_free_voice()
        self.assertTrue(idx == -1, 'Should not find free voice when all active')
        live_player.voices[2].start_time = -1.0
        idx = live_player._steal_oldest_voice()
        self.assertTrue(idx == 2, f'Should steal oldest voice (idx 2), got {idx}')

    def test_note_on_off(self):
        live_player = player.LiveMidiPlayer(num_voices=4, profile_name='Neutral')
        live_player._note_on(60, 100, 0)
        active_count = sum((1 for v in live_player.voices if v.active))
        self.assertTrue(active_count == 1, f'Expected 1 active voice, got {active_count}')
        voice = next((v for v in live_player.voices if v.active))
        self.assertTrue(voice.note == 60)
        self.assertTrue(voice.channel == 0)
        self.assertTrue(voice.velocity == 100 / 127.0)
        self.assertTrue(voice.env_phase == 0)
        live_player._note_off(60, 0)
        self.assertTrue(voice.env_phase == 3, f'Expected release phase (3), got {voice.env_phase}')

    def test_pitch_bend(self):
        live_player = player.LiveMidiPlayer(num_voices=4, profile_name='Neutral')
        live_player._note_on(60, 100, 0)
        live_player._handle_pitch_bend(8191, 0)
        voice = next((v for v in live_player.voices if v.active))
        expected_bend = 8191 / 8192.0 * 2.0
        self.assertTrue(abs(voice.pitch_bend - expected_bend) < 0.01, f'Expected {expected_bend}, got {voice.pitch_bend}')

    def test_mod_wheel(self):
        live_player = player.LiveMidiPlayer(num_voices=4, profile_name='Neutral')
        live_player._note_on(60, 100, 0)
        live_player._handle_cc(1, 127, 0)
        voice = next((v for v in live_player.voices if v.active))
        self.assertTrue(voice.vibrato_depth == 1.0, f'Expected 1.0, got {voice.vibrato_depth}')

    def test_sustain(self):
        live_player = player.LiveMidiPlayer(num_voices=4, profile_name='Neutral')
        live_player._handle_cc(64, 127, 0)
        live_player._note_on(60, 100, 0)
        live_player._note_off(60, 0)
        voice = next((v for v in live_player.voices if v.active))
        self.assertTrue(voice.sustained, 'Voice should be marked as sustained')
        self.assertTrue(voice.env_phase != 3, 'Voice should NOT be in release phase while sustained')
        live_player._handle_cc(64, 0, 0)
        self.assertTrue(not voice.sustained, 'Voice should no longer be sustained')
        self.assertTrue(voice.env_phase == 3, 'Voice should now be in release phase')

    def test_panic(self):
        live_player = player.LiveMidiPlayer(num_voices=4, profile_name='Neutral')
        live_player._note_on(60, 100, 0)
        live_player._note_on(64, 100, 0)
        live_player._note_on(67, 100, 0)
        active_before = sum((1 for v in live_player.voices if v.active))
        self.assertTrue(active_before == 3)
        live_player.panic()
        active_after = sum((1 for v in live_player.voices if v.active))
        self.assertTrue(active_after == 0, f'Expected 0 active voices after panic, got {active_after}')

    def test_profile_switching(self):
        live_player = player.LiveMidiPlayer(num_voices=4, profile_name='Neutral')
        self.assertTrue(live_player.profile_name == 'Neutral')
        live_player.set_profile('NES-ish')
        self.assertTrue(live_player.profile_name == 'NES-ish')
        live_player.set_profile('GB-ish')
        self.assertTrue(live_player.profile_name == 'GB-ish')

    def test_envelope(self):
        live_player = player.LiveMidiPlayer(num_voices=4, profile_name='Neutral')
        voice = live_player.voices[0]
        voice.active = True
        voice.env_phase = 0
        voice.env_pos = 0
        voice.env_level = 0.0
        frames = 256
        env, new_phase, new_pos, new_level, still_active = live_player._apply_envelope(voice, frames)
        self.assertTrue(env.shape == (frames,))
        self.assertTrue(still_active, 'Voice should still be active during attack')
        self.assertTrue(new_level > 0, 'Envelope level should increase during attack')


if __name__ == "__main__":
    unittest.main(verbosity=2)
