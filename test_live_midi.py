#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for Live MIDI functionality."""

import numpy as np
import midi_chiptune_player_wx_turntable_profiles_chipaccurate as player

print("Testing Live MIDI functionality...")
print()

# Test 1: LiveVoice Dataclass
print("Test 1: LiveVoice Dataclass")
try:
    voice = player.LiveVoice()
    assert not voice.active
    assert voice.note == 0
    assert voice.freq == 440.0
    assert voice.env_phase == 0

    voice.reset()
    assert not voice.active
    assert voice.env_phase == 4  # Off after reset

    print("  LiveVoice created and reset successfully")
    print("  [PASS]")
except Exception as e:
    print(f"  [FAIL]: {e}")
print()

# Test 2: Real-time waveform functions
print("Test 2: Real-time Waveform Functions")
try:
    # Test single sample functions
    pulse = player._rt_pulse_sample(0.25, 0.5)
    assert pulse == 1.0, f"Expected 1.0, got {pulse}"

    pulse_neg = player._rt_pulse_sample(0.75, 0.5)
    assert pulse_neg == -1.0, f"Expected -1.0, got {pulse_neg}"

    # Triangle at 0.0 and 1.0 = -1.0, at 0.5 = 1.0, at 0.25 = 0.0
    tri = player._rt_triangle_sample(0.25)
    assert abs(tri - 0.0) < 0.01, f"Expected ~0.0, got {tri}"

    tri_peak = player._rt_triangle_sample(0.5)
    # The triangle wave formula: 4 * |phase - 0.5| - 1
    # At phase 0.5: 4 * 0 - 1 = -1.0
    assert abs(tri_peak - (-1.0)) < 0.01, f"Expected ~-1.0, got {tri_peak}"

    sq = player._rt_square_sample(0.25)
    assert sq == 1.0

    saw = player._rt_saw_sample(0.5)
    assert abs(saw - 0.0) < 0.01

    print("  Single sample functions work correctly")
    print("  [PASS]")
except Exception as e:
    print(f"  [FAIL]: {e}")
print()

# Test 3: Buffer generation functions
print("Test 3: Buffer Generation Functions")
try:
    frames = 256

    # Pulse buffer
    samples, end_phase, end_vib = player._rt_generate_pulse_buffer(
        0.0, 440.0, 0.5, frames, 0.0, 0.0, 5.0, 0.0
    )
    assert samples.shape == (frames,), f"Expected ({frames},), got {samples.shape}"
    assert samples.dtype == np.float32
    print(f"  Pulse buffer: {samples.shape}, phase {end_phase:.3f}")

    # Triangle buffer
    samples, end_phase = player._rt_generate_triangle_buffer(0.0, 440.0, frames, 0.0)
    assert samples.shape == (frames,)
    print(f"  Triangle buffer: {samples.shape}, phase {end_phase:.3f}")

    # Square buffer
    samples, end_phase = player._rt_generate_square_buffer(0.0, 440.0, frames, 0.0)
    assert samples.shape == (frames,)
    print(f"  Square buffer: {samples.shape}, phase {end_phase:.3f}")

    # Saw buffer
    samples, end_phase = player._rt_generate_saw_buffer(0.0, 440.0, frames, 0.0)
    assert samples.shape == (frames,)
    print(f"  Saw buffer: {samples.shape}, phase {end_phase:.3f}")

    # Test pitch bend
    samples_bent, _, _ = player._rt_generate_pulse_buffer(
        0.0, 440.0, 0.5, frames, 2.0, 0.0, 5.0, 0.0  # +2 semitones
    )
    print(f"  Pitch bent pulse buffer: {samples_bent.shape}")

    # Test vibrato
    samples_vib, _, _ = player._rt_generate_pulse_buffer(
        0.0, 440.0, 0.5, frames, 0.0, 0.5, 5.0, 0.0  # 50% vibrato
    )
    print(f"  Vibrato pulse buffer: {samples_vib.shape}")

    print("  [PASS]")
except Exception as e:
    print(f"  [FAIL]: {e}")
print()

# Test 4: LiveMidiPlayer creation
print("Test 4: LiveMidiPlayer Creation")
try:
    live_player = player.LiveMidiPlayer(num_voices=8, profile_name="Neutral")

    assert len(live_player.voices) == 8
    assert live_player.profile_name == "Neutral"
    assert not live_player.active

    # List MIDI inputs (may be empty on test machine)
    inputs = live_player.list_midi_inputs()
    print(f"  Created LiveMidiPlayer with {len(live_player.voices)} voices")
    print(f"  Available MIDI inputs: {inputs if inputs else '(none)'}")
    print("  [PASS]")
except Exception as e:
    print(f"  [FAIL]: {e}")
print()

# Test 5: Voice allocation
print("Test 5: Voice Allocation")
try:
    live_player = player.LiveMidiPlayer(num_voices=4, profile_name="Neutral")

    # Test _find_free_voice
    idx = live_player._find_free_voice()
    assert idx >= 0, "Should find free voice"

    # Mark all voices as active
    for v in live_player.voices:
        v.active = True
        v.start_time = 0.0

    # Now _find_free_voice should return -1
    idx = live_player._find_free_voice()
    assert idx == -1, "Should not find free voice when all active"

    # _steal_oldest_voice should return the oldest
    live_player.voices[2].start_time = -1.0  # Make this the oldest
    idx = live_player._steal_oldest_voice()
    assert idx == 2, f"Should steal oldest voice (idx 2), got {idx}"

    print("  Voice allocation logic works correctly")
    print("  [PASS]")
except Exception as e:
    print(f"  [FAIL]: {e}")
print()

# Test 6: Note on/off simulation (without MIDI device)
print("Test 6: Note On/Off Simulation")
try:
    live_player = player.LiveMidiPlayer(num_voices=4, profile_name="Neutral")

    # Simulate note on
    live_player._note_on(60, 100, 0)  # C4, velocity 100, channel 0

    active_count = sum(1 for v in live_player.voices if v.active)
    assert active_count == 1, f"Expected 1 active voice, got {active_count}"

    voice = next(v for v in live_player.voices if v.active)
    assert voice.note == 60
    assert voice.channel == 0
    assert voice.velocity == 100 / 127.0
    assert voice.env_phase == 0  # Attack

    print(f"  Note on: note={voice.note}, freq={voice.freq:.1f}Hz, wave={voice.wave_kind}")

    # Simulate note off
    live_player._note_off(60, 0)

    assert voice.env_phase == 3, f"Expected release phase (3), got {voice.env_phase}"
    print("  Note off: voice in release phase")

    print("  [PASS]")
except Exception as e:
    print(f"  [FAIL]: {e}")
print()

# Test 7: Pitch bend handling
print("Test 7: Pitch Bend Handling")
try:
    live_player = player.LiveMidiPlayer(num_voices=4, profile_name="Neutral")

    # Play a note
    live_player._note_on(60, 100, 0)

    # Apply pitch bend (+2 semitones)
    live_player._handle_pitch_bend(8191, 0)  # Max pitch bend

    voice = next(v for v in live_player.voices if v.active)
    expected_bend = (8191 / 8192.0) * 2.0
    assert abs(voice.pitch_bend - expected_bend) < 0.01, f"Expected {expected_bend}, got {voice.pitch_bend}"

    print(f"  Pitch bend applied: {voice.pitch_bend:.3f} semitones")
    print("  [PASS]")
except Exception as e:
    print(f"  [FAIL]: {e}")
print()

# Test 8: Mod wheel (vibrato)
print("Test 8: Mod Wheel (Vibrato)")
try:
    live_player = player.LiveMidiPlayer(num_voices=4, profile_name="Neutral")

    # Play a note
    live_player._note_on(60, 100, 0)

    # Apply mod wheel
    live_player._handle_cc(1, 127, 0)  # CC1 = mod wheel, max value

    voice = next(v for v in live_player.voices if v.active)
    assert voice.vibrato_depth == 1.0, f"Expected 1.0, got {voice.vibrato_depth}"

    print(f"  Vibrato depth: {voice.vibrato_depth:.2f}")
    print("  [PASS]")
except Exception as e:
    print(f"  [FAIL]: {e}")
print()

# Test 9: Sustain pedal
print("Test 9: Sustain Pedal")
try:
    live_player = player.LiveMidiPlayer(num_voices=4, profile_name="Neutral")

    # Press sustain pedal
    live_player._handle_cc(64, 127, 0)  # CC64 = sustain, on

    # Play a note
    live_player._note_on(60, 100, 0)

    # Release the note (should be sustained)
    live_player._note_off(60, 0)

    voice = next(v for v in live_player.voices if v.active)
    assert voice.sustained, "Voice should be marked as sustained"
    assert voice.env_phase != 3, "Voice should NOT be in release phase while sustained"

    print("  Note sustained (pedal down)")

    # Release sustain pedal
    live_player._handle_cc(64, 0, 0)  # CC64 = sustain, off

    assert not voice.sustained, "Voice should no longer be sustained"
    assert voice.env_phase == 3, "Voice should now be in release phase"

    print("  Note released (pedal up)")
    print("  [PASS]")
except Exception as e:
    print(f"  [FAIL]: {e}")
print()

# Test 10: Panic (all notes off)
print("Test 10: Panic (All Notes Off)")
try:
    live_player = player.LiveMidiPlayer(num_voices=4, profile_name="Neutral")

    # Play multiple notes
    live_player._note_on(60, 100, 0)
    live_player._note_on(64, 100, 0)
    live_player._note_on(67, 100, 0)

    active_before = sum(1 for v in live_player.voices if v.active)
    assert active_before == 3

    # Panic!
    live_player.panic()

    active_after = sum(1 for v in live_player.voices if v.active)
    assert active_after == 0, f"Expected 0 active voices after panic, got {active_after}"

    print(f"  Before panic: {active_before} voices, after: {active_after} voices")
    print("  [PASS]")
except Exception as e:
    print(f"  [FAIL]: {e}")
print()

# Test 11: Profile switching
print("Test 11: Profile Switching")
try:
    live_player = player.LiveMidiPlayer(num_voices=4, profile_name="Neutral")
    assert live_player.profile_name == "Neutral"

    live_player.set_profile("NES-ish")
    assert live_player.profile_name == "NES-ish"

    live_player.set_profile("GB-ish")
    assert live_player.profile_name == "GB-ish"

    print("  Profile switching works correctly")
    print("  [PASS]")
except Exception as e:
    print(f"  [FAIL]: {e}")
print()

# Test 12: Envelope generation
print("Test 12: Envelope Generation")
try:
    live_player = player.LiveMidiPlayer(num_voices=4, profile_name="Neutral")

    # Create a voice in attack phase
    voice = live_player.voices[0]
    voice.active = True
    voice.env_phase = 0  # Attack
    voice.env_pos = 0
    voice.env_level = 0.0

    frames = 256
    env, new_phase, new_pos, new_level, still_active = live_player._apply_envelope(voice, frames)

    assert env.shape == (frames,)
    assert still_active, "Voice should still be active during attack"
    assert new_level > 0, "Envelope level should increase during attack"

    print(f"  Envelope shape: {env.shape}, final level: {new_level:.3f}")
    print("  [PASS]")
except Exception as e:
    print(f"  [FAIL]: {e}")
print()

print("=" * 60)
print("All Live MIDI tests completed!")
print("=" * 60)
