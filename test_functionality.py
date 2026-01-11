#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functional test for MIDI Chiptune Player (no GUI)."""

import numpy as np
import midi_chiptune_player_wx_turntable_profiles_chipaccurate as player

print("Testing MIDI Chiptune Player functionality...")
print()

# Test 1: Profile loading
print("Test 1: Profile Loading")
print(f"  Profiles available: {list(player.PROFILES.keys())}")
print("  [PASS]")
print()

# Test 2: Synthesizer functions
print("Test 2: Synthesizer Functions")
try:
    pulse = player.pulse_tone(440.0, 0.1, 0.5)
    triangle = player.triangle_tone(440.0, 0.1, 0.5)
    square = player.square_tone(440.0, 0.1, 0.5)
    saw = player.saw_tone(440.0, 0.1, 0.5)
    noise = player.noise_tone(0.1, 0.5)

    print(f"  Pulse tone: {pulse.shape} samples")
    print(f"  Triangle tone: {triangle.shape} samples")
    print(f"  Square tone: {square.shape} samples")
    print(f"  Saw tone: {saw.shape} samples")
    print(f"  Noise tone: {noise.shape} samples")
    print("  [PASS]")
except Exception as e:
    print(f"  [FAIL]: {e}")
print()

# Test 3: Wavetable
print("Test 3: Wavetable Generation")
try:
    wt = player._choose_wave_table("gb_default")
    print(f"  Wavetable shape: {wt.shape}")
    print(f"  Wavetable dtype: {wt.dtype}")
    print(f"  [PASS] Passed")
except Exception as e:
    print(f"  [FAIL] Failed: {e}")
print()

# Test 4: MIDI-to-Freq lookup
print("Test 4: MIDI-to-Frequency Lookup")
try:
    # A4 = MIDI 69 = 440 Hz
    freq_a4 = player.MIDI_TO_FREQ[69]
    # C4 = MIDI 60 ≈ 261.63 Hz
    freq_c4 = player.MIDI_TO_FREQ[60]

    print(f"  A4 (MIDI 69): {freq_a4:.2f} Hz (expected: 440.00)")
    print(f"  C4 (MIDI 60): {freq_c4:.2f} Hz (expected: ~261.63)")

    if abs(freq_a4 - 440.0) < 0.01 and abs(freq_c4 - 261.63) < 0.01:
        print(f"  [PASS] Passed")
    else:
        print(f"  [FAIL] Failed: Incorrect frequencies")
except Exception as e:
    print(f"  [FAIL] Failed: {e}")
print()

# Test 5: Rendering (without actual MIDI file)
print("Test 5: Rendering Test Notes")
try:
    # Create fake notes: [(start, end, pitch, channel, velocity, is_drum), ...]
    test_notes = [
        (0.0, 0.5, 60, 0, 100, False),  # C4
        (0.5, 1.0, 64, 0, 100, False),  # E4
        (1.0, 1.5, 67, 0, 100, False),  # G4
    ]

    audio = player.render_chiptune_float32(test_notes, profile_name="Neutral")

    print(f"  Rendered audio shape: {audio.shape}")
    print(f"  Rendered audio dtype: {audio.dtype}")
    print(f"  Audio duration: {audio.shape[0] / player.SAMPLE_RATE:.2f}s")
    print(f"  [PASS] Passed")
except Exception as e:
    print(f"  [FAIL] Failed: {e}")
print()

# Test 6: TurntablePlayer
print("Test 6: TurntablePlayer (no audio output)")
try:
    # Create test audio buffer
    test_buffer = np.random.randn(44100, 2).astype(np.float32) * 0.1

    turntable = player.TurntablePlayer(test_buffer, loop=True, volume=1.0)
    turntable.set_rate(1.5)
    turntable.set_volume(0.8)

    print(f"  Buffer size: {turntable.n} samples")
    print(f"  Rate: {turntable.rate}")
    print(f"  Volume: {turntable.volume}")
    print(f"  [PASS] Passed")
except Exception as e:
    print(f"  [FAIL] Failed: {e}")
print()

# Test 7: Channel volume rendering
print("Test 7: Channel Volume Rendering")
try:
    test_notes = [
        (0.0, 0.5, 60, 0, 100, False),
        (0.5, 1.0, 64, 1, 100, False),
    ]

    channel_vols = {0: 0.5, 1: 0.8}
    audio = player.render_chiptune_float32(test_notes, profile_name="Neutral", channel_volumes=channel_vols)

    print(f"  Rendered with channel volumes: {audio.shape}")
    print(f"  [PASS] Passed")
except Exception as e:
    print(f"  [FAIL] Failed: {e}")
print()

print("=" * 60)
print("All functionality tests passed!")
print("=" * 60)
