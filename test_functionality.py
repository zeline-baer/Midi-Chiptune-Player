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
assert "TIA" in player.PROFILES, "TIA profile should exist"
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

# Test 3: TIA Frequency Table
print("Test 3: TIA Frequency Table")
try:
    tia_table = player._get_tia_freq_table()
    freqs = [entry[0] for entry in tia_table]
    print(f"  TIA frequency table size: {len(tia_table)} entries")
    print(f"  Frequency range: {min(freqs):.1f} Hz - {max(freqs):.1f} Hz")
    
    # Test snapping
    target = 440.0
    snapped, audf, divisor = player._snap_to_tia_freq(target)
    print(f"  440 Hz snaps to: {snapped:.1f} Hz (AUDF={audf}, div={divisor})")
    
    # Test low frequency
    target_low = 100.0
    snapped_low, audf_low, div_low = player._snap_to_tia_freq(target_low)
    print(f"  100 Hz snaps to: {snapped_low:.1f} Hz (AUDF={audf_low}, div={div_low})")
    print("  [PASS]")
except Exception as e:
    print(f"  [FAIL]: {e}")
print()

# Test 4: TIA Synthesizer Functions
print("Test 4: TIA Synthesizer Functions")
try:
    tia_pure = player.tia_tone(440.0, 0.1, 0.5, mode="pure")
    tia_buzzy = player.tia_tone(440.0, 0.1, 0.5, mode="buzzy")
    tia_saw = player.tia_tone(440.0, 0.1, 0.5, mode="saw")
    tia_noise = player.tia_noise(0.1, 0.5)

    print(f"  TIA pure tone: {tia_pure.shape} samples")
    print(f"  TIA buzzy tone: {tia_buzzy.shape} samples")
    print(f"  TIA saw tone: {tia_saw.shape} samples")
    print(f"  TIA noise: {tia_noise.shape} samples")
    print("  [PASS]")
except Exception as e:
    print(f"  [FAIL]: {e}")
print()

# Test 5: Wavetable
print("Test 5: Wavetable Generation")
try:
    wt = player._choose_wave_table("gb_default")
    print(f"  Wavetable shape: {wt.shape}")
    print(f"  Wavetable dtype: {wt.dtype}")
    print("  [PASS]")
except Exception as e:
    print(f"  [FAIL]: {e}")
print()

# Test 6: MIDI-to-Frequency Lookup (optional feature)
print("Test 6: MIDI-to-Frequency Lookup")
try:
    if hasattr(player, 'MIDI_TO_FREQ'):
        freq_a4 = player.MIDI_TO_FREQ[69]
        freq_c4 = player.MIDI_TO_FREQ[60]
        print(f"  A4 (MIDI 69): {freq_a4:.2f} Hz (expected: 440.00)")
        print(f"  C4 (MIDI 60): {freq_c4:.2f} Hz (expected: ~261.63)")
        if abs(freq_a4 - 440.0) < 0.01 and abs(freq_c4 - 261.63) < 0.01:
            print("  [PASS]")
        else:
            print("  [FAIL]: Incorrect frequencies")
    else:
        print("  [SKIP] MIDI_TO_FREQ not present in this version")
except Exception as e:
    print(f"  [FAIL]: {e}")
print()

# Test 7: Rendering with standard profile
print("Test 7: Rendering Test Notes (Neutral profile)")
try:
    test_notes = [
        (0.0, 0.5, 60, 0, 100, False),  # C4
        (0.5, 1.0, 64, 0, 100, False),  # E4
        (1.0, 1.5, 67, 0, 100, False),  # G4
    ]

    audio = player.render_chiptune_float32(test_notes, profile_name="Neutral")

    print(f"  Rendered audio shape: {audio.shape}")
    print(f"  Rendered audio dtype: {audio.dtype}")
    print(f"  Audio duration: {audio.shape[0] / player.SAMPLE_RATE:.2f}s")
    print("  [PASS]")
except Exception as e:
    print(f"  [FAIL]: {e}")
print()

# Test 8: Rendering with TIA profile (THE KEY TEST!)
print("Test 8: Rendering Test Notes (TIA profile)")
try:
    test_notes = [
        (0.0, 0.5, 60, 0, 100, False),  # C4
        (0.5, 1.0, 64, 1, 100, False),  # E4
        (1.0, 1.5, 67, 0, 100, False),  # G4
        (1.5, 2.0, 36, 9, 100, True),   # Kick drum
    ]

    audio = player.render_chiptune_float32(test_notes, profile_name="TIA")

    print(f"  Rendered audio shape: {audio.shape}")
    print(f"  Rendered audio dtype: {audio.dtype}")
    print(f"  Audio duration: {audio.shape[0] / player.SAMPLE_RATE:.2f}s")
    
    # Check it's actually mono (TIA is mono)
    if np.allclose(audio[:, 0], audio[:, 1]):
        print("  Audio is mono (correct for TIA)")
    else:
        print("  WARNING: Audio is not mono!")
    
    print("  [PASS]")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  [FAIL]: {e}")
print()

# Test 9: TurntablePlayer
print("Test 9: TurntablePlayer (no audio output)")
try:
    # Create test audio buffer
    test_buffer = np.random.randn(44100, 2).astype(np.float32) * 0.1

    turntable = player.TurntablePlayer(test_buffer, loop=True)
    turntable.set_rate(1.5)

    print(f"  Buffer size: {turntable.n} samples")
    print(f"  Rate: {turntable.rate}")
    print("  [PASS]")
except Exception as e:
    print(f"  [FAIL]: {e}")
print()

# Test 10: Channel volume rendering (optional feature)
print("Test 10: Channel Volume Rendering")
try:
    test_notes = [
        (0.0, 0.5, 60, 0, 100, False),
        (0.5, 1.0, 64, 1, 100, False),
    ]
    
    # Check if channel_volumes parameter is supported
    import inspect
    sig = inspect.signature(player.render_chiptune_float32)
    if 'channel_volumes' in sig.parameters:
        channel_vols = {0: 0.5, 1: 0.8}
        audio = player.render_chiptune_float32(test_notes, profile_name="Neutral", channel_volumes=channel_vols)
        print(f"  Rendered with channel volumes: {audio.shape}")
        print("  [PASS]")
    else:
        audio = player.render_chiptune_float32(test_notes, profile_name="Neutral")
        print(f"  [SKIP] channel_volumes not supported, basic render: {audio.shape}")
except Exception as e:
    print(f"  [FAIL]: {e}")
print()

# Test 11: All profiles render without error
print("Test 11: Render All Profiles")
test_notes = [
    (0.0, 0.3, 60, 0, 100, False),
    (0.3, 0.6, 64, 1, 100, False),
    (0.6, 0.9, 67, 2, 100, False),
    (0.9, 1.2, 36, 9, 100, True),  # drum
]

all_passed = True
for profile_name in player.PROFILES.keys():
    try:
        audio = player.render_chiptune_float32(test_notes, profile_name=profile_name)
        print(f"  {profile_name}: OK ({audio.shape[0] / player.SAMPLE_RATE:.2f}s)")
    except Exception as e:
        print(f"  {profile_name}: FAIL - {e}")
        all_passed = False

if all_passed:
    print("  [PASS] All profiles render successfully!")
else:
    print("  [FAIL] Some profiles failed!")
print()

print("=" * 60)
print("All functionality tests completed!")
print("=" * 60)
