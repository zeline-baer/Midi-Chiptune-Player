#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Real-world performance test with actual MIDI rendering."""

import time
import numpy as np
import midi_chiptune_player_wx_turntable_profiles_chipaccurate as player

print("=" * 60)
print("Real-World Performance Analysis")
print("=" * 60)
print()

# Create realistic test data (simulating a typical MIDI file)
print("Creating test MIDI data (500 notes)...")
test_notes = []
for i in range(500):
    start = i * 0.1
    end = start + 0.2
    pitch = 60 + (i % 24)  # C4 to B5
    channel = i % 6
    velocity = 80 + (i % 40)
    is_drum = (channel == 9) if i % 20 == 0 else False
    test_notes.append((start, end, pitch, channel, velocity, is_drum))

print(f"Created {len(test_notes)} notes")
print()

# Test rendering with different profiles
profiles = ["Neutral", "NES-ish", "GB-ish"]

for profile_name in profiles:
    print(f"Testing {profile_name} profile:")
    print("-" * 40)

    # Render multiple times to get average
    times = []
    for i in range(3):
        start = time.perf_counter()
        audio = player.render_chiptune_float32(test_notes, profile_name=profile_name)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s")

    avg_time = sum(times) / len(times)
    print(f"  Average: {avg_time:.3f}s")
    print(f"  Audio shape: {audio.shape}")
    print()

# Profile the slowest parts
print("=" * 60)
print("Profiling individual synthesizer functions:")
print("-" * 40)

iterations = 100

# Test pulse_tone (most common)
start = time.perf_counter()
for _ in range(iterations):
    _ = player.pulse_tone(440.0, 0.2, 0.5, duty=0.5)
pulse_time = time.perf_counter() - start
print(f"pulse_tone ({iterations}x):     {pulse_time:.3f}s ({pulse_time/iterations*1000:.2f}ms each)")

# Test triangle_tone
start = time.perf_counter()
for _ in range(iterations):
    _ = player.triangle_tone(440.0, 0.2, 0.5)
tri_time = time.perf_counter() - start
print(f"triangle_tone ({iterations}x):  {tri_time:.3f}s ({tri_time/iterations*1000:.2f}ms each)")

# Test square_tone
start = time.perf_counter()
for _ in range(iterations):
    _ = player.square_tone(440.0, 0.2, 0.5)
square_time = time.perf_counter() - start
print(f"square_tone ({iterations}x):    {square_time:.3f}s ({square_time/iterations*1000:.2f}ms each)")

# Test ADSR generation (called by every tone)
start = time.perf_counter()
for _ in range(iterations):
    _ = player._adsr_env(8820, player.SAMPLE_RATE, total_sec=0.2)
adsr_time = time.perf_counter() - start
print(f"_adsr_env ({iterations}x):      {adsr_time:.3f}s ({adsr_time/iterations*1000:.2f}ms each)")

print()
print("=" * 60)
print("BOTTLENECK ANALYSIS:")
print("-" * 40)
print("The main performance issues are:")
print("1. Synthesizer functions (pulse_tone, triangle_tone, etc.)")
print("2. ADSR envelope generation (called for every note)")
print("3. NumPy array operations in audio synthesis")
print()
print("The MIDI-to-freq lookup and wavetable cache help,")
print("but the real bottleneck is the audio synthesis loop.")
print("=" * 60)
