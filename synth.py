"""Pure synthesis engine: waveforms, ADSR, drums, chip profiles, MIDI parsing, render, export.

This module has no dependency on wx or any GUI toolkit.  Everything here is
either a stateless function or a data-only constant, so it can be imported
and tested without a display.

Dependency map
--------------
synth  <-  live_midi, metronome, main GUI module
          (none of those flow back into this file)
"""

import math
import wave
import tempfile
import subprocess
import shutil
from pathlib import Path
from functools import lru_cache

import numpy as np
import mido

# Optional Numba JIT — falls back to identity decorator when not installed.
try:
    from numba import jit, njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if args and callable(args[0]):
            return args[0]
        return decorator

    njit = jit


# ---------------------------------------------------------------------------
# Global audio constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 44100
MASTER_GAIN = 0.85  # internal overall gain applied during synthesis

# Pre-computed MIDI note → frequency table (avoids repeated pow() calls)
MIDI_TO_FREQ = np.array(
    [440.0 * (2.0 ** ((i - 69.0) / 12.0)) for i in range(128)],
    dtype=np.float32,
)

_TWO_PI = 2.0 * np.pi
_INV_PI = 1.0 / np.pi


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _pan_gains(pan: float):
    """Equal-power panning.  pan=-1 → hard left, 0 → centre, +1 → hard right."""
    p = max(-1.0, min(1.0, float(pan)))
    angle = (p + 1.0) * math.pi / 4.0
    return math.cos(angle), math.sin(angle)


def _stereo_from_mono(mono: np.ndarray, pan: float):
    gL, gR = _pan_gains(pan)
    return np.stack([mono * gL, mono * gR], axis=1)


def _adsr_env(n: int, sr: int,
              attack=0.008, decay=0.050, sustain=0.65, release=0.120,
              total_sec=None):
    """ADSR envelope as a float32 array of length *n* (or derived from total_sec)."""
    if total_sec is not None:
        n = max(2, int(sr * total_sec))

    a = max(1, int(sr * attack))
    d = max(1, int(sr * decay))
    r = max(1, int(sr * release))
    s_len = max(0, n - (a + d + r))

    total_len = a + d + s_len + r

    env = np.empty(total_len, dtype=np.float32)
    env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    env[a:a + d] = np.linspace(1.0, sustain, d, dtype=np.float32)
    env[a + d:a + d + s_len] = sustain
    env[a + d + s_len:] = np.linspace(sustain, 0.0, r, dtype=np.float32)

    if total_len < n:
        env = np.pad(env, (0, n - total_len), mode='edge')
    elif total_len > n:
        env = env[:n]

    return env


# ---------------------------------------------------------------------------
# Offline waveform generators (used for pre-render, not real-time)
# ---------------------------------------------------------------------------
def pulse_tone(freq=440.0, dur=0.2, vol=0.4, duty=0.5,
               vib_rate=0.0, vib_depth_cents=0.0):
    n = max(2, int(SAMPLE_RATE * float(dur)))
    t = np.arange(n, dtype=np.float32) * (1.0 / SAMPLE_RATE)
    if vib_rate > 0.0 and vib_depth_cents != 0.0:
        lfo = np.sin(_TWO_PI * float(vib_rate) * t)
        ratio = 2.0 ** ((lfo * float(vib_depth_cents)) / 1200.0)
        inst_f = float(freq) * ratio
    else:
        inst_f = float(freq)

    if isinstance(inst_f, float):
        phase = _TWO_PI * float(freq) * t
    else:
        phase = _TWO_PI * np.cumsum(inst_f) / SAMPLE_RATE

    frac = np.mod(phase, _TWO_PI) * _INV_PI * 0.5
    w = np.where(frac < float(duty), 1.0, -1.0).astype(np.float32)
    env = _adsr_env(n, SAMPLE_RATE, total_sec=dur)
    return w * env * (float(vol) * MASTER_GAIN)


def triangle_tone(freq=220.0, dur=0.2, vol=0.4):
    n = max(2, int(SAMPLE_RATE * float(dur)))
    t = np.arange(n, dtype=np.float32) * (1.0 / SAMPLE_RATE)
    tri = (2.0 * _INV_PI) * np.arcsin(np.sin(_TWO_PI * float(freq) * t)).astype(np.float32)
    env = _adsr_env(n, SAMPLE_RATE, total_sec=dur)
    return tri * env * (float(vol) * MASTER_GAIN)


def square_tone(freq=440.0, dur=0.2, vol=0.4):
    n = max(2, int(SAMPLE_RATE * float(dur)))
    t = np.arange(n, dtype=np.float32) * (1.0 / SAMPLE_RATE)
    w = np.sign(np.sin(_TWO_PI * float(freq) * t)).astype(np.float32)
    env = _adsr_env(n, SAMPLE_RATE, total_sec=dur)
    return w * env * (float(vol) * MASTER_GAIN)


def saw_tone(freq=440.0, dur=0.2, vol=0.4):
    n = max(2, int(SAMPLE_RATE * float(dur)))
    t = np.arange(n, dtype=np.float32) * (1.0 / SAMPLE_RATE)
    frac = np.mod(float(freq) * t, 1.0)
    w = (frac * 2.0 - 1.0).astype(np.float32)
    env = _adsr_env(n, SAMPLE_RATE, total_sec=dur)
    return w * env * (float(vol) * MASTER_GAIN)


def noise_tone(dur=0.2, vol=0.35):
    n = max(2, int(SAMPLE_RATE * float(dur)))
    w = (np.random.rand(n).astype(np.float32) * 2.0 - 1.0)
    env = _adsr_env(n, SAMPLE_RATE,
                    attack=0.002, decay=0.05, sustain=0.4, release=0.08,
                    total_sec=dur)
    return w * env * (float(vol) * MASTER_GAIN)


def wavetable_tone(freq=440.0, dur=0.2, vol=0.4, table=None):
    if table is None:
        table = np.sin(np.linspace(0, 2 * np.pi, 32, endpoint=False)).astype(np.float32)
    table = np.asarray(table, dtype=np.float32)
    if table.ndim != 1 or table.size < 2:
        raise ValueError("wavetable must be 1D with >=2 samples")
    n = max(2, int(SAMPLE_RATE * float(dur)))
    phase = (np.cumsum(np.full(n, float(freq) / SAMPLE_RATE, dtype=np.float32)) % 1.0)
    idx = (phase * table.size).astype(np.int32)
    w = table[idx]
    env = _adsr_env(n, SAMPLE_RATE,
                    attack=0.002, decay=0.010, sustain=1.0, release=0.020,
                    total_sec=dur)
    return w.astype(np.float32) * env * (float(vol) * MASTER_GAIN)


# ---------------------------------------------------------------------------
# Real-time waveform helpers (used by LiveMidiPlayer's audio callback)
#
# The single-sample versions (_rt_*_sample) exist for testing only; the
# vectorised buffer versions (_rt_generate_*_buffer) are what the callback
# actually calls for performance.
# ---------------------------------------------------------------------------
def _rt_pulse_sample(phase: float, duty: float) -> float:
    return 1.0 if (phase % 1.0) < duty else -1.0


def _rt_triangle_sample(phase: float) -> float:
    p = phase % 1.0
    return 4.0 * abs(p - 0.5) - 1.0


def _rt_square_sample(phase: float) -> float:
    return 1.0 if (phase % 1.0) < 0.5 else -1.0


def _rt_saw_sample(phase: float) -> float:
    return 2.0 * (phase % 1.0) - 1.0


def _rt_generate_pulse_buffer(
        start_phase: float, freq: float, duty: float, frames: int,
        pitch_bend_semitones: float = 0.0,
        vibrato_depth: float = 0.0, vibrato_rate: float = 5.0,
        vibrato_phase: float = 0.0) -> tuple[np.ndarray, float, float]:
    """Pulse buffer with optional pitch bend and vibrato.

    Returns (samples, end_phase, end_vib_phase).
    """
    t = np.arange(frames, dtype=np.float32) / SAMPLE_RATE

    bent_freq = freq * (2.0 ** (pitch_bend_semitones / 12.0))

    if vibrato_depth > 0.0:
        vib_t = vibrato_phase + vibrato_rate * t
        lfo = np.sin(_TWO_PI * vib_t)
        cents = lfo * vibrato_depth * 50.0
        inst_freq = bent_freq * (2.0 ** (cents / 1200.0))
        phases = start_phase + np.cumsum(inst_freq / SAMPLE_RATE)
        end_vib_phase = float(vib_t[-1]) % 1.0 if frames > 0 else vibrato_phase
    else:
        phases = start_phase + bent_freq * t
        end_vib_phase = vibrato_phase

    frac = np.mod(phases, 1.0)
    samples = np.where(frac < duty, 1.0, -1.0).astype(np.float32)
    end_phase = float(phases[-1]) % 1.0 if frames > 0 else start_phase
    return samples, end_phase, end_vib_phase


def _rt_generate_triangle_buffer(
        start_phase: float, freq: float, frames: int,
        pitch_bend_semitones: float = 0.0) -> tuple[np.ndarray, float]:
    t = np.arange(frames, dtype=np.float32) / SAMPLE_RATE
    bent_freq = freq * (2.0 ** (pitch_bend_semitones / 12.0))
    phases = start_phase + bent_freq * t
    frac = np.mod(phases, 1.0)
    samples = (4.0 * np.abs(frac - 0.5) - 1.0).astype(np.float32)
    end_phase = float(phases[-1]) % 1.0 if frames > 0 else start_phase
    return samples, end_phase


def _rt_generate_square_buffer(
        start_phase: float, freq: float, frames: int,
        pitch_bend_semitones: float = 0.0) -> tuple[np.ndarray, float]:
    t = np.arange(frames, dtype=np.float32) / SAMPLE_RATE
    bent_freq = freq * (2.0 ** (pitch_bend_semitones / 12.0))
    phases = start_phase + bent_freq * t
    frac = np.mod(phases, 1.0)
    samples = np.where(frac < 0.5, 1.0, -1.0).astype(np.float32)
    end_phase = float(phases[-1]) % 1.0 if frames > 0 else start_phase
    return samples, end_phase


def _rt_generate_saw_buffer(
        start_phase: float, freq: float, frames: int,
        pitch_bend_semitones: float = 0.0) -> tuple[np.ndarray, float]:
    t = np.arange(frames, dtype=np.float32) / SAMPLE_RATE
    bent_freq = freq * (2.0 ** (pitch_bend_semitones / 12.0))
    phases = start_phase + bent_freq * t
    frac = np.mod(phases, 1.0)
    samples = (2.0 * frac - 1.0).astype(np.float32)
    end_phase = float(phases[-1]) % 1.0 if frames > 0 else start_phase
    return samples, end_phase


# ---------------------------------------------------------------------------
# Drum sounds
# ---------------------------------------------------------------------------
def _drum_kick_mono(dur=0.10, vol=0.72):
    base = square_tone(60.0, dur, vol * 0.9)
    overt = square_tone(120.0, dur * 0.7, vol * 0.45)
    n = min(base.shape[0], overt.shape[0])
    return (base[:n] + overt[:n] * 0.6) * 0.95


def _drum_snare_mono(dur=0.09, vol=0.62):
    n = max(2, int(SAMPLE_RATE * dur))
    noise = (np.random.rand(n).astype(np.float32) * 2.0 - 1.0)
    env = _adsr_env(n, SAMPLE_RATE,
                    attack=0.002, decay=0.06, sustain=0.3, release=0.08,
                    total_sec=dur)
    body = square_tone(200.0, dur * 0.6, vol=0.35)
    m = min(n, body.shape[0])
    return (noise[:m] * env[:m] * vol * MASTER_GAIN * 0.9 + body[:m] * 0.5)


def _drum_hat_mono(dur=0.02, vol=0.25):
    n = max(2, int(SAMPLE_RATE * dur))
    m = (np.random.rand(n).astype(np.float32) * 2.0 - 1.0)
    env = _adsr_env(n, SAMPLE_RATE,
                    attack=0.001, decay=0.02, sustain=0.0, release=0.01,
                    total_sec=dur)
    return m * env * (vol * MASTER_GAIN)


def _drum_tom_mono(freq=110.0, dur=0.15, vol=0.5):
    return square_tone(freq, dur, vol)


# ---------------------------------------------------------------------------
# Chip profiles
# ---------------------------------------------------------------------------
def get_profiles():
    return {
        "Neutral": {
            "pan_map": {0: -0.25, 1: 0.25, 2: -0.4, 3: 0.4, 4: -0.15, 5: 0.15},
            "timbre": {
                0: dict(wave="pulse", duty=0.50, vib_rate=5.5, vib_depth_cents=12.0),
                1: dict(wave="pulse", duty=0.25, vib_rate=0.0,  vib_depth_cents=0.0),
                2: dict(wave="pulse", duty=0.75, vib_rate=0.0,  vib_depth_cents=0.0),
                3: dict(wave="pulse", duty=0.62, vib_rate=4.0,  vib_depth_cents=6.0),
                4: dict(wave="pulse", duty=0.38, vib_rate=0.0,  vib_depth_cents=0.0),
                5: dict(wave="pulse", duty=0.50, vib_rate=0.0,  vib_depth_cents=0.0),
            },
            "stereo_width": 1.0,
            "quantize": None,
            "chip": None,
        },
        "NES-ish": {
            "pan_map": {"pulse0": -0.25, "pulse1": 0.25, "tri": 0.0},
            "timbre": {
                "pulse0": dict(wave="pulse",    duty=0.125, vib_rate=5.0, vib_depth_cents=8.0),
                "pulse1": dict(wave="pulse",    duty=0.25,  vib_rate=0.0, vib_depth_cents=0.0),
                "tri":    dict(wave="triangle", duty=None,  vib_rate=0.0, vib_depth_cents=0.0),
            },
            "stereo_width": 0.9,
            "quantize": None,
            "chip": {
                "voices": [("pulse", 2), ("triangle", 1)],
                "mono": False,
                "vol_steps": 16,
                "duty_set": [0.125, 0.25, 0.5, 0.75],
                "tri_no_adsr": True,
                "voice_policy": "steal_oldest",
                "tri_split_note": 55,
            },
        },
        "GB-ish": {
            "pan_map": {"pulse0": 0.0, "pulse1": 0.0, "wave": 0.0},
            "timbre": {
                "pulse0": dict(wave="pulse",     duty=0.125, vib_rate=0.0, vib_depth_cents=0.0),
                "pulse1": dict(wave="pulse",     duty=0.5,   vib_rate=0.0, vib_depth_cents=0.0),
                "wave":   dict(wave="wavetable", duty=None,  vib_rate=0.0, vib_depth_cents=0.0),
            },
            "stereo_width": 0.0,
            "quantize": "8bit",
            "chip": {
                "voices": [("pulse", 2), ("wavetable", 1)],
                "mono": True,
                "vol_steps": 4,
                "duty_set": [0.125, 0.25, 0.5, 0.75],
                "tri_no_adsr": True,
                "voice_policy": "steal_oldest",
                "wave_table": "gb_default",
                "wave_split_note": 50,
            },
        },
        "C64-ish": {
            "pan_map": {"v0": -0.25, "v1": 0.25, "v2": 0.0},
            "timbre": {
                "v0": dict(wave="pulse", duty=0.5,  vib_rate=5.0, vib_depth_cents=14.0),
                "v1": dict(wave="saw",   duty=None, vib_rate=3.0, vib_depth_cents=8.0),
                "v2": dict(wave="pulse", duty=0.75, vib_rate=0.0, vib_depth_cents=0.0),
            },
            "stereo_width": 1.0,
            "quantize": None,
            "chip": {
                "voices": [("sid", 3)],
                "mono": False,
                "vol_steps": 16,
                "duty_set": [0.125, 0.25, 0.5, 0.75],
                "tri_no_adsr": False,
                "voice_policy": "steal_oldest",
                "sid_adsr_jitter": 0.06,
                "sid_detune_cents": 3.0,
            },
        },
        "Beeper": {
            "pan_map": {"b": 0.0},
            "timbre": {"b": dict(wave="square", duty=None, vib_rate=0.0, vib_depth_cents=0.0)},
            "stereo_width": 0.0,
            "quantize": "8bit",
            "chip": {
                "voices": [("square", 1)],
                "mono": True,
                "vol_steps": 2,
                "duty_set": None,
                "tri_no_adsr": True,
                "voice_policy": "steal_oldest",
            },
        },
    }


PROFILES = get_profiles()


# ---------------------------------------------------------------------------
# Chip-voice helpers
# ---------------------------------------------------------------------------
def _quantize_unit(x: float, steps: int):
    if steps <= 1:
        return 0.0
    x = max(0.0, min(1.0, float(x)))
    return round(x * (steps - 1)) / (steps - 1)


@lru_cache(maxsize=16)
def _choose_wave_table(name: str):
    if name == "gb_default":
        t = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        w = (0.65 * np.sin(t) + 0.25 * np.sin(2 * t) + 0.10 * np.sin(3 * t))
        w = np.clip(w, -1.0, 1.0)
        w = np.round(((w + 1.0) * 7.5)) / 7.5 - 1.0
        return w.astype(np.float32)
    return np.sin(np.linspace(0, 2 * np.pi, 32, endpoint=False)).astype(np.float32)


def _flatten_voice_spec(voice_spec):
    out = []
    for kind, cnt in voice_spec:
        if kind == "pulse":
            for i in range(cnt):
                out.append(f"pulse{i}")
        elif kind == "triangle":
            for _ in range(cnt):
                out.append("tri")
        elif kind == "wavetable":
            for _ in range(cnt):
                out.append("wave")
        elif kind == "square":
            for _ in range(cnt):
                out.append("b")
        elif kind == "sid":
            for i in range(cnt):
                out.append(f"v{i}")
        else:
            for i in range(cnt):
                out.append(f"{kind}{i}")
    return out


def assign_chip_voices(notes, profile):
    chip = profile.get("chip")
    if not chip:
        return None

    voice_keys = _flatten_voice_spec(chip["voices"])
    free_at = {vk: -1.0 for vk in voice_keys}
    tri_split = chip.get("tri_split_note", None)
    wave_split = chip.get("wave_split_note", None)

    assigned = []
    for (start, end, pitch, ch, vel, is_drum) in notes:
        if is_drum:
            assigned.append(dict(start=start, end=end, pitch=pitch, ch=ch, vel=vel,
                                 is_drum=True, voice_key="drum", wave_kind="drum"))
            continue

        desired = None
        if "tri" in voice_keys and tri_split is not None and pitch < tri_split:
            desired = "tri"
        if "wave" in voice_keys and wave_split is not None and pitch < wave_split:
            desired = "wave"
        if desired is None:
            if any(vk.startswith("pulse") for vk in voice_keys):
                desired = "pulse"
            elif any(vk.startswith("v") for vk in voice_keys):
                desired = "v"
            else:
                desired = voice_keys[0]

        if desired == "pulse":
            cands = [vk for vk in voice_keys if vk.startswith("pulse")]
        elif desired == "v":
            cands = [vk for vk in voice_keys if vk.startswith("v")]
        else:
            cands = [vk for vk in voice_keys if vk == desired] or voice_keys

        vk = None
        for c in cands:
            if free_at[c] <= start:
                vk = c
                break
        if vk is None:
            vk = min(cands, key=lambda k: free_at[k])

        free_at[vk] = end
        wave_kind = profile["timbre"].get(vk, {}).get("wave", "pulse")
        assigned.append(dict(start=start, end=end, pitch=pitch, ch=ch, vel=vel,
                             is_drum=False, voice_key=vk, wave_kind=wave_kind))
    return assigned


def apply_stereo_width(stereo: np.ndarray, width: float):
    width = float(max(0.0, min(1.5, width)))
    L = stereo[:, 0].copy()
    R = stereo[:, 1].copy()
    M = (L + R) * 0.5
    stereo[:, 0] = M + width * (L - M)
    stereo[:, 1] = M + width * (R - M)
    return stereo


def apply_quantize(stereo: np.ndarray, mode: str | None):
    if mode is None:
        return stereo
    if mode == "8bit":
        stereo[:] = np.round(stereo * 127.0) / 127.0
    return stereo


# ---------------------------------------------------------------------------
# Faithful MIDI parsing
# ---------------------------------------------------------------------------
def collect_notes_faithful(mid: mido.MidiFile):
    """Return a list of (start_sec, end_sec, note, channel, velocity, is_drum)."""
    tpq = mid.ticks_per_beat or 480
    tempo = 500000
    merged = mido.merge_tracks(mid.tracks)
    sec = 0.0
    on: dict = {}
    out = []
    for msg in merged:
        if msg.time:
            sec += (msg.time * (tempo / 1_000_000.0)) / tpq
        if msg.type == 'set_tempo':
            tempo = msg.tempo
        elif msg.type == 'note_on' and msg.velocity > 0:
            ch = getattr(msg, 'channel', 0)
            on[(ch, msg.note)] = (sec, msg.velocity)
        elif msg.type in ('note_off',) or (msg.type == 'note_on' and msg.velocity == 0):
            ch = getattr(msg, 'channel', 0)
            key = (ch, msg.note)
            if key in on:
                start_sec, vel = on.pop(key)
                out.append((start_sec, sec, msg.note, ch, vel, ch == 9))
    out.sort(key=lambda x: x[0])
    return out


def channels_in_notes(notes):
    used = set()
    for (_, _, _, ch, _, _) in notes:
        used.add(int(ch))
    return sorted(used)


# ---------------------------------------------------------------------------
# Renderer — float32 stereo output
# ---------------------------------------------------------------------------
def render_chiptune_float32(notes, profile_name="Neutral", channel_volumes=None):
    if channel_volumes is None:
        channel_volumes = {}

    profile = PROFILES.get(profile_name, PROFILES["Neutral"])
    pan_map = profile["pan_map"]
    timbre = profile["timbre"]
    stereo_width = profile["stereo_width"]
    quantize = profile["quantize"]
    chip = profile.get("chip")

    if not notes:
        return None

    total_sec = max((end for (start, end, *_) in notes)) + 0.2
    N = int(SAMPLE_RATE * total_sec)
    mixL = np.zeros(N, dtype=np.float32)
    mixR = np.zeros(N, dtype=np.float32)

    assigned = assign_chip_voices(notes, profile) if chip else None

    # Drums
    for start, end, pitch, ch, vel, is_drum in notes:
        if not is_drum:
            continue
        ch_vol = float(channel_volumes.get(int(ch), 1.0))
        if ch_vol <= 0.0001:
            continue
        t0 = int(min(N - 1, max(0, start * SAMPLE_RATE)))
        if pitch in (35, 36):
            mono = _drum_kick_mono(0.11, 0.75)
        elif pitch in (38, 40):
            mono = _drum_snare_mono(0.09, 0.62)
        elif pitch in (42, 44):
            mono = _drum_hat_mono(0.02, 0.22)
        elif pitch in (46, 49, 51):
            mono = _drum_hat_mono(0.06, 0.28)
        elif pitch in (41, 43, 45, 47, 48, 50):
            base = {41: 110.0, 43: 130.8, 45: 146.8, 47: 164.8,
                    48: 174.6, 50: 196.0}.get(pitch, 130.8)
            mono = _drum_tom_mono(base, 0.16, 0.48)
        else:
            n = max(2, int(SAMPLE_RATE * 0.04))
            env = _adsr_env(n, SAMPLE_RATE,
                            attack=0.002, decay=0.03, sustain=0.0, release=0.02,
                            total_sec=0.04)
            mono = (np.random.rand(n).astype(np.float32) * 2.0 - 1.0) * env * (0.25 * MASTER_GAIN)

        mono = mono * ch_vol
        endi = min(N, t0 + mono.shape[0])
        ln = endi - t0
        if ln > 0:
            mixL[t0:endi] += mono[:ln] * 0.9
            mixR[t0:endi] += mono[:ln] * 0.9

    # Pitched notes — Neutral / no chip
    if assigned is None:
        for start, end, pitch, ch, vel, is_drum in notes:
            if is_drum:
                continue
            ch_vol = float(channel_volumes.get(int(ch), 1.0))
            if ch_vol <= 0.0001:
                continue

            dur = max(0.02, float(end - start))
            freq = float(MIDI_TO_FREQ[pitch])
            v = (vel / 127.0) * ch_vol

            settings = timbre.get(
                ch % 6,
                dict(wave="pulse", duty=0.5, vib_rate=0.0, vib_depth_cents=0.0),
            )
            wave_kind = settings.get("wave", "pulse")

            if wave_kind == "triangle":
                mono = triangle_tone(freq, dur, vol=0.48 * v)
            elif wave_kind == "square":
                mono = square_tone(freq, dur, vol=0.48 * v)
            else:
                mono = pulse_tone(freq, dur, vol=0.48 * v,
                                  duty=settings.get('duty', 0.5),
                                  vib_rate=settings.get('vib_rate', 0.0),
                                  vib_depth_cents=settings.get('vib_depth_cents', 0.0))

            pan = pan_map.get(ch % 6, 0.0)
            stereo = _stereo_from_mono(mono, pan)

            s = int(start * SAMPLE_RATE)
            e = min(N, s + stereo.shape[0])
            ln = e - s
            if ln > 0:
                mixL[s:e] += stereo[:ln, 0]
                mixR[s:e] += stereo[:ln, 1]

    else:
        # Chip-accurate voice assignment
        vol_steps = int(chip.get("vol_steps", 16))
        duty_set = chip.get("duty_set", None)
        tri_no_adsr = bool(chip.get("tri_no_adsr", False))

        wave_table = None
        if chip.get("wave_table"):
            wave_table = _choose_wave_table(str(chip["wave_table"]))

        sid_jitter = float(chip.get("sid_adsr_jitter", 0.0))
        sid_detune = float(chip.get("sid_detune_cents", 0.0))

        for item in assigned:
            if item["is_drum"]:
                continue

            ch = int(item.get("ch", 0))
            ch_vol = float(channel_volumes.get(ch, 1.0))
            if ch_vol <= 0.0001:
                continue

            start = float(item["start"])
            end   = float(item["end"])
            pitch = int(item["pitch"])
            vel   = int(item["vel"])
            vk    = item["voice_key"]
            dur   = max(0.02, float(end - start))

            freq = float(MIDI_TO_FREQ[pitch])
            v = _quantize_unit(vel / 127.0, vol_steps) * ch_vol

            settings = timbre.get(
                vk,
                dict(wave="pulse", duty=0.5, vib_rate=0.0, vib_depth_cents=0.0),
            )
            wave_kind = settings.get("wave", "pulse")

            duty = settings.get("duty", 0.5)
            if duty_set and duty is not None:
                duty = min(duty_set, key=lambda x: abs(x - float(duty)))

            if sid_detune and vk.startswith("v"):
                cents = np.random.uniform(-sid_detune, sid_detune)
                freq = freq * (2.0 ** (cents / 1200.0))

            if wave_kind == "triangle":
                if tri_no_adsr:
                    mono = triangle_tone(freq, dur, vol=0.62 * v) * 1.15
                else:
                    mono = triangle_tone(freq, dur, vol=0.48 * v)
            elif wave_kind == "square":
                mono = square_tone(freq, dur, vol=0.55 * v)
            elif wave_kind == "saw":
                mono = saw_tone(freq, dur, vol=0.50 * v)
            elif wave_kind == "noise":
                mono = noise_tone(dur, vol=0.45 * v)
            elif wave_kind == "wavetable":
                mono = wavetable_tone(freq, dur, vol=0.60 * v, table=wave_table)
            else:
                mono = pulse_tone(freq, dur, vol=0.50 * v,
                                  duty=duty if duty is not None else 0.5,
                                  vib_rate=settings.get('vib_rate', 0.0),
                                  vib_depth_cents=settings.get('vib_depth_cents', 0.0))

            if sid_jitter and vk.startswith("v"):
                mono = mono * np.random.uniform(1.0 - sid_jitter, 1.0 + sid_jitter)

            pan = pan_map.get(vk, 0.0)
            stereo = _stereo_from_mono(mono, pan)

            s = int(start * SAMPLE_RATE)
            e = min(N, s + stereo.shape[0])
            ln = e - s
            if ln > 0:
                mixL[s:e] += stereo[:ln, 0]
                mixR[s:e] += stereo[:ln, 1]

    # Soft normalise
    peak = max(1e-6, float(max(
        abs(mixL.max()), abs(mixL.min()), abs(mixR.max()), abs(mixR.min()),
    )))
    if peak > 0.98:
        mixL *= 0.98 / peak
        mixR *= 0.98 / peak

    stereo = np.stack([mixL, mixR], axis=1).astype(np.float32, copy=False)
    stereo = apply_stereo_width(stereo, stereo_width)
    stereo = np.clip(stereo, -1.0, 1.0)
    stereo = apply_quantize(stereo, quantize)

    if chip and chip.get("mono", False):
        M = (stereo[:, 0] + stereo[:, 1]) * 0.5
        stereo[:, 0] = M
        stereo[:, 1] = M

    return stereo.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------
def export_wav_float32(path: str, stereo_f32: np.ndarray):
    stereo_i16 = (np.clip(stereo_f32, -1.0, 1.0) * 32767.0).astype(np.int16, copy=False)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(stereo_i16.tobytes())


def export_mp3_via_ffmpeg(path: str, stereo_f32: np.ndarray, quality_q: int = 2):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg not found in PATH")

    with tempfile.TemporaryDirectory() as td:
        tmp_wav = str(Path(td) / "tmp.wav")
        export_wav_float32(tmp_wav, stereo_f32)

        cmd = [
            ffmpeg, "-y",
            "-i", tmp_wav,
            "-codec:a", "libmp3lame",
            "-q:a", str(int(quality_q)),
            str(path),
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            raise RuntimeError(p.stderr.strip() or "ffmpeg failed")
