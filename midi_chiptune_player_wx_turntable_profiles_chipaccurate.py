# MIDI Chiptune Player — Turntable Pitch (LIVE) + Chip Profiles + WAV/MP3 Export
# + Master Volume Slider (live, NVDA-friendly)
# + Per-used-MIDI-Channel Volume Sliders (auto re-render on change, resume position)
# + Seek (scrub) slider + elapsed/total time (NVDA-friendly)
# + MP3 export via FFmpeg if installed (otherwise shows install hint)
# + Space key is NOT hijacked globally (works normally on focused controls)

import math
import threading
import wave
import time
import tempfile
import subprocess
import shutil
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass, field

import numpy as np
import mido
import wx
import sounddevice as sd

# Optional Numba JIT compilation for performancee
try:
    from numba import jit, njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Fallback: no-op decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if args and callable(args[0]):
            return args[0]
        return decorator

    njit = jit

# --------------------- Audio/Render config ---------------------
SAMPLE_RATE = 44100
MASTER_GAIN = 0.85  # internal overall gain used during synthesis (not the user master volume)

# Pre-computed MIDI note to frequency lookup table (performance optimization)
MIDI_TO_FREQ = np.array([440.0 * (2.0 ** ((i - 69.0) / 12.0)) for i in range(128)], dtype=np.float32)

# Pre-computed sin/cos lookups for common operations
_TWO_PI = 2.0 * np.pi
_INV_PI = 1.0 / np.pi


# --------------------- Accessibility helpers ---------------------
def _a11y(win: wx.Window, name: str, help_text: str | None = None):
    """Set NVDA-friendly label/name/helptext on controls."""
    try:
        win.SetName(name)
    except Exception:
        pass
    try:
        win.SetHelpText(help_text or name)
    except Exception:
        pass


# --------------------- Synth helpers ---------------------
def _pan_gains(pan: float):
    p = max(-1.0, min(1.0, float(pan)))
    angle = (p + 1.0) * math.pi / 4.0  # equal-power
    return math.cos(angle), math.sin(angle)


def _stereo_from_mono(mono: np.ndarray, pan: float):
    gL, gR = _pan_gains(pan)
    return np.stack([mono * gL, mono * gR], axis=1)


def _adsr_env(n: int, sr: int, attack=0.008, decay=0.050, sustain=0.65, release=0.120, total_sec=None):
    """Optimized ADSR envelope with pre-allocation (faster than concatenate)."""
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


def pulse_tone(freq=440.0, dur=0.2, vol=0.4, duty=0.5, vib_rate=0.0, vib_depth_cents=0.0):
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
    env = _adsr_env(n, SAMPLE_RATE, attack=0.002, decay=0.05, sustain=0.4, release=0.08, total_sec=dur)
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
    env = _adsr_env(n, SAMPLE_RATE, attack=0.002, decay=0.010, sustain=1.0, release=0.020, total_sec=dur)
    return w.astype(np.float32) * env * (float(vol) * MASTER_GAIN)


# --------------------- Real-time waveform functions (for live MIDI) ---------------------
def _rt_pulse_sample(phase: float, duty: float) -> float:
    """Single pulse sample at given phase (0.0-1.0)."""
    return 1.0 if (phase % 1.0) < duty else -1.0


def _rt_triangle_sample(phase: float) -> float:
    """Single triangle sample at given phase (0.0-1.0)."""
    p = phase % 1.0
    return 4.0 * abs(p - 0.5) - 1.0


def _rt_square_sample(phase: float) -> float:
    """Single square sample at given phase (0.0-1.0)."""
    return 1.0 if (phase % 1.0) < 0.5 else -1.0


def _rt_saw_sample(phase: float) -> float:
    """Single sawtooth sample at given phase (0.0-1.0)."""
    return 2.0 * (phase % 1.0) - 1.0


# Vectorized versions for buffer generation (much faster)
def _rt_generate_pulse_buffer(start_phase: float, freq: float, duty: float,
                               frames: int, pitch_bend_semitones: float = 0.0,
                               vibrato_depth: float = 0.0, vibrato_rate: float = 5.0,
                               vibrato_phase: float = 0.0) -> tuple[np.ndarray, float, float]:
    """Generate pulse buffer with optional pitch bend and vibrato. Returns (samples, end_phase, end_vib_phase)."""
    t = np.arange(frames, dtype=np.float32) / SAMPLE_RATE

    # Apply pitch bend (±2 semitones)
    bent_freq = freq * (2.0 ** (pitch_bend_semitones / 12.0))

    # Apply vibrato (LFO)
    if vibrato_depth > 0.0:
        vib_t = vibrato_phase + vibrato_rate * t
        lfo = np.sin(_TWO_PI * vib_t)
        # Vibrato depth in cents (max ~50 cents)
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


def _rt_generate_triangle_buffer(start_phase: float, freq: float, frames: int,
                                  pitch_bend_semitones: float = 0.0) -> tuple[np.ndarray, float]:
    """Generate triangle buffer. Returns (samples, end_phase)."""
    t = np.arange(frames, dtype=np.float32) / SAMPLE_RATE
    bent_freq = freq * (2.0 ** (pitch_bend_semitones / 12.0))
    phases = start_phase + bent_freq * t
    frac = np.mod(phases, 1.0)
    samples = (4.0 * np.abs(frac - 0.5) - 1.0).astype(np.float32)
    end_phase = float(phases[-1]) % 1.0 if frames > 0 else start_phase
    return samples, end_phase


def _rt_generate_square_buffer(start_phase: float, freq: float, frames: int,
                                pitch_bend_semitones: float = 0.0) -> tuple[np.ndarray, float]:
    """Generate square buffer. Returns (samples, end_phase)."""
    t = np.arange(frames, dtype=np.float32) / SAMPLE_RATE
    bent_freq = freq * (2.0 ** (pitch_bend_semitones / 12.0))
    phases = start_phase + bent_freq * t
    frac = np.mod(phases, 1.0)
    samples = np.where(frac < 0.5, 1.0, -1.0).astype(np.float32)
    end_phase = float(phases[-1]) % 1.0 if frames > 0 else start_phase
    return samples, end_phase


def _rt_generate_saw_buffer(start_phase: float, freq: float, frames: int,
                             pitch_bend_semitones: float = 0.0) -> tuple[np.ndarray, float]:
    """Generate sawtooth buffer. Returns (samples, end_phase)."""
    t = np.arange(frames, dtype=np.float32) / SAMPLE_RATE
    bent_freq = freq * (2.0 ** (pitch_bend_semitones / 12.0))
    phases = start_phase + bent_freq * t
    frac = np.mod(phases, 1.0)
    samples = (2.0 * frac - 1.0).astype(np.float32)
    end_phase = float(phases[-1]) % 1.0 if frames > 0 else start_phase
    return samples, end_phase


# --------------------- Drums ---------------------
def _drum_kick_mono(dur=0.10, vol=0.72):
    base = square_tone(60.0, dur, vol * 0.9)
    overt = square_tone(120.0, dur * 0.7, vol * 0.45)
    n = min(base.shape[0], overt.shape[0])
    return (base[:n] + overt[:n] * 0.6) * 0.95


def _drum_snare_mono(dur=0.09, vol=0.62):
    n = max(2, int(SAMPLE_RATE * dur))
    noise = (np.random.rand(n).astype(np.float32) * 2.0 - 1.0)
    env = _adsr_env(n, SAMPLE_RATE, attack=0.002, decay=0.06, sustain=0.3, release=0.08, total_sec=dur)
    body = square_tone(200.0, dur * 0.6, vol=0.35)
    m = min(n, body.shape[0])
    return (noise[:m] * env[:m] * vol * MASTER_GAIN * 0.9 + body[:m] * 0.5)


def _drum_hat_mono(dur=0.02, vol=0.25):
    n = max(2, int(SAMPLE_RATE * dur))
    m = (np.random.rand(n).astype(np.float32) * 2.0 - 1.0)
    env = _adsr_env(n, SAMPLE_RATE, attack=0.001, decay=0.02, sustain=0.0, release=0.01, total_sec=dur)
    return m * env * (vol * MASTER_GAIN)


def _drum_tom_mono(freq=110.0, dur=0.15, vol=0.5):
    return square_tone(freq, dur, vol)


# --------------------- Profiles ---------------------
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
                "pulse0": dict(wave="pulse", duty=0.125, vib_rate=5.0, vib_depth_cents=8.0),
                "pulse1": dict(wave="pulse", duty=0.25,  vib_rate=0.0, vib_depth_cents=0.0),
                "tri":    dict(wave="triangle", duty=None, vib_rate=0.0, vib_depth_cents=0.0),
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
                "pulse0": dict(wave="pulse", duty=0.125, vib_rate=0.0, vib_depth_cents=0.0),
                "pulse1": dict(wave="pulse", duty=0.5,   vib_rate=0.0, vib_depth_cents=0.0),
                "wave":   dict(wave="wavetable", duty=None, vib_rate=0.0, vib_depth_cents=0.0),
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
        }
    }


PROFILES = get_profiles()


# --------------------- Live MIDI Voice ---------------------
@dataclass
class LiveVoice:
    """Single voice state for real-time MIDI synthesis."""
    active: bool = False
    note: int = 0
    channel: int = 0
    velocity: float = 0.0
    freq: float = 440.0
    phase: float = 0.0
    env_phase: int = 0      # 0=attack, 1=decay, 2=sustain, 3=release, 4=off
    env_pos: int = 0
    env_level: float = 0.0
    wave_kind: str = "pulse"
    duty: float = 0.5
    start_time: float = 0.0
    # MIDI controller state
    pitch_bend: float = 0.0      # -1.0 to +1.0 (±2 semitones)
    vibrato_depth: float = 0.0   # 0.0 to 1.0 (from mod wheel)
    sustained: bool = False      # Held by sustain pedal

    def reset(self):
        """Reset voice to inactive state."""
        self.active = False
        self.note = 0
        self.channel = 0
        self.velocity = 0.0
        self.freq = 440.0
        self.phase = 0.0
        self.env_phase = 4
        self.env_pos = 0
        self.env_level = 0.0
        self.wave_kind = "pulse"
        self.duty = 0.5
        self.start_time = 0.0
        self.pitch_bend = 0.0
        self.vibrato_depth = 0.0
        self.sustained = False


# --------------------- Chip-ish helpers ---------------------
def _quantize_unit(x: float, steps: int):
    if steps <= 1:
        return 0.0
    x = max(0.0, min(1.0, float(x)))
    return round(x * (steps - 1)) / (steps - 1)


@lru_cache(maxsize=16)
def _choose_wave_table(name: str):
    """Cached wavetable generation (performance optimization)."""
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
            assigned.append(dict(start=start, end=end, pitch=pitch, ch=ch, vel=vel, is_drum=True,
                                 voice_key="drum", wave_kind="drum"))
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
        assigned.append(dict(start=start, end=end, pitch=pitch, ch=ch, vel=vel, is_drum=False,
                             voice_key=vk, wave_kind=wave_kind))
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


# --------------------- Faithful MIDI parsing ---------------------
def collect_notes_faithful(mid: mido.MidiFile):
    tpq = mid.ticks_per_beat or 480
    tempo = 500000
    merged = mido.merge_tracks(mid.tracks)
    sec = 0.0
    on = {}
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


# --------------------- Render to float32 stereo (with profile) ---------------------
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

    # -------- Drums --------
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
            base = {41: 110.0, 43: 130.8, 45: 146.8, 47: 164.8, 48: 174.6, 50: 196.0}.get(pitch, 130.8)
            mono = _drum_tom_mono(base, 0.16, 0.48)
        else:
            n = max(2, int(SAMPLE_RATE * 0.04))
            env = _adsr_env(n, SAMPLE_RATE, attack=0.002, decay=0.03, sustain=0.0, release=0.02, total_sec=0.04)
            mono = (np.random.rand(n).astype(np.float32) * 2.0 - 1.0) * env * (0.25 * MASTER_GAIN)

        mono = mono * ch_vol
        endi = min(N, t0 + mono.shape[0])
        ln = endi - t0
        if ln > 0:
            mixL[t0:endi] += mono[:ln] * 0.9
            mixR[t0:endi] += mono[:ln] * 0.9

    # -------- Pitched notes --------
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

            settings = timbre.get(ch % 6, dict(wave="pulse", duty=0.5, vib_rate=0.0, vib_depth_cents=0.0))
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
            end = float(item["end"])
            pitch = int(item["pitch"])
            vel = int(item["vel"])
            vk = item["voice_key"]
            dur = max(0.02, float(end - start))

            freq = float(MIDI_TO_FREQ[pitch])
            v = _quantize_unit(vel / 127.0, vol_steps) * ch_vol

            settings = timbre.get(vk, dict(wave="pulse", duty=0.5, vib_rate=0.0, vib_depth_cents=0.0))
            wave_kind = settings.get("wave", "pulse")

            duty = settings.get("duty", 0.5)
            if duty_set and duty is not None:
                duty = min(duty_set, key=lambda x: abs(x - float(duty)))

            if sid_detune and vk.startswith("v"):
                cents = np.random.uniform(-sid_detune, sid_detune)
                freq = freq * (2.0 ** (cents / 1200.0))

            if wave_kind == "triangle":
                if tri_no_adsr:
                    mono = triangle_tone(freq, dur, vol=0.62 * v)
                    mono *= 1.15
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

    # -------- Soft normalize --------
    peak = max(1e-6, float(max(abs(mixL.max()), abs(mixL.min()), abs(mixR.max()), abs(mixR.min()))))
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


# --------------------- Export helpers ---------------------
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


# --------------------- Live turntable streamer ---------------------
class TurntablePlayer:
    def __init__(self, data: np.ndarray, loop: bool = True, volume: float = 1.0):
        assert data.ndim == 2 and data.shape[1] == 2, "data must be (N,2)"
        self.data = data
        self.n = data.shape[0]
        self.loop = loop
        self.rate = 1.0
        self.idx = 0.0
        self.volume = float(volume)
        self.stream: sd.OutputStream | None = None
        self.lock = threading.Lock()
        self.playing = False

    def set_rate(self, rate: float):
        rate = max(0.1, min(3.0, float(rate)))
        with self.lock:
            self.rate = rate

    def set_loop(self, loop: bool):
        with self.lock:
            self.loop = bool(loop)

    def set_volume(self, volume: float):
        volume = max(0.0, min(2.0, float(volume)))
        with self.lock:
            self.volume = volume

    def stop(self, reset_position: bool = False):
        with self.lock:
            self.playing = False
            if reset_position:
                self.idx = 0.0
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        finally:
            self.stream = None

    def pause(self):
        with self.lock:
            self.playing = False
        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:
                pass

    def resume(self):
        """Resume if paused. If stream was closed, recreate it."""
        try:
            if self.stream is None:
                self.stream = sd.OutputStream(
                    channels=2,
                    dtype='float32',
                    samplerate=SAMPLE_RATE,
                    blocksize=512,
                    callback=self._callback
                )
            self.stream.start()
            with self.lock:
                self.playing = True
            return True
        except Exception:
            return False

    def is_playing(self):
        return self.playing and self.stream is not None

    def get_fraction(self) -> float:
        with self.lock:
            if self.n <= 1:
                return 0.0
            return float((self.idx % (self.n - 1)) / (self.n - 1))

    def set_fraction(self, frac: float):
        frac = max(0.0, min(1.0, float(frac)))
        with self.lock:
            self.idx = frac * (self.n - 1)

    def get_time_seconds(self) -> float:
        with self.lock:
            return float(self.idx / SAMPLE_RATE)

    def get_total_seconds(self) -> float:
        return float(self.n / SAMPLE_RATE)

    def _callback(self, outdata, frames, _time_info, status):
        """Optimized callback with vectorized operations for better performance."""
        out = outdata.view(dtype=np.float32).reshape((-1, 2))
        buf = self.data
        n = self.n

        with self.lock:
            rate = float(self.rate)
            loop = bool(self.loop)
            idx = float(self.idx)
            vol = float(self.volume)

        indices = idx + np.arange(frames, dtype=np.float32) * rate

        if loop:
            indices = np.mod(indices, n - 1)
            i0 = indices.astype(np.int32)
            i1 = np.mod(i0 + 1, n)
            frac = (indices - i0).reshape(-1, 1)
            out[:] = ((1.0 - frac) * buf[i0] + frac * buf[i1]) * vol

            final_idx = float(indices[-1] + rate)
            if final_idx >= n - 1:
                final_idx = np.mod(final_idx, n - 1)
        else:
            valid_mask = indices < (n - 1)
            valid_count = int(np.sum(valid_mask))

            if valid_count == 0:
                out.fill(0.0)
                final_idx = n - 1
                wx.CallAfter(self._later_stop_safe)
            elif valid_count < frames:
                valid_indices = indices[:valid_count]
                i0 = valid_indices.astype(np.int32)
                i1 = np.minimum(i0 + 1, n - 1)
                frac = (valid_indices - i0).reshape(-1, 1)

                out[:valid_count] = ((1.0 - frac) * buf[i0] + frac * buf[i1]) * vol
                out[valid_count:] = 0.0
                final_idx = n - 1
                wx.CallAfter(self._later_stop_safe)
            else:
                i0 = indices.astype(np.int32)
                i1 = np.minimum(i0 + 1, n - 1)
                frac = (indices - i0).reshape(-1, 1)
                out[:] = ((1.0 - frac) * buf[i0] + frac * buf[i1]) * vol
                final_idx = float(indices[-1] + rate)

        with self.lock:
            self.idx = float(final_idx)

    def _later_stop_safe(self):
        try:
            if self.stream is not None:
                self.stream.stop()
        except Exception:
            pass
        with self.lock:
            self.playing = False

    def play(self, start_at_fraction: float | None = None):
        self.stop(reset_position=False)
        if start_at_fraction is not None:
            with self.lock:
                self.idx = float(start_at_fraction % 1.0) * (self.n - 1)

        self.stream = sd.OutputStream(
            channels=2,
            dtype='float32',
            samplerate=SAMPLE_RATE,
            blocksize=512,
            callback=self._callback
        )
        self.stream.start()
        with self.lock:
            self.playing = True
        return True


# --------------------- Live MIDI Player ---------------------
class LiveMidiPlayer:
    """Real-time MIDI input synthesizer for live keyboard playing."""

    # ADSR timing in samples (at 44100 Hz)
    ATTACK_SAMPLES = int(0.008 * SAMPLE_RATE)   # 8ms
    DECAY_SAMPLES = int(0.050 * SAMPLE_RATE)    # 50ms
    SUSTAIN_LEVEL = 0.65
    RELEASE_SAMPLES = int(0.120 * SAMPLE_RATE)  # 120ms

    def __init__(self, num_voices: int = 12, profile_name: str = "Neutral"):
        self.num_voices = num_voices
        self.voices: list[LiveVoice] = [LiveVoice() for _ in range(num_voices)]
        self.profile_name = profile_name
        self.volume = 0.8
        self.active = False

        self.midi_port: mido.ports.BaseInput | None = None
        self.stream: sd.OutputStream | None = None
        self.lock = threading.Lock()

        # Per-channel state for controllers
        self._channel_pitch_bend: dict[int, float] = {ch: 0.0 for ch in range(16)}
        self._channel_mod_wheel: dict[int, float] = {ch: 0.0 for ch in range(16)}
        self._channel_sustain: dict[int, bool] = {ch: False for ch in range(16)}

        # Vibrato phase tracking per voice (for continuous LFO)
        self._voice_vib_phase: list[float] = [0.0 for _ in range(num_voices)]

        # Status callback for UI updates
        self._status_callback = None

    def set_status_callback(self, callback):
        """Set callback for status updates (active voice count, etc.)."""
        self._status_callback = callback

    def _notify_status(self):
        """Notify status callback if set."""
        if self._status_callback:
            with self.lock:
                active_count = sum(1 for v in self.voices if v.active)
            try:
                wx.CallAfter(self._status_callback, active_count)
            except Exception:
                pass

    # ----- MIDI Port Management -----
    def list_midi_inputs(self) -> list[str]:
        """List available MIDI input ports."""
        try:
            return mido.get_input_names()
        except Exception:
            return []

    def open_midi_input(self, port_name: str | None = None) -> bool:
        """Open MIDI input port. Returns True on success."""
        self.close_midi_input()
        try:
            if port_name is None:
                ports = self.list_midi_inputs()
                if not ports:
                    return False
                port_name = ports[0]

            self.midi_port = mido.open_input(port_name, callback=self._midi_callback)
            return True
        except Exception as e:
            print(f"MIDI open error: {e}")
            return False

    def close_midi_input(self):
        """Close MIDI input port."""
        if self.midi_port is not None:
            try:
                self.midi_port.close()
            except Exception:
                pass
            self.midi_port = None

    def get_midi_port_name(self) -> str | None:
        """Get currently open MIDI port name."""
        if self.midi_port is not None:
            try:
                return self.midi_port.name
            except Exception:
                pass
        return None

    # ----- Audio Stream Management -----
    def start(self) -> bool:
        """Start audio output stream."""
        if self.active:
            return True
        try:
            self.stream = sd.OutputStream(
                channels=2,
                dtype='float32',
                samplerate=SAMPLE_RATE,
                blocksize=256,  # Low latency
                latency='low',
                callback=self._audio_callback
            )
            self.stream.start()
            self.active = True
            return True
        except Exception as e:
            print(f"Audio stream error: {e}")
            return False

    def stop(self):
        """Stop audio output stream."""
        self.active = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        # Clear all voices
        with self.lock:
            for v in self.voices:
                v.reset()

    def set_volume(self, volume: float):
        """Set live playback volume (0.0 to 2.0)."""
        with self.lock:
            self.volume = max(0.0, min(2.0, float(volume)))

    def set_profile(self, profile_name: str):
        """Set chip profile for timbre."""
        with self.lock:
            self.profile_name = profile_name

    # ----- MIDI Callback (runs on MIDI thread) -----
    def _midi_callback(self, msg: mido.Message):
        """Handle incoming MIDI messages."""
        if msg.type == 'note_on' and msg.velocity > 0:
            self._note_on(msg.note, msg.velocity, msg.channel)
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            self._note_off(msg.note, msg.channel)
        elif msg.type == 'control_change':
            self._handle_cc(msg.control, msg.value, msg.channel)
        elif msg.type == 'pitchwheel':
            self._handle_pitch_bend(msg.pitch, msg.channel)

    def _handle_cc(self, control: int, value: int, channel: int):
        """Handle Control Change messages."""
        with self.lock:
            if control == 1:  # Mod Wheel
                self._channel_mod_wheel[channel] = value / 127.0
                # Update active voices on this channel
                for v in self.voices:
                    if v.active and v.channel == channel:
                        v.vibrato_depth = value / 127.0
            elif control == 64:  # Sustain Pedal
                sustain_on = value >= 64
                old_sustain = self._channel_sustain.get(channel, False)
                self._channel_sustain[channel] = sustain_on

                if not sustain_on and old_sustain:
                    # Pedal released - release all sustained notes on this channel
                    for v in self.voices:
                        if v.active and v.channel == channel and v.sustained:
                            v.sustained = False
                            v.env_phase = 3  # Start release
                            v.env_pos = 0

    def _handle_pitch_bend(self, pitch: int, channel: int):
        """Handle Pitch Bend messages."""
        # pitch is -8192 to +8191, map to ±2 semitones
        bend_semitones = (pitch / 8192.0) * 2.0
        with self.lock:
            self._channel_pitch_bend[channel] = bend_semitones
            # Update active voices on this channel
            for v in self.voices:
                if v.active and v.channel == channel:
                    v.pitch_bend = bend_semitones

    def _note_on(self, note: int, velocity: int, channel: int):
        """Activate a voice for a new note."""
        with self.lock:
            # Find free voice or steal oldest
            voice_idx = self._find_free_voice()
            if voice_idx < 0:
                voice_idx = self._steal_oldest_voice()

            voice = self.voices[voice_idx]

            # Get timbre from profile
            profile = PROFILES.get(self.profile_name, PROFILES["Neutral"])
            timbre = profile["timbre"]
            chip = profile.get("chip")

            # Determine voice key and settings
            if chip:
                voice_keys = _flatten_voice_spec(chip["voices"])
                # Simple round-robin assignment based on channel
                vk = voice_keys[channel % len(voice_keys)]
                settings = timbre.get(vk, {"wave": "pulse", "duty": 0.5})
            else:
                settings = timbre.get(channel % 6, {"wave": "pulse", "duty": 0.5})

            # Initialize voice
            voice.active = True
            voice.note = note
            voice.channel = channel
            voice.velocity = velocity / 127.0
            voice.freq = MIDI_TO_FREQ[note] if 0 <= note < 128 else 440.0
            voice.phase = 0.0
            voice.env_phase = 0  # Attack
            voice.env_pos = 0
            voice.env_level = 0.0
            voice.start_time = time.time()

            voice.wave_kind = settings.get("wave", "pulse")
            voice.duty = settings.get("duty", 0.5) or 0.5

            # Apply current controller state
            voice.pitch_bend = self._channel_pitch_bend.get(channel, 0.0)
            voice.vibrato_depth = self._channel_mod_wheel.get(channel, 0.0)
            voice.sustained = False

            # Reset vibrato phase for this voice
            self._voice_vib_phase[voice_idx] = 0.0

        self._notify_status()

    def _note_off(self, note: int, channel: int):
        """Release a note (start release phase or mark as sustained)."""
        with self.lock:
            sustain_on = self._channel_sustain.get(channel, False)

            # Release ALL voices with this note/channel (not just the first one)
            for v in self.voices:
                if v.active and v.note == note and v.channel == channel and v.env_phase < 3:
                    if sustain_on:
                        # Mark as sustained, don't release yet
                        v.sustained = True
                    else:
                        # Start release phase
                        v.env_phase = 3
                        v.env_pos = 0
                    # Don't break - release all matching voices

        self._notify_status()

    def _find_free_voice(self) -> int:
        """Find an inactive voice or one in release phase. Returns -1 if none available."""
        # First, look for completely inactive voices
        for i, v in enumerate(self.voices):
            if not v.active:
                return i

        # Second, look for voices in release phase (can be reused)
        for i, v in enumerate(self.voices):
            if v.active and v.env_phase >= 3:  # Release or off
                return i

        return -1

    def _steal_oldest_voice(self) -> int:
        """Steal the oldest active voice, preferring those in release phase."""
        # First try to steal a voice in release phase
        oldest_release_idx = -1
        oldest_release_time = float('inf')

        oldest_idx = 0
        oldest_time = float('inf')

        for i, v in enumerate(self.voices):
            if v.active:
                if v.env_phase >= 3 and v.start_time < oldest_release_time:
                    oldest_release_time = v.start_time
                    oldest_release_idx = i
                if v.start_time < oldest_time:
                    oldest_time = v.start_time
                    oldest_idx = i

        # Prefer stealing a voice already in release
        if oldest_release_idx >= 0:
            return oldest_release_idx
        return oldest_idx

    # ----- Audio Callback (runs on audio thread) -----
    def _audio_callback(self, outdata, frames, time_info, status):
        """Generate audio for all active voices."""
        out = outdata.view(dtype=np.float32).reshape((-1, 2))
        out.fill(0.0)

        if not self.active:
            return

        with self.lock:
            vol = self.volume
            profile = PROFILES.get(self.profile_name, PROFILES["Neutral"])
            pan_map = profile["pan_map"]
            stereo_width = profile["stereo_width"]

            # Process each voice
            for i, voice in enumerate(self.voices):
                if not voice.active:
                    continue

                # Generate waveform buffer
                samples, new_phase, new_vib_phase = self._generate_voice_buffer(
                    voice, frames, self._voice_vib_phase[i]
                )

                # Apply envelope
                env_samples, new_env_phase, new_env_pos, new_env_level, still_active = \
                    self._apply_envelope(voice, frames)

                # Combine waveform with envelope
                samples = samples * env_samples * voice.velocity * vol * MASTER_GAIN

                # Apply panning
                vk = self._get_voice_key(voice)
                pan = pan_map.get(vk, pan_map.get(voice.channel % 6, 0.0))
                gL, gR = _pan_gains(pan)

                # Mix into output
                out[:, 0] += samples * gL
                out[:, 1] += samples * gR

                # Update voice state
                voice.phase = new_phase
                voice.env_phase = new_env_phase
                voice.env_pos = new_env_pos
                voice.env_level = new_env_level
                self._voice_vib_phase[i] = new_vib_phase

                if not still_active:
                    voice.reset()

        # Apply stereo width
        if stereo_width != 1.0:
            L = out[:, 0].copy()
            R = out[:, 1].copy()
            M = (L + R) * 0.5
            out[:, 0] = M + stereo_width * (L - M)
            out[:, 1] = M + stereo_width * (R - M)

        # Soft clip to prevent harsh clipping
        np.clip(out, -1.0, 1.0, out=out)

    def _get_voice_key(self, voice: LiveVoice) -> str:
        """Get profile voice key for panning lookup."""
        profile = PROFILES.get(self.profile_name, PROFILES["Neutral"])
        chip = profile.get("chip")
        if chip:
            voice_keys = _flatten_voice_spec(chip["voices"])
            return voice_keys[voice.channel % len(voice_keys)]
        return voice.channel % 6

    def _generate_voice_buffer(self, voice: LiveVoice, frames: int,
                                vib_phase: float) -> tuple[np.ndarray, float, float]:
        """Generate waveform buffer for a voice."""
        wave = voice.wave_kind
        freq = voice.freq
        phase = voice.phase
        duty = voice.duty
        pitch_bend = voice.pitch_bend
        vib_depth = voice.vibrato_depth

        if wave == "pulse":
            samples, new_phase, new_vib_phase = _rt_generate_pulse_buffer(
                phase, freq, duty, frames, pitch_bend, vib_depth, 5.0, vib_phase
            )
        elif wave == "triangle":
            samples, new_phase = _rt_generate_triangle_buffer(phase, freq, frames, pitch_bend)
            new_vib_phase = vib_phase
        elif wave == "square":
            samples, new_phase = _rt_generate_square_buffer(phase, freq, frames, pitch_bend)
            new_vib_phase = vib_phase
        elif wave == "saw":
            samples, new_phase = _rt_generate_saw_buffer(phase, freq, frames, pitch_bend)
            new_vib_phase = vib_phase
        else:
            # Default to pulse
            samples, new_phase, new_vib_phase = _rt_generate_pulse_buffer(
                phase, freq, 0.5, frames, pitch_bend, vib_depth, 5.0, vib_phase
            )

        return samples, new_phase, new_vib_phase

    def _apply_envelope(self, voice: LiveVoice, frames: int) -> tuple[np.ndarray, int, int, float, bool]:
        """Apply ADSR envelope to voice. Returns (env_samples, new_phase, new_pos, new_level, still_active)."""
        env = np.empty(frames, dtype=np.float32)
        phase = voice.env_phase
        pos = voice.env_pos
        level = voice.env_level

        for i in range(frames):
            if phase == 0:  # Attack
                level = pos / max(1, self.ATTACK_SAMPLES)
                pos += 1
                if pos >= self.ATTACK_SAMPLES:
                    phase = 1
                    pos = 0
            elif phase == 1:  # Decay
                level = 1.0 - (1.0 - self.SUSTAIN_LEVEL) * (pos / max(1, self.DECAY_SAMPLES))
                pos += 1
                if pos >= self.DECAY_SAMPLES:
                    phase = 2
                    pos = 0
                    level = self.SUSTAIN_LEVEL
            elif phase == 2:  # Sustain
                level = self.SUSTAIN_LEVEL
            elif phase == 3:  # Release
                start_level = voice.env_level if pos == 0 else level
                level = start_level * (1.0 - pos / max(1, self.RELEASE_SAMPLES))
                pos += 1
                if pos >= self.RELEASE_SAMPLES or level <= 0.001:
                    phase = 4
                    level = 0.0
            else:  # Off
                level = 0.0

            env[i] = level

        still_active = phase < 4
        return env, phase, pos, level, still_active

    def get_active_voice_count(self) -> int:
        """Get number of currently active voices."""
        with self.lock:
            return sum(1 for v in self.voices if v.active)

    def panic(self):
        """All notes off - emergency stop all voices."""
        with self.lock:
            for v in self.voices:
                v.reset()
            # Reset all controller states
            for ch in range(16):
                self._channel_pitch_bend[ch] = 0.0
                self._channel_mod_wheel[ch] = 0.0
                self._channel_sustain[ch] = False
        self._notify_status()


# --------------------- wx GUI ---------------------
ID_OPEN = wx.NewIdRef()
ID_EXPORT = wx.NewIdRef()
ID_STOP = wx.NewIdRef()
ID_LOOP = wx.NewIdRef()
ID_EXIT = wx.NewIdRef()
ID_SPEED_RESET = wx.NewIdRef()
ID_SPEED_SLOWER = wx.NewIdRef()
ID_SPEED_FASTER = wx.NewIdRef()

ID_PROF_NEUTRAL = wx.NewIdRef()
ID_PROF_NES = wx.NewIdRef()
ID_PROF_GB = wx.NewIdRef()
ID_PROF_C64 = wx.NewIdRef()
ID_PROF_BEEPER = wx.NewIdRef()

# Live MIDI IDs
ID_LIVE_TOGGLE = wx.NewIdRef()
ID_LIVE_REFRESH = wx.NewIdRef()
ID_LIVE_PANIC = wx.NewIdRef()

PROFILE_ID_MAP = {
    ID_PROF_NEUTRAL: "Neutral",
    ID_PROF_NES: "NES-ish",
    ID_PROF_GB: "GB-ish",
    ID_PROF_C64: "C64-ish",
    ID_PROF_BEEPER: "Beeper",
}


def _fmt_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    s = int(sec + 0.5)
    m = s // 60
    ss = s % 60
    h = m // 60
    mm = m % 60
    if h > 0:
        return f"{h}:{mm:02d}:{ss:02d}"
    return f"{mm}:{ss:02d}"


class MainFrame(wx.Frame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.SetTitle("MIDI Chiptune Player — Turntable + Chip Profiles")
        self.SetSize((980, 640))
        self.Centre()

        self.filepath: Path | None = None
        self.notes = None
        self.audio_f32: np.ndarray | None = None
        self.player: TurntablePlayer | None = None
        self.profile_name: str = "Neutral"

        self.master_volume = 1.0  # 0..2
        self.channel_volumes = {ch: 1.0 for ch in range(16)}
        self.used_channels = []

        self._scrubbing = False

        # Live MIDI player
        self.live_midi_player: LiveMidiPlayer | None = None
        self._live_midi_enabled = False

        self._rerender_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_rerender_timer, self._rerender_timer)
        self._rerender_pending = False

        self._ui_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_ui_timer, self._ui_timer)

        self._build_menu()
        self._build_body()

        self._ui_timer.Start(200)

        wx.CallAfter(self.on_open)

    def _build_menu(self):
        menubar = wx.MenuBar()

        file_menu = wx.Menu()
        file_menu.Append(ID_OPEN, "&Open MIDI\tCtrl+O")
        file_menu.Append(ID_EXPORT, "Export Audio...\tCtrl+S")
        file_menu.AppendSeparator()
        file_menu.Append(ID_EXIT, "E&xit\tCtrl+Q")
        menubar.Append(file_menu, "&File")

        play_menu = wx.Menu()
        play_menu.Append(ID_STOP, "&Stop")
        play_menu.AppendCheckItem(ID_LOOP, "&Loop\tCtrl+L")
        play_menu.Check(ID_LOOP, True)
        menubar.Append(play_menu, "&Playback")

        speed_menu = wx.Menu()
        speed_menu.Append(ID_SPEED_SLOWER, "Slower\tCtrl+,")
        speed_menu.Append(ID_SPEED_FASTER, "Faster\tCtrl+.")
        speed_menu.Append(ID_SPEED_RESET, "Reset 100%\tCtrl+0")
        menubar.Append(speed_menu, "&Speed")

        prof_menu = wx.Menu()
        prof_menu.AppendRadioItem(ID_PROF_NEUTRAL, "Neutral\tCtrl+1")
        prof_menu.AppendRadioItem(ID_PROF_NES, "NES-ish\tCtrl+2")
        prof_menu.AppendRadioItem(ID_PROF_GB, "GB-ish\tCtrl+3")
        prof_menu.AppendRadioItem(ID_PROF_C64, "C64-ish\tCtrl+4")
        prof_menu.AppendRadioItem(ID_PROF_BEEPER, "Beeper\tCtrl+5")
        prof_menu.Check(ID_PROF_NEUTRAL, True)
        menubar.Append(prof_menu, "&Profile")

        # Live MIDI menu
        self.live_menu = wx.Menu()
        self.live_menu.AppendCheckItem(ID_LIVE_TOGGLE, "&Enable Live MIDI\tCtrl+M")
        self.live_menu.Append(ID_LIVE_REFRESH, "&Refresh MIDI Devices")
        self.live_menu.Append(ID_LIVE_PANIC, "&Panic (All Notes Off)")
        self.live_menu.AppendSeparator()
        # Device submenu will be populated dynamically
        self._midi_device_ids = []
        menubar.Append(self.live_menu, "&Live MIDI")

        self.SetMenuBar(menubar)

        # IMPORTANT: no Space accelerator. Space stays normal for focused controls.
        accel = wx.AcceleratorTable([
            (wx.ACCEL_CTRL, ord('O'), ID_OPEN),
            (wx.ACCEL_CTRL, ord('S'), ID_EXPORT),
            (wx.ACCEL_CTRL, ord('Q'), ID_EXIT),
            (wx.ACCEL_CTRL, ord('L'), ID_LOOP),
            (wx.ACCEL_CTRL, ord(','), ID_SPEED_SLOWER),
            (wx.ACCEL_CTRL, ord('.'), ID_SPEED_FASTER),
            (wx.ACCEL_CTRL, ord('0'), ID_SPEED_RESET),
            (wx.ACCEL_CTRL, ord('1'), ID_PROF_NEUTRAL),
            (wx.ACCEL_CTRL, ord('2'), ID_PROF_NES),
            (wx.ACCEL_CTRL, ord('3'), ID_PROF_GB),
            (wx.ACCEL_CTRL, ord('4'), ID_PROF_C64),
            (wx.ACCEL_CTRL, ord('5'), ID_PROF_BEEPER),
            (wx.ACCEL_CTRL, ord('M'), ID_LIVE_TOGGLE),
        ])
        self.SetAcceleratorTable(accel)

        self.Bind(wx.EVT_MENU, self.on_open, id=ID_OPEN)
        self.Bind(wx.EVT_MENU, self.on_export, id=ID_EXPORT)
        self.Bind(wx.EVT_MENU, self.on_stop, id=ID_STOP)
        self.Bind(wx.EVT_MENU, self.on_toggle_loop, id=ID_LOOP)
        self.Bind(wx.EVT_MENU, self.on_exit, id=ID_EXIT)
        self.Bind(wx.EVT_MENU, lambda e: self.nudge_speed(0.95), id=ID_SPEED_SLOWER)
        self.Bind(wx.EVT_MENU, lambda e: self.nudge_speed(1.05), id=ID_SPEED_FASTER)
        self.Bind(wx.EVT_MENU, lambda e: self.set_speed(1.0), id=ID_SPEED_RESET)

        self.Bind(wx.EVT_MENU, self.on_select_profile, id=ID_PROF_NEUTRAL)
        self.Bind(wx.EVT_MENU, self.on_select_profile, id=ID_PROF_NES)
        self.Bind(wx.EVT_MENU, self.on_select_profile, id=ID_PROF_GB)
        self.Bind(wx.EVT_MENU, self.on_select_profile, id=ID_PROF_C64)
        self.Bind(wx.EVT_MENU, self.on_select_profile, id=ID_PROF_BEEPER)

        # Live MIDI bindings
        self.Bind(wx.EVT_MENU, self.on_live_toggle, id=ID_LIVE_TOGGLE)
        self.Bind(wx.EVT_MENU, self.on_live_refresh_devices, id=ID_LIVE_REFRESH)
        self.Bind(wx.EVT_MENU, self.on_live_panic, id=ID_LIVE_PANIC)

        # Populate MIDI devices on startup
        wx.CallAfter(self._populate_midi_devices)

    def _build_body(self):
        panel = wx.Panel(self)
        root = wx.BoxSizer(wx.VERTICAL)

        self.lbl_file = wx.StaticText(panel, label="No file loaded.")
        self.lbl_status = wx.StaticText(panel, label="Ready.")
        f = self.lbl_file.GetFont()
        f.SetWeight(wx.FONTWEIGHT_BOLD)
        self.lbl_file.SetFont(f)

        # Buttons row
        btn_grid = wx.FlexGridSizer(rows=1, cols=4, vgap=8, hgap=10)
        btn_grid.AddGrowableCol(0, 1)
        btn_grid.AddGrowableCol(1, 1)
        btn_grid.AddGrowableCol(2, 1)
        btn_grid.AddGrowableCol(3, 1)

        self.btn_open = wx.Button(panel, label="Open MIDI")
        self.btn_play_toggle = wx.ToggleButton(panel, label="Play")
        self.btn_stop = wx.Button(panel, label="Stop")
        self.btn_export = wx.Button(panel, label="Export Audio...")

        _a11y(self.btn_open, "Open MIDI button")
        _a11y(self.btn_play_toggle, "Play/Pause toggle button", "Toggle playback. Space key works normally on focused controls.")
        _a11y(self.btn_stop, "Stop button")
        _a11y(self.btn_export, "Export audio button")

        for b in (self.btn_play_toggle, self.btn_stop, self.btn_export):
            b.Enable(False)

        btn_grid.Add(self.btn_open, 0, wx.EXPAND)
        btn_grid.Add(self.btn_play_toggle, 0, wx.EXPAND)
        btn_grid.Add(self.btn_stop, 0, wx.EXPAND)
        btn_grid.Add(self.btn_export, 0, wx.EXPAND)

        # Loop checkbox
        self.chk_loop = wx.CheckBox(panel, label="Loop")
        self.chk_loop.SetValue(True)
        _a11y(self.chk_loop, "Loop checkbox", "If checked, playback loops.")

        # --- Slider block with explicit labels (NVDA-friendly) ---
        sliders = wx.FlexGridSizer(rows=2, cols=3, vgap=10, hgap=10)
        sliders.AddGrowableCol(1, 1)

        # TURNABLE SPEED (tempo+pitch) -> 50..200
        self.lbl_speed_title = wx.StaticText(panel, label="Turntable speed (pitch+tempo)")
        self.slider_speed = wx.Slider(panel, value=100, minValue=50, maxValue=200, style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS)
        self.slider_speed.SetTickFreq(10)
        self.lbl_speed_value = wx.StaticText(panel, label="100% (1.00x)")

        _a11y(self.slider_speed, "Turntable speed slider (pitch and tempo)", "Turntable speed 50 to 200 percent.")

        # MASTER VOLUME -> 0..200
        self.lbl_master_title = wx.StaticText(panel, label="Master volume")
        self.slider_master = wx.Slider(panel, value=100, minValue=0, maxValue=200, style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS)
        self.slider_master.SetTickFreq(10)
        self.lbl_master_value = wx.StaticText(panel, label="100% (1.00x)")

        _a11y(self.slider_master, "Master volume slider", "Master volume 0 to 200 percent. Affects playback and export.")

        sliders.Add(self.lbl_speed_title, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        sliders.Add(self.slider_speed, 0, wx.EXPAND)
        sliders.Add(self.lbl_speed_value, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)

        sliders.Add(self.lbl_master_title, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        sliders.Add(self.slider_master, 0, wx.EXPAND)
        sliders.Add(self.lbl_master_value, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)

        # --- Seek row ---
        seek_box = wx.StaticBoxSizer(wx.StaticBox(panel, label="Position"), wx.VERTICAL)
        self.slider_seek = wx.Slider(panel, value=0, minValue=0, maxValue=1000, style=wx.SL_HORIZONTAL)
        self.lbl_time = wx.StaticText(panel, label="Time: 0:00 / 0:00")
        _a11y(self.slider_seek, "Playback position slider", "Scrub playback position. 0 to 1000.")
        _a11y(self.lbl_time, "Time label")

        seek_box.Add(self.slider_seek, 0, wx.EXPAND | wx.ALL, 6)
        seek_box.Add(self.lbl_time, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        # Profile label
        self.lbl_profile = wx.StaticText(panel, label="Profile: Neutral")
        _a11y(self.lbl_profile, "Profile label")

        # --- Per-channel volume section (scroll) ---
        self.channel_box = wx.StaticBox(panel, label="Per used MIDI channel volume (auto re-render)")
        self.channel_sizer = wx.StaticBoxSizer(self.channel_box, wx.VERTICAL)

        self.scroll = wx.ScrolledWindow(panel, style=wx.VSCROLL)
        self.scroll.SetScrollRate(10, 10)
        self.scroll_sizer = wx.BoxSizer(wx.VERTICAL)
        self.scroll.SetSizer(self.scroll_sizer)
        self.channel_sizer.Add(self.scroll, 1, wx.EXPAND | wx.ALL, 6)

        # --- Live MIDI Keyboard section ---
        live_box = wx.StaticBoxSizer(wx.StaticBox(panel, label="Live MIDI Keyboard"), wx.VERTICAL)

        # Device row
        device_row = wx.BoxSizer(wx.HORIZONTAL)
        device_lbl = wx.StaticText(panel, label="Device:")
        self.cmb_midi_device = wx.Choice(panel, choices=["(No MIDI devices)"])
        self.btn_refresh_midi = wx.Button(panel, label="Refresh")
        _a11y(self.cmb_midi_device, "MIDI input device selector", "Select MIDI keyboard or controller for live playing")
        _a11y(self.btn_refresh_midi, "Refresh MIDI devices button")

        device_row.Add(device_lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        device_row.Add(self.cmb_midi_device, 1, wx.EXPAND | wx.RIGHT, 8)
        device_row.Add(self.btn_refresh_midi, 0)

        # Enable checkbox
        self.chk_live_enable = wx.CheckBox(panel, label="Enable Live Input")
        _a11y(self.chk_live_enable, "Enable live MIDI input checkbox",
              "When checked, incoming MIDI notes will play through the chiptune synthesizer")

        # Live volume row
        live_vol_row = wx.BoxSizer(wx.HORIZONTAL)
        live_vol_lbl = wx.StaticText(panel, label="Live Volume:")
        self.slider_live_volume = wx.Slider(panel, value=80, minValue=0, maxValue=200,
                                             style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS)
        self.slider_live_volume.SetTickFreq(10)
        self.lbl_live_volume = wx.StaticText(panel, label="80%")
        _a11y(self.slider_live_volume, "Live MIDI volume slider", "Volume for live MIDI keyboard input, 0 to 200 percent")

        live_vol_row.Add(live_vol_lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        live_vol_row.Add(self.slider_live_volume, 1, wx.EXPAND | wx.RIGHT, 8)
        live_vol_row.Add(self.lbl_live_volume, 0, wx.ALIGN_CENTER_VERTICAL)

        # Status label
        self.lbl_live_status = wx.StaticText(panel, label="Status: Not connected")
        _a11y(self.lbl_live_status, "Live MIDI status label")

        live_box.Add(device_row, 0, wx.EXPAND | wx.ALL, 6)
        live_box.Add(self.chk_live_enable, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)
        live_box.Add(live_vol_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)
        live_box.Add(self.lbl_live_status, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        # Layout
        root.Add(self.lbl_file, 0, wx.ALL, 10)
        root.Add(self.lbl_status, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        root.Add(btn_grid, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        root.Add(self.chk_loop, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        root.Add(sliders, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        root.Add(seek_box, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        root.Add(self.lbl_profile, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        root.Add(live_box, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        root.Add(self.channel_sizer, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        panel.SetSizer(root)

        # Bind
        self.btn_open.Bind(wx.EVT_BUTTON, self.on_open)
        self.btn_play_toggle.Bind(wx.EVT_TOGGLEBUTTON, self.on_play_toggle)
        self.btn_stop.Bind(wx.EVT_BUTTON, self.on_stop)
        self.btn_export.Bind(wx.EVT_BUTTON, self.on_export)
        self.chk_loop.Bind(wx.EVT_CHECKBOX, self.on_toggle_loop)
        self.slider_speed.Bind(wx.EVT_SLIDER, self.on_speed_change)
        self.slider_master.Bind(wx.EVT_SLIDER, self.on_master_change)

        # Seek events
        self.slider_seek.Bind(wx.EVT_SCROLL_THUMBTRACK, self.on_seek_track)
        self.slider_seek.Bind(wx.EVT_SCROLL_THUMBRELEASE, self.on_seek_release)
        self.slider_seek.Bind(wx.EVT_SLIDER, self.on_seek_release)  # fallback

        # Live MIDI events
        self.cmb_midi_device.Bind(wx.EVT_CHOICE, self.on_midi_device_change)
        self.btn_refresh_midi.Bind(wx.EVT_BUTTON, self.on_live_refresh_devices)
        self.chk_live_enable.Bind(wx.EVT_CHECKBOX, self.on_live_enable_toggle)
        self.slider_live_volume.Bind(wx.EVT_SLIDER, self.on_live_volume_change)

        # Tooltips (optional)
        self.slider_speed.SetToolTip("Turntable speed: changes pitch and tempo like vinyl (50%..200%)")
        self.slider_master.SetToolTip("Master volume (0%..200%), affects playback and export")
        self.slider_seek.SetToolTip("Scrub/seek position")

        # initial labels
        self._refresh_speed_label()
        self._refresh_master_label()
        self._refresh_time_label(0.0, 0.0)

        # init channel list UI
        self.build_channel_sliders()

    # ---------- Helpers ----------
    def set_status(self, text: str):
        self.lbl_status.SetLabel(text)

    def get_speed_rate(self) -> float:
        return self.slider_speed.GetValue() / 100.0

    def get_master_rate(self) -> float:
        return self.slider_master.GetValue() / 100.0

    def _refresh_speed_label(self):
        rate = self.get_speed_rate()
        self.lbl_speed_value.SetLabel(f"{int(rate * 100)}% ({rate:.2f}x)")

    def _refresh_master_label(self):
        vol = self.get_master_rate()
        self.lbl_master_value.SetLabel(f"{int(vol * 100)}% ({vol:.2f}x)")

    def _refresh_time_label(self, cur_sec: float, total_sec: float):
        self.lbl_time.SetLabel(f"Time: {_fmt_time(cur_sec)} / {_fmt_time(total_sec)}")

    def set_speed(self, rate: float):
        rate = max(0.1, min(3.0, float(rate)))
        val = int(round(rate * 100))
        val = max(50, min(200, val))
        self.slider_speed.SetValue(val)
        self._refresh_speed_label()
        if self.player:
            self.player.set_rate(val / 100.0)

    def nudge_speed(self, mul: float):
        self.set_speed(self.get_speed_rate() * mul)

    def stop_player(self, reset_position: bool = False):
        if self.player:
            try:
                self.player.stop(reset_position=reset_position)
            except Exception:
                pass
        self.player = None

    def _set_play_toggle_ui(self, playing: bool):
        self.btn_play_toggle.SetValue(bool(playing))
        self.btn_play_toggle.SetLabel("Pause" if playing else "Play")

    def build_channel_sliders(self):
        self.scroll.Freeze()
        try:
            for child in self.scroll.GetChildren():
                child.Destroy()
            self.scroll_sizer.Clear(delete_windows=False)

            if not self.used_channels:
                info = wx.StaticText(self.scroll, label="No MIDI loaded. Channel sliders appear after opening a file.")
                self.scroll_sizer.Add(info, 0, wx.ALL, 6)
                self.scroll.Layout()
                self.scroll.FitInside()
                return

            g = wx.FlexGridSizer(rows=max(1, len(self.used_channels)), cols=3, vgap=6, hgap=10)
            g.AddGrowableCol(1, 1)

            self._channel_widgets = {}

            for ch in self.used_channels:
                if ch == 9:
                    label_txt = "Channel 10 (Drums)"
                else:
                    label_txt = f"Channel {ch + 1}"

                lbl = wx.StaticText(self.scroll, label=label_txt)
                s = wx.Slider(
                    self.scroll,
                    value=int(round(self.channel_volumes.get(ch, 1.0) * 100)),
                    minValue=0,
                    maxValue=200,
                    style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS
                )
                s.SetTickFreq(10)
                val_lbl = wx.StaticText(self.scroll, label=f"{s.GetValue()}%")

                _a11y(s, f"Volume slider for {label_txt}", f"Volume for {label_txt}. 0 to 200 percent. Changing re-renders automatically.")
                s.SetToolTip(f"Volume for {label_txt} (0%..200%). Changing it re-renders.")

                def make_handler(channel):
                    def _on(evt):
                        self.on_channel_volume_change(channel)
                    return _on

                s.Bind(wx.EVT_SLIDER, make_handler(ch))

                g.Add(lbl, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
                g.Add(s, 0, wx.EXPAND)
                g.Add(val_lbl, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)

                self._channel_widgets[ch] = (s, val_lbl)

            self.scroll_sizer.Add(g, 0, wx.EXPAND | wx.ALL, 8)
            self.scroll.Layout()
            self.scroll.FitInside()
        finally:
            self.scroll.Thaw()

    def schedule_rerender(self):
        if self.notes is None:
            return
        self._rerender_pending = True
        if self._rerender_timer.IsRunning():
            self._rerender_timer.Stop()
        self._rerender_timer.StartOnce(250)

    def _on_rerender_timer(self, event):
        if not self._rerender_pending:
            return
        self._rerender_pending = False
        self.rerender_current(keep_playback_position=True, reason="Channel volume change")

    def rerender_current(self, keep_playback_position=True, reason="Re-render"):
        if self.notes is None:
            return

        old_playing = bool(self.player and self.player.is_playing())
        old_frac = self.player.get_fraction() if self.player else 0.0
        old_rate = self.get_speed_rate()

        self.set_status(f"{reason}: rendering...")
        self.stop_player(reset_position=False)

        chan_vol = {int(k): float(v) for k, v in self.channel_volumes.items()}

        def worker():
            err = None
            audio = None
            try:
                audio = render_chiptune_float32(self.notes, profile_name=self.profile_name, channel_volumes=chan_vol)
            except Exception as e:
                err = e

            def done():
                if audio is None:
                    self.set_status(f"Render error: {err}")
                    wx.MessageBox(f"Could not render MIDI:\n{err}", "Error", wx.OK | wx.ICON_ERROR, self)
                    return

                self.audio_f32 = audio
                self.set_status("Ready.")

                if old_playing and keep_playback_position:
                    try:
                        self.player = TurntablePlayer(
                            self.audio_f32.copy(),
                            loop=self.chk_loop.GetValue(),
                            volume=self.get_master_rate()
                        )
                        self.player.set_rate(old_rate)
                        self.player.play(start_at_fraction=old_frac)
                        self.set_status(f"Playing ({self.profile_name}).")
                        self._set_play_toggle_ui(True)
                    except Exception as e2:
                        self.set_status(f"Audio error after re-render: {e2}")
                        self._set_play_toggle_ui(False)
                else:
                    self._set_play_toggle_ui(False)

            wx.CallAfter(done)

        threading.Thread(target=worker, daemon=True).start()

    # ---------- Events ----------
    def on_open(self, event=None):
        with wx.FileDialog(
            self,
            "Choose MIDI file",
            wildcard="MIDI files (*.mid;*.midi)|*.mid;*.midi",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
        ) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            path = dlg.GetPath()

        self.filepath = Path(path)
        self.lbl_file.SetLabel(f"File: {self.filepath.name}")
        self.set_status("Parsing + rendering...")

        for b in (self.btn_play_toggle, self.btn_stop, self.btn_export):
            b.Enable(False)
        self._set_play_toggle_ui(False)

        self.stop_player(reset_position=True)

        def worker():
            err = None
            audio = None
            notes = None
            used = []
            try:
                mid = mido.MidiFile(path)
                notes = collect_notes_faithful(mid)
                used = channels_in_notes(notes)
                for ch in used:
                    self.channel_volumes.setdefault(ch, 1.0)
                audio = render_chiptune_float32(notes, profile_name=self.profile_name, channel_volumes=self.channel_volumes)
            except Exception as e:
                err = e

            def done():
                if audio is None:
                    self.audio_f32 = None
                    self.notes = None
                    self.used_channels = []
                    self.build_channel_sliders()
                    self.set_status(f"Render error: {err}")
                    wx.MessageBox(f"Could not render MIDI:\n{err}", "Error", wx.OK | wx.ICON_ERROR, self)
                    return

                self.notes = notes
                self.used_channels = used
                self.build_channel_sliders()

                self.audio_f32 = audio
                self.set_status("Ready.")
                for b in (self.btn_play_toggle, self.btn_stop, self.btn_export):
                    b.Enable(True)

                # reset seek display
                self.slider_seek.SetValue(0)
                total = self.audio_f32.shape[0] / SAMPLE_RATE
                self._refresh_time_label(0.0, total)

            wx.CallAfter(done)

        threading.Thread(target=worker, daemon=True).start()

    def on_play_toggle(self, event=None):
        if self.audio_f32 is None:
            self._set_play_toggle_ui(False)
            return

        want_play = bool(self.btn_play_toggle.GetValue())

        if not want_play:
            # Pause
            if self.player:
                self.player.pause()
            self.set_status("Paused.")
            self._set_play_toggle_ui(False)
            return

        # Play / Resume
        try:
            if self.player is not None and not self.player.is_playing():
                self.player.set_loop(self.chk_loop.GetValue())
                self.player.set_rate(self.get_speed_rate())
                self.player.set_volume(self.get_master_rate())
                if self.player.resume():
                    self.set_status(f"Playing ({self.profile_name}).")
                    self._set_play_toggle_ui(True)
                    return

            # Start new
            self.player = TurntablePlayer(
                self.audio_f32.copy(),
                loop=self.chk_loop.GetValue(),
                volume=self.get_master_rate()
            )
            self.player.set_rate(self.get_speed_rate())
            self.player.play(start_at_fraction=self.slider_seek.GetValue() / 1000.0)
            self.set_status(f"Playing (turntable mode, {self.profile_name}).")
            self._set_play_toggle_ui(True)
        except Exception as e:
            self.set_status(f"Audio error: {e}")
            wx.MessageBox(f"Audio error:\n{e}", "Audio error", wx.OK | wx.ICON_ERROR, self)
            self._set_play_toggle_ui(False)

    def on_stop(self, event=None):
        self.stop_player(reset_position=True)
        self.set_status("Stopped.")
        self._set_play_toggle_ui(False)
        self.slider_seek.SetValue(0)
        total = (self.audio_f32.shape[0] / SAMPLE_RATE) if self.audio_f32 is not None else 0.0
        self._refresh_time_label(0.0, total)

    def on_toggle_loop(self, event=None):
        if self.player:
            self.player.set_loop(self.chk_loop.GetValue())
        self.set_status(f"Loop {'on' if self.chk_loop.GetValue() else 'off'}.")

    def on_speed_change(self, event):
        self._refresh_speed_label()
        if self.player:
            self.player.set_rate(self.get_speed_rate())

    def on_master_change(self, event):
        self._refresh_master_label()
        if self.player:
            self.player.set_volume(self.get_master_rate())

    def on_seek_track(self, event):
        if self.audio_f32 is None:
            return
        self._scrubbing = True
        frac = self.slider_seek.GetValue() / 1000.0
        total = self.audio_f32.shape[0] / SAMPLE_RATE
        self._refresh_time_label(frac * total, total)

    def on_seek_release(self, event):
        if self.audio_f32 is None:
            self._scrubbing = False
            return
        frac = self.slider_seek.GetValue() / 1000.0
        total = self.audio_f32.shape[0] / SAMPLE_RATE
        self._refresh_time_label(frac * total, total)

        if self.player:
            self.player.set_fraction(frac)
            # If currently playing, keep going
            if self.btn_play_toggle.GetValue():
                self.player.resume()

        self._scrubbing = False

    def _on_ui_timer(self, event):
        if self.audio_f32 is None or self._scrubbing:
            return

        total = self.audio_f32.shape[0] / SAMPLE_RATE
        cur = 0.0
        if self.player:
            cur = self.player.get_time_seconds()
            if total > 0 and self.chk_loop.GetValue():
                cur = cur % total

        frac = 0.0 if total <= 0 else max(0.0, min(1.0, cur / total))
        self.slider_seek.SetValue(int(round(frac * 1000.0)))
        self._refresh_time_label(cur, total)

        # reflect stopped playback in toggle
        if self.player and not self.player.is_playing() and self.btn_play_toggle.GetValue():
            self._set_play_toggle_ui(False)

    def on_channel_volume_change(self, ch: int):
        if hasattr(self, "_channel_widgets") and ch in self._channel_widgets:
            s, val_lbl = self._channel_widgets[ch]
            v = s.GetValue() / 100.0
            self.channel_volumes[int(ch)] = float(v)
            val_lbl.SetLabel(f"{s.GetValue()}%")
        self.schedule_rerender()

    def on_export(self, event=None):
        if self.audio_f32 is None:
            wx.MessageBox("Nothing to export.", "Info", wx.OK | wx.ICON_INFORMATION, self)
            return

        base = (self.filepath.stem if self.filepath else "output") + f"_{self.profile_name}"
        default = base + ".wav"

        wildcard = "WAV files (*.wav)|*.wav|MP3 files (*.mp3)|*.mp3"
        with wx.FileDialog(
            self,
            "Export Audio",
            wildcard=wildcard,
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
            defaultFile=default
        ) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            out = Path(dlg.GetPath())
            filter_idx = dlg.GetFilterIndex()  # 0=wav, 1=mp3

        want_ext = ".wav" if filter_idx == 0 else ".mp3"
        if out.suffix.lower() not in (".wav", ".mp3"):
            out = out.with_suffix(want_ext)
        elif out.suffix.lower() != want_ext:
            out = out.with_suffix(want_ext)

        # Apply master volume to export as well (so export matches what you hear).
        export_buf = self.audio_f32 * float(self.get_master_rate())

        try:
            if out.suffix.lower() == ".wav":
                export_wav_float32(str(out), export_buf)
                self.set_status(f"WAV saved: {out.name}")
            else:
                try:
                    export_mp3_via_ffmpeg(str(out), export_buf, quality_q=2)
                    self.set_status(f"MP3 saved: {out.name}")
                except FileNotFoundError:
                    wx.MessageBox(
                        "MP3 export needs FFmpeg.\n\n"
                        "Install FFmpeg and make sure 'ffmpeg' is in PATH.\n\n"
                        "Windows (winget):\n"
                        "  winget install -e --id Gyan.FFmpeg\n\n"
                        "Then restart this app and try again.",
                        "FFmpeg not found",
                        wx.OK | wx.ICON_WARNING,
                        self
                    )
                except Exception as e:
                    wx.MessageBox(f"Could not export MP3:\n{e}", "Export error", wx.OK | wx.ICON_ERROR, self)
        except Exception as e:
            wx.MessageBox(f"Could not write file:\n{e}", "Export error", wx.OK | wx.ICON_ERROR, self)

    def on_select_profile(self, event):
        new_name = PROFILE_ID_MAP.get(event.GetId(), "Neutral")
        if new_name == self.profile_name:
            return

        old_playing = bool(self.player and self.player.is_playing())
        old_frac = self.player.get_fraction() if self.player else 0.0
        old_rate = self.get_speed_rate()

        self.profile_name = new_name
        self.lbl_profile.SetLabel(f"Profile: {self.profile_name}")

        # Update live MIDI player profile
        if self.live_midi_player:
            self.live_midi_player.set_profile(new_name)

        self.set_status(f"Rendering with profile '{self.profile_name}'...")
        self.stop_player(reset_position=False)
        self._set_play_toggle_ui(False)

        if self.notes is None:
            self.set_status("No MIDI loaded; profile will apply to next file.")
            return

        chan_vol = {int(k): float(v) for k, v in self.channel_volumes.items()}

        def worker():
            err = None
            audio = None
            try:
                audio = render_chiptune_float32(self.notes, profile_name=self.profile_name, channel_volumes=chan_vol)
            except Exception as e:
                err = e

            def done():
                if audio is None:
                    self.set_status(f"Render error: {err}")
                    wx.MessageBox(f"Could not render MIDI:\n{err}", "Error", wx.OK | wx.ICON_ERROR, self)
                    return

                self.audio_f32 = audio
                self.set_status(f"Profile '{self.profile_name}' ready.")

                if old_playing:
                    try:
                        self.player = TurntablePlayer(
                            self.audio_f32.copy(),
                            loop=self.chk_loop.GetValue(),
                            volume=self.get_master_rate()
                        )
                        self.player.set_rate(old_rate)
                        self.player.play(start_at_fraction=old_frac)
                        self.set_status(f"Playing ({self.profile_name}).")
                        self._set_play_toggle_ui(True)
                    except Exception as e2:
                        self.set_status(f"Audio error after profile switch: {e2}")
                        self._set_play_toggle_ui(False)

            wx.CallAfter(done)

        threading.Thread(target=worker, daemon=True).start()

    # ---------- Live MIDI Methods ----------
    def _populate_midi_devices(self):
        """Populate the MIDI device dropdown."""
        self.cmb_midi_device.Clear()

        # Create temporary player just to list devices
        temp_player = LiveMidiPlayer()
        devices = temp_player.list_midi_inputs()

        if devices:
            for dev in devices:
                self.cmb_midi_device.Append(dev)
            self.cmb_midi_device.SetSelection(0)
        else:
            self.cmb_midi_device.Append("(No MIDI devices found)")
            self.cmb_midi_device.SetSelection(0)

        self._update_live_menu_devices(devices)

    def _update_live_menu_devices(self, devices: list[str]):
        """Update the Live MIDI menu with device list."""
        # Remove old device menu items
        for item_id in self._midi_device_ids:
            try:
                self.Unbind(wx.EVT_MENU, id=item_id)
                item = self.live_menu.FindItemById(item_id)
                if item:
                    self.live_menu.Remove(item_id)
            except Exception:
                pass
        self._midi_device_ids.clear()

        # Add new device menu items
        if devices:
            for i, dev in enumerate(devices):
                item_id = wx.NewIdRef()
                self._midi_device_ids.append(item_id)
                self.live_menu.AppendRadioItem(item_id, dev)
                self.Bind(wx.EVT_MENU, lambda e, d=dev: self._select_midi_device(d), id=item_id)
                if i == 0:
                    self.live_menu.Check(item_id, True)

    def _select_midi_device(self, device_name: str):
        """Select MIDI device from menu."""
        # Update dropdown to match
        idx = self.cmb_midi_device.FindString(device_name)
        if idx != wx.NOT_FOUND:
            self.cmb_midi_device.SetSelection(idx)

        # If live is enabled, reconnect to new device
        if self._live_midi_enabled and self.live_midi_player:
            self.live_midi_player.close_midi_input()
            if self.live_midi_player.open_midi_input(device_name):
                self.lbl_live_status.SetLabel(f"Status: Connected to {device_name}")
            else:
                self.lbl_live_status.SetLabel(f"Status: Failed to connect to {device_name}")

    def on_midi_device_change(self, event=None):
        """Handle MIDI device dropdown change."""
        sel = self.cmb_midi_device.GetSelection()
        if sel == wx.NOT_FOUND:
            return

        device_name = self.cmb_midi_device.GetString(sel)
        if device_name.startswith("("):
            return  # Placeholder text

        # Update menu radio items to match
        for i, item_id in enumerate(self._midi_device_ids):
            if i < self.cmb_midi_device.GetCount():
                dev = self.cmb_midi_device.GetString(i)
                if dev == device_name:
                    self.live_menu.Check(item_id, True)
                    break

        # If live is enabled, reconnect
        if self._live_midi_enabled and self.live_midi_player:
            self.live_midi_player.close_midi_input()
            if self.live_midi_player.open_midi_input(device_name):
                self.lbl_live_status.SetLabel(f"Status: Connected to {device_name}")
            else:
                self.lbl_live_status.SetLabel(f"Status: Failed to connect to {device_name}")

    def on_live_refresh_devices(self, event=None):
        """Refresh MIDI device list."""
        self._populate_midi_devices()
        self.set_status("MIDI devices refreshed.")

    def on_live_toggle(self, event=None):
        """Toggle live MIDI from menu."""
        # Sync checkbox with menu
        menu_item = self.live_menu.FindItemById(ID_LIVE_TOGGLE)
        if menu_item:
            is_checked = menu_item.IsChecked()
            self.chk_live_enable.SetValue(is_checked)
            self._toggle_live_midi(is_checked)

    def on_live_enable_toggle(self, event=None):
        """Handle live enable checkbox toggle."""
        is_checked = self.chk_live_enable.GetValue()
        # Sync menu with checkbox
        self.live_menu.Check(ID_LIVE_TOGGLE, is_checked)
        self._toggle_live_midi(is_checked)

    def _toggle_live_midi(self, enable: bool):
        """Enable or disable live MIDI playback."""
        if enable:
            # Create and start live MIDI player
            if self.live_midi_player is None:
                self.live_midi_player = LiveMidiPlayer(num_voices=12, profile_name=self.profile_name)
                self.live_midi_player.set_status_callback(self._on_live_voice_status)

            # Set volume
            vol = self.slider_live_volume.GetValue() / 100.0
            self.live_midi_player.set_volume(vol)

            # Get selected device
            sel = self.cmb_midi_device.GetSelection()
            device_name = None
            if sel != wx.NOT_FOUND:
                device_name = self.cmb_midi_device.GetString(sel)
                if device_name.startswith("("):
                    device_name = None

            # Start audio stream
            if not self.live_midi_player.start():
                self.lbl_live_status.SetLabel("Status: Audio stream error")
                self.chk_live_enable.SetValue(False)
                self.live_menu.Check(ID_LIVE_TOGGLE, False)
                self._live_midi_enabled = False
                return

            # Open MIDI input
            if device_name:
                if self.live_midi_player.open_midi_input(device_name):
                    self.lbl_live_status.SetLabel(f"Status: Connected to {device_name}")
                else:
                    self.lbl_live_status.SetLabel(f"Status: Failed to connect to {device_name}")
            else:
                # Try default device
                if self.live_midi_player.open_midi_input():
                    port = self.live_midi_player.get_midi_port_name() or "default"
                    self.lbl_live_status.SetLabel(f"Status: Connected to {port}")
                else:
                    self.lbl_live_status.SetLabel("Status: No MIDI device available")

            self._live_midi_enabled = True
            self.set_status("Live MIDI enabled.")
        else:
            # Stop live MIDI player
            if self.live_midi_player:
                self.live_midi_player.close_midi_input()
                self.live_midi_player.stop()

            self._live_midi_enabled = False
            self.lbl_live_status.SetLabel("Status: Not connected")
            self.set_status("Live MIDI disabled.")

    def _on_live_voice_status(self, active_count: int):
        """Callback for live voice status updates."""
        if self._live_midi_enabled:
            device = self.live_midi_player.get_midi_port_name() if self.live_midi_player else "unknown"
            self.lbl_live_status.SetLabel(f"Status: {device} - {active_count} voice{'s' if active_count != 1 else ''} active")

    def on_live_volume_change(self, event=None):
        """Handle live volume slider change."""
        vol = self.slider_live_volume.GetValue()
        self.lbl_live_volume.SetLabel(f"{vol}%")
        if self.live_midi_player:
            self.live_midi_player.set_volume(vol / 100.0)

    def on_live_panic(self, event=None):
        """All notes off - panic button."""
        if self.live_midi_player:
            self.live_midi_player.panic()
            self.set_status("Live MIDI: All notes off.")

    def on_exit(self, event=None):
        # Stop live MIDI player
        if self.live_midi_player:
            self.live_midi_player.close_midi_input()
            self.live_midi_player.stop()
        self.stop_player(reset_position=False)
        self.Close()


class App(wx.App):
    def OnInit(self):
        self.frame = MainFrame(None)
        self.frame.Show(True)
        return True


if __name__ == "__main__":
    app = App(False)
    app.MainLoop()
