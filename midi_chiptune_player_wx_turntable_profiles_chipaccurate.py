# MIDI Chiptune Player — Turntable Pitch (LIVE) + Chip Profiles + WAV Export
# + Master Volume Slider (live, NVDA-friendly)
# + Per-MIDI-Channel Volume Sliders (re-render on change, resume position)
# + Performance optimizations (Numba JIT, LRU cache, vectorization)

import math
import threading
import wave
from pathlib import Path
from functools import lru_cache

import numpy as np
import mido
import wx
import sounddevice as sd

# Optional Numba JIT compilation for performance
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

    # Pre-allocate array (faster than concatenate)
    env = np.empty(total_len, dtype=np.float32)

    # Fill segments in-place
    env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    env[a:a+d] = np.linspace(1.0, sustain, d, dtype=np.float32)
    env[a+d:a+d+s_len] = sustain
    env[a+d+s_len:] = np.linspace(sustain, 0.0, r, dtype=np.float32)

    # Handle size mismatch
    if total_len < n:
        env = np.pad(env, (0, n - total_len), mode='edge')
    elif total_len > n:
        env = env[:n]

    return env

def pulse_tone(freq=440.0, dur=0.2, vol=0.4, duty=0.5, vib_rate=0.0, vib_depth_cents=0.0):
    n = max(2, int(SAMPLE_RATE * float(dur)))
    t = np.arange(n, dtype=np.float32) * (1.0 / SAMPLE_RATE)  # Faster than division
    if vib_rate > 0.0 and vib_depth_cents != 0.0:
        lfo = np.sin(_TWO_PI * float(vib_rate) * t)
        ratio = 2.0 ** ((lfo * float(vib_depth_cents)) / 1200.0)
        inst_f = float(freq) * ratio
    else:
        inst_f = float(freq)  # Scalar instead of array when constant

    if isinstance(inst_f, float):
        # Fast path for constant frequency
        phase = _TWO_PI * float(freq) * t
    else:
        phase = _TWO_PI * np.cumsum(inst_f) / SAMPLE_RATE

    frac = np.mod(phase, _TWO_PI) * _INV_PI * 0.5  # Avoid division by two_pi
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

    total_sec = max((end for (start, end, *_ ) in notes)) + 0.2
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
            # Use pre-computed lookup table for performance
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

            # Use pre-computed lookup table for performance
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

def export_wav_float32(path: str, stereo_f32: np.ndarray):
    stereo_i16 = (np.clip(stereo_f32, -1.0, 1.0) * 32767.0).astype(np.int16, copy=False)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(stereo_i16.tobytes())

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

    def stop(self):
        with self.lock:
            self.playing = False
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

    def is_playing(self):
        return self.playing and self.stream is not None

    def get_fraction(self) -> float:
        with self.lock:
            if self.n <= 1:
                return 0.0
            return float((self.idx % (self.n - 1)) / (self.n - 1))

    def _callback(self, outdata, frames, time, status):
        """Optimized callback with vectorized operations for better performance."""
        out = outdata.view(dtype=np.float32).reshape((-1, 2))
        buf = self.data
        n = self.n

        with self.lock:
            rate = float(self.rate)
            loop = self.loop
            idx = float(self.idx)
            vol = float(self.volume)

        # Vectorized approach: compute all indices at once
        indices = idx + np.arange(frames, dtype=np.float32) * rate

        if loop:
            # Handle looping with modulo
            indices = np.mod(indices, n - 1)
            i0 = indices.astype(np.int32)
            i1 = np.mod(i0 + 1, n)
            frac = (indices - i0).reshape(-1, 1)

            # Linear interpolation (vectorized)
            out[:] = ((1.0 - frac) * buf[i0] + frac * buf[i1]) * vol

            # Update index
            final_idx = float(indices[-1] + rate)
            if final_idx >= n - 1:
                final_idx = np.mod(final_idx, n - 1)
        else:
            # Non-looping: check if we exceed buffer
            valid_mask = indices < (n - 1)
            valid_count = np.sum(valid_mask)

            if valid_count == 0:
                out.fill(0.0)
                final_idx = n - 1
            elif valid_count < frames:
                # Some samples are valid, rest are silence
                valid_indices = indices[:valid_count]
                i0 = valid_indices.astype(np.int32)
                i1 = np.minimum(i0 + 1, n - 1)
                frac = (valid_indices - i0).reshape(-1, 1)

                out[:valid_count] = ((1.0 - frac) * buf[i0] + frac * buf[i1]) * vol
                out[valid_count:] = 0.0

                # Stop playback
                def _later_stop(stream=self.stream):
                    try:
                        if stream is not None:
                            stream.stop()
                    except Exception:
                        pass
                wx.CallAfter(_later_stop)
                final_idx = n - 1
            else:
                # All samples valid
                i0 = indices.astype(np.int32)
                i1 = np.minimum(i0 + 1, n - 1)
                frac = (indices - i0).reshape(-1, 1)

                out[:] = ((1.0 - frac) * buf[i0] + frac * buf[i1]) * vol
                final_idx = float(indices[-1] + rate)

        with self.lock:
            self.idx = final_idx

    def play(self, start_at_fraction: float | None = None):
        self.stop()
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

# --------------------- wx GUI ---------------------
ID_OPEN = wx.NewIdRef()
ID_EXPORT = wx.NewIdRef()
ID_PLAYPAUSE = wx.NewIdRef()
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

PROFILE_ID_MAP = {
    ID_PROF_NEUTRAL: "Neutral",
    ID_PROF_NES: "NES-ish",
    ID_PROF_GB: "GB-ish",
    ID_PROF_C64: "C64-ish",
    ID_PROF_BEEPER: "Beeper",
}

def _set_accessible_name(win: wx.Window, name: str):
    try:
        win.SetName(name)
    except Exception:
        pass

class MainFrame(wx.Frame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.SetTitle("MIDI Chiptune Player — Turntable + Chip Profiles")
        self.SetSize((980, 560))
        self.Centre()

        self.filepath: Path | None = None
        self.notes = None
        self.audio_f32: np.ndarray | None = None
        self.player: TurntablePlayer | None = None
        self.profile_name: str = "Neutral"

        self.master_volume = 1.0  # 0..2
        self.channel_volumes = {ch: 1.0 for ch in range(16)}
        self.used_channels = []

        self._rerender_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_rerender_timer, self._rerender_timer)
        self._rerender_pending = False

        self._build_menu()
        self._build_body()

        wx.CallAfter(self.on_open)

    def _build_menu(self):
        menubar = wx.MenuBar()

        file_menu = wx.Menu()
        file_menu.Append(ID_OPEN, "&Open MIDI\tCtrl+O")
        file_menu.Append(ID_EXPORT, "Export &WAV\tCtrl+S")
        file_menu.AppendSeparator()
        file_menu.Append(ID_EXIT, "E&xit\tCtrl+Q")
        menubar.Append(file_menu, "&File")

        play_menu = wx.Menu()
        play_menu.Append(ID_PLAYPAUSE, "&Play/Pause\tSpace")
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

        self.SetMenuBar(menubar)

        accel = wx.AcceleratorTable([
            (wx.ACCEL_CTRL, ord('O'), ID_OPEN),
            (wx.ACCEL_CTRL, ord('S'), ID_EXPORT),
            (wx.ACCEL_CTRL, ord('Q'), ID_EXIT),
            (wx.ACCEL_CTRL, ord('L'), ID_LOOP),
            (0, wx.WXK_SPACE, ID_PLAYPAUSE),
            (wx.ACCEL_CTRL, ord(','), ID_SPEED_SLOWER),
            (wx.ACCEL_CTRL, ord('.'), ID_SPEED_FASTER),
            (wx.ACCEL_CTRL, ord('0'), ID_SPEED_RESET),
            (wx.ACCEL_CTRL, ord('1'), ID_PROF_NEUTRAL),
            (wx.ACCEL_CTRL, ord('2'), ID_PROF_NES),
            (wx.ACCEL_CTRL, ord('3'), ID_PROF_GB),
            (wx.ACCEL_CTRL, ord('4'), ID_PROF_C64),
            (wx.ACCEL_CTRL, ord('5'), ID_PROF_BEEPER),
        ])
        self.SetAcceleratorTable(accel)

        self.Bind(wx.EVT_MENU, self.on_open, id=ID_OPEN)
        self.Bind(wx.EVT_MENU, self.on_export, id=ID_EXPORT)
        self.Bind(wx.EVT_MENU, self.on_playpause, id=ID_PLAYPAUSE)
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
        self.btn_play = wx.Button(panel, label="Play / Pause")
        self.btn_stop = wx.Button(panel, label="Stop")
        self.btn_export = wx.Button(panel, label="Export WAV")

        for b in (self.btn_play, self.btn_stop, self.btn_export):
            b.Enable(False)

        btn_grid.Add(self.btn_open, 0, wx.EXPAND)
        btn_grid.Add(self.btn_play, 0, wx.EXPAND)
        btn_grid.Add(self.btn_stop, 0, wx.EXPAND)
        btn_grid.Add(self.btn_export, 0, wx.EXPAND)

        # Loop checkbox
        self.chk_loop = wx.CheckBox(panel, label="Loop")
        self.chk_loop.SetValue(True)

        # --- Slider block with explicit labels (NVDA-friendly) ---
        sliders = wx.FlexGridSizer(rows=2, cols=3, vgap=10, hgap=10)
        sliders.AddGrowableCol(1, 1)

        # TURNABLE SPEED (tempo+pitch) -> MUST be 50..200
        self.lbl_speed_title = wx.StaticText(panel, label="Turntable speed (pitch+tempo)")
        self.slider_speed = wx.Slider(
            panel,
            value=100,
            minValue=50,
            maxValue=200,
            style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS
        )
        self.slider_speed.SetTickFreq(10)
        self.lbl_speed_value = wx.StaticText(panel, label="100% (1.00x)")

        # MASTER VOLUME -> MUST be 0..200
        self.lbl_master_title = wx.StaticText(panel, label="Master volume")
        self.slider_master = wx.Slider(
            panel,
            value=100,
            minValue=0,
            maxValue=200,
            style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS
        )
        self.slider_master.SetTickFreq(10)
        self.lbl_master_value = wx.StaticText(panel, label="100% (1.00x)")

        sliders.Add(self.lbl_speed_title, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        sliders.Add(self.slider_speed, 0, wx.EXPAND)
        sliders.Add(self.lbl_speed_value, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)

        sliders.Add(self.lbl_master_title, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        sliders.Add(self.slider_master, 0, wx.EXPAND)
        sliders.Add(self.lbl_master_value, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)

        # Profile label
        self.lbl_profile = wx.StaticText(panel, label="Profile: Neutral")

        # --- Per-channel volume section (scroll) ---
        self.channel_box = wx.StaticBox(panel, label="Per MIDI channel volume (re-renders)")
        self.channel_sizer = wx.StaticBoxSizer(self.channel_box, wx.VERTICAL)

        self.scroll = wx.ScrolledWindow(panel, style=wx.VSCROLL)
        self.scroll.SetScrollRate(10, 10)
        self.scroll_sizer = wx.BoxSizer(wx.VERTICAL)
        self.scroll.SetSizer(self.scroll_sizer)
        self.channel_sizer.Add(self.scroll, 1, wx.EXPAND | wx.ALL, 6)

        # Layout
        root.Add(self.lbl_file, 0, wx.ALL, 10)
        root.Add(self.lbl_status, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        root.Add(btn_grid, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        root.Add(self.chk_loop, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        root.Add(sliders, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        root.Add(self.lbl_profile, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        root.Add(self.channel_sizer, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        panel.SetSizer(root)

        # Bind
        self.btn_open.Bind(wx.EVT_BUTTON, self.on_open)
        self.btn_play.Bind(wx.EVT_BUTTON, self.on_playpause)
        self.btn_stop.Bind(wx.EVT_BUTTON, self.on_stop)
        self.btn_export.Bind(wx.EVT_BUTTON, self.on_export)
        self.chk_loop.Bind(wx.EVT_CHECKBOX, self.on_toggle_loop)
        self.slider_speed.Bind(wx.EVT_SLIDER, self.on_speed_change)
        self.slider_master.Bind(wx.EVT_SLIDER, self.on_master_change)

        # Tooltips + NVDA names
        self.slider_speed.SetToolTip("Turntable speed: changes pitch and tempo like vinyl (50%..200%)")
        _set_accessible_name(self.slider_speed, "Turntable speed slider (pitch and tempo)")

        self.slider_master.SetToolTip("Master volume (0%..200%), affects playback volume")
        _set_accessible_name(self.slider_master, "Master volume slider")

        # initial labels
        self._refresh_speed_label()
        self._refresh_master_label()

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

    def set_speed(self, rate: float):
        rate = max(0.1, min(3.0, float(rate)))
        val = int(round(rate * 100))
        # keep in slider bounds (50..200)
        val = max(50, min(200, val))
        self.slider_speed.SetValue(val)
        self._refresh_speed_label()
        if self.player:
            self.player.set_rate(val / 100.0)

    def nudge_speed(self, mul: float):
        self.set_speed(self.get_speed_rate() * mul)

    def stop_player(self):
        if self.player:
            try:
                self.player.stop()
            except Exception:
                pass
        self.player = None

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

                _set_accessible_name(s, f"Volume slider for {label_txt}")
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
        old_frac = self.player.get_fraction() if (self.player) else 0.0
        old_rate = self.get_speed_rate()

        self.set_status(f"{reason}: rendering...")
        self.stop_player()

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
                    return

                self.audio_f32 = audio
                self.set_status("Ready.")

                if old_playing and keep_playback_position:
                    try:
                        self.player = TurntablePlayer(self.audio_f32.copy(), loop=self.chk_loop.GetValue(),
                                                     volume=self.get_master_rate())
                        self.player.set_rate(old_rate)
                        self.player.play(start_at_fraction=old_frac)
                        self.set_status(f"Playing ({self.profile_name}).")
                    except Exception as e2:
                        self.set_status(f"Audio error after re-render: {e2}")

            wx.CallAfter(done)

        threading.Thread(target=worker, daemon=True).start()

    # ---------- Events ----------
    def on_open(self, event=None):
        with wx.FileDialog(self, "Choose MIDI file",
                           wildcard="MIDI files (*.mid;*.midi)|*.mid;*.midi",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            path = dlg.GetPath()

        self.filepath = Path(path)
        self.lbl_file.SetLabel(f"File: {self.filepath.name}")
        self.set_status("Parsing + rendering...")
        for b in (self.btn_play, self.btn_stop, self.btn_export):
            b.Enable(False)
        self.stop_player()

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
                audio = render_chiptune_float32(notes, profile_name=self.profile_name,
                                                channel_volumes=self.channel_volumes)
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
                for b in (self.btn_play, self.btn_stop, self.btn_export):
                    b.Enable(True)
                self.on_playpause()

            wx.CallAfter(done)

        threading.Thread(target=worker, daemon=True).start()

    def on_playpause(self, event=None):
        if self.audio_f32 is None:
            return
        if self.player and self.player.is_playing():
            try:
                self.player.pause()
            except Exception:
                pass
            self.set_status("Paused.")
            return
        try:
            self.player = TurntablePlayer(self.audio_f32.copy(), loop=self.chk_loop.GetValue(),
                                          volume=self.get_master_rate())
            self.player.set_rate(self.get_speed_rate())
            self.player.play()
            self.set_status(f"Playing (turntable mode, {self.profile_name}).")
        except Exception as e:
            self.set_status(f"Audio error: {e}")
            wx.MessageBox(f"Audio error:\n{e}", "Audio error", wx.OK | wx.ICON_ERROR, self)

    def on_stop(self, event=None):
        self.stop_player()
        self.set_status("Stopped.")

    def on_toggle_loop(self, event=None):
        if self.player:
            self.player.set_loop(self.chk_loop.GetValue())
        self.set_status(f"Loop {'on' if self.chk_loop.GetValue() else 'off'}.")

    def on_speed_change(self, event):
        # This is the TURNABLE SPEED slider (50..200)
        self._refresh_speed_label()
        if self.player:
            self.player.set_rate(self.get_speed_rate())

    def on_master_change(self, event):
        # This is the MASTER VOLUME slider (0..200)
        self._refresh_master_label()
        if self.player:
            self.player.set_volume(self.get_master_rate())

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
        default = (self.filepath.stem if self.filepath else "output") + f"_{self.profile_name}.wav"
        with wx.FileDialog(self, "Save WAV", wildcard="WAV files (*.wav)|*.wav",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT, defaultFile=default) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            out = dlg.GetPath()
        try:
            export_wav_float32(out, self.audio_f32 * float(self.get_master_rate()))
        except Exception as e:
            wx.MessageBox(f"Could not write WAV:\n{e}", "Export error", wx.OK | wx.ICON_ERROR, self)
            return
        self.set_status(f"WAV saved: {Path(out).name}")

    def on_select_profile(self, event):
        new_name = PROFILE_ID_MAP.get(event.GetId(), "Neutral")
        if new_name == self.profile_name:
            return

        old_playing = bool(self.player and self.player.is_playing())
        old_frac = self.player.get_fraction() if (self.player) else 0.0
        old_rate = self.get_speed_rate()

        self.profile_name = new_name
        self.lbl_profile.SetLabel(f"Profile: {self.profile_name}")
        self.set_status(f"Rendering with profile '{self.profile_name}'...")
        self.stop_player()

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
                    return

                self.audio_f32 = audio
                self.set_status(f"Profile '{self.profile_name}' ready.")

                if old_playing:
                    try:
                        self.player = TurntablePlayer(self.audio_f32.copy(), loop=self.chk_loop.GetValue(),
                                                     volume=self.get_master_rate())
                        self.player.set_rate(old_rate)
                        self.player.play(start_at_fraction=old_frac)
                        self.set_status(f"Playing ({self.profile_name}).")
                    except Exception as e2:
                        self.set_status(f"Audio error after profile switch: {e2}")

            wx.CallAfter(done)

        threading.Thread(target=worker, daemon=True).start()

    def on_exit(self, event=None):
        self.stop_player()
        self.Close()

class App(wx.App):
    def OnInit(self):
        self.frame = MainFrame(None)
        self.frame.Show(True)
        return True

if __name__ == "__main__":
    app = App(False)
    app.MainLoop()
