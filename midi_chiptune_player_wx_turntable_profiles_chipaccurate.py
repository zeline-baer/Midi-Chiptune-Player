# MIDI Chiptune Player — Turntable Pitch (LIVE) + Chip Profiles + WAV Export
# - Renders MIDI->float32 once; plays via sounddevice with live SPEED (pitch+tempo)
# - "Chip Profiles" (Neutral, NES-ish, GB-ish, C64-ish, Beeper) affecting timbre, stereo width, quantization
# - Re-render on profile change; resumes playback near same position
#
# Install (Windows, pip):
#   py -m pip install --upgrade pip
#   py -m pip install --only-binary=:all: wxPython
#   py -m pip install numpy mido sounddevice
#
# Run:
#   py midi_chiptune_player_wx_turntable_profiles.py

import math
import threading
import wave
from pathlib import Path

import numpy as np
import mido
import wx
import sounddevice as sd

# --------------------- Audio/Render config ---------------------
SAMPLE_RATE = 44100
MASTER_GAIN = 0.85

# --------------------- Synth helpers ---------------------
def _pan_gains(pan: float):
    p = max(-1.0, min(1.0, float(pan)))
    angle = (p + 1.0) * math.pi / 4.0  # equal-power
    return math.cos(angle), math.sin(angle)

def _stereo_from_mono(mono: np.ndarray, pan: float):
    gL, gR = _pan_gains(pan)
    return np.stack([mono * gL, mono * gR], axis=1)

def _adsr_env(n: int, sr: int, attack=0.008, decay=0.050, sustain=0.65, release=0.120, total_sec=None):
    if total_sec is not None:
        n = max(2, int(sr*total_sec))
    a = max(1, int(sr*attack))
    d = max(1, int(sr*decay))
    r = max(1, int(sr*release))
    s_len = max(0, n - (a + d + r))
    env = np.concatenate([
        np.linspace(0.0, 1.0, a, dtype=np.float32),
        np.linspace(1.0, sustain, d, dtype=np.float32),
        np.full(s_len, sustain, dtype=np.float32),
        np.linspace(sustain, 0.0, r, dtype=np.float32)
    ]).astype(np.float32)
    if env.shape[0] < n:
        env = np.pad(env, (0, n - env.shape[0]), mode='edge')
    elif env.shape[0] > n:
        env = env[:n]
    return env

def pulse_tone(freq=440.0, dur=0.2, vol=0.4, duty=0.5, vib_rate=0.0, vib_depth_cents=0.0):
    n = max(2, int(SAMPLE_RATE * float(dur)))
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    if vib_rate > 0.0 and vib_depth_cents != 0.0:
        lfo = np.sin(2.0*np.pi*float(vib_rate)*t)
        ratio = 2.0 ** ((lfo * float(vib_depth_cents)) / 1200.0)
        inst_f = float(freq) * ratio
    else:
        inst_f = np.full_like(t, float(freq), dtype=np.float32)
    phase = 2.0*np.pi * np.cumsum(inst_f) / SAMPLE_RATE
    two_pi = 2.0*np.pi
    frac = np.mod(phase, two_pi) / two_pi
    wave = np.where(frac < float(duty), 1.0, -1.0).astype(np.float32)
    env = _adsr_env(n, SAMPLE_RATE, total_sec=dur)
    mono = wave * env * (float(vol) * MASTER_GAIN)
    return mono

def triangle_tone(freq=220.0, dur=0.2, vol=0.4):
    n = max(2, int(SAMPLE_RATE * float(dur)))
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    tri = (2.0/np.pi) * np.arcsin(np.sin(2.0*np.pi*float(freq)*t)).astype(np.float32)
    env = _adsr_env(n, SAMPLE_RATE, total_sec=dur)
    return tri * env * (float(vol) * MASTER_GAIN)

def square_tone(freq=440.0, dur=0.2, vol=0.4):
    n = max(2, int(SAMPLE_RATE * float(dur)))
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    wave = np.sign(np.sin(2.0*np.pi*float(freq)*t)).astype(np.float32)
    env = _adsr_env(n, SAMPLE_RATE, total_sec=dur)
    mono = wave * env * (float(vol) * MASTER_GAIN)
    return mono


# Extra chip-ish waves
def saw_tone(freq=440.0, dur=0.2, vol=0.4):
    n = max(2, int(SAMPLE_RATE * float(dur)))
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    # naive saw (we embrace aliasing for "chip" grit)
    frac = np.mod(float(freq) * t, 1.0).astype(np.float32)
    wave = (2.0 * frac - 1.0).astype(np.float32)
    env = _adsr_env(n, SAMPLE_RATE, total_sec=dur)
    return wave * env * (float(vol) * MASTER_GAIN)

def noise_tone(dur=0.2, vol=0.35):
    n = max(2, int(SAMPLE_RATE * float(dur)))
    w = (np.random.rand(n).astype(np.float32) * 2.0 - 1.0)
    env = _adsr_env(n, SAMPLE_RATE, attack=0.002, decay=0.05, sustain=0.4, release=0.08, total_sec=dur)
    return w * env * (float(vol) * MASTER_GAIN)

def wavetable_tone(freq=440.0, dur=0.2, vol=0.4, table=None):
    """Simple 32-sample wavetable osc (Game Boy-ish CH3 style)."""
    if table is None:
        table = np.sin(np.linspace(0, 2*np.pi, 32, endpoint=False)).astype(np.float32)
    table = np.asarray(table, dtype=np.float32)
    if table.ndim != 1 or table.size < 2:
        raise ValueError("wavetable must be 1D with >=2 samples")
    n = max(2, int(SAMPLE_RATE * float(dur)))
    # phase accumulator
    phase = (np.cumsum(np.full(n, float(freq)/SAMPLE_RATE, dtype=np.float32)) % 1.0)
    idx = (phase * table.size).astype(np.int32)
    wave = table[idx]
    # CH3 is more "gated" than ADSR; use a short click-safe gate
    env = _adsr_env(n, SAMPLE_RATE, attack=0.002, decay=0.010, sustain=1.0, release=0.020, total_sec=dur)
    return wave.astype(np.float32) * env * (float(vol) * MASTER_GAIN)

# --------------------- Drums ---------------------
def _drum_kick_mono(dur=0.10, vol=0.72):
    base = square_tone(60.0, dur, vol*0.9)
    overt = square_tone(120.0, dur*0.7, vol*0.45)
    n = min(base.shape[0], overt.shape[0])
    return (base[:n] + overt[:n]*0.6) * 0.95

def _drum_snare_mono(dur=0.09, vol=0.62):
    n = max(2, int(SAMPLE_RATE*dur))
    noise = (np.random.rand(n).astype(np.float32)*2.0 - 1.0)
    env = _adsr_env(n, SAMPLE_RATE, attack=0.002, decay=0.06, sustain=0.3, release=0.08, total_sec=dur)
    body = square_tone(200.0, dur*0.6, vol=0.35)
    m = min(n, body.shape[0])
    return (noise[:m]*env[:m]*vol*MASTER_GAIN*0.9 + body[:m]*0.5)

def _drum_hat_mono(dur=0.02, vol=0.25):
    n = max(2, int(SAMPLE_RATE*dur))
    m = (np.random.rand(n).astype(np.float32)*2.0 - 1.0)
    env = _adsr_env(n, SAMPLE_RATE, attack=0.001, decay=0.02, sustain=0.0, release=0.01, total_sec=dur)
    return m * env * (vol * MASTER_GAIN)

def _drum_tom_mono(freq=110.0, dur=0.15, vol=0.5):
    return square_tone(freq, dur, vol)

# --------------------- Profiles ---------------------

def get_profiles():
    # Chip-ish constraints live in "chip": voice limits + volume steps + mono etc.
    # If chip is None, renderer uses legacy "per MIDI channel" behavior.
    return {
        "Neutral": {
            "pan_map": {0:-0.25, 1:0.25, 2:-0.4, 3:0.4, 4:-0.15, 5:0.15},
            "timbre": {
                0: dict(wave="pulse", duty=0.50, vib_rate=5.5, vib_depth_cents=12.0),
                1: dict(wave="pulse", duty=0.25, vib_rate=0.0,  vib_depth_cents=0.0),
                2: dict(wave="pulse", duty=0.75, vib_rate=0.0,  vib_depth_cents=0.0),  # bass
                3: dict(wave="pulse", duty=0.62, vib_rate=4.0,  vib_depth_cents=6.0),
                4: dict(wave="pulse", duty=0.38, vib_rate=0.0,  vib_depth_cents=0.0),
                5: dict(wave="pulse", duty=0.50, vib_rate=0.0,  vib_depth_cents=0.0),
            },
            "stereo_width": 1.0,
            "quantize": None,
            "chip": None,
        },
        "NES-ish": {
            # Rough 2A03 mapping: 2 pulse + triangle (noise reserved for drums here)
            "pan_map": {"pulse0":-0.25, "pulse1":0.25, "tri":0.0},
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
                "vol_steps": 16,      # 0..15
                "duty_set": [0.125, 0.25, 0.5, 0.75],
                "tri_no_adsr": True,
                "voice_policy": "steal_oldest",
                # heuristic: low notes go to triangle
                "tri_split_note": 55,  # MIDI note < 55 => triangle
            },
        },
        "GB-ish": {
            # DMG: 2 pulse + 1 wavetable + noise (we keep drums separate), true mono
            "pan_map": {"pulse0":0.0, "pulse1":0.0, "wave":0.0},
            "timbre": {
                "pulse0": dict(wave="pulse", duty=0.125, vib_rate=0.0, vib_depth_cents=0.0),
                "pulse1": dict(wave="pulse", duty=0.5,   vib_rate=0.0, vib_depth_cents=0.0),
                "wave":   dict(wave="wavetable", duty=None, vib_rate=0.0, vib_depth_cents=0.0),
            },
            "stereo_width": 0.0,
            "quantize": "8bit",  # extra roughness
            "chip": {
                "voices": [("pulse", 2), ("wavetable", 1)],
                "mono": True,
                "vol_steps": 4,       # 0..3 (2-bit)
                "duty_set": [0.125, 0.25, 0.5, 0.75],
                "tri_no_adsr": True,
                "voice_policy": "steal_oldest",
                "wave_table": "gb_default",
                "wave_split_note": 50,  # low notes to wave channel
            },
        },
        "C64-ish": {
            # SID: 3 voices, 4-bit volume, "analog-ish" ADSR jitter
            "pan_map": {"v0":-0.25, "v1":0.25, "v2":0.0},
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
                "vol_steps": 16,      # 0..15
                "duty_set": [0.125, 0.25, 0.5, 0.75],
                "tri_no_adsr": False,
                "voice_policy": "steal_oldest",
                "sid_adsr_jitter": 0.06,   # +/- 6%
                "sid_detune_cents": 3.0,   # tiny instability
            },
        },
        "Beeper": {
            "pan_map": {"b":0.0},
            "timbre": {
                "b": dict(wave="square", duty=None, vib_rate=0.0, vib_depth_cents=0.0),
            },
            "stereo_width": 0.0,
            "quantize": "8bit",
            "chip": {
                "voices": [("square", 1)],
                "mono": True,
                "vol_steps": 2,       # 0..1
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

def _choose_wave_table(name: str):
    # 32-sample tables, roughly 4-bit shaped like DMG CH3.
    if name == "gb_default":
        # A slightly "hollow" waveform, quantized to 16 levels (4-bit)
        t = np.linspace(0, 2*np.pi, 32, endpoint=False)
        w = (0.65*np.sin(t) + 0.25*np.sin(2*t) + 0.10*np.sin(3*t))
        w = np.clip(w, -1.0, 1.0)
        w = np.round(((w + 1.0) * 7.5)) / 7.5 - 1.0  # 16-ish steps centered
        return w.astype(np.float32)
    # fallback
    return np.sin(np.linspace(0, 2*np.pi, 32, endpoint=False)).astype(np.float32)

def _flatten_voice_spec(voice_spec):
    # voice_spec: [("pulse",2), ("triangle",1), ...] -> ["pulse0","pulse1","tri",...]
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
    """Assign notes to a limited set of voices (with simple voice stealing).

    Returns list of dict items:
      {start,end,pitch,vel,is_drum, voice_key, wave_kind}
    """
    chip = profile.get("chip")
    if not chip:
        return None

    voice_keys = _flatten_voice_spec(chip["voices"])
    # track when each voice becomes free
    free_at = {vk: -1.0 for vk in voice_keys}

    tri_split = chip.get("tri_split_note", None)
    wave_split = chip.get("wave_split_note", None)

    assigned = []
    for (start, end, pitch, ch, vel, is_drum) in notes:
        if is_drum:
            assigned.append(dict(start=start, end=end, pitch=pitch, vel=vel, is_drum=True,
                                 voice_key="drum", wave_kind="drum"))
            continue

        # pick a desired class
        desired = None
        if "tri" in voice_keys and tri_split is not None and pitch < tri_split:
            desired = "tri"
        if "wave" in voice_keys and wave_split is not None and pitch < wave_split:
            desired = "wave"
        if desired is None:
            # prefer pulse voices for NES/GB, else SID voices, else whatever exists
            if any(vk.startswith("pulse") for vk in voice_keys):
                desired = "pulse"
            elif any(vk.startswith("v") for vk in voice_keys):
                desired = "v"
            else:
                desired = voice_keys[0]

        # candidate voices
        if desired == "pulse":
            cands = [vk for vk in voice_keys if vk.startswith("pulse")]
        elif desired == "v":
            cands = [vk for vk in voice_keys if vk.startswith("v")]
        else:
            cands = [vk for vk in voice_keys if vk == desired] or voice_keys

        # find a free voice
        vk = None
        for c in cands:
            if free_at[c] <= start:
                vk = c
                break

        # steal oldest (earliest free_at) among candidates
        if vk is None:
            vk = min(cands, key=lambda k: free_at[k])
        free_at[vk] = end

        wave_kind = profile["timbre"].get(vk, {}).get("wave", "pulse")
        assigned.append(dict(start=start, end=end, pitch=pitch, vel=vel, is_drum=False,
                             voice_key=vk, wave_kind=wave_kind))

    return assigned


def apply_stereo_width(stereo: np.ndarray, width: float):
    width = float(max(0.0, min(1.5, width)))
    L = stereo[:,0].copy()
    R = stereo[:,1].copy()
    M = (L + R) * 0.5
    stereo[:,0] = M + width * (L - M)
    stereo[:,1] = M + width * (R - M)
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
                out.append((start_sec, sec, msg.note, ch, vel, ch==9))
    out.sort(key=lambda x: x[0])
    return out

# --------------------- Render to float32 stereo (with profile) ---------------------

def render_chiptune_float32(notes, profile_name="Neutral"):
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

    # Pre-assign voices if this profile is chip-limited
    assigned = assign_chip_voices(notes, profile) if chip else None

    # -------- Drums (GM channel 10) --------
    # We keep a dedicated drum renderer; on NES/GB this approximates noise channel percussion.
    for start, end, pitch, ch, vel, is_drum in notes:
        if not is_drum:
            continue
        t0 = int(min(N-1, max(0, start * SAMPLE_RATE)))
        if pitch in (35, 36):
            mono = _drum_kick_mono(0.11, 0.75)
        elif pitch in (38, 40):
            mono = _drum_snare_mono(0.09, 0.62)
        elif pitch in (42, 44):
            mono = _drum_hat_mono(0.02, 0.22)
        elif pitch in (46, 49, 51):
            mono = _drum_hat_mono(0.06, 0.28)
        elif pitch in (41,43,45,47,48,50):
            base = {41:110.0,43:130.8,45:146.8,47:164.8,48:174.6,50:196.0}.get(pitch, 130.8)
            mono = _drum_tom_mono(base, 0.16, 0.48)
        else:
            n = max(2, int(SAMPLE_RATE*0.04))
            env = _adsr_env(n, SAMPLE_RATE, attack=0.002, decay=0.03, sustain=0.0, release=0.02, total_sec=0.04)
            mono = (np.random.rand(n).astype(np.float32)*2.0-1.0) * env * (0.25*MASTER_GAIN)

        endi = min(N, t0 + mono.shape[0]); ln = endi - t0
        if ln > 0:
            # drums mostly center
            mixL[t0:endi] += mono[:ln]*0.9
            mixR[t0:endi] += mono[:ln]*0.9

    # -------- Pitched notes --------
    if assigned is None:
        # Legacy mode: map by MIDI channel modulo 6
        for start, end, pitch, ch, vel, is_drum in notes:
            if is_drum:
                continue
            dur = max(0.02, float(end - start))
            freq = 440.0 * (2.0 ** ((float(pitch) - 69.0)/12.0))
            v = (vel/127.0)
            settings = timbre.get(ch % 6, dict(wave="pulse", duty=0.5, vib_rate=0.0, vib_depth_cents=0.0))
            wave_kind = settings.get("wave", "pulse")
            if wave_kind == "triangle":
                mono = triangle_tone(freq, dur, vol=0.48*v)
            elif wave_kind == "square":
                mono = square_tone(freq, dur, vol=0.48*v)
            else:
                mono = pulse_tone(freq, dur, vol=0.48*v,
                                  duty=settings.get('duty', 0.5),
                                  vib_rate=settings.get('vib_rate', 0.0),
                                  vib_depth_cents=settings.get('vib_depth_cents', 0.0))
            pan = pan_map.get(ch % 6, 0.0)
            stereo = _stereo_from_mono(mono, pan)
            s = int(start * SAMPLE_RATE)
            e = min(N, s + stereo.shape[0]); ln = e - s
            if ln > 0:
                mixL[s:e] += stereo[:ln,0]
                mixR[s:e] += stereo[:ln,1]
    else:
        # Chip-limited mode: strict voice counts + coarse volume
        vol_steps = int(chip.get("vol_steps", 16))
        duty_set = chip.get("duty_set", None)
        tri_no_adsr = bool(chip.get("tri_no_adsr", False))

        # GB wavetable
        wave_table = None
        if chip.get("wave_table"):
            wave_table = _choose_wave_table(str(chip["wave_table"]))

        sid_jitter = float(chip.get("sid_adsr_jitter", 0.0))
        sid_detune = float(chip.get("sid_detune_cents", 0.0))

        for item in assigned:
            if item["is_drum"]:
                continue
            start = float(item["start"]); end = float(item["end"])
            pitch = int(item["pitch"]); vel = int(item["vel"])
            vk = item["voice_key"]
            dur = max(0.02, float(end - start))

            # base frequency
            freq = 440.0 * (2.0 ** ((float(pitch) - 69.0)/12.0))

            # quantized volume
            v = _quantize_unit(vel/127.0, vol_steps)

            settings = timbre.get(vk, dict(wave="pulse", duty=0.5, vib_rate=0.0, vib_depth_cents=0.0))
            wave_kind = settings.get("wave", "pulse")

            # Quantize duty to classic sets when available
            duty = settings.get("duty", 0.5)
            if duty_set and duty is not None:
                duty = min(duty_set, key=lambda x: abs(x - float(duty)))

            # SID-ish slight instability (very subtle)
            if sid_detune and vk.startswith("v"):
                cents = np.random.uniform(-sid_detune, sid_detune)
                freq = freq * (2.0 ** (cents / 1200.0))

            if wave_kind == "triangle":
                if tri_no_adsr:
                    # gate-like envelope (click-safe)
                    mono = triangle_tone(freq, dur, vol=0.62*v)
                    # flatten envelope feel a bit
                    mono *= 1.15
                else:
                    mono = triangle_tone(freq, dur, vol=0.48*v)

            elif wave_kind == "square":
                mono = square_tone(freq, dur, vol=0.55*v)

            elif wave_kind == "saw":
                mono = saw_tone(freq, dur, vol=0.50*v)

            elif wave_kind == "noise":
                mono = noise_tone(dur, vol=0.45*v)

            elif wave_kind == "wavetable":
                mono = wavetable_tone(freq, dur, vol=0.60*v, table=wave_table)

            else:
                mono = pulse_tone(freq, dur, vol=0.50*v,
                                  duty=duty if duty is not None else 0.5,
                                  vib_rate=settings.get('vib_rate', 0.0),
                                  vib_depth_cents=settings.get('vib_depth_cents', 0.0))

            # SID-ish ADSR jitter: introduce tiny amplitude wobble (approximation)
            if sid_jitter and vk.startswith("v"):
                mono = mono * np.random.uniform(1.0 - sid_jitter, 1.0 + sid_jitter)

            pan = pan_map.get(vk, 0.0)
            stereo = _stereo_from_mono(mono, pan)
            s = int(start * SAMPLE_RATE)
            e = min(N, s + stereo.shape[0]); ln = e - s
            if ln > 0:
                mixL[s:e] += stereo[:ln,0]
                mixR[s:e] += stereo[:ln,1]

    # -------- Soft normalize --------
    peak = max(1e-6, float(max(abs(mixL.max()), abs(mixL.min()), abs(mixR.max()), abs(mixR.min()))))
    if peak > 0.98:
        mixL *= 0.98/peak
        mixR *= 0.98/peak

    stereo = np.stack([mixL, mixR], axis=1).astype(np.float32, copy=False)
    stereo = apply_stereo_width(stereo, stereo_width)
    stereo = np.clip(stereo, -1.0, 1.0)
    stereo = apply_quantize(stereo, quantize)

    # True mono for GB-ish / Beeper
    if chip and chip.get("mono", False):
        M = (stereo[:,0] + stereo[:,1]) * 0.5
        stereo[:,0] = M
        stereo[:,1] = M

    return stereo.astype(np.float32, copy=False)

def export_wav_float32(path: str, stereo_f32: np.ndarray):
    stereo_i16 = (np.clip(stereo_f32, -1.0, 1.0) * 32767.0).astype(np.int16, copy=False)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(stereo_i16.tobytes())

# --------------------- Live turntable streamer ---------------------
class TurntablePlayer:
    """Stream a stereo float32 buffer with variable speed (rate), looping."""
    def __init__(self, data: np.ndarray, loop: bool = True):
        assert data.ndim == 2 and data.shape[1] == 2, "data must be (N,2)"
        self.data = data
        self.n = data.shape[0]
        self.loop = loop
        self.rate = 1.0
        self.idx = 0.0
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
        if status:
            pass
        out = outdata.view(dtype=np.float32).reshape((-1, 2))
        out.fill(0.0)
        buf = self.data
        n = self.n
        with self.lock:
            rate = float(self.rate)
            loop = self.loop
            idx = float(self.idx)
        for i in range(frames):
            i0 = int(idx)
            frac = idx - i0
            if i0 >= n - 1:
                if loop:
                    i0 %= (n - 1)
                else:
                    break
            i1 = i0 + 1 if i0 + 1 < n else (0 if loop else i0)
            s0 = buf[i0]; s1 = buf[i1]
            out[i] = (1.0 - frac) * s0 + frac * s1
            idx += rate
            if loop:
                if idx >= n - 1:
                    idx -= (n - 1)
            else:
                if idx >= n - 1:
                    for k in range(i+1, frames):
                        out[k] = 0.0
                    def _later_stop(stream=self.stream):
                        try:
                            if stream is not None:
                                stream.stop()
                        except Exception:
                            pass
                    wx.CallAfter(_later_stop)
                    break
        with self.lock:
            self.idx = idx

    def play(self, start_at_fraction: float | None = None):
        self.stop()
        if start_at_fraction is not None:
            with self.lock:
                self.idx = float(start_at_fraction % 1.0) * (self.n - 1)
        try:
            self.stream = sd.OutputStream(channels=2, dtype='float32',
                                          samplerate=SAMPLE_RATE, blocksize=512,
                                          callback=self._callback)
            self.stream.start()
            with self.lock:
                self.playing = True
            return True
        except Exception as e:
            self.stream = None
            raise e

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

class MainFrame(wx.Frame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.SetTitle("MIDI Chiptune Player — Turntable + Chip Profiles")
        self.SetSize((880, 380))
        self.Centre()

        # State
        self.filepath: Path | None = None
        self.notes = None
        self.audio_f32: np.ndarray | None = None
        self.player: TurntablePlayer | None = None
        self.profile_name: str = "Neutral"

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
        vbox = wx.BoxSizer(wx.VERTICAL)

        self.lbl_file = wx.StaticText(panel, label="No file loaded.")
        self.lbl_status = wx.StaticText(panel, label="Ready.")
        font_bold = self.lbl_file.GetFont()
        font_bold.SetWeight(wx.FONTWEIGHT_BOLD)
        self.lbl_file.SetFont(font_bold)

        grid = wx.FlexGridSizer(rows=4, cols=4, vgap=8, hgap=8)
        self.btn_open = wx.Button(panel, label="Open MIDI")
        self.btn_play = wx.Button(panel, label="Play / Pause")
        self.btn_stop = wx.Button(panel, label="Stop")
        self.btn_export = wx.Button(panel, label="Export WAV")
        self.chk_loop = wx.CheckBox(panel, label="Loop")
        self.chk_loop.SetValue(True)

        self.slider_speed = wx.Slider(panel, value=100, minValue=50, maxValue=200,
                                      style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS)
        self.slider_speed.SetTickFreq(10)
        self.lbl_speed = wx.StaticText(panel, label="Speed: 100% (1.00x)")

        self.lbl_profile = wx.StaticText(panel, label="Profile: Neutral")

        for b in (self.btn_play, self.btn_stop, self.btn_export):
            b.Enable(False)

        grid.Add(self.btn_open, 0, wx.EXPAND)
        grid.Add(self.btn_play, 0, wx.EXPAND)
        grid.Add(self.btn_stop, 0, wx.EXPAND)
        grid.Add(self.btn_export, 0, wx.EXPAND)
        grid.Add(self.chk_loop, 0, wx.ALIGN_LEFT)
        grid.Add((0,0)); grid.Add((0,0)); grid.Add((0,0))
        grid.Add(self.slider_speed, 0, wx.EXPAND)
        grid.Add(self.lbl_speed, 0, wx.ALIGN_LEFT|wx.ALIGN_CENTER_VERTICAL)
        grid.Add((0,0)); grid.Add((0,0))
        grid.Add(self.lbl_profile, 0, wx.ALIGN_LEFT)
        grid.Add((0,0)); grid.Add((0,0)); grid.Add((0,0))

        for c in range(4):
            grid.AddGrowableCol(c, 1)

        vbox.Add(self.lbl_file, 0, wx.ALL, 10)
        vbox.Add(self.lbl_status, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        vbox.Add(grid, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        panel.SetSizer(vbox)

        self.btn_open.Bind(wx.EVT_BUTTON, self.on_open)
        self.btn_play.Bind(wx.EVT_BUTTON, self.on_playpause)
        self.btn_stop.Bind(wx.EVT_BUTTON, self.on_stop)
        self.btn_export.Bind(wx.EVT_BUTTON, self.on_export)
        self.chk_loop.Bind(wx.EVT_CHECKBOX, self.on_toggle_loop)
        self.slider_speed.Bind(wx.EVT_SLIDER, self.on_speed_change)

        self.btn_open.SetToolTip("Open a MIDI file (Ctrl+O)")
        self.btn_play.SetToolTip("Play/Pause (Space)")
        self.btn_stop.SetToolTip("Stop playback")
        self.btn_export.SetToolTip("Export current render to WAV (Ctrl+S)")
        self.chk_loop.SetToolTip("Loop playback (Ctrl+L)")
        self.slider_speed.SetToolTip("Turntable speed (50%..200%), changes pitch and tempo like vinyl")

    # ---------- Helpers ----------
    def set_status(self, text: str):
        self.lbl_status.SetLabel(text)

    def set_speed(self, rate: float):
        rate = max(0.1, min(3.0, float(rate)))
        val = int(round(rate * 100))
        self.slider_speed.SetValue(val)
        self.lbl_speed.SetLabel(f"Speed: {val}% ({rate:.2f}x)")
        if self.player:
            self.player.set_rate(rate)

    def nudge_speed(self, mul: float):
        cur = self.slider_speed.GetValue() / 100.0
        self.set_speed(cur * mul)

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
            try:
                mid = mido.MidiFile(path)
                notes = collect_notes_faithful(mid)
                audio = render_chiptune_float32(notes, profile_name=self.profile_name)
                self.notes = notes
            except Exception as e:
                err = e

            def done():
                if audio is None:
                    self.audio_f32 = None
                    self.set_status(f"Render error: {err}")
                    wx.MessageBox(f"Could not render MIDI:\n{err}", "Error", wx.OK | wx.ICON_ERROR, self)
                else:
                    self.audio_f32 = audio
                    self.set_status("Ready.")
                    for b in (self.btn_play, self.btn_stop, self.btn_export):
                        b.Enable(True)
                    self.on_playpause()

            wx.CallAfter(done)

        threading.Thread(target=worker, daemon=True).start()

    def stop_player(self):
        if self.player:
            try:
                self.player.stop()
            except Exception:
                pass
        self.player = None

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
            self.player = TurntablePlayer(self.audio_f32.copy(), loop=self.chk_loop.GetValue())
            rate = self.slider_speed.GetValue() / 100.0
            self.player.set_rate(rate)
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
        rate = self.slider_speed.GetValue() / 100.0
        self.lbl_speed.SetLabel(f"Speed: {int(rate*100)}% ({rate:.2f}x)")
        if self.player:
            self.player.set_rate(rate)

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
            export_wav_float32(out, self.audio_f32)
        except Exception as e:
            wx.MessageBox(f"Could not write WAV:\n{e}", "Export error", wx.OK | wx.ICON_ERROR, self)
            return
        self.set_status(f"WAV saved: {Path(out).name}")

    def on_select_profile(self, event):
        PROFILE_ID_MAP = {
            ID_PROF_NEUTRAL: "Neutral",
            ID_PROF_NES: "NES-ish",
            ID_PROF_GB: "GB-ish",
            ID_PROF_C64: "C64-ish",
            ID_PROF_BEEPER: "Beeper",
        }
        new_name = PROFILE_ID_MAP.get(event.GetId(), "Neutral")
        if new_name == self.profile_name:
            return
        old_playing = bool(self.player and self.player.is_playing())
        old_frac = self.player.get_fraction() if (self.player) else 0.0
        old_rate = self.slider_speed.GetValue() / 100.0
        self.profile_name = new_name
        self.lbl_profile.SetLabel(f"Profile: {self.profile_name}")
        self.set_status(f"Rendering with profile '{self.profile_name}'...")
        self.stop_player()

        if self.notes is None:
            self.set_status("No MIDI loaded; profile will apply to next file.")
            return

        def worker():
            err = None
            audio = None
            try:
                audio = render_chiptune_float32(self.notes, profile_name=self.profile_name)
            except Exception as e:
                err = e

            def done():
                if audio is None:
                    self.set_status(f"Render error: {err}")
                else:
                    self.audio_f32 = audio
                    self.set_status(f"Profile '{self.profile_name}' ready.")
                    if old_playing:
                        try:
                            self.player = TurntablePlayer(self.audio_f32.copy(), loop=self.chk_loop.GetValue())
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
