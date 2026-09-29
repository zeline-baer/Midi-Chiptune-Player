"""Real-time MIDI keyboard synthesiser.

LiveVoice  — dataclass that holds per-voice state (note, phase, ADSR, …)
LiveMidiPlayer — opens a MIDI input port and an audio output stream,
                 synthesises voices in the sounddevice callback.

Imports from synth.py for all waveform generation so that the audio
logic stays in one place.

Dependency map
--------------
synth  →  live_midi  →  main GUI module
"""

import time
import threading
from dataclasses import dataclass

import numpy as np
import mido
import sounddevice as sd
import wx  # wx.CallAfter is used for thread-safe UI status updates only

from synth import (
    SAMPLE_RATE, MASTER_GAIN, MIDI_TO_FREQ, PROFILES,
    _TWO_PI, _pan_gains, _flatten_voice_spec,
    _rt_generate_pulse_buffer,
    _rt_generate_triangle_buffer,
    _rt_generate_square_buffer,
    _rt_generate_saw_buffer,
)


# ---------------------------------------------------------------------------
# Voice state
# ---------------------------------------------------------------------------
@dataclass
class LiveVoice:
    """Mutable state for one polyphonic voice."""
    active: bool = False
    note: int = 0
    channel: int = 0
    velocity: float = 0.0
    freq: float = 440.0
    phase: float = 0.0
    env_phase: int = 0   # 0 attack · 1 decay · 2 sustain · 3 release · 4 off
    env_pos: int = 0
    env_level: float = 0.0
    wave_kind: str = "pulse"
    duty: float = 0.5
    start_time: float = 0.0
    pitch_bend: float = 0.0     # semitones, ±2
    vibrato_depth: float = 0.0  # 0..1 from mod wheel
    sustained: bool = False     # held by sustain pedal

    def reset(self):
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


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------
class LiveMidiPlayer:
    """Real-time MIDI input synthesiser for live keyboard playing."""

    # ADSR timing in samples at 44100 Hz
    ATTACK_SAMPLES  = int(0.008 * SAMPLE_RATE)
    DECAY_SAMPLES   = int(0.050 * SAMPLE_RATE)
    SUSTAIN_LEVEL   = 0.65
    RELEASE_SAMPLES = int(0.120 * SAMPLE_RATE)

    def __init__(self, num_voices: int = 12, profile_name: str = "Neutral"):
        self.num_voices = num_voices
        self.voices: list[LiveVoice] = [LiveVoice() for _ in range(num_voices)]
        self.profile_name = profile_name
        self.volume = 0.8
        self.active = False

        self.midi_port: mido.ports.BaseInput | None = None
        self.stream: sd.OutputStream | None = None
        self.lock = threading.Lock()

        self._channel_pitch_bend: dict[int, float] = {ch: 0.0 for ch in range(16)}
        self._channel_mod_wheel:  dict[int, float] = {ch: 0.0 for ch in range(16)}
        self._channel_sustain:    dict[int, bool]  = {ch: False for ch in range(16)}
        self._voice_vib_phase: list[float] = [0.0] * num_voices

        self._status_callback = None

    def set_status_callback(self, callback):
        self._status_callback = callback

    def _notify_status(self):
        if self._status_callback:
            with self.lock:
                active_count = sum(1 for v in self.voices if v.active)
            try:
                wx.CallAfter(self._status_callback, active_count)
            except Exception:
                pass

    # ----- MIDI port management -----

    def list_midi_inputs(self) -> list[str]:
        try:
            return mido.get_input_names()
        except Exception:
            return []

    def open_midi_input(self, port_name: str | None = None) -> bool:
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
        if self.midi_port is not None:
            try:
                self.midi_port.close()
            except Exception:
                pass
            self.midi_port = None

    def get_midi_port_name(self) -> str | None:
        if self.midi_port is not None:
            try:
                return self.midi_port.name
            except Exception:
                pass
        return None

    # ----- Audio stream management -----

    def start(self) -> bool:
        if self.active:
            return True
        try:
            self.stream = sd.OutputStream(
                channels=2, dtype='float32', samplerate=SAMPLE_RATE,
                blocksize=256, latency='low', callback=self._audio_callback,
            )
            self.stream.start()
            self.active = True
            return True
        except Exception as e:
            print(f"Audio stream error: {e}")
            return False

    def stop(self):
        self.active = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        with self.lock:
            for v in self.voices:
                v.reset()

    def set_volume(self, volume: float):
        with self.lock:
            self.volume = max(0.0, min(2.0, float(volume)))

    def set_profile(self, profile_name: str):
        with self.lock:
            self.profile_name = profile_name

    # ----- MIDI callback (runs on MIDI thread) -----

    def _midi_callback(self, msg: mido.Message):
        if msg.type == 'note_on' and msg.velocity > 0:
            self._note_on(msg.note, msg.velocity, msg.channel)
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            self._note_off(msg.note, msg.channel)
        elif msg.type == 'control_change':
            self._handle_cc(msg.control, msg.value, msg.channel)
        elif msg.type == 'pitchwheel':
            self._handle_pitch_bend(msg.pitch, msg.channel)

    def _handle_cc(self, control: int, value: int, channel: int):
        with self.lock:
            if control == 1:  # Mod Wheel
                self._channel_mod_wheel[channel] = value / 127.0
                for v in self.voices:
                    if v.active and v.channel == channel:
                        v.vibrato_depth = value / 127.0
            elif control == 64:  # Sustain Pedal
                sustain_on = value >= 64
                old_sustain = self._channel_sustain.get(channel, False)
                self._channel_sustain[channel] = sustain_on
                if not sustain_on and old_sustain:
                    for v in self.voices:
                        if v.active and v.channel == channel and v.sustained:
                            v.sustained = False
                            v.env_phase = 3
                            v.env_pos = 0

    def _handle_pitch_bend(self, pitch: int, channel: int):
        bend_semitones = (pitch / 8192.0) * 2.0
        with self.lock:
            self._channel_pitch_bend[channel] = bend_semitones
            for v in self.voices:
                if v.active and v.channel == channel:
                    v.pitch_bend = bend_semitones

    def _note_on(self, note: int, velocity: int, channel: int):
        with self.lock:
            voice_idx = self._find_free_voice()
            if voice_idx < 0:
                voice_idx = self._steal_oldest_voice()

            voice = self.voices[voice_idx]

            profile = PROFILES.get(self.profile_name, PROFILES["Neutral"])
            timbre = profile["timbre"]
            chip   = profile.get("chip")

            if chip:
                voice_keys = _flatten_voice_spec(chip["voices"])
                vk = voice_keys[channel % len(voice_keys)]
                settings = timbre.get(vk, {"wave": "pulse", "duty": 0.5})
            else:
                settings = timbre.get(channel % 6, {"wave": "pulse", "duty": 0.5})

            voice.active       = True
            voice.note         = note
            voice.channel      = channel
            voice.velocity     = velocity / 127.0
            voice.freq         = MIDI_TO_FREQ[note] if 0 <= note < 128 else 440.0
            voice.phase        = 0.0
            voice.env_phase    = 0
            voice.env_pos      = 0
            voice.env_level    = 0.0
            voice.start_time   = time.time()
            voice.wave_kind    = settings.get("wave", "pulse")
            voice.duty         = settings.get("duty", 0.5) or 0.5
            voice.pitch_bend   = self._channel_pitch_bend.get(channel, 0.0)
            voice.vibrato_depth = self._channel_mod_wheel.get(channel, 0.0)
            voice.sustained    = False

            self._voice_vib_phase[voice_idx] = 0.0

        self._notify_status()

    def _note_off(self, note: int, channel: int):
        with self.lock:
            sustain_on = self._channel_sustain.get(channel, False)
            for v in self.voices:
                if v.active and v.note == note and v.channel == channel and v.env_phase < 3:
                    if sustain_on:
                        v.sustained = True
                    else:
                        v.env_phase = 3
                        v.env_pos = 0
        self._notify_status()

    def _find_free_voice(self) -> int:
        for i, v in enumerate(self.voices):
            if not v.active:
                return i
        for i, v in enumerate(self.voices):
            if v.active and v.env_phase >= 3:
                return i
        return -1

    def _steal_oldest_voice(self) -> int:
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
        return oldest_release_idx if oldest_release_idx >= 0 else oldest_idx

    # ----- Audio callback (runs on audio thread) -----

    def _audio_callback(self, outdata, frames, time_info, status):
        out = outdata.view(dtype=np.float32).reshape((-1, 2))
        out.fill(0.0)

        if not self.active:
            return

        with self.lock:
            vol = self.volume
            profile = PROFILES.get(self.profile_name, PROFILES["Neutral"])
            pan_map = profile["pan_map"]
            stereo_width = profile["stereo_width"]

            for i, voice in enumerate(self.voices):
                if not voice.active:
                    continue

                samples, new_phase, new_vib_phase = self._generate_voice_buffer(
                    voice, frames, self._voice_vib_phase[i],
                )
                env_samples, new_env_phase, new_env_pos, new_env_level, still_active = \
                    self._apply_envelope(voice, frames)

                samples = samples * env_samples * voice.velocity * vol * MASTER_GAIN

                vk = self._get_voice_key(voice)
                pan = pan_map.get(vk, pan_map.get(voice.channel % 6, 0.0))
                gL, gR = _pan_gains(pan)

                out[:, 0] += samples * gL
                out[:, 1] += samples * gR

                voice.phase     = new_phase
                voice.env_phase = new_env_phase
                voice.env_pos   = new_env_pos
                voice.env_level = new_env_level
                self._voice_vib_phase[i] = new_vib_phase

                if not still_active:
                    voice.reset()

        if stereo_width != 1.0:
            L = out[:, 0].copy()
            R = out[:, 1].copy()
            M = (L + R) * 0.5
            out[:, 0] = M + stereo_width * (L - M)
            out[:, 1] = M + stereo_width * (R - M)

        np.clip(out, -1.0, 1.0, out=out)

    def _get_voice_key(self, voice: LiveVoice) -> str:
        profile = PROFILES.get(self.profile_name, PROFILES["Neutral"])
        chip = profile.get("chip")
        if chip:
            voice_keys = _flatten_voice_spec(chip["voices"])
            return voice_keys[voice.channel % len(voice_keys)]
        return voice.channel % 6

    def _generate_voice_buffer(self, voice: LiveVoice, frames: int,
                                vib_phase: float) -> tuple[np.ndarray, float, float]:
        wave  = voice.wave_kind
        freq  = voice.freq
        phase = voice.phase
        duty  = voice.duty
        pitch_bend  = voice.pitch_bend
        vib_depth   = voice.vibrato_depth

        if wave == "pulse":
            samples, new_phase, new_vib_phase = _rt_generate_pulse_buffer(
                phase, freq, duty, frames, pitch_bend, vib_depth, 5.0, vib_phase,
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
            samples, new_phase, new_vib_phase = _rt_generate_pulse_buffer(
                phase, freq, 0.5, frames, pitch_bend, vib_depth, 5.0, vib_phase,
            )

        return samples, new_phase, new_vib_phase

    def _apply_envelope(self, voice: LiveVoice, frames: int
                        ) -> tuple[np.ndarray, int, int, float, bool]:
        """Per-sample ADSR.  Returns (env_array, new_phase, new_pos, new_level, still_active)."""
        env   = np.empty(frames, dtype=np.float32)
        phase = voice.env_phase
        pos   = voice.env_pos
        level = voice.env_level

        for i in range(frames):
            if phase == 0:   # Attack
                level = pos / max(1, self.ATTACK_SAMPLES)
                pos += 1
                if pos >= self.ATTACK_SAMPLES:
                    phase, pos = 1, 0
            elif phase == 1:  # Decay
                level = 1.0 - (1.0 - self.SUSTAIN_LEVEL) * (pos / max(1, self.DECAY_SAMPLES))
                pos += 1
                if pos >= self.DECAY_SAMPLES:
                    phase, pos, level = 2, 0, self.SUSTAIN_LEVEL
            elif phase == 2:  # Sustain
                level = self.SUSTAIN_LEVEL
            elif phase == 3:  # Release
                start_level = voice.env_level if pos == 0 else level
                level = start_level * (1.0 - pos / max(1, self.RELEASE_SAMPLES))
                pos += 1
                if pos >= self.RELEASE_SAMPLES or level <= 0.001:
                    phase, level = 4, 0.0
            else:
                level = 0.0
            env[i] = level

        return env, phase, pos, level, phase < 4

    def get_active_voice_count(self) -> int:
        with self.lock:
            return sum(1 for v in self.voices if v.active)

    def panic(self):
        """All notes off — emergency stop."""
        with self.lock:
            for v in self.voices:
                v.reset()
            for ch in range(16):
                self._channel_pitch_bend[ch] = 0.0
                self._channel_mod_wheel[ch]  = 0.0
                self._channel_sustain[ch]    = False
        self._notify_status()
