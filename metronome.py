"""Standalone metronome driven by a sounddevice output callback.

Design
------
A single sounddevice.OutputStream runs permanently while the metronome is
active.  Its callback walks a beat-position counter forward frame by frame
and injects a short square-burst click whenever a beat boundary is crossed.

Beat 1 of each bar gets an accented click (1200 Hz); all other beats get a
softer click (800 Hz).  Both clicks are pre-computed as float32 arrays so
the callback never allocates memory.

If a click straddles a callback boundary (the click is ~5 ms, a typical
256-frame block at 44100 Hz is ~5.8 ms) the leftover samples are stored in
_pending_click and mixed into the next buffer before any new click is fired.

All shared state that controls musical parameters (bpm, beats_per_bar,
volume, the click buffers) is protected by a threading.Lock.  The per-
callback counters (_beat_pos, _beat_num, _pending_*) are only ever written
by the audio thread, so they need no lock.

Dependency map
--------------
synth  →  metronome  →  main GUI module
"""

import threading

import numpy as np
import sounddevice as sd

from synth import SAMPLE_RATE, MASTER_GAIN, _TWO_PI


class Metronome:
    _CLICK_SAMPLES = max(2, int(SAMPLE_RATE * 0.005))  # 5 ms click burst

    def __init__(self):
        self.bpm: float = 120.0
        self.beats_per_bar: int = 4
        self.volume: float = 0.7

        self._active = False
        self._stream: sd.OutputStream | None = None
        self._lock = threading.Lock()

        # Audio-thread-only state — never touched from the main thread while active
        self._beat_pos: int = 0   # samples elapsed within the current beat
        self._beat_num: int = 0   # total beat count (mod beats_per_bar → accent)
        self._pending_click: np.ndarray | None = None  # leftover click from split buffer
        self._pending_off: int = 0

        self._rebuild_clicks()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_bpm(self, bpm: float):
        with self._lock:
            self.bpm = max(20.0, min(300.0, float(bpm)))

    def set_beats_per_bar(self, n: int):
        with self._lock:
            self.beats_per_bar = max(1, int(n))

    def set_volume(self, vol: float):
        with self._lock:
            self.volume = max(0.0, min(2.0, float(vol)))
            self._rebuild_clicks()

    def start(self) -> bool:
        """Open the audio stream and start ticking.  Returns False on error."""
        if self._active:
            return True
        self._beat_pos = 0
        self._beat_num = 0
        self._pending_click = None
        self._pending_off = 0
        try:
            self._stream = sd.OutputStream(
                channels=2, dtype='float32', samplerate=SAMPLE_RATE,
                blocksize=256, latency='low', callback=self._callback,
            )
            self._stream.start()
            self._active = True
            return True
        except Exception:
            return False

    def stop(self):
        """Stop ticking and close the audio stream."""
        self._active = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def is_active(self) -> bool:
        return self._active

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_clicks(self):
        """Pre-compute accent and normal click buffers from current volume."""
        n = self._CLICK_SAMPLES
        t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
        # Exponential decay envelope: ~2 ms half-life gives a snappy click
        decay = np.exp(-t / 0.002).astype(np.float32)
        v = float(self.volume) * MASTER_GAIN
        self._accent     = (np.sign(np.sin(_TWO_PI * 1200.0 * t)) * decay * v      ).astype(np.float32)
        self._beat_click = (np.sign(np.sin(_TWO_PI * 800.0  * t)) * decay * v * 0.65).astype(np.float32)

    def _callback(self, outdata, frames, time_info, status):
        out = outdata.view(np.float32).reshape(-1, 2)
        out.fill(0.0)

        # Snapshot musical parameters under the lock so set_bpm() from the
        # main thread never races with the calculation below.
        with self._lock:
            beat_len   = max(1, int(SAMPLE_RATE * 60.0 / self.bpm))
            bpb        = self.beats_per_bar
            accent     = self._accent
            beat_click = self._beat_click

        beat_pos = self._beat_pos
        beat_num = self._beat_num

        # Finish any click that started at the end of the previous callback
        if self._pending_click is not None:
            rem = len(self._pending_click) - self._pending_off
            n = min(rem, frames)
            chunk = self._pending_click[self._pending_off:self._pending_off + n]
            out[:n, 0] += chunk
            out[:n, 1] += chunk
            self._pending_off += n
            if self._pending_off >= len(self._pending_click):
                self._pending_click = None
                self._pending_off = 0

        # Walk through the frame; fire a click at every beat boundary
        i = 0
        while i < frames:
            if beat_pos == 0:
                click = accent if beat_num % bpb == 0 else beat_click
                fit = min(len(click), frames - i)
                out[i:i + fit, 0] += click[:fit]
                out[i:i + fit, 1] += click[:fit]
                leftover = len(click) - fit
                if leftover > 0:
                    # Click overflows into the next callback; stash the tail
                    self._pending_click = click
                    self._pending_off = fit

            step = min(beat_len - beat_pos, frames - i)
            beat_pos += step
            i += step
            if beat_pos >= beat_len:
                beat_pos = 0
                beat_num += 1

        self._beat_pos = beat_pos
        self._beat_num = beat_num
