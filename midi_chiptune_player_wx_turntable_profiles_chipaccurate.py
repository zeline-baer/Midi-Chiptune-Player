# MIDI Chiptune Player â€” Turntable Pitch (LIVE) + Chip Profiles + WAV/MP3 Export
# + Master Volume Slider (live, NVDA-friendly)
# + Per-used-MIDI-Channel Volume Sliders (auto re-render on change, resume position)
# + Seek (scrub) slider + elapsed/total time (NVDA-friendly)
# + MP3 export via FFmpeg if installed (otherwise shows install hint)
# + Space key is NOT hijacked globally (works normally on focused controls)
# + Metronome with tap tempo and accent on beat 1
#
# Entry point and GUI layer.  Pure audio logic lives in the modules below:
#   synth.py      â€” waveforms, ADSR, chip profiles, MIDI parser, renderer, export
#   live_midi.py  â€” LiveVoice dataclass + LiveMidiPlayer (real-time keyboard)
#   metronome.py  â€” Metronome (sounddevice callback, BPM, tap tempo)

import threading
import time
from pathlib import Path

import numpy as np
import mido
import wx
import sounddevice as sd

# Re-export everything that tests and callers reference via this module's
# namespace (e.g. `import midi_chiptune_player_â€¦ as player; player.PROFILES`).
from synth import (
    SAMPLE_RATE, MASTER_GAIN, MIDI_TO_FREQ, PROFILES,
    _TWO_PI, _INV_PI,
    pulse_tone, triangle_tone, square_tone, saw_tone, noise_tone, wavetable_tone,
    _choose_wave_table,
    _rt_pulse_sample, _rt_triangle_sample, _rt_square_sample, _rt_saw_sample,
    _rt_generate_pulse_buffer, _rt_generate_triangle_buffer,
    _rt_generate_square_buffer, _rt_generate_saw_buffer,
    collect_notes_faithful, channels_in_notes,
    render_chiptune_float32, export_wav_float32, export_mp3_via_ffmpeg,
    _pan_gains,
)
from live_midi import LiveVoice, LiveMidiPlayer
from metronome import Metronome


# ---------------------------------------------------------------------------
# Accessibility helper â€” wx-dependent, so it lives here rather than synth.py
# ---------------------------------------------------------------------------
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

# Metronome IDs
ID_METRO_TOGGLE = wx.NewIdRef()

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
        self.SetTitle("MIDI Chiptune Player â€” Turntable + Chip Profiles")
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

        # Metronome
        self._metronome = Metronome()
        self._tap_times: list[float] = []

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
        play_menu.AppendSeparator()
        play_menu.AppendCheckItem(ID_METRO_TOGGLE, "&Metronome\tCtrl+T")
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
            (wx.ACCEL_CTRL, ord('T'), ID_METRO_TOGGLE),
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

        # Metronome bindings
        self.Bind(wx.EVT_MENU, self.on_metro_toggle, id=ID_METRO_TOGGLE)

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
        seek_parent = seek_box.GetStaticBox()
        self.slider_seek = wx.Slider(seek_parent, value=0, minValue=0, maxValue=1000, style=wx.SL_HORIZONTAL)
        self.lbl_time = wx.StaticText(seek_parent, label="Time: 0:00 / 0:00")
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

        self.scroll = wx.ScrolledWindow(self.channel_box, style=wx.VSCROLL)
        self.scroll.SetScrollRate(10, 10)
        self.scroll_sizer = wx.BoxSizer(wx.VERTICAL)
        self.scroll.SetSizer(self.scroll_sizer)
        self.channel_sizer.Add(self.scroll, 1, wx.EXPAND | wx.ALL, 6)

        # --- Live MIDI Keyboard section ---
        live_box = wx.StaticBoxSizer(wx.StaticBox(panel, label="Live MIDI Keyboard"), wx.VERTICAL)
        live_parent = live_box.GetStaticBox()

        # Device row
        device_row = wx.BoxSizer(wx.HORIZONTAL)
        device_lbl = wx.StaticText(live_parent, label="Device:")
        self.cmb_midi_device = wx.Choice(live_parent, choices=["(No MIDI devices)"])
        self.btn_refresh_midi = wx.Button(live_parent, label="Refresh")
        _a11y(self.cmb_midi_device, "MIDI input device selector", "Select MIDI keyboard or controller for live playing")
        _a11y(self.btn_refresh_midi, "Refresh MIDI devices button")

        device_row.Add(device_lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        device_row.Add(self.cmb_midi_device, 1, wx.EXPAND | wx.RIGHT, 8)
        device_row.Add(self.btn_refresh_midi, 0)

        # Enable checkbox
        self.chk_live_enable = wx.CheckBox(live_parent, label="Enable Live Input")
        _a11y(self.chk_live_enable, "Enable live MIDI input checkbox",
              "When checked, incoming MIDI notes will play through the chiptune synthesizer")

        # Live volume row
        live_vol_row = wx.BoxSizer(wx.HORIZONTAL)
        live_vol_lbl = wx.StaticText(live_parent, label="Live Volume:")
        self.slider_live_volume = wx.Slider(live_parent, value=80, minValue=0, maxValue=200,
                                             style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS)
        self.slider_live_volume.SetTickFreq(10)
        self.lbl_live_volume = wx.StaticText(live_parent, label="80%")
        _a11y(self.slider_live_volume, "Live MIDI volume slider", "Volume for live MIDI keyboard input, 0 to 200 percent")

        live_vol_row.Add(live_vol_lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        live_vol_row.Add(self.slider_live_volume, 1, wx.EXPAND | wx.RIGHT, 8)
        live_vol_row.Add(self.lbl_live_volume, 0, wx.ALIGN_CENTER_VERTICAL)

        # Status label
        self.lbl_live_status = wx.StaticText(live_parent, label="Status: Not connected")
        _a11y(self.lbl_live_status, "Live MIDI status label")

        live_box.Add(device_row, 0, wx.EXPAND | wx.ALL, 6)
        live_box.Add(self.chk_live_enable, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)
        live_box.Add(live_vol_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)
        live_box.Add(self.lbl_live_status, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        # --- Metronome section ---
        metro_box = wx.StaticBoxSizer(wx.StaticBox(panel, label="Metronome"), wx.VERTICAL)
        metro_parent = metro_box.GetStaticBox()

        # Row 1: enable + tap tempo
        metro_row1 = wx.BoxSizer(wx.HORIZONTAL)
        self.chk_metro = wx.CheckBox(metro_parent, label="Enable")
        self.btn_tap = wx.Button(metro_parent, label="Tap Tempo")
        _a11y(self.chk_metro, "Metronome enable checkbox", "Start or stop the metronome click track")
        _a11y(self.btn_tap, "Tap tempo button", "Click several times in rhythm to set the BPM")
        metro_row1.Add(self.chk_metro, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 16)
        metro_row1.Add(self.btn_tap, 0)

        # Row 2: BPM slider
        metro_bpm_row = wx.BoxSizer(wx.HORIZONTAL)
        metro_bpm_lbl = wx.StaticText(metro_parent, label="BPM:")
        self.slider_metro_bpm = wx.Slider(metro_parent, value=120, minValue=20, maxValue=300,
                                           style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS)
        self.slider_metro_bpm.SetTickFreq(20)
        self.lbl_metro_bpm = wx.StaticText(metro_parent, label="120 BPM")
        _a11y(self.slider_metro_bpm, "Metronome BPM slider", "Beats per minute, 20 to 300")
        metro_bpm_row.Add(metro_bpm_lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        metro_bpm_row.Add(self.slider_metro_bpm, 1, wx.EXPAND | wx.RIGHT, 8)
        metro_bpm_row.Add(self.lbl_metro_bpm, 0, wx.ALIGN_CENTER_VERTICAL)

        # Row 3: beats per bar
        metro_bar_row = wx.BoxSizer(wx.HORIZONTAL)
        metro_bar_lbl = wx.StaticText(metro_parent, label="Beats/bar:")
        self.cmb_metro_beats = wx.Choice(metro_parent, choices=["2", "3", "4", "6"])
        self.cmb_metro_beats.SetSelection(2)  # default: 4/4
        _a11y(self.cmb_metro_beats, "Beats per bar selector", "Number of beats in one bar (time signature numerator)")
        metro_bar_row.Add(metro_bar_lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        metro_bar_row.Add(self.cmb_metro_beats, 0)

        # Row 4: volume
        metro_vol_row = wx.BoxSizer(wx.HORIZONTAL)
        metro_vol_lbl = wx.StaticText(metro_parent, label="Volume:")
        self.slider_metro_vol = wx.Slider(metro_parent, value=70, minValue=0, maxValue=100,
                                           style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS)
        self.slider_metro_vol.SetTickFreq(10)
        self.lbl_metro_vol = wx.StaticText(metro_parent, label="70%")
        _a11y(self.slider_metro_vol, "Metronome volume slider", "Click volume 0 to 100 percent")
        metro_vol_row.Add(metro_vol_lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        metro_vol_row.Add(self.slider_metro_vol, 1, wx.EXPAND | wx.RIGHT, 8)
        metro_vol_row.Add(self.lbl_metro_vol, 0, wx.ALIGN_CENTER_VERTICAL)

        metro_box.Add(metro_row1, 0, wx.ALL, 6)
        metro_box.Add(metro_bpm_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)
        metro_box.Add(metro_bar_row, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)
        metro_box.Add(metro_vol_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        # Layout
        root.Add(self.lbl_file, 0, wx.ALL, 10)
        root.Add(self.lbl_status, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        root.Add(btn_grid, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        root.Add(self.chk_loop, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        root.Add(sliders, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        root.Add(seek_box, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        root.Add(self.lbl_profile, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        root.Add(live_box, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        root.Add(metro_box, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
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

        # Metronome events
        self.chk_metro.Bind(wx.EVT_CHECKBOX, self.on_metro_toggle)
        self.btn_tap.Bind(wx.EVT_BUTTON, self.on_metro_tap)
        self.slider_metro_bpm.Bind(wx.EVT_SLIDER, self.on_metro_bpm_change)
        self.cmb_metro_beats.Bind(wx.EVT_CHOICE, self.on_metro_beats_change)
        self.slider_metro_vol.Bind(wx.EVT_SLIDER, self.on_metro_vol_change)

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

    # ---------- Metronome handlers ----------
    def on_metro_toggle(self, event=None):
        """Toggle metronome on/off â€” called from checkbox or menu item."""
        # Determine desired state from whichever control was triggered
        if isinstance(event, wx.CommandEvent) and event.GetId() == ID_METRO_TOGGLE:
            # Triggered from menu: sync checkbox
            checked = self.GetMenuBar().FindItemById(ID_METRO_TOGGLE).IsChecked()
            self.chk_metro.SetValue(checked)
        else:
            # Triggered from checkbox: sync menu
            checked = self.chk_metro.GetValue()
            mb = self.GetMenuBar()
            if mb:
                mb.FindItemById(ID_METRO_TOGGLE).Check(checked)

        if checked:
            if not self._metronome.start():
                self.set_status("Metronome: could not open audio stream.")
                self.chk_metro.SetValue(False)
                mb = self.GetMenuBar()
                if mb:
                    mb.FindItemById(ID_METRO_TOGGLE).Check(False)
        else:
            self._metronome.stop()

    def on_metro_bpm_change(self, event=None):
        bpm = self.slider_metro_bpm.GetValue()
        self.lbl_metro_bpm.SetLabel(f"{bpm} BPM")
        self._metronome.set_bpm(float(bpm))
        self._tap_times.clear()

    def on_metro_beats_change(self, event=None):
        sel = self.cmb_metro_beats.GetStringSelection()
        try:
            self._metronome.set_beats_per_bar(int(sel))
        except ValueError:
            pass

    def on_metro_vol_change(self, event=None):
        vol = self.slider_metro_vol.GetValue()
        self.lbl_metro_vol.SetLabel(f"{vol}%")
        self._metronome.set_volume(vol / 100.0)

    def on_metro_tap(self, event=None):
        """Record a tap and recalculate BPM from the average tap interval."""
        now = time.time()
        self._tap_times.append(now)
        # Discard taps older than 3 seconds (stale rhythm)
        self._tap_times = [t for t in self._tap_times if now - t < 3.0]
        if len(self._tap_times) >= 2:
            intervals = [self._tap_times[i + 1] - self._tap_times[i]
                         for i in range(len(self._tap_times) - 1)]
            avg = sum(intervals) / len(intervals)
            bpm = int(round(max(20.0, min(300.0, 60.0 / avg))))
            self.slider_metro_bpm.SetValue(bpm)
            self.lbl_metro_bpm.SetLabel(f"{bpm} BPM")
            self._metronome.set_bpm(float(bpm))

    def on_exit(self, event=None):
        # Stop live MIDI player
        if self.live_midi_player:
            self.live_midi_player.close_midi_input()
            self.live_midi_player.stop()
        self._metronome.stop()
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
