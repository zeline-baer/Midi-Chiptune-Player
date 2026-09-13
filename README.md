# Midi Chiptune Player

Desktop MIDI player with five chiptune profiles, live MIDI keyboard input, turntable
speed control, and WAV/MP3 export.

## Windows setup

Tested with **64-bit CPython 3.12.10** on Windows. Use a project virtual environment
so these packages do not conflict with other Python applications. Run these commands
in PowerShell from the repository folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

The requirements pin the direct and transitive package versions tested together on
2026-09-13. Windows wheels are available for this Python version; no C++ compiler
is needed. Other Python versions and operating systems have not been verified.

Start the player by double-clicking **start_player.bat**, or run:

```powershell
.\.venv\Scripts\python.exe .\midi_chiptune_player_wx_turntable_profiles_chipaccurate.py
```

No virtual environment activation is required. If you see `ModuleNotFoundError`,
check that you are using the `.venv` interpreter above and have installed the
requirements into that environment.

## Testing

The automated tests cover the five implemented profiles (Neutral, NES-ish, GB-ish,
C64-ish, Beeper), synthesis, MIDI parsing and tempo changes, playback callbacks,
channel volumes, WAV export, and live MIDI controller logic. They return a nonzero
exit code on failure and do not play audio or open a MIDI input. The old TIA test
expectations did not match this version of the player and have been replaced with
checks for the implemented profiles.

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_functionality test_live_midi
.\.venv\Scripts\python.exe test_real_performance.py
```

Use the explicit test module names above: `test_midi_debug.py` is an interactive
hardware diagnostic, and the performance scripts are standalone benchmarks.

For a listening test, open a MIDI file, try Play/Pause, Stop, seeking, the speed and
volume sliders, and switch profiles with Ctrl+1 through Ctrl+5. Export a WAV and
play it back. To test a connected MIDI keyboard, select its input in **Live MIDI
Keyboard**, enable **Live Input**, and try notes, sustain, pitch bend and mod wheel.
Logic tests cannot verify the physical keyboard or the sound at your speakers.

## Optional MP3 export

WAV export works with the Python requirements alone. MP3 export additionally needs
`ffmpeg` on `PATH`; its automated test is skipped when FFmpeg is missing. To install
it with Windows Package Manager:

```powershell
winget install --id Gyan.FFmpeg --exact
```

Restart the terminal/player after installation and check `ffmpeg -version`.

## Checking future updates

```powershell
.\.venv\Scripts\python.exe -m pip list --outdated
```

Test upgrades in a separate virtual environment, run `pip check` and the automated
and listening tests, then update the pins in `requirements.txt` together. Keep the
NumPy, Numba and llvmlite versions compatible with each other. A newer Python
interpreter is not required for the tested setup.

