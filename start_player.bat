@echo off
setlocal
pushd "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Create the project environment first:
    echo   py -3.12 -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    popd
    exit /b 1
)
".venv\Scripts\python.exe" "midi_chiptune_player_wx_turntable_profiles_chipaccurate.py"
set "PLAYER_EXIT_CODE=%ERRORLEVEL%"
if not "%PLAYER_EXIT_CODE%"=="0" pause
popd
exit /b %PLAYER_EXIT_CODE%
