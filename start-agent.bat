@echo off
setlocal EnableExtensions

cd /d "%~dp0"
title Berangaria Agent

if defined BERANGARIA_AGENT_PYTHON (
    set "AGENT_PYTHON=%BERANGARIA_AGENT_PYTHON%"
) else (
    set "AGENT_PYTHON=%CD%\.venv\Scripts\python.exe"
)
set "AGENT_REQUIREMENTS=%CD%\requirements.txt"

echo.
echo  Berangaria Agent
echo  =================
echo.

if not exist "%AGENT_PYTHON%" (
    echo [1/3] Creating a Python 3.11 environment...

    where uv >nul 2>nul
    if not errorlevel 1 (
        uv venv "%CD%\.venv" --python 3.11
        if errorlevel 1 goto :setup_error
        set "AGENT_PYTHON=%CD%\.venv\Scripts\python.exe"
    ) else (
        where py >nul 2>nul
        if errorlevel 1 goto :python_missing
        py -3.11 -m venv "%CD%\.venv"
        if errorlevel 1 goto :python_missing
        set "AGENT_PYTHON=%CD%\.venv\Scripts\python.exe"
    )

    echo [2/3] Installing dependencies...
    call :install_requirements
    if errorlevel 1 goto :setup_error
) else (
    echo [1/3] Python environment found.
    "%AGENT_PYTHON%" -c "import faster_whisper, httpx, dotenv, mss, PIL, sounddevice, webrtcvad, yaml" >nul 2>nul
    if errorlevel 1 (
        echo [2/3] Repairing dependencies...
        call :install_requirements
        if errorlevel 1 goto :setup_error
    ) else (
        echo [2/3] Dependencies are ready.
    )
)

if /i "%~1"=="--help" goto :launch

if not exist "%CD%\.env" (
    echo [3/3] Creating .env...
    copy /y "%CD%\.env.example" "%CD%\.env" >nul
    if errorlevel 1 goto :setup_error

    echo.
    echo Notepad will open now.
    echo Fill at least OPENROUTER_API_KEY.
    echo OPENROUTER_API_KEY enables Luna vision and microphone transcription.
    echo FISH_API_KEY and FISH_VOICE_ID enable spoken replies.
    echo Save the file and close Notepad to continue.
    echo.
    start "" /wait notepad.exe "%CD%\.env"
) else (
    echo [3/3] Configuration file found.
)

if not exist "%CD%\config.yaml" (
    copy /y "%CD%\config.example.yaml" "%CD%\config.yaml" >nul
    if errorlevel 1 goto :setup_error
    echo Created config.yaml from config.example.yaml.
)

:launch
echo.
echo Starting the local agent. Press Ctrl+C to stop it.
echo.
if "%~1"=="" (
    "%AGENT_PYTHON%" -m berangaria_agent --gui
) else (
    "%AGENT_PYTHON%" -m berangaria_agent %*
)
set "AGENT_EXIT=%ERRORLEVEL%"

if not "%AGENT_EXIT%"=="0" (
    echo.
    echo The agent exited with error %AGENT_EXIT%.
    pause
)
exit /b %AGENT_EXIT%

:install_requirements
where uv >nul 2>nul
if not errorlevel 1 (
    uv pip install --python "%AGENT_PYTHON%" -r "%AGENT_REQUIREMENTS%"
) else (
    "%AGENT_PYTHON%" -m pip install -r "%AGENT_REQUIREMENTS%"
)
exit /b %ERRORLEVEL%

:python_missing
echo.
echo Python 3.11 and uv were not found.
echo Install uv from https://docs.astral.sh/uv/ or Python 3.11 from python.org.
pause
exit /b 1

:setup_error
echo.
echo Failed to prepare the environment or install dependencies.
echo Check the messages above and your internet connection.
pause
exit /b 1
