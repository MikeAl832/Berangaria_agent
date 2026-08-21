@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "AGENT_PYTHONW=%CD%\.venv\Scripts\pythonw.exe"
if not exist "%AGENT_PYTHONW%" (
    echo Python environment not found. Run start-agent.bat once first.
    pause
    exit /b 1
)

start "Berangaria Agent" "%AGENT_PYTHONW%" -m berangaria_agent --gui
exit /b 0
