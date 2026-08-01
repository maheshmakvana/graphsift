@echo off
rem Portable launcher for the graphsift MCP server (stdio).
rem Avoids hardcoding any interpreter path. Probes, in order:
rem   .venv\Scripts\python.exe  venv\Scripts\python.exe  python.exe  py.exe
rem Each candidate must be able to import graphsift before it is used.
rem With no venv present, python/py resolve however the user's
rem environment is configured (conda, pyenv, system, etc.).
setlocal
cd /d "%~dp0\.."

set "PY="
for %%P in (".venv\Scripts\python.exe" "venv\Scripts\python.exe" "python.exe" "py.exe") do (
    if not defined PY (
        %%P -c "import graphsift" >nul 2>&1 && set "PY=%%P"
    )
)

if not defined PY (
    echo graphsift-mcp: no Python with graphsift installed was found. 1>&2
    echo Install it with: pip install -e .  or: pip install graphsift 1>&2
    exit /b 1
)

rem py.exe needs the -3 flag to select a Python.
if /i "%PY%"=="py.exe" set "PY=py -3"

%PY% -m graphsift.mcp_server %*
set "EC=%ERRORLEVEL%"
endlocal & exit /b %EC%
