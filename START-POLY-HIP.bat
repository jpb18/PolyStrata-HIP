@echo off
rem strata-poly on an AMD card (HIP, experimental: AMD_PORT_PHASE2.md), started with a double-click.
rem
rem   START-POLY-HIP.bat                  the server on this PC only (http://127.0.0.1:8090/), and its page in the browser
rem   START-POLY-HIP.bat chat             a chat in this window (the server starts by itself)
rem   START-POLY-HIP.bat C:\models\X.gguf the first time: writes the run config for this GGUF, then starts
rem   START-POLY-HIP.bat check            says what it would start, and starts nothing
rem
rem It builds the engine when build-hip-poly-win\strata-poly.exe is missing (tools\hip\build_windows.bat poly, with
rem the llama.cpp in third_party\llama.cpp-3cf0325 when there is one), and creates .venv when it is missing.
rem Settings: POLY_CONFIG (a strata-<name>.json; default: the first strata-*-hip.json here), POLY_NO_OPEN=1 (no browser).
setlocal EnableDelayedExpansion
title PolyStrata (AMD, HIP)
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
set "MODE=server"
set "GGUF="
if /i "%~1"=="chat" set "MODE=chat"
if /i "%~1"=="check" set "MODE=check"
if /i "%~x1"==".gguf" set "GGUF=%~f1"

rem --- Python, with the server's packages
if exist "%PY%" goto deps
echo  Creating the Python environment in .venv ...
set "SYSPY="
where py >nul 2>nul && set "SYSPY=py -3"
if not defined SYSPY where python >nul 2>nul && set "SYSPY=python"
if not defined SYSPY (
  echo  Python 3.10 or newer is not installed: run START-HERE.bat once, or install it from python.org.
  goto fail
)
%SYSPY% -m venv .venv
if not exist "%PY%" (echo  Could not create .venv & goto fail)
:deps
"%PY%" -c "import jinja2" >nul 2>nul
if errorlevel 1 (
  echo  Installing the server's Python packages ...
  "%PY%" -m pip install --disable-pip-version-check -q -r requirements.txt
  "%PY%" -c "import jinja2" >nul 2>nul
  if errorlevel 1 (echo  Could not install requirements.txt into .venv & goto fail)
)

rem --- the engine
if exist "build-hip-poly-win\strata-poly.exe" goto config
if /i "%MODE%"=="check" (echo  engine: missing, would be built with tools\hip\build_windows.bat poly & goto config)
echo  build-hip-poly-win\strata-poly.exe is missing: building it (10-30 minutes, once) ...
if not defined STRATA_GGML_DIR if exist "third_party\llama.cpp-3cf0325\ggml" set "STRATA_GGML_DIR=%CD%\third_party\llama.cpp-3cf0325"
call tools\hip\build_windows.bat poly
if not exist "build-hip-poly-win\strata-poly.exe" (echo  The build failed: see the messages above. & goto fail)

rem --- the run config
:config
if defined GGUF (
  if not exist "%GGUF%" (echo  No such file: %GGUF% & goto fail)
  for /f %%C in ('powershell -NoProfile -Command "(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum"') do set "CORES=%%C"
  if not defined CORES set "CORES=6"
  echo  Writing the run config for %GGUF% ^(context 49152, !CORES! CPU threads^) ...
  "%PY%" tools\hip\poly_config.py --model "%GGUF%" --context 49152 --threads !CORES! > "%TEMP%\poly-config.txt"
  if errorlevel 1 (type "%TEMP%\poly-config.txt" & goto fail)
  for /f "tokens=1,*" %%V in ('findstr /b /c:"written:" "%TEMP%\poly-config.txt"') do set "POLY_CONFIG=%%~nxW"
)
if not defined POLY_CONFIG for %%F in (strata-*-hip.json) do if not defined POLY_CONFIG set "POLY_CONFIG=%%F"
if not defined POLY_CONFIG (
  echo.
  echo  No model is set up yet. Give the GGUF once, for example:
  echo    START-POLY-HIP.bat C:\Users\%USERNAME%\models\Qwen3.5-35B-A3B-Q3_K_M.gguf
  echo  ^(or drop the .gguf file onto this script^)
  goto fail
)
if not exist "%POLY_CONFIG%" (echo  No such run config: %POLY_CONFIG% & goto fail)
for %%F in ("%POLY_CONFIG%") do set "NAME=%%~nF"
set "NAME=!NAME:~7!"
set "PORT=8090"
for /f "usebackq" %%P in (`%PY% -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8')).get('port',8090))" "%POLY_CONFIG%"`) do set "PORT=%%P"
set "URL=http://127.0.0.1:%PORT%"

if /i "%MODE%"=="check" (
  echo  python: %PY%
  echo  config: %POLY_CONFIG%   model: !NAME!   port: %PORT%
  echo  would start: "%PY%" serve\server.py --engine strata --config "%POLY_CONFIG%" --port %PORT%
  exit /b 0
)

rem --- a server already running on the port: use it
curl -s -m 2 -o nul "%URL%/health" >nul 2>nul
if not errorlevel 1 (
  echo  The server is already running: %URL%/
  if /i "%MODE%"=="chat" ("%PY%" polystrata.py --model "!NAME!" --port %PORT% & goto end)
  if not "%POLY_NO_OPEN%"=="1" start "" "%URL%/"
  goto end
)

if /i "%MODE%"=="chat" (
  "%PY%" polystrata.py --model "!NAME!" --port %PORT%
  goto end
)
echo  Starting !NAME! on %URL%/ ^(this PC only^). Close this window to stop it.
set "OPEN=--open"
if "%POLY_NO_OPEN%"=="1" set "OPEN="
"%PY%" serve\server.py --engine strata --config "%POLY_CONFIG%" --port %PORT% %OPEN%
goto end

:fail
echo.
pause
exit /b 1
:end
endlocal
