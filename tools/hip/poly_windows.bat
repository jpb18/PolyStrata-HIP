@echo off
rem strata-poly for AMD on Windows (STRATA_POLY_HIP, AMD_PORT_IMPLEMENTATION.md): runs what
rem "tools\hip\build_windows.bat poly" built, with the ROCm SDK's libraries on the PATH and its HIP runtime beside the
rem programs (the build copies it there).
rem
rem   tools\hip\poly_windows.bat probe                       the card, launches, round trips, graphs, pinned memory
rem   tools\hip\poly_windows.bat tests                       the tests that need no model (ctest -R poly^|conversation)
rem   tools\hip\poly_windows.bat parity MODEL.gguf           every expert format of the file: GPU and CPU vs FP64
rem   tools\hip\poly_windows.bat run --model MODEL.gguf ...  strata-poly with these arguments
rem   tools\hip\poly_windows.bat cmd ...                     any command in this environment
rem
rem Settings: ROCM_VENV (<repo>\.rocm-win), BUILD_DIR (<repo>\build-hip-poly-win)
setlocal EnableDelayedExpansion
for %%I in ("%~dp0..\..") do set "SRC=%%~fI"
if not defined ROCM_VENV set "ROCM_VENV=%SRC%\.rocm-win"
if not defined BUILD_DIR set "BUILD_DIR=%SRC%\build-hip-poly-win"
if not exist "%ROCM_VENV%\Scripts\rocm-sdk.exe" (echo No ROCm SDK in %ROCM_VENV%: run tools\hip\build_windows.bat poly first & exit /b 1)
for /f "delims=" %%R in ('"%ROCM_VENV%\Scripts\rocm-sdk.exe" path --root') do set "ROCM=%%R"
if not exist "%BUILD_DIR%\strata-poly.exe" (echo No %BUILD_DIR%\strata-poly.exe: run tools\hip\build_windows.bat poly first & exit /b 1)
set "PATH=%BUILD_DIR%;%ROCM%\bin;%PATH%;%ROCM_VENV%\Scripts"
set "WHAT=%~1"
if "%WHAT%"=="" (echo usage: tools\hip\poly_windows.bat probe^|tests^|parity MODEL^|run ARGS^|cmd ARGS & exit /b 2)
shift
set "REST="
:collect
if "%~1"=="" goto go
set "REST=!REST! %1"
shift
goto collect
:go
if /i "%WHAT%"=="probe" (
  "%BUILD_DIR%\strata-device.exe" --list-devices
  "%BUILD_DIR%\hip_poly_probe.exe"
  exit /b !errorlevel!
)
if /i "%WHAT%"=="tests" (
  ctest --test-dir "%BUILD_DIR%" --output-on-failure -R "poly|conversation"
  exit /b !errorlevel!
)
if /i "%WHAT%"=="parity" (
  "%BUILD_DIR%\poly_expert_parity.exe" !REST!
  exit /b !errorlevel!
)
if /i "%WHAT%"=="run" (
  "%BUILD_DIR%\strata-poly.exe" !REST!
  exit /b !errorlevel!
)
if /i "%WHAT%"=="cmd" (
  !REST!
  exit /b !errorlevel!
)
echo unknown: %WHAT% & exit /b 2
