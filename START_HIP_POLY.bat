@echo off
rem strata-poly on an AMD card (HIP, experimental: AMD_PORT_PHASE2.md, AMD_KV_STREAMING.md), started with a double-click.
rem
rem   START_HIP_POLY.bat                  a menu: the model, when more than one is set up (here Qwen3.6-35B-A3B
rem                                       UD-Q3_K_M and Empero's Qwen3.8-35B-A3B-Distill IQ3_M), and the context (48K,
rem                                       128K, 262K, 262K with KV streaming), then the server on this PC only
rem                                       (http://127.0.0.1:8090/) and its page in the browser
rem   START_HIP_POLY.bat 128k             no menu: that profile (48k, 128k, 262k, 262k-stream)
rem   START_HIP_POLY.bat qwen3.6 [128k]   no model menu: the model set up here whose name has that text in it
rem   START_HIP_POLY.bat chat [128k]      a chat in this window (the server starts by itself)
rem   START_HIP_POLY.bat C:\models\X.gguf the first time: sets up this GGUF, then the menu
rem   START_HIP_POLY.bat check [128k]     says what it would start, and starts nothing
rem   START_HIP_POLY.bat q8_0 [128k]      voluntary Q8: 32K default, 48K/128K/262K, no parking
rem   START_HIP_POLY.bat fp16 [128k]      stable engine (also the default, never remembers Q8)
rem
rem   The Q8 expert cache has a measured ceiling per context (5 000 experts at 32K/48K, 3 500 at 128K, 2 000 at
rem   262K).  After the context the menu asks whether to keep it - Enter keeps the measured value - or to start
rem   without it: "START_HIP_POLY.bat q8_0 [128k] nocap" sends no ceiling at all and the engine sizes the cache to
rem   the free VRAM it finds.  Measured at 262K: 4 330 to 5 097 experts against the 2 000 with the ceiling, which
rem   raised the hit rate and the tokens/s (AMD_HIP_VRAM_FLOOR.md).
rem
rem Prerequisites are installed explicitly: see README.md. This launcher never installs packages or builds.
rem Each profile of a model has its own run config (tools\hip\poly_profile.py writes it once); the last choices are the
rem menus' defaults.  Settings: POLY_CONFIG (a strata-<name>.json: skips the menus), POLY_NO_OPEN=1 (no browser).
rem There is one engine, build-hip-poly-win\strata-poly.exe, for every menu; a change to it is in the next start.
setlocal EnableDelayedExpansion
title PolyStrata (AMD, HIP)
pushd "%~dp0" || exit /b 1
set "PY=.venv\Scripts\python.exe"
set "MODE=server"
set "GGUF="
set "PROFILE="
set "Q8_PROFILE="
set "Q8_CAPS="
set "MODEL="
set "KV=fp16"
set "KV_EXPLICIT="
set "HAS_ARGS=%~1"
if /i "%~1"=="--help" goto help
if /i "%~1"=="help" goto help
for %%A in (%*) do (
  set "KNOWN="
  if /i "%%~A"=="fp16" (set "KV=fp16" & set "KV_EXPLICIT=1" & set "KNOWN=1")
  if /i "%%~A"=="q8_0" (set "KV=q8_0" & set "KV_EXPLICIT=1" & set "KNOWN=1")
  if /i "%%~A"=="chat" (set "MODE=chat" & set "KNOWN=1")
  if /i "%%~A"=="check" (set "MODE=check" & set "KNOWN=1")
  if /i "%%~xA"==".gguf" (set "GGUF=%%~fA" & set "KNOWN=1")
  if /i "%%~A"=="32k" (set "Q8_PROFILE=32k" & set "KNOWN=1")
  if /i "%%~A"=="nocap" (set "Q8_CAPS=uncapped" & set "KNOWN=1")
  for %%P in (48k 128k 262k 262k-stream) do if /i "%%~A"=="%%P" (set "PROFILE=%%P" & set "KNOWN=1")
  if not defined KNOWN set "MODEL=%%~A"
)

rem --- Python, with the server's packages
if not exist "%PY%" (
  echo Python environment missing. From the repository, run:
  echo   py -3 -m venv .venv
  echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
  echo See README.md for the explicit HIP build and model configuration steps.
  goto fail
)
"%PY%" -c "import sys; assert sys.version_info >= (3,10); import jinja2,numpy,regex,psutil,requests,yaml" >nul 2>nul
if errorlevel 1 (
  echo Python 3.10+ and the packages in requirements.txt are required.
  echo Run .venv\Scripts\python.exe -m pip install -r requirements.txt
  goto fail
)

rem --- KV selection: Enter/EOF/timeout means FP16; argument-based launches keep their old defaults.
if defined KV_EXPLICIT goto kv_selected
if defined HAS_ARGS goto kv_selected
set "KV="
for /f "delims=" %%K in ('call "%PY%" tools\hip\poly_launch.py --choose-kv') do set "KV=%%K"
if not defined KV goto fail
:kv_selected
rem --- Select the model before either KV branch chooses its context.
if not defined POLY_CONFIG if not defined GGUF if not defined MODEL (
  for /f "delims=" %%M in ('call "%PY%" tools\hip\poly_launch.py --choose-model --kv !KV! --mode !MODE!') do set "MODEL=%%M"
  if not defined MODEL goto fail
)
if /i "%KV%"=="q8_0" goto q8_config

rem --- the FP16 engine
if defined POLY_CONFIG goto config
if exist "build-hip-poly-win\strata-poly.exe" goto config
echo Missing build-hip-poly-win\strata-poly.exe. Build explicitly with:
echo   tools\hip\build_windows.bat poly
echo This build downloads ROCm and llama.cpp dependencies. See README.md first.
goto fail

:q8_config
if defined PROFILE set "Q8_PROFILE=!PROFILE!"
if not defined Q8_PROFILE (
  if /i "%MODE%"=="check" (
    set "Q8_PROFILE=32k"
  ) else (
    for /f "delims=" %%P in ('call "%PY%" tools\hip\poly_launch.py --choose-context') do set "Q8_PROFILE=%%P"
  )
)
if not defined Q8_PROFILE goto fail
rem --- the cache ceiling: the question comes after the context, Enter keeps the measured value
if not defined Q8_CAPS (
  if /i "%MODE%"=="check" (
    set "Q8_CAPS=capped"
  ) else (
    for /f "delims=" %%K in ('call "%PY%" tools\hip\poly_launch.py --choose-caps') do set "Q8_CAPS=%%K"
  )
)
if not defined Q8_CAPS goto fail
set "QARGS="
set "QARGS=--profile !Q8_PROFILE! --caps !Q8_CAPS!"
if defined POLY_CONFIG set "QARGS=!QARGS! --config "!POLY_CONFIG!""
if defined MODEL set "QARGS=!QARGS! --base "!MODEL!""
if defined GGUF set "QARGS=!QARGS! --gguf "!GGUF!""
set "POLY_CONFIG="
for /f "delims=" %%C in ('call "%PY%" tools\hip\poly_launch.py --prepare-q8 !QARGS!') do set "POLY_CONFIG=%%C"
if not defined POLY_CONFIG (echo  Q8 nao foi preparado. FP16 intacto. & goto fail)
goto have_config

rem --- the run config: POLY_CONFIG, else the profile's (from the menu)
:config
if defined POLY_CONFIG goto have_config
if defined GGUF if not exist "%GGUF%" (echo  O ficheiro nao existe: %GGUF% & goto fail)
if not defined GGUF (
  set "ANY="
  for %%F in (strata-*-hip*.json) do set "ANY=1"
  if not defined ANY (
    echo.
    echo  Ainda nao ha nenhum modelo configurado. Indica o GGUF uma vez, por exemplo:
    echo    START_HIP_POLY.bat "models\Qwen3.6-35B-A3B-UD-Q3_K_M.gguf"
    echo  ^(ou arrasta o ficheiro .gguf para cima deste script^)
    goto fail
  )
)
set "LASTM=build-hip-poly-win\last-model.txt"
set "LAST=build-hip-poly-win\last-profile.txt"
if not defined PROFILE (
  set "DEF=1"
  if exist "%LAST%" (
    set /p PREV=<"%LAST%"
    if /i "!PREV!"=="128k" set "DEF=2"
    if /i "!PREV!"=="262k" set "DEF=3"
    if /i "!PREV!"=="262k-stream" set "DEF=4"
  )
  rem the speeds of Qwen3.6-35B-A3B UD-Q3_K_M measured on an RX 9060 XT (AMD_QWEN36_Q3_MIGRATION.md), and of the
  rem Empero Qwen3.8 Distill IQ3_M at 48K (AMD_EMPERO_QWEN38.md); another model's were not measured here
  set "S1=velocidade nao medida"
  set "S2=velocidade nao medida"
  set "S3=velocidade nao medida"
  if defined MODEL if not "!MODEL:qwen3.6-35b-a3b-ud-q3_k_m=!"=="!MODEL!" (
    set "S1=~93 tok/s numa conversa curta"
    set "S2=~27 tok/s com 128K preenchidos"
    set "S3=~14 tok/s com 261K preenchidos"
  )
  if defined MODEL if not "!MODEL:qwen3.8-35b-a3b-distill-iq3_m=!"=="!MODEL!" set "S1=~92 tok/s numa conversa curta"
  echo.
  echo   PolyStrata no AMD - que tamanho de contexto?
  echo.
  echo    1  48K          o mais rapido para conversas e codigo           ^(!S1!^)
  echo    2  128K         documentos longos, a KV toda na VRAM            ^(!S2!^)
  echo    3  262K         o maximo do modelo, a KV toda na VRAM           ^(!S3!^)
  echo    4  262K stream  262K com parte da KV na RAM: so quando outros   ^(rapido ate 48K, muito lento acima^)
  echo                    programas ocupam a VRAM e a opcao 3 nao carrega
  echo.
  if /i "%MODE%"=="check" (
    set "CH=!DEF!"
  ) else (
    choice /c 1234 /n /t 20 /d !DEF! /m "  Escolhe 1-4 (em 20 s fica a !DEF!): "
    set "CH=!errorlevel!"
  )
  if "!CH!"=="1" set "PROFILE=48k"
  if "!CH!"=="2" set "PROFILE=128k"
  if "!CH!"=="3" set "PROFILE=262k"
  if "!CH!"=="4" set "PROFILE=262k-stream"
)
if not defined PROFILE set "PROFILE=48k"
set "GARG="
if defined MODEL set "GARG=--base !MODEL!"
if defined GGUF set "GARG=--gguf "%GGUF%""
if defined GGUF echo  A configurar %GGUF% ...
set "POLY_CONFIG="
set "PROFILE_GUARD=--no-overwrite"
if /i "%MODE%"=="check" set "PROFILE_GUARD=--no-overwrite --read-only"
for /f "delims=" %%C in ('call "%PY%" tools\hip\poly_profile.py !PROFILE! !GARG! !PROFILE_GUARD!') do set "POLY_CONFIG=%%C"
if not defined POLY_CONFIG (echo  Nao foi possivel escrever a configuracao do perfil !PROFILE!: ve as mensagens acima. & goto fail)
if /i not "%MODE%"=="check" >"%LAST%" echo !PROFILE!
if /i not "%MODE%"=="check" if defined MODEL >"%LASTM%" echo !MODEL!

:have_config
if not exist "%POLY_CONFIG%" (echo  A configuracao nao existe: %POLY_CONFIG% & goto fail)
for %%F in ("%POLY_CONFIG%") do set "NAME=%%~nF"
set "NAME=!NAME:~7!"
set "PORT=8090"
for /f "usebackq" %%P in (`%PY% -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8')).get('port',8090))" "%POLY_CONFIG%"`) do set "PORT=%%P"
set "URL=http://127.0.0.1:%PORT%"

rem The adapter checks every port owner (including non-HTTP), never asks a resident server to switch,
rem and confirms Q8 INFO/READY + health before announcing that it is active. Server fail-closed is unchanged.
set "OPEN=--open"
if "%POLY_NO_OPEN%"=="1" set "OPEN="
"%PY%" tools\hip\poly_launch.py --config "%POLY_CONFIG%" --kv %KV% --mode %MODE% %OPEN%
if errorlevel 1 goto fail
goto end

:fail
echo.
if not defined HAS_ARGS if not "%POLY_NO_PAUSE%"=="1" pause
popd
exit /b 1
:help
echo Usage: START_HIP_POLY.bat [model-substring^|"model.gguf"] [48k^|128k^|262k^|262k-stream]
echo        START_HIP_POLY.bat [fp16^|q8_0] [chat^|check] [context] [nocap]
echo POLY_CONFIG selects an existing JSON config; its engine arguments are preserved.
echo POLY_NO_OPEN=1 disables the browser; POLY_NO_PAUSE=1 disables error pauses.
echo Q8 uses the profile described in docs\HIP_LAUNCH_KV.md, with the same engine as FP16.
echo Installation, builds and downloads are explicit steps in README.md.
:end
popd
endlocal
exit /b 0
