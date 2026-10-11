@echo off
rem strata-poly on an AMD card (HIP, experimental: AMD_PORT_PHASE2.md, AMD_KV_STREAMING.md), started with a double-click.
rem
rem   START-POLY-HIP.bat                  a menu: the model, when more than one is set up (Qwen3.5, Qwen3.6), and the
rem                                       context (48K, 128K, 262K, 262K with KV streaming), then the server on this PC
rem                                       only (http://127.0.0.1:8090/) and its page in the browser
rem   START-POLY-HIP.bat 128k             no menu: that profile (48k, 128k, 262k, 262k-stream)
rem   START-POLY-HIP.bat qwen3.6 [128k]   no model menu: the model set up here whose name has that text in it
rem   START-POLY-HIP.bat chat [128k]      a chat in this window (the server starts by itself)
rem   START-POLY-HIP.bat C:\models\X.gguf the first time: sets up this GGUF, then the menu
rem   START-POLY-HIP.bat check [128k]     says what it would start, and starts nothing
rem
rem It builds the engine when build-hip-poly-win\strata-poly.exe is missing (tools\hip\build_windows.bat poly, with
rem the llama.cpp in third_party\llama.cpp-3cf0325 when there is one), and creates .venv when it is missing. Each
rem profile of a model has its own run config (tools\hip\poly_profile.py writes it once); the last choices are the
rem menus' defaults.  Settings: POLY_CONFIG (a strata-<name>.json: skips the menus), POLY_NO_OPEN=1 (no browser).
setlocal EnableDelayedExpansion
title PolyStrata (AMD, HIP)
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
set "MODE=server"
set "GGUF="
set "PROFILE="
set "MODEL="
for %%A in (%*) do (
  set "KNOWN="
  if /i "%%~A"=="chat" (set "MODE=chat" & set "KNOWN=1")
  if /i "%%~A"=="check" (set "MODE=check" & set "KNOWN=1")
  if /i "%%~xA"==".gguf" (set "GGUF=%%~fA" & set "KNOWN=1")
  for %%P in (48k 128k 262k 262k-stream) do if /i "%%~A"=="%%P" (set "PROFILE=%%P" & set "KNOWN=1")
  if not defined KNOWN set "MODEL=%%~A"
)

rem --- Python, with the server's packages
if exist "%PY%" goto deps
echo  A criar o ambiente Python em .venv ...
set "SYSPY="
where py >nul 2>nul && set "SYSPY=py -3"
if not defined SYSPY where python >nul 2>nul && set "SYSPY=python"
if not defined SYSPY (
  echo  O Python 3.10 ou mais recente nao esta instalado: corre o START-HERE.bat uma vez, ou instala-o de python.org.
  goto fail
)
%SYSPY% -m venv .venv
if not exist "%PY%" (echo  Nao foi possivel criar o .venv & goto fail)
:deps
"%PY%" -c "import jinja2" >nul 2>nul
if errorlevel 1 (
  echo  A instalar os pacotes Python do servidor ...
  "%PY%" -m pip install --disable-pip-version-check -q -r requirements.txt
  "%PY%" -c "import jinja2" >nul 2>nul
  if errorlevel 1 (echo  Nao foi possivel instalar o requirements.txt no .venv & goto fail)
)

rem --- the engine
if exist "build-hip-poly-win\strata-poly.exe" goto config
if /i "%MODE%"=="check" (echo  motor: em falta, seria compilado com tools\hip\build_windows.bat poly & goto config)
echo  Falta build-hip-poly-win\strata-poly.exe: a compilar ^(10 a 30 minutos, uma vez^) ...
if not defined STRATA_GGML_DIR if exist "third_party\llama.cpp-3cf0325\ggml" set "STRATA_GGML_DIR=%CD%\third_party\llama.cpp-3cf0325"
call tools\hip\build_windows.bat poly
if not exist "build-hip-poly-win\strata-poly.exe" (echo  A compilacao falhou: ve as mensagens acima. & goto fail)

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
    echo    START-POLY-HIP.bat C:\Users\%USERNAME%\models\Qwen3.5-35B-A3B-Q3_K_M.gguf
    echo  ^(ou arrasta o ficheiro .gguf para cima deste script^)
    goto fail
  )
)
rem --- the model, when more than one is set up and none was named (the last one chosen is the default)
set "LASTM=build-hip-poly-win\last-model.txt"
if not defined GGUF if not defined MODEL (
  set "N=0"
  for /f "tokens=1,2 delims=|" %%M in ('call "%PY%" tools\hip\poly_profile.py --list') do (
    set /a N+=1
    set "BASE!N!=%%M"
    set "DISP!N!=%%N"
  )
  if !N! GTR 1 (
    set "DEFM=1"
    set "KEYS="
    if exist "%LASTM%" set /p PREVM=<"%LASTM%"
    for /l %%I in (1,1,!N!) do (
      set "KEYS=!KEYS!%%I"
      if /i "!BASE%%I!"=="!PREVM!" set "DEFM=%%I"
    )
    echo.
    echo   PolyStrata no AMD - que modelo?
    echo.
    for /l %%I in (1,1,!N!) do echo    %%I  !DISP%%I!
    echo.
    if /i "%MODE%"=="check" (
      set "CHM=!DEFM!"
    ) else (
      choice /c !KEYS! /n /t 20 /d !DEFM! /m "  Escolhe 1-!N! (em 20 s fica o !DEFM!): "
      set "CHM=!errorlevel!"
    )
    for %%I in (!CHM!) do set "MODEL=!BASE%%I!"
  )
)
set "LAST=build-hip-poly-win\last-profile.txt"
if not defined PROFILE (
  set "DEF=1"
  if exist "%LAST%" (
    set /p PREV=<"%LAST%"
    if /i "!PREV!"=="128k" set "DEF=2"
    if /i "!PREV!"=="262k" set "DEF=3"
    if /i "!PREV!"=="262k-stream" set "DEF=4"
  )
  echo.
  echo   PolyStrata no AMD - que tamanho de contexto?
  echo.
  echo    1  48K          o mais rapido para conversas e codigo           ^(~68 tok/s numa conversa curta^)
  echo    2  128K         documentos longos, a KV toda na VRAM            ^(~29 tok/s com 128K preenchidos^)
  echo    3  262K         o maximo do modelo, a KV toda na VRAM           ^(~17 tok/s com 261K preenchidos^)
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
for /f "delims=" %%C in ('call "%PY%" tools\hip\poly_profile.py !PROFILE! !GARG!') do set "POLY_CONFIG=%%C"
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

if /i "%MODE%"=="check" (
  echo  python: %PY%
  echo  configuracao: %POLY_CONFIG%   modelo: !NAME!   porta: %PORT%
  echo  arrancaria: "%PY%" serve\server.py --engine strata --config "%POLY_CONFIG%" --port %PORT%
  exit /b 0
)

rem --- a server already running on the port: use it when it has this model, else say which one it has
curl -s -m 2 -o nul "%URL%/health" >nul 2>nul
if not errorlevel 1 (
  set "RUNNING="
  for /f "usebackq delims=" %%M in (`curl -s -m 5 "%URL%/v1/models" ^| "%PY%" -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])"`) do set "RUNNING=%%M"
  if /i not "!RUNNING!"=="!NAME!" (
    echo.
    echo  Ja esta a correr outro servidor em %URL%/ com "!RUNNING!".
    echo  Fecha a janela dele e volta a abrir este script para arrancar "!NAME!".
    goto fail
  )
  echo  O servidor ja esta a correr: %URL%/
  if /i "%MODE%"=="chat" ("%PY%" polystrata.py --model "!NAME!" --port %PORT% & goto end)
  if not "%POLY_NO_OPEN%"=="1" start "" "%URL%/"
  goto end
)

if /i "%MODE%"=="chat" (
  "%PY%" polystrata.py --model "!NAME!" --port %PORT%
  goto end
)
echo.
echo  A arrancar !NAME! em %URL%/ ^(so neste PC^). Fecha esta janela para o parar.
echo  A primeira resposta demora mais: o modelo carrega em 15 a 30 s.
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
