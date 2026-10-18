@echo off
rem Compatibility name: the launcher and all its options live in START_HIP_POLY.bat.
call "%~dp0START_HIP_POLY.bat" %*
exit /b %errorlevel%
