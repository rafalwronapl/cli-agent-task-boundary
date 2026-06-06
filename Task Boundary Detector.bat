@echo off
REM Task Boundary Detector — Windows GUI launcher.
REM Tries common Python install locations; falls back to pythonw on PATH.
cd /d %~dp0
set PYW=

REM 1. Miniconda / Anaconda in user profile (the most common install)
if exist "%USERPROFILE%\miniconda3\pythonw.exe" set PYW=%USERPROFILE%\miniconda3\pythonw.exe
if "%PYW%"=="" if exist "%USERPROFILE%\anaconda3\pythonw.exe" set PYW=%USERPROFILE%\anaconda3\pythonw.exe

REM 2. Standard python.org installs
if "%PYW%"=="" if exist "%LOCALAPPDATA%\Programs\Python\Python313\pythonw.exe" set PYW=%LOCALAPPDATA%\Programs\Python\Python313\pythonw.exe
if "%PYW%"=="" if exist "%LOCALAPPDATA%\Programs\Python\Python312\pythonw.exe" set PYW=%LOCALAPPDATA%\Programs\Python\Python312\pythonw.exe
if "%PYW%"=="" if exist "%LOCALAPPDATA%\Programs\Python\Python311\pythonw.exe" set PYW=%LOCALAPPDATA%\Programs\Python\Python311\pythonw.exe
if "%PYW%"=="" if exist "%LOCALAPPDATA%\Programs\Python\Python310\pythonw.exe" set PYW=%LOCALAPPDATA%\Programs\Python\Python310\pythonw.exe

REM 3. Whatever pythonw is on PATH (last resort; may be the Microsoft Store stub)
if "%PYW%"=="" set PYW=pythonw

start "" "%PYW%" "%~dp0gui.py"
