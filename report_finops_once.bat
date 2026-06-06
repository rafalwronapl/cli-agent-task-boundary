@echo off
setlocal
cd /d "%~dp0"
python reporter.py --server-url http://127.0.0.1:8787
endlocal
