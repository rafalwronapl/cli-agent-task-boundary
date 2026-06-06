@echo off
setlocal
cd /d "%~dp0"
echo Starting Task Boundary FinOps at http://127.0.0.1:8787
python -m server
endlocal
