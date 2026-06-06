@echo off
REM Install Layer 2 — sentence-transformers (80MB model + numpy + torch).
REM First run downloads all-MiniLM-L6-v2 model (~80MB) to ~/.cache/task-boundary-detector/
echo Installing Layer 2 dependencies (sentence-transformers, numpy, torch)...
echo This may take 2-5 minutes.
pip install sentence-transformers numpy
if errorlevel 1 (
    echo.
    echo INSTALLATION FAILED. Check Python version, pip, and internet connection.
    pause
    exit /b 1
)
echo.
echo Done. Layer 2 should now show as ON in the GUI.
echo First Analyze call will download the model (~30s, one-time).
pause
