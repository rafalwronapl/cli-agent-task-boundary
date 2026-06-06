#!/usr/bin/env bash
# Cross-platform launcher (Linux / macOS). Windows users — see "Task Boundary Detector.bat".
set -e
cd "$(dirname "$0")"

# Pick a Python interpreter. Prefer python3, fall back to python.
PY="$(command -v python3 || command -v python)"
if [ -z "$PY" ]; then
  echo "No python3/python on PATH. Install Python 3.10+ and try again." >&2
  exit 1
fi

# Run the GUI. PYTHONIOENCODING=utf-8 is harmless on Linux/macOS and prevents
# the emoji crash some Polish-locale Windows consoles hit (Cp1250).
PYTHONIOENCODING=utf-8 exec "$PY" gui.py "$@"
