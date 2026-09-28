#!/usr/bin/env bash
# Rytability Handwriting Magic - one-command launcher (macOS / Linux).
#
#   ./run.sh              start the demo and open it in your browser
#   ./run.sh --lan        also serve it on your Wi-Fi so an iPad + Apple Pencil can use it
#   ./run.sh --reader apple   use macOS's built-in handwriting recogniser (experimental)
#
# First run: creates .venv, installs dependencies (~1 GB) and downloads the
# models (~4.5 GB handwriting reader + 43 MB handwriting synthesiser).
set -euo pipefail
cd "$(dirname "$0")"

pick_python() {
  for p in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$p" >/dev/null 2>&1 && "$p" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      echo "$p"; return 0
    fi
  done
  return 1
}

if [ ! -x .venv/bin/python ]; then
  echo "==> Setting up (first run only)"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 .venv
    PIP=(uv pip install --python .venv/bin/python)
  else
    PY="$(pick_python)" || {
      echo "Python 3.10+ is needed. Install it with:  brew install python@3.12   (or install uv: https://docs.astral.sh/uv/)"
      exit 1
    }
    "$PY" -m venv .venv
    .venv/bin/python -m pip install --upgrade pip >/dev/null
    PIP=(.venv/bin/python -m pip install)
  fi
  "${PIP[@]}" -r requirements.txt
  touch .venv/.deps-ok
elif [ requirements.txt -nt .venv/.deps-ok ]; then
  echo "==> requirements.txt changed, updating dependencies"
  if command -v uv >/dev/null 2>&1; then uv pip install --python .venv/bin/python -r requirements.txt
  else .venv/bin/python -m pip install -r requirements.txt; fi
  touch .venv/.deps-ok
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HUB_DISABLE_TELEMETRY=1
exec .venv/bin/python server/app.py "$@"
