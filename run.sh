#!/usr/bin/env bash
# Rytability Handwriting Magic - one-command launcher (macOS / Linux).
#
#   ./run.sh                  start the demo and open it in your browser
#   ./run.sh --lan            also serve it on your Wi-Fi so an iPad + Apple Pencil can use it
#   ./run.sh --reader apple   use macOS's built-in handwriting recogniser (experimental)
#   ./run.sh --reset          delete the Python environment and set it up again
#
# First run: creates .venv, installs dependencies (~1 GB) and downloads the
# models (~4.5 GB handwriting reader + 43 MB handwriting synthesiser).
# Everything this script prints is also saved to logs/last-run.log.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
exec > >(tee logs/last-run.log) 2>&1

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
fail() {
  printf '\n\033[1;31mSetup failed: %s\033[0m\n' "$*"
  echo "The full log is in: $(pwd)/logs/last-run.log  (send it to Claude to get this fixed)"
  exit 1
}

ARGS=()
for a in "$@"; do
  if [ "$a" = "--reset" ]; then say "Removing the old environment"; rm -rf .venv; else ARGS+=("$a"); fi
done

say "System: $(uname -s) $(uname -r) $(uname -m)$( [ "$(uname -s)" = Darwin ] && echo " / macOS $(sw_vers -productVersion)")"
if [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = x86_64 ] && [ "$(sysctl -n sysctl.proc_translated 2>/dev/null)" = 1 ]; then
  echo "Note: this Terminal is running under Rosetta (Intel mode). It works, but Apple Silicon native is much faster."
fi

# --- find a Python >= 3.10, or get one via uv --------------------------------
find_uv() {
  for u in uv "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" /opt/homebrew/bin/uv /usr/local/bin/uv; do
    if command -v "$u" >/dev/null 2>&1; then command -v "$u"; return 0; fi
  done
  return 1
}
pick_python() {
  for p in python3.13 python3.12 python3.11 python3.10 /opt/homebrew/bin/python3 /usr/local/bin/python3 python3; do
    if command -v "$p" >/dev/null 2>&1 && "$p" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      echo "$p"; return 0
    fi
  done
  return 1
}

if [ ! -x .venv/bin/python ] || ! .venv/bin/python -c 'import sys' >/dev/null 2>&1; then
  say "Setting up (first run only - this can take 5-10 minutes)"
  rm -rf .venv
  UV="$(find_uv || true)"
  if [ -z "$UV" ] && ! pick_python >/dev/null; then
    say "Python 3.10+ not found - installing uv (a small Python manager, into ~/.local/bin, no admin needed)"
    curl -LsSf https://astral.sh/uv/install.sh | sh || fail "could not install uv. Install Python instead: brew install python@3.12"
    UV="$(find_uv || true)"
    [ -n "$UV" ] || fail "uv installed but not found on PATH"
  fi
  if [ -n "$UV" ]; then
    echo "Using uv: $UV"
    "$UV" venv --python 3.12 .venv || fail "uv could not create the Python environment"
  else
    PY="$(pick_python)"
    echo "Using $PY ($("$PY" --version))"
    "$PY" -m venv .venv || fail "could not create the Python environment with $PY"
    .venv/bin/python -m pip install --upgrade pip >/dev/null || true
  fi
  rm -f .venv/.deps-ok
fi

UV="$(find_uv || true)"
pip_install() {
  if [ -n "$UV" ]; then "$UV" pip install --python .venv/bin/python "$@"
  else .venv/bin/python -m pip install "$@"; fi
}

READER_ARGS=()
if [ ! -f .venv/.deps-ok ] || [ requirements.txt -nt .venv/.deps-ok ]; then
  say "Installing core packages"
  pip_install numpy scipy pillow || fail "could not install numpy/scipy/pillow"
  say "Installing the AI packages (PyTorch, Transformers - about 1 GB)"
  if pip_install -r requirements.txt; then
    touch .venv/.deps-ok
  else
    printf '\n\033[1;33mThe AI packages failed to install (details above), so starting in TIDY-ONLY mode.\033[0m\n'
    echo "Tidy mode works fully. For Magic mode, send logs/last-run.log to Claude."
    READER_ARGS=(--reader none)
  fi
fi

if ! .venv/bin/python -c "import numpy, scipy, PIL" 2>/dev/null; then
  fail "core packages are missing even after installing"
fi
if [ ${#READER_ARGS[@]} -eq 0 ] && ! .venv/bin/python -c "import torch, transformers" 2>/dev/null; then
  echo "PyTorch/Transformers not importable - starting in TIDY-ONLY mode (try ./run.sh --reset)."
  READER_ARGS=(--reader none)
fi

say "Starting (first start downloads the models; the page works in Tidy mode meanwhile)"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HUB_DISABLE_TELEMETRY=1
.venv/bin/python server/app.py ${READER_ARGS[@]+"${READER_ARGS[@]}"} ${ARGS[@]+"${ARGS[@]}"}
status=$?
[ $status -eq 0 ] || fail "the server stopped with an error (exit code $status)"
