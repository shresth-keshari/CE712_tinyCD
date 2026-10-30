#!/usr/bin/env bash
# A local, repeatable installation. No administrator access or shell-profile edits.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
DEMO_ONLY=0
CUDA=0
for arg in "$@"; do
  case "$arg" in
    --demo-only) DEMO_ONLY=1 ;;
    --cuda) CUDA=1 ;;
    --help|-h)
      printf '%s\n' 'Usage: bash setup.sh [--demo-only] [--cuda]' \
        'Default: local Python 3.12, CPU PyTorch, all dependencies, checkpoints and full LEVIR-CD256.' \
        '--demo-only: omit the 2.48 GB dataset archive; fixed demos/uploads/Powai still work.' \
        '--cuda: install the CUDA 12.6 PyTorch build (requires a compatible NVIDIA driver).'
      exit 0 ;;
    *) printf 'Unknown option: %s\n' "$arg" >&2; exit 2 ;;
  esac
done
trap 'printf "Setup stopped at line %s. Fix the reported error and rerun bash setup.sh.\n" "$LINENO" >&2' ERR
export PYTHONUTF8=1
export PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS:$ARCH" in
  Linux:x86_64) TARGET=x86_64-unknown-linux-gnu; SHA=ce4f6d0a6cfdb3bed6e44c2f9affd85943c76a42e25ad3a259002f31d6fb8a20 ;;
  Linux:aarch64|Linux:arm64) TARGET=aarch64-unknown-linux-gnu; SHA=bb4b6392670c5d3f93876996fcaa19858a4a413a0c7720144c18b0d6f1f0db2d ;;
  Darwin:arm64) TARGET=aarch64-apple-darwin; SHA=353d5f1b4fd2715e7053f12f068435d5bbad16b91e057006b5d01e0925709a7e ;;
  MINGW*:x86_64|MSYS*:x86_64|CYGWIN*:x86_64) TARGET=x86_64-pc-windows-msvc; SHA=1f2ab53880fc67ea323398c6388e05af28addc99a8e389209f08a51fdff4b149 ;;
  *) printf 'Unsupported platform: %s %s. Use Linux/WSL, Apple Silicon macOS, or Windows x64 Git Bash.\n' "$OS" "$ARCH" >&2; exit 1 ;;
esac
if [[ "$CUDA" == 1 && ( "$OS" == Darwin || "$ARCH" != x86_64 ) ]]; then
  printf '%s\n' '--cuda requires x86_64 Linux/WSL or Windows with an NVIDIA driver.' >&2; exit 2
fi
RUNTIME=".tools/python-$TARGET"
BASE_PY="$RUNTIME/python/bin/python3.12"
VENV_PY='.venv/bin/python'
if [[ "$TARGET" == *windows* ]]; then
  BASE_PY="$RUNTIME/python/python.exe"
  VENV_PY='.venv/Scripts/python.exe'
fi
if [[ ! -f "$BASE_PY" ]]; then
  command -v curl >/dev/null || { printf '%s\n' 'curl is required to bootstrap Python.' >&2; exit 1; }
  command -v tar >/dev/null || { printf '%s\n' 'tar is required to unpack Python.' >&2; exit 1; }
  mkdir -p "$RUNTIME"
  ARCHIVE="$RUNTIME/runtime.tar.gz"
  URL="https://github.com/astral-sh/python-build-standalone/releases/download/20261026/cpython-3.12.16%2B20261026-$TARGET-install_only_stripped.tar.gz"
  printf '%s\n' 'Downloading a checksum-pinned Python 3.12 runtime into .tools/ ...'
  curl --fail --location --retry 3 --connect-timeout 30 "$URL" -o "$ARCHIVE"
  if command -v sha256sum >/dev/null; then
    printf '%s  %s\n' "$SHA" "$ARCHIVE" | sha256sum --check --status
  else
    printf '%s  %s\n' "$SHA" "$ARCHIVE" | shasum -a 256 --check --status
  fi
  tar -xzf "$ARCHIVE" -C "$RUNTIME"
fi
if [[ ! -f "$VENV_PY" ]]; then
  if [[ -d .venv ]]; then
    printf '%s\n' '.venv belongs to another OS or is incomplete. Use a separate clone for Windows and WSL.' >&2
    exit 1
  fi
  "$BASE_PY" -m venv .venv
fi
"$VENV_PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Expected Python 3.12 in .venv"'
"$VENV_PY" -m pip install 'pip==26.2.2'
TORCH_INDEX='https://download.pytorch.org/whl/cpu'
if [[ "$CUDA" == 1 ]]; then TORCH_INDEX='https://download.pytorch.org/whl/cu126'; fi
if [[ "$OS" == Darwin ]]; then
  "$VENV_PY" -m pip install 'torch==2.7.0' 'torchvision==0.22.0'
else
  BUILD=cpu
  if [[ "$CUDA" == 1 ]]; then BUILD=cu126; fi
  VISION="torchvision==0.22.0+$BUILD"
  if [[ "$ARCH" == aarch64 || "$ARCH" == arm64 ]]; then VISION='torchvision==0.22.0'; fi
  "$VENV_PY" -m pip install "torch==2.7.0+$BUILD" "$VISION" --index-url "$TORCH_INDEX"
fi
"$VENV_PY" -m pip install -r requirements.txt -r requirements-usgs.txt
"$VENV_PY" -m pip check
"$VENV_PY" scripts/bootstrap_bit.py
if [[ "$DEMO_ONLY" == 1 ]]; then
  "$VENV_PY" scripts/download.py --weights
else
  "$VENV_PY" scripts/download.py
fi
"$VENV_PY" scripts/check_install.py
printf '\n%s\n' 'Setup complete. Start the app from this folder:' './streamlit run app/app.py'
