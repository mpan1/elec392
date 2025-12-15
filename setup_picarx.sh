#!/usr/bin/env bash
#
# setup_picarx.sh
# - Clones SunFounder robot-hat + picar-x (v2.0 branch by default)
# - Installs into a dedicated Python venv (recommended)
# - Performs quick sanity checks (import + I2C presence)
#
# Usage:
#   ./setup_picarx.sh
#
# Optional environment variables:
#   INSTALL_DIR    Where to put cloned repos (default: ~/dev/sunfounder)
#   VENV_DIR       Where to create venv (default: ./venv-picarx)
#   ROBOT_HAT_REF  Branch/tag/commit (default: v2.0)
#   PICARX_REF     Branch/tag/commit (default: v2.0)
#   PYTHON_BIN     Python interpreter (default: python3)
#
set -euo pipefail

###############################################################################
#                                                                             #
#      ███████╗██╗     ███████╗ ██████╗        ██████╗  █████╗ ██████╗        #
#      ██╔════╝██║     ██╔════╝██╔════╝       ██╔════╝ ██╔══██╗██╔══██╗       #
#      █████╗  ██║     █████╗  ██║            ██║      ███████║██████╔╝       #
#      ██╔══╝  ██║     ██╔══╝  ██║            ██║      ██╔══██║██╔══██╗       #
#      ███████╗███████╗███████╗╚██████╗       ╚██████╗ ██║  ██║██║  ██║       #
#      ╚══════╝╚══════╝╚══════╝ ╚═════╝        ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝       #
#                                                                             #
#   ELEC 392 – Engineering Design & Development                               #
#   Picar-X Environment Setup                                   #
#                                                                             #
#   Target OS : Debian 12 (Bookworm)                                          #
#   Hardware  : Raspberry Pi 4B                                               #
#                                                                             #
###############################################################################

print_centered() {
  local cols="${COLUMNS:-80}"
  while IFS= read -r line; do
    # Strip trailing whitespace for accurate width
    local clean="${line%"${line##*[![:space:]]}"}"
    local len="${#clean}"
    if (( len < cols )); then
      printf "%*s%s\n" $(( (cols - len) / 2 )) "" "$clean"
    else
      printf "%s\n" "$clean"
    fi
  done
}

print_centered <<'EOF'

             ':looooooooooooooooooo;
          ':lxOOOOOOOOOOOOOOOOOOOkd;
       ':lxOOOOOOOOOOOOOOOOOOOkdc,
     ;lxOOOOOOOOOOOOOOOOOOOkdc,
    :kOOOOOOOOOOOOOOOOOOOdl;'
    ';;;;;;;:cdkOOOOOOOOkdooooooooo;
          ':oxOOOOOOOOOOOOOOOOOOOkd;
       ':lxOOOOOOOOOOOOOOOOOOOkdc,
    ';lxOOOOOOOOOOOOOOOOOOOkdc,
    :kOOOOOOOOOOOOOOOOOOOdl;
    ';;;;;;;cdkOOOOOOOOkxooooooooo;
          ':oxOOOOOOOOOOO0OOOOOOOko;
       ':oxOOOOOOOOOOOOOOOOOOOkdc,
    ':oxOOOOOOOOOOOOOOOOO0Okdc,
    :kOOOOOOOOOOOOOOOOOOkdc,
    ';;;;;;;;;;;;;;;;;;;,

ELEC 392 – Engineering Design & Development
Queen’s University | Smith Engineering

PiCar-X Environment Setup

EOF

# -------------------------
# Config (override via env)
# -------------------------
INSTALL_DIR="${INSTALL_DIR:-$HOME/dev/sunfounder}"
VENV_DIR="${VENV_DIR:-./venv-picarx}"
ROBOT_HAT_REF="${ROBOT_HAT_REF:-v2.0}"
PICARX_REF="${PICARX_REF:-v2.0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ROBOT_HAT_REPO="https://github.com/sunfounder/robot-hat.git"
PICARX_REPO="https://github.com/sunfounder/picar-x.git"
VILIB_REF="${VILIB_REF:-picamera2}"
VILIB_REPO="https://github.com/sunfounder/vilib.git"


# -------------------------
# Helpers
# -------------------------
log()  { echo -e "\n\033[1;34m[INFO]\033[0m $*"; }
warn() { echo -e "\n\033[1;33m[WARN]\033[0m $*"; }
err()  { echo -e "\n\033[1;31m[ERR ]\033[0m $*"; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { err "Missing required command: $1"; exit 1; }
}

is_pi() {
  grep -qi "raspberry pi" /proc/cpuinfo 2>/dev/null
}

# Clone or update a repo to a specific ref
sync_repo() {
  local url="$1"
  local dir="$2"
  local ref="$3"

  if [[ -d "$dir/.git" ]]; then
    log "Updating existing repo: $dir"
    git -C "$dir" fetch --all --tags --prune
  else
    log "Cloning repo: $url -> $dir"
    git clone "$url" "$dir"
  fi

  log "Checking out ref '$ref' in $dir"
  git -C "$dir" checkout "$ref" >/dev/null 2>&1 || {
    err "Failed to checkout '$ref' in $dir. Available refs:"
    git -C "$dir" branch -a || true
    git -C "$dir" tag -l || true
    exit 1
  }

  # If it's a branch, pull latest. If it's a tag/commit, no pull.
  if git -C "$dir" show-ref --verify --quiet "refs/heads/$ref"; then
    git -C "$dir" pull --ff-only
  fi

  log "Pinned $dir at: $(git -C "$dir" rev-parse --short HEAD)"
}

# -------------------------
# Start
# -------------------------
log "PiCar-X setup (SunFounder robot-hat + picar-x)"
log "Install dir: $INSTALL_DIR"
log "Venv dir:    $VENV_DIR"
log "Refs:        robot-hat=$ROBOT_HAT_REF, picar-x=$PICARX_REF"

need_cmd git
need_cmd "$PYTHON_BIN"

if ! is_pi; then
  warn "This does not look like a Raspberry Pi. Continuing anyway."
fi

# -------------------------
# System deps
# -------------------------
log "Installing system packages (requires sudo)"
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  python3-pip \
  python3-dev \
  python3-setuptools \
  python3-smbus \
  git i2c-tools \
  libatlas-base-dev \
  build-essential

# Optional: enable I2C tools visibility check
if [[ -e /dev/i2c-1 ]]; then
  log "I2C device present: /dev/i2c-1"
else
  warn "/dev/i2c-1 not found. I2C may be disabled."
  warn "Enable it with: sudo raspi-config  -> Interface Options -> I2C -> Enable"
fi

# -------------------------
# Create venv
# -------------------------
log "Creating Python virtual environment"
if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

log "Upgrading pip tooling in venv"
python -m pip install --upgrade pip setuptools wheel

# -------------------------
# Clone repos (pinned)
# -------------------------
mkdir -p "$INSTALL_DIR"
ROBOT_HAT_DIR="$INSTALL_DIR/robot-hat"
VILIB_DIR="$INSTALL_DIR/vilib"
PICARX_DIR="$INSTALL_DIR/picar-x"

sync_repo "$ROBOT_HAT_REPO" "$ROBOT_HAT_DIR" "$ROBOT_HAT_REF"
sync_repo "$PICARX_REPO"     "$PICARX_DIR"     "$PICARX_REF"

if [[ -d "$VILIB_DIR/.git" ]]; then
  log "Updating existing repo: $VILIB_DIR"
  git -C "$VILIB_DIR" fetch origin "$VILIB_REF"
  git -C "$VILIB_DIR" checkout "$VILIB_REF"
else
  log "Cloning vilib (branch=$VILIB_REF, depth=1)"
  git clone -b "$VILIB_REF" --depth 1 "$VILIB_REPO" "$VILIB_DIR"
fi

log "Pinned vilib at: $(git -C "$VILIB_DIR" rev-parse --short HEAD)"

# -------------------------
# Install (editable) into venv
# -------------------------
log "Installing robot-hat into venv (editable)"
python -m pip install -e "$ROBOT_HAT_DIR"

log "Installing vilib into venv (editable)"
python -m pip install -e "$VILIB_DIR"

log "Installing picar-x into venv (editable)"
python -m pip install -e "$PICARX_DIR"

# -------------------------
# Sanity checks
# -------------------------
log "Sanity check: Python imports"
python - <<'PY'
import sys
print("Python:", sys.version)

# robot-hat import check (module name varies by version; try a couple)
ok_hat = False
for name in ("robot_hat", "robot_hat.core", "robot_hat.utils"):
    try:
        __import__(name)
        print(f"OK: import {name}")
        ok_hat = True
        break
    except Exception as e:
        last = e
if not ok_hat:
    print("WARN: Could not import robot_hat modules cleanly. (May still work on target)")
    print("      Last error:", repr(last))

# vilib import check
try:
    import vilib
    print("OK: import vilib")
except Exception as e:
    print("WARN: import vilib failed (camera stack may not be ready yet)")
    print("      Error:", repr(e))

# picar-x import check
try:
    from picarx import Picarx
    print("OK: from picarx import Picarx")
except Exception as e:
    print("FAIL: from picarx import Picarx")
    raise
PY

log "Sanity check: I2C probe (won't fail the install)"
if command -v i2cdetect >/dev/null 2>&1; then
  sudo i2cdetect -y 1 || true
fi

# -------------------------
# Wrap up
# -------------------------
cat <<EOF

============================================================
DONE.

To use the PiCar-X Python environment later:

  source "$VENV_DIR/bin/activate"

Repos were installed from:
  $ROBOT_HAT_DIR  (ref: $ROBOT_HAT_REF)
  $PICARX_DIR     (ref: $PICARX_REF)

If I2C was missing:
  sudo raspi-config -> Interface Options -> I2C -> Enable -> Reboot
============================================================

EOF
