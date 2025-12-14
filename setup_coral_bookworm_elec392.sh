#!/usr/bin/env bash
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
#   Coral USB Accelerator Environment Setup                                   #
#                                                                             #
#   Target OS : Debian 12 (Bookworm)                                          #
#   Hardware  : Raspberry Pi 4B                                               #
#   TPU       : Coral USB Accelerator                                         #
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

Coral USB Accelerator Environment Setup

EOF

# ---------------------- Configuration -----------------------
PYTHON_VERSION="3.9.20"
PROJECT_DIR="${HOME}/dev/coral"
DO_UPGRADE=/

# ---------------------- Helpers ------------------------------
log()  { printf "\n\033[1;32m==>\033[0m %s\n" "$*"; }
warn() { printf "\n\033[1;33m[warn]\033[0m %s\n" "$*"; }
die()  { printf "\n\033[1;31m[error]\033[0m %s\n" "$*"; exit 1; }

append_if_missing() {
  local line="$1"
  local file="$2"
  grep -qxF "$line" "$file" 2>/dev/null || echo "$line" >> "$file"
}

# ---------------------- Sanity checks ------------------------
if [[ "$EUID" -eq 0 ]]; then
  die "Run this script as a regular user, not with sudo."
fi

command -v sudo >/dev/null || die "sudo is required"
command -v curl >/dev/null || die "curl is required"
command -v git  >/dev/null || die "git is required"

# ---------------------- System update ------------------------
log "Updating system packages"
sudo apt update -y
if [[ "$DO_UPGRADE" -eq 1 ]]; then
  sudo apt full-upgrade -y
fi

# ---------------------- Build deps ---------------------------
log "Installing Python build dependencies (for pyenv)"
sudo apt install -y \
  build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev \
  libncursesw5-dev xz-utils tk-dev \
  libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
  ca-certificates

# ---------------------- pyenv -------------------------------
if [[ ! -d "${HOME}/.pyenv" ]]; then
  log "Installing pyenv"
  curl -fsSL https://pyenv.run | bash
else
  log "pyenv already installed"
fi

log "Configuring ~/.bashrc for pyenv"
BASHRC="${HOME}/.bashrc"
append_if_missing 'export PYENV_ROOT="$HOME/.pyenv"' "$BASHRC"
append_if_missing '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' "$BASHRC"
append_if_missing 'eval "$(pyenv init -)"' "$BASHRC"

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# ---------------------- Python -------------------------------
if ! pyenv versions --bare | grep -qx "$PYTHON_VERSION"; then
  log "Installing Python ${PYTHON_VERSION} (this will take a while)"
  pyenv install -v "$PYTHON_VERSION"
else
  log "Python ${PYTHON_VERSION} already installed"
fi

log "Setting Python ${PYTHON_VERSION} as global default"
pyenv global "$PYTHON_VERSION"
pyenv rehash

# ---------------------- pip + pipenv -------------------------
log "Installing pipenv"
python -m pip install --upgrade pip pipenv
append_if_missing 'export PIPENV_VENV_IN_PROJECT=1' "$BASHRC"

# ---------------------- Coral runtime ------------------------
log "Installing libedgetpu (standard USB runtime)"

sudo install -d -m 0755 /etc/apt/keyrings
if [[ ! -f /etc/apt/keyrings/coral.gpg ]]; then
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | sudo gpg --dearmor -o /etc/apt/keyrings/coral.gpg
fi

echo "deb [signed-by=/etc/apt/keyrings/coral.gpg] \
https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
| sudo tee /etc/apt/sources.list.d/coral-edgetpu.list >/dev/null

sudo apt update -y
sudo apt install -y libedgetpu1-std

# ---------------------- Project setup ------------------------
log "Creating ELEC 392 Coral directory at ${PROJECT_DIR}"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

log "Creating Python virtual environment"
python -m venv .venv
source .venv/bin/activate

log "Installing pycoral into virtual environment"
pip install --upgrade pip
pip install numpy==1.26.4
pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0

# ---------------------- TPU test -----------------------------
log "Creating TPU detection test script"

cat > list_tpus.py <<'EOF'
from pycoral.utils.edgetpu import list_edge_tpus

if __name__ == "__main__":
    for tpu in list_edge_tpus():
        print(tpu)
EOF

# ---------------------- Examples -----------------------------
log "Cloning pycoral examples"
git clone --recurse-submodules https://github.com/google-coral/pycoral.git || true
cd pycoral
bash examples/install_requirements.sh
cd ..

log "Cloning ELEC392 Coral Starter Kit"
git clone https://github.com/mpan1/elec392-coral-starter-kit.git || true
#cd coral-startup-kit


# ---------------------- Done --------------------------------
echo
echo "============================================================"
echo " Setup complete!"
echo
echo " Next steps:"
echo "   1) Plug in the Coral USB Accelerator"
echo "   2) Activate environment:"
echo "        source ${PROJECT_DIR}/.venv/bin/activate"
echo "   3) Verify TPU:"
echo "        python ${PROJECT_DIR}/list_tpus.py"
echo "   4) Run example:"
echo "        cd ${PROJECT_DIR}/pycoral"
echo "        python3 examples/classify_image.py \\"
echo "          --model test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite \\"
echo "          --labels test_data/inat_bird_labels.txt \\"
echo "          --input test_data/parrot.jpg"
echo
echo " ELEC 392 Coral environment ready. Script complete."
echo "============================================================"

