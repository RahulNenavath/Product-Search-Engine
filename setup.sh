# Creates a clean Conda env + installs pip requirements + registers a Jupyter kernel.

# Use -e and pipefail; avoid -u because some conda hooks reference unset vars.
set -e -o pipefail

# -------- Config --------
ENV_NAME="${ENV_NAME:-product_search_env}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
KERNEL_NAME="${KERNEL_NAME:-Product Search (Py3.12)}"
# Comma-separated list (default: conda-forge). Example: CHANNELS="conda-forge,defaults"
CHANNELS="${CHANNELS:-conda-forge}"

echo "==> Environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
echo "==> Channels: ${CHANNELS}"

# -------- Preconditions --------
if ! command -v conda &>/dev/null; then
  echo "conda not found in PATH. Install Miniconda/Anaconda first."
  exit 1
fi

# Load conda shell functions (required for 'conda activate')
source "$(conda info --base)/etc/profile.d/conda.sh"

# -------- Optional: Accept ToS for defaults channels --------
# Harmless if you only use conda-forge.
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >/dev/null 2>&1 || true

# -------- Configure solver & channels --------
echo "==> Configuring solver and channel priority..."
conda update -n base -y conda conda-libmamba-solver >/dev/null 2>&1 || true
conda config --set solver libmamba
conda config --set channel_priority strict

# Set channels explicitly (avoid slow/ambiguous solves)
conda config --remove-key channels >/dev/null 2>&1 || true
IFS=',' read -ra CHS <<<"$CHANNELS"
for ch in "${CHS[@]}"; do
  conda config --add channels "$(echo "$ch" | xargs)"
done

# -------- Create fresh environment --------
echo "==> Recreating Conda environment: ${ENV_NAME}"
conda env remove -n "${ENV_NAME}" -y >/dev/null 2>&1 || true
conda create -n "${ENV_NAME}" -y "python=${PYTHON_VERSION}" pip

# -------- Activate --------
echo "==> Activating environment..."
conda activate "${ENV_NAME}"

# -------- Install pip packages --------
if [[ -f "requirements.txt" ]]; then
  echo "==> Installing pip packages from requirements.txt..."
  python -m pip install --upgrade pip wheel setuptools
  python -m pip install -r requirements.txt
else
  echo "requirements.txt not found; skipping pip install."
fi

# -------- Register Jupyter kernel --------
echo "==> Registering Jupyter kernel..."
python -m pip install -q ipykernel
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "${KERNEL_NAME}"

echo
echo "Setup complete."
echo "Activate with: conda activate ${ENV_NAME}"