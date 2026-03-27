#!/bin/bash
# =============================================================================
# NVIDIA Earth-2 Learning Environment Setup
# =============================================================================
# This script sets up everything you need to work through the exercises.
# Run: chmod +x setup.sh && ./setup.sh
# =============================================================================

set -e

echo "========================================"
echo "  NVIDIA Earth-2 Learning Environment"
echo "========================================"
echo ""

# --- Step 1: Check for Homebrew ---
if ! command -v brew &> /dev/null; then
    echo "[1/5] Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add brew to PATH for Apple Silicon
    if [ -f /opt/homebrew/bin/brew ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
    fi
else
    echo "[1/5] Homebrew already installed ✓"
fi

# --- Step 2: Install Python 3.12 ---
if ! python3.12 --version &> /dev/null 2>&1; then
    echo "[2/5] Installing Python 3.12 via Homebrew..."
    brew install python@3.12
else
    echo "[2/5] Python 3.12 already installed ✓"
fi

# Determine correct python path
PYTHON=$(command -v python3.12 || command -v python3)
echo "    Using: $PYTHON ($($PYTHON --version))"

# --- Step 3: Create virtual environment ---
VENV_DIR="$(cd "$(dirname "$0")" && pwd)/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "[3/5] Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
else
    echo "[3/5] Virtual environment already exists ✓"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
echo "    Activated: $VENV_DIR"

# --- Step 4: Install core dependencies ---
echo "[4/5] Installing dependencies..."

pip install --upgrade pip setuptools wheel

# Core inference library
# Note: [all] extras require CUDA/Linux. On macOS, install base package only.
pip install earth2studio

# Visualization & data handling
pip install matplotlib cartopy xarray zarr netcdf4 cfgrib

# Jupyter for notebooks
pip install jupyterlab ipywidgets

# Evaluation tools
pip install scipy scikit-learn pandas

# Spherical harmonics library
pip install torch-harmonics

# Progress bars
pip install tqdm

echo ""

# --- Step 5: Verify installation ---
echo "[5/5] Verifying installation..."

$PYTHON -c "
import earth2studio
print(f'  earth2studio: {earth2studio.__version__}')
" 2>/dev/null && echo "  earth2studio ✓" || echo "  earth2studio: install may need GPU — see notes below"

$PYTHON -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
else:
    print('  (CPU-only mode — Level 1 exercises will work, Level 2+ need GPU)')
"

$PYTHON -c "import matplotlib; print('  matplotlib ✓')"
$PYTHON -c "import xarray; print('  xarray ✓')"
$PYTHON -c "import torch_harmonics; print('  torch-harmonics ✓')" 2>/dev/null || echo "  torch-harmonics: optional, needed for Level 3"

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "To activate this environment in future sessions:"
echo "  source $(pwd)/.venv/bin/activate"
echo ""
echo "To start Jupyter:"
echo "  jupyter lab --notebook-dir=notebooks/"
echo ""
echo "Start learning:"
echo "  python exercises/level1_beginner/ex01_first_forecast.py"
echo ""
echo "NOTE: If you don't have an NVIDIA GPU, Level 1 exercises"
echo "work on CPU (slower). Level 2+ exercises require a GPU."
echo "Consider using:"
echo "  - Google Colab (free T4 GPU)"
echo "  - Lambda Labs / RunPod (rent A100s)"
echo "  - NVIDIA build.nvidia.com API (cloud, no GPU needed)"
echo "========================================"
