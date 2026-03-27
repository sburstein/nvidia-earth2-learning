"""
Exercise 3.1: Train a Mini CorrDiff on Custom Data
====================================================
Level: Advanced | Time: 2-4 hours | GPU: A100 or better (multi-GPU recommended)

GOAL: Train your own CorrDiff downscaling model using PhysicsNeMo. This teaches
      you the full training pipeline for generative diffusion weather models.

CONCEPTS:
  - PhysicsNeMo: NVIDIA's training framework (successor to Modulus)
  - Two-stage training: regression UNet first, then diffusion model
  - HRRR: NOAA's High-Resolution Rapid Refresh (3km over CONUS) — the target
  - GEFS: Global Ensemble Forecast System (~25km) — the input
  - Data curation: the hardest part of training any weather AI model

PREREQUISITES:
  - PhysicsNeMo installed: pip install "nvidia-physicsnemo[cu12,nn-extras]"
  - NVIDIA A100 or better (multi-GPU recommended for reasonable training time)
  - ~100GB storage for training data
  - Clone: git clone https://github.com/NVIDIA/physicsnemo.git

ESTIMATED RESOURCES:
  - Mini config: ~50 GPU-hours on A100 for usable results
  - Full config: ~500 GPU-hours on A100 for publication quality
"""

import os

# ===========================================================================
# STEP 0: Environment Setup (run in terminal)
# ===========================================================================
SETUP_COMMANDS = """
# ============================================================
# Run these commands in your terminal BEFORE this script
# ============================================================

# 1. Clone PhysicsNeMo
git clone https://github.com/NVIDIA/physicsnemo.git
cd physicsnemo

# 2. Install with weather extras
pip install -e ".[weather]"

# 3. Navigate to CorrDiff examples
cd examples/weather/corrdiff

# 4. Check GPU availability
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
"""

print(SETUP_COMMANDS)


# ===========================================================================
# STEP 1: Understanding the data format
# ===========================================================================

DATA_FORMAT_EXPLANATION = """
CorrDiff Training Data Format
==============================

CorrDiff needs PAIRED samples of:
  - Coarse input (e.g., GEFS at 25km)
  - Fine target (e.g., HRRR at 3km)

For the SAME time step, you need both resolutions.

Directory structure:
  data/
  ├── coarse/          # GEFS fields, regridded to training grid
  │   ├── 2020010100.npy
  │   ├── 2020010106.npy
  │   └── ...
  ├── fine/            # HRRR fields at native resolution
  │   ├── 2020010100.npy
  │   ├── 2020010106.npy
  │   └── ...
  └── stats/           # Normalization statistics
      ├── coarse_mean.npy
      ├── coarse_std.npy
      ├── fine_mean.npy
      └── fine_std.npy

Minimum recommended: 50,000+ paired samples
For testing/learning: Use the HRRR-Mini config (~1000 samples)
"""

print(DATA_FORMAT_EXPLANATION)


# ===========================================================================
# STEP 2: Training Configuration
# ===========================================================================

TRAINING_CONFIG = """
# File: conf/config_training_hrrr_mini_regression.yaml
# This is the mini config for learning — modify paths to your data

defaults:
  - base/model/regression
  - base/dataset/hrrr_mini
  - base/training/regression

training:
  hp:
    total_batch_size: 64        # Reduce if OOM
    training_duration: 500000   # Training steps
    lr: 2e-4
    weight_decay: 0.0

  io:
    save_checkpoint_path: ./checkpoints/regression/
    print_progress_freq: 100

dataset:
  data_path: /path/to/your/data/  # CHANGE THIS
  n_train: 45000                   # Number of training samples
  n_valid: 5000                    # Number of validation samples
"""

print(TRAINING_CONFIG)


# ===========================================================================
# STEP 3: Training pipeline script
# ===========================================================================

def print_training_pipeline():
    """Print the complete training pipeline with commands."""

    pipeline = """
    ============================================================
    COMPLETE CORRDIFF TRAINING PIPELINE
    ============================================================

    STAGE 1: Train Regression Model (~30% of total compute)
    --------------------------------------------------------
    This model learns to predict the conditional MEAN of the
    fine-resolution output. It's a deterministic baseline.

    cd physicsnemo/examples/weather/corrdiff

    # Single GPU
    python train.py --config-name=config_training_hrrr_mini_regression.yaml

    # Multi-GPU (recommended)
    torchrun --nproc_per_node=4 train.py \\
        --config-name=config_training_hrrr_mini_regression.yaml

    # Monitor with MLflow
    mlflow ui -p 2458  # Open http://localhost:2458


    STAGE 2: Train Diffusion Model (~70% of total compute)
    --------------------------------------------------------
    This model learns to predict the RESIDUAL (correction)
    between the regression output and the actual fine field.
    It generates diverse, realistic fine-scale details.

    python train.py \\
        --config-name=config_training_hrrr_mini_diffusion.yaml \\
        ++training.io.regression_checkpoint_path=/path/to/regression/best.pth


    STAGE 3: Generate Predictions
    --------------------------------------------------------
    Combine both models to produce downscaled output.

    python generate.py \\
        --config-name="config_generate_hrrr_mini.yaml" \\
        ++generation.io.res_ckpt_filename=/path/to/diffusion/best.pth \\
        ++generation.io.reg_ckpt_filename=/path/to/regression/best.pth \\
        ++generation.num_samples=10

    Output will be in outputs/corrdiff_generate/


    STAGE 4: Evaluate
    --------------------------------------------------------
    Compare against held-out HRRR verification data.

    python evaluate.py \\
        --predictions_path=outputs/corrdiff_generate/ \\
        --verification_path=/path/to/hrrr/verification/
    """

    print(pipeline)

print_training_pipeline()


# ===========================================================================
# STEP 4: Monitoring training (Python helper)
# ===========================================================================

def create_training_monitor():
    """Create a training monitoring script."""

    monitor_code = '''
#!/usr/bin/env python3
"""Monitor CorrDiff training progress."""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_training_curves(log_dir, save_path="training_curves.png"):
    """Plot loss curves from PhysicsNeMo training logs."""

    # PhysicsNeMo logs to MLflow or console
    # This reads the MLflow metrics directory
    metrics_dir = Path(log_dir) / "mlruns"

    if not metrics_dir.exists():
        print(f"No MLflow logs found at {metrics_dir}")
        print("Looking for console logs instead...")
        return

    # Find all metric files
    train_loss = []
    val_loss = []
    steps = []

    for run_dir in metrics_dir.glob("*/*/metrics/train_loss"):
        with open(run_dir) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    train_loss.append(float(parts[1]))
                    steps.append(int(parts[2]))

    if not train_loss:
        print("No training data found yet. Training may still be starting.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(steps, train_loss, 'b-', alpha=0.5, label='Train Loss')

    # Smoothed curve
    window = min(50, len(train_loss) // 10)
    if window > 1:
        smoothed = np.convolve(train_loss, np.ones(window)/window, mode='valid')
        ax.semilogy(steps[window-1:], smoothed, 'r-', linewidth=2,
                    label=f'Smoothed (window={window})')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('CorrDiff Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()

if __name__ == "__main__":
    import sys
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    plot_training_curves(log_dir)
'''

    monitor_path = os.path.join(os.path.dirname(__file__), 'training_monitor.py')
    with open(monitor_path, 'w') as f:
        f.write(monitor_code)
    print(f"Created: {monitor_path}")
    print("Usage: python training_monitor.py /path/to/training/dir")

create_training_monitor()


# ===========================================================================
# EXERCISES
# ===========================================================================
"""
✅ Exercise 3.1a: Clone PhysicsNeMo and explore the CorrDiff example
   directory structure. Read the README and config files to understand
   the training hyperparameters.

✅ Exercise 3.1b: Download the HRRR-Mini dataset (see PhysicsNeMo docs).
   This is a small subset designed for testing the training pipeline
   without needing 100GB+ of data.

✅ Exercise 3.1c: Train the regression model for 10,000 steps (not full
   convergence). Watch the loss curve — when does it plateau?

✅ Exercise 3.1d: After regression training, freeze the checkpoint and
   train the diffusion model for 10,000 steps. Compare the regression-only
   output vs regression+diffusion. Where does diffusion add the most value?

🔥 Challenge: Adapt CorrDiff for a custom domain (not CONUS). You'll need
   to create your own paired coarse-fine training dataset. Consider using
   ERA5 (coarse) → WRF (fine) or GFS → any regional model for your domain.
"""
