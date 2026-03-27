"""
Exercise 3.2: Fine-Tune FCN3 on a Regional Domain
===================================================
Level: Advanced | Time: 4-8 hours | GPU: Multi-GPU (4+ A100s recommended)

GOAL: Start from the pretrained FCN3 checkpoint and fine-tune on ERA5 data
      for a specific region or application. Transfer learning lets you
      specialize a global model for your use case with much less compute
      than training from scratch.

CONCEPTS:
  - Transfer learning: start from pretrained weights, adapt to new task
  - PhysicsNeMo unified recipe: one training script for SFNO/AFNO/FCN3
  - ERA5 data pipeline: downloading, curating, and normalizing reanalysis data
  - Multi-GPU training with mpirun/torchrun

USE CASES FOR FINE-TUNING:
  - Regional specialization (e.g., monsoon prediction over South Asia)
  - Variable addition (add new output channels the base model doesn't predict)
  - Temporal resolution (adapt from 6hr to 1hr steps)
  - Bias correction (train on operational analyses rather than reanalysis)
"""

import os

# ===========================================================================
# STEP 1: Environment Setup
# ===========================================================================

SETUP = """
============================================================
ENVIRONMENT SETUP
============================================================

# Clone PhysicsNeMo
git clone https://github.com/NVIDIA/physicsnemo.git
cd physicsnemo/examples/weather/unified_recipe

# Install dependencies
pip install -r requirements.txt

# Check GPU setup
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'GPUs available: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    print(f'  Memory: {torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB')
"
"""

print(SETUP)


# ===========================================================================
# STEP 2: Download ERA5 Training Data
# ===========================================================================

ERA5_DOWNLOAD = """
============================================================
ERA5 DATA DOWNLOAD
============================================================

Option A: ARCO ERA5 (recommended — fast, no API key needed)
-----------------------------------------------------------
ARCO provides ERA5 in cloud-optimized zarr format on Google Cloud.
No CDS API registration needed. Much faster for bulk downloads.

python download_era5_arco.py \\
    --years 2018 2019 2020 2021 \\
    --output_dir ./data/era5/ \\
    --variables t2m u10m v10m msl z500 t500 u500 v500 z850 t850 u850 v850

Option B: CDS API (official ECMWF source — slower, requires account)
--------------------------------------------------------------------
1. Register at https://cds.climate.copernicus.eu/
2. Install: pip install cdsapi
3. Place API key in ~/.cdsapirc

python download_era5_cds.py \\
    --years 2018 2019 2020 2021 \\
    --output_dir ./data/era5/

Note: CDS can be very slow (hours to days for large requests).
      ARCO is 10-100x faster for training data downloads.

Data size: ~50-100GB for a multi-year training dataset at 0.25°
"""

print(ERA5_DOWNLOAD)


# ===========================================================================
# STEP 3: Curate Training Data
# ===========================================================================

CURATE = """
============================================================
DATA CURATION
============================================================

The unified recipe expects data in zarr format with specific structure:

python curate_era5.py \\
    --input_dir ./data/era5/ \\
    --output_dir ./data/era5_curated/ \\
    --train_years 2018 2019 2020 \\
    --val_years 2021

This script:
1. Reads raw ERA5 data
2. Selects the variables your model needs
3. Computes per-variable normalization statistics
4. Writes training/validation splits in zarr format
5. Stores metadata for the training pipeline

Output structure:
  data/era5_curated/
  ├── train.zarr          # (time, channel, lat, lon)
  ├── valid.zarr
  ├── stats/
  │   ├── global_means.npy
  │   ├── global_stds.npy
  │   └── time_means.npy  # Climatological means
  └── metadata.json       # Variable names, pressure levels, grid info
"""

print(CURATE)


# ===========================================================================
# STEP 4: Fine-Tuning Configuration
# ===========================================================================

FINETUNE_CONFIG = """
============================================================
FINE-TUNING CONFIGURATION
============================================================

# Modify conf/model.yaml to select your architecture:

model:
  name: sfno            # Options: sfno, afno, graphcast
  pretrained: true       # Load pretrained checkpoint
  checkpoint_path: /path/to/pretrained/fcn3_checkpoint.pth

  # Freeze early layers (optional — reduces compute, keeps learned features)
  freeze_encoder: false  # Set true to freeze encoder, only train decoder

  # Architecture params (keep defaults unless you know what you're doing)
  embed_dim: 384
  num_layers: 8
  num_heads: 8

training:
  epochs: 50             # Fine-tuning needs fewer epochs than training from scratch
  lr: 1e-5               # Lower LR for fine-tuning (1/10 to 1/100 of scratch)
  weight_decay: 1e-5
  warmup_steps: 1000
  batch_size: 4          # Per-GPU batch size

  # Loss function
  loss: mse              # Options: mse, weighted_mse, spectral

  # Optional: add spectral loss to preserve small-scale features
  spectral_loss_weight: 0.1  # Penalizes spectral energy loss

data:
  train_path: ./data/era5_curated/train.zarr
  valid_path: ./data/era5_curated/valid.zarr
  stats_path: ./data/era5_curated/stats/
"""

print(FINETUNE_CONFIG)


# ===========================================================================
# STEP 5: Training Commands
# ===========================================================================

TRAINING_COMMANDS = """
============================================================
FINE-TUNING COMMANDS
============================================================

# Single GPU (testing)
python train.py

# Multi-GPU (recommended for real fine-tuning)
mpirun -np 4 python train.py

# With torchrun (alternative to mpirun)
torchrun --nproc_per_node=4 train.py

# Multi-node (cluster)
mpirun -np 32 --hostfile hosts.txt python train.py

# Monitor training
mlflow ui -p 2458
# Open http://localhost:2458 in browser

============================================================
EXPECTED TRAINING TIMELINE (4x A100)
============================================================
- Epochs 1-5:   Loss drops rapidly (learning new domain features)
- Epochs 5-20:  Gradual improvement (refinement)
- Epochs 20-50: Diminishing returns (convergence)
- Validation loss should decrease monotonically; if it increases,
  you're overfitting → reduce epochs or increase regularization.

Total time: ~4-8 hours on 4x A100 for 50 epochs on 3 years of ERA5
"""

print(TRAINING_COMMANDS)


# ===========================================================================
# STEP 6: Evaluation Script
# ===========================================================================

def create_evaluation_script():
    """Generate a script to evaluate your fine-tuned model."""

    eval_code = '''#!/usr/bin/env python3
"""
Evaluate a fine-tuned weather model against ERA5 verification.

Usage:
    python evaluate_finetune.py --checkpoint /path/to/best.pth
"""

import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from earth2studio.models.px import FCN3
from earth2studio.data import ARCO
from earth2studio.io import ZarrBackend
from earth2studio import run

def evaluate_model(checkpoint_path, init_dates, nsteps=40):
    """Run forecasts and compare to ERA5 verification."""

    # Load fine-tuned model
    # Note: loading custom checkpoints requires modifying Earth2Studio's
    # model loading — see PhysicsNeMo docs for export-to-earth2studio
    print(f"Loading checkpoint: {checkpoint_path}")

    # For standard evaluation, use the default model
    package = FCN3.load_default_package()
    model = FCN3.load_model(package)

    data = ARCO()  # ERA5 for both init conditions and verification

    results = {}
    for date in init_dates:
        print(f"\\nForecasting from {date}...")
        io = run.deterministic(
            time=[date],
            nsteps=nsteps,
            model=model,
            data=data,
            io=ZarrBackend(f"eval_{date[:10]}.zarr")
        )

        ds = xr.open_zarr(f"eval_{date[:10]}.zarr")

        # Compute RMSE against ERA5 at each lead time
        # (In practice, you'd fetch ERA5 verification data separately)
        results[date] = ds

    return results


def compute_scorecard(results, variables=["t2m", "z500", "u10m"]):
    """Compute standard verification metrics."""

    print("\\n" + "=" * 60)
    print("VERIFICATION SCORECARD")
    print("=" * 60)

    # Standard metrics:
    # - RMSE: Root Mean Square Error
    # - ACC: Anomaly Correlation Coefficient
    # - Bias: Mean systematic error
    # - CRPS: Continuous Ranked Probability Score (for ensembles)

    print("\\nNote: Full verification requires ERA5 truth data at each")
    print("forecast lead time. Use WeatherBench2 for standardized evaluation:")
    print("  pip install weatherbench2")
    print("  https://weatherbench2.readthedocs.io/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    init_dates = [
        "2024-01-15T00:00:00",
        "2024-04-15T00:00:00",
        "2024-07-15T00:00:00",
        "2024-10-15T00:00:00",
    ]

    results = evaluate_model(args.checkpoint, init_dates)
    compute_scorecard(results)
'''

    eval_path = os.path.join(os.path.dirname(__file__), 'evaluate_finetune.py')
    with open(eval_path, 'w') as f:
        f.write(eval_code)
    print(f"\nCreated: {eval_path}")

create_evaluation_script()


# ===========================================================================
# EXERCISES
# ===========================================================================
"""
✅ Exercise 3.2a: Download ERA5 data for 2018-2021 using ARCO. Start
   with just 10 variables to keep data size manageable (~20GB).

✅ Exercise 3.2b: Run the unified recipe training script for 5 epochs
   to verify your data pipeline works end-to-end. Don't worry about
   accuracy yet — just make sure training runs without errors.

✅ Exercise 3.2c: Compare fine-tuned vs base model RMSE for a test case.
   Does fine-tuning help? (It should, especially for the region/season
   your training data focuses on.)

✅ Exercise 3.2d: Experiment with freezing the encoder vs training all
   layers. Freezing the encoder trains faster but limits how much the
   model can adapt. What's the sweet spot?

🔥 Challenge: Fine-tune specifically for extreme weather events. Create a
   training dataset weighted toward hurricane, heat wave, and cold wave
   cases. Evaluate whether the fine-tuned model performs better for these
   high-impact events.
"""
