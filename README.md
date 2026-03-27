# NVIDIA Earth-2 Climate AI — Learning Workspace

Hands-on exercises for learning NVIDIA's Earth-2 climate and weather AI ecosystem.

## Quick Start

```bash
# Activate the virtual environment
source .venv/bin/activate

# Verify installation
python -c "import earth2studio; print(earth2studio.__version__)"

# Run your first forecast
python exercises/level1_beginner/ex01_first_forecast.py
```

## Learning Roadmap

### Level 1: Beginner (No GPU, 1-2 hours)

Start here. These exercises use Earth2Studio's inference API to run pretrained
models. Everything works on CPU (just slower than GPU).

| Exercise | File | What You'll Learn |
|----------|------|-------------------|
| 1.1 First Forecast | `ex01_first_forecast.py` | Model-Data-IO-Run pattern, FCN3, GFS data |
| 1.2 Visualization | `ex02_visualize_forecast.py` | xarray, matplotlib, weather maps, regional plots |
| 1.3 Model Comparison | `ex03_compare_models.py` | FCN3 vs DLWP, RMSE, error growth, spectral analysis |

### Level 2: Intermediate (GPU required, 3-5 hours)

Ensemble forecasting, downscaling, custom diagnostics, and seasonal prediction.

| Exercise | File | What You'll Learn |
|----------|------|-------------------|
| 2.1 Ensemble Forecast | `ex04_ensemble_forecast.py` | 50-member ensemble, spread, probability maps, spaghetti plots |
| 2.2 CorrDiff Downscaling | `ex05_corrdiff_downscaling.py` | 25km→3km, NIM containers, two-stage architecture |
| 2.3 Custom Diagnostics | `ex06_custom_diagnostic.py` | Wind speed, heat index, daily extremes, extension pattern |
| 2.4 DLESyM Seasonal | `ex07_dlesym_seasonal.py` | S2S forecasting, coupled atmosphere-ocean, tercile outlooks |

### Level 3: Advanced (Multi-GPU, 10+ hours)

Training, fine-tuning, and building production pipelines.

| Exercise | File | What You'll Learn |
|----------|------|-------------------|
| 3.1 Train CorrDiff | `ex08_train_corrdiff.py` | PhysicsNeMo, regression + diffusion training |
| 3.2 Fine-Tune FCN3 | `ex09_finetune_fcn3.py` | Transfer learning, ERA5 data pipeline, unified recipe |
| 3.3 HENS Pipeline | `ex10_hens_pipeline.py` | 1000+ member ensembles, bred vectors, tail risk |

## Project Structure

```
nvidia-earth2-learning/
├── README.md              ← You are here
├── setup.sh               ← Environment setup script
├── exercises/
│   ├── level1_beginner/   ← Start here (no GPU needed)
│   ├── level2_intermediate/ ← GPU exercises
│   └── level3_advanced/   ← Multi-GPU training
├── templates/
│   ├── earth2_utils.py    ← Reusable utilities (data loading, metrics, plotting)
│   └── cheatsheet.md      ← Product landscape quick reference
├── outputs/               ← Forecast outputs (zarr, png)
├── data/                  ← Training data (ERA5, etc.)
└── notebooks/             ← Jupyter notebooks
```

## Key Resources

| Resource | URL |
|----------|-----|
| Earth2Studio Docs | https://nvidia.github.io/earth2studio/ |
| PhysicsNeMo Docs | https://docs.nvidia.com/physicsnemo/latest/ |
| Earth2Studio GitHub | https://github.com/NVIDIA/earth2studio |
| PhysicsNeMo GitHub | https://github.com/NVIDIA/physicsnemo |
| NVIDIA Earth-2 Blog | https://blogs.nvidia.com/blog/nvidia-earth-2-open-models/ |
| HuggingFace Models | https://huggingface.co/nvidia |
| Comprehensive Guide | ~/agent-knowledge/nvidia-earth2-climate-ai.md |

## Important Notes

- **macOS users**: Use `pip install earth2studio` (not `[all]`). CUDA extras are Linux-only.
- **No GPU?** Level 1 exercises work on CPU. For Level 2+, consider Google Colab (free T4).
- **Models auto-download** weights on first use (~500MB per model). Be patient.
- **Longitude convention**: Earth2Studio uses 0-360°. Convert: `lon_360 = 360 - lon_west`.
