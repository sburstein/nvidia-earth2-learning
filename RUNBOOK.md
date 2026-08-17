# Runbook: What Runs Where

Honest capability map for this machine (Apple M2, 8 GB RAM, no NVIDIA GPU)
and exact steps for the cloud path when you outgrow it.

## What runs locally, today

| Task | Status | Command |
|------|--------|---------|
| DLWP 5-day global forecast + verification | Works, ~15s warm, ~1 min cold | `python exercises/level1_beginner/ex00_local_dlwp_forecast.py` |
| GFS / ARCO ERA5 data pulls (no account) | Works | any `earth2studio.data` source that uses S3/GCS |
| Plotting, verification, zarr wrangling | Works | see `templates/earth2_utils.py` |

Measured on this Mac (2026-08-16, init 2026-08-08 00Z, 20 x 6h steps):
model load 1.9s, full 5-day global rollout 10.1s on CPU, verification fetch
1.4s (warm cache). First run adds a one-time 67 MB weight download from NGC
and GFS grib byte-range fetches, roughly one extra minute on this connection.

Known DLWP behavior you will see in the outputs: z500 (the large-scale flow)
beats persistence at every lead. t850 and t2m develop a warm drift over land
by day 3 to 5. That is a real, documented limitation of this small 2017-era
architecture, not a bug in the pipeline.

## What does NOT run locally, and why

| Model | Blocker on this Mac |
|-------|---------------------|
| FCN3 / SFNO | Needs `makani` + `torch-harmonics` CUDA path; weights and activations far exceed 8 GB RAM at 0.25 degree (721x1440x73 channels) |
| Pangu, FengWu, FuXi | ONNX models need `onnxruntime-gpu` per earth2studio extras; CPU ONNX at 0.25 degree needs more RAM than 8 GB |
| GraphCast | JAX with `jax[cuda12]` pin; CPU inference at 0.25 degree also RAM-bound |
| AIFS | Requires `flash-attn` (CUDA only) |
| CorrDiff, StormCast, ensemble (Level 2+) | CUDA-only kernels (cupy, cucim) |

## Cloud path A: Google Colab (free, fastest to results)

1. Open https://colab.research.google.com and create a new notebook.
2. Runtime > Change runtime type > T4 GPU (free tier). For FCN3 pick A100
   (Colab Pro) or use SFNO on T4; FCN3 weights want more than 16 GB VRAM.
3. First cell:
   ```
   !pip install "earth2studio[sfno]"   # or [fcn3] on an A100
   ```
   Restart the runtime when pip finishes (Colab caches old numpy).
4. Second cell, same pattern as ex00 with the model swapped:
   ```python
   from earth2studio.models.px import SFNO
   from earth2studio.data import GFS
   from earth2studio.io import ZarrBackend
   from earth2studio import run

   package = SFNO.load_default_package()
   model = SFNO.load_model(package)
   io = run.deterministic(
       time=["2026-08-08"], nsteps=20,
       prognostic=model, data=GFS(), io=ZarrBackend("forecast.zarr"),
   )
   ```
   Note the keyword is `prognostic=`, not `model=` (earth2studio >= 0.13).
5. Weight downloads from NGC are anonymous for public models. No account
   needed. GFS pulls from NOAA's open S3 bucket, also no account.
6. Copy the verification cells from `ex00_local_dlwp_forecast.py`
   unchanged; they are model-agnostic.

Expected T4 timings: SFNO ~2 min for weights, ~10 s per 6h step.

## Cloud path B: NVIDIA build.nvidia.com API (no GPU anywhere)

1. Sign up free at https://build.nvidia.com (email only, gives ~1000 credits).
2. Generate an API key from any Earth-2 model card (e.g. FourCastNet).
3. `export NGC_API_KEY=...` and call the hosted NIM endpoint over HTTPS.
   This runs inference on NVIDIA's GPUs and returns arrays to this Mac.
4. Good for CorrDiff downscaling demos (Level 2.2), which have a hosted NIM.

## Cloud path C: rented GPU (Lambda, RunPod) for Level 2 and 3

1. Rent a single A100 80GB (about 1 to 2 USD per hour).
2. `pip install "earth2studio[fcn3]"` inside the provider's PyTorch image.
3. Everything in `exercises/level2_intermediate/` runs as written.
4. Level 3 training exercises want multiple A100s; treat those as read-along
   material unless you have a real budget.

## ERA5 without an account

The `ARCO` data source in earth2studio reads Google's public
`gs://gcp-public-data-arco-era5` zarr anonymously and works from this Mac.
Use it instead of `CDS` (which requires a free Copernicus signup and an API
token in `~/.cdsapirc`) whenever you need historical initial conditions.
