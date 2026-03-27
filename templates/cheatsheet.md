# NVIDIA Earth-2 Product Landscape Cheat Sheet

## The Pipeline (Data → Forecast → Detail → Nowcast)

```
Observations → [HealDA] → Global Forecast → [CorrDiff] → Local Detail → [StormScope] → Storm-Scale
  (satellites,    (AI DA)    (Atlas/FCN3)    (diffusion     (2-3 km)     (generative    (0-6 hr)
   stations)                  (0-15 days)    downscaling)                 nowcasting)
```

## Quick Reference: Which Tool When

| I want to...                          | Use this                     | Install                           |
|---------------------------------------|------------------------------|-----------------------------------|
| Run inference (forecasts)             | Earth2Studio                 | `pip install earth2studio`        |
| Train or fine-tune models             | PhysicsNeMo                  | `pip install nvidia-physicsnemo`  |
| Deploy production API                 | NIM Microservices            | `docker pull nvcr.io/nim/nvidia/` |
| Research new architectures            | Makani                       | `git clone NVIDIA/makani`         |
| Differentiable spherical transforms   | torch-harmonics              | `pip install torch-harmonics`     |
| Visualize weather in 3D               | Omniverse Blueprint          | NVIDIA Omniverse                  |

## Models at a Glance

| Model       | Type     | Resolution | Timescale    | Speed vs NWP | Key Feature              |
|-------------|----------|------------|--------------|--------------|--------------------------|
| **Atlas**   | Global   | 0.25°      | 1-15 days    | ~5000x       | Best accuracy (2026)     |
| **FCN3**    | Global   | 0.25°      | 0-60 days    | 8x GenCast   | Fastest large ensembles  |
| **FCNv2**   | Global   | 0.25°      | 0-14 days    | ~3000x       | Stable year-long rollout |
| **DLESyM**  | Global   | ~1°        | 2wk-2yr      | ~100x        | Coupled ocean            |
| **CorrDiff**| Downscale| 25→2-3 km  | Post-process | 22x          | Generative uncertainty   |
| **cBottle** | Climate  | 5-100 km   | Scenarios    | N/A          | Foundation model         |
| **StormScope**| Nowcast| ~1 km      | 0-6 hours    | ~10x         | Beats NWP nowcasting     |
| **HealDA**  | DA       | ~25 km     | N/A          | ~1000x       | Coming late 2026         |

## GitHub Repos

| Repo | URL | Purpose |
|------|-----|---------|
| earth2studio | github.com/NVIDIA/earth2studio | Inference framework |
| physicsnemo | github.com/NVIDIA/physicsnemo | Training framework |
| makani | github.com/NVIDIA/makani | Research training |
| FourCastNet | github.com/NVlabs/FourCastNet | Original FCN code |
| torch-harmonics | github.com/NVIDIA/torch-harmonics | Spherical harmonics |
| earth2mip | github.com/NVIDIA/earth2mip | Legacy intercomparison |

## HuggingFace Models

| Model | HuggingFace ID | Use Case |
|-------|----------------|----------|
| Atlas | `nvidia/atlas-era5` | Medium-range forecast |
| FCN3 | `nvidia/fourcastnet3` | Fast ensemble forecast |
| DLESyM | `nvidia/dlesym-v1-era5` | Seasonal forecast |
| StormScope | `nvidia/stormscope-goes-mrms` | Nowcasting |
| CorrDiff | `nvidia/corrdiff-cmip6-era5` | Climate downscaling |

## Earth2Studio Code Patterns

### Deterministic Forecast
```python
from earth2studio.models.px import FCN3
from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio import run

model = FCN3.load_model(FCN3.load_default_package())
io = run.deterministic(["2025-06-01T00:00:00"], 20, model, GFS(), ZarrBackend("out.zarr"))
```

### Ensemble Forecast
```python
from earth2studio.perturbation import SphericalGaussian

io = run.ensemble(
    time=["2025-06-01T00:00:00"], nsteps=40, nensemble=50,
    model=model, data=GFS(), io=ZarrBackend("ens.zarr"),
    perturbation=SphericalGaussian(noise_amplitude=0.05)
)
```

### Read Output
```python
import xarray as xr
ds = xr.open_zarr("out.zarr")
t2m = ds.sel(variable="t2m").isel(time=0, lead_time=8)  # T+48h
```

## Key Variables

| Variable | Description | Units | Common Use |
|----------|-------------|-------|------------|
| t2m | 2m temperature | K | Surface weather |
| u10m | 10m U-wind | m/s | Surface wind |
| v10m | 10m V-wind | m/s | Surface wind |
| msl | Mean sea level pressure | Pa | Cyclone tracking |
| z500 | 500hPa geopotential | m²/s² | Synoptic patterns |
| t850 | 850hPa temperature | K | Air mass analysis |
| u250 | 250hPa U-wind | m/s | Jet stream |

## Common Unit Conversions

```python
celsius = kelvin - 273.15
fahrenheit = celsius * 9/5 + 32
decameters = geopotential / (9.81 * 10)
knots = meters_per_second * 1.944
lon_360 = 360 - lon_west  # e.g., 74°W → 286°E
```

## Data Sources

| Source | Access | Best For |
|--------|--------|----------|
| GFS | Real-time, free, no API key | Initial conditions |
| ARCO ERA5 | Google Cloud zarr, no API key | Training data (fast) |
| CDS ERA5 | ECMWF API, requires account | Official ERA5 |
| HRRR | AWS open data | CONUS high-res verification |

## NIM Deployment (Production)

```bash
# CorrDiff NIM
docker pull nvcr.io/nim/nvidia/corrdiff:1.1.0
docker run --rm --runtime=nvidia --gpus all --shm-size 4g \
    -p 8000:8000 -e NGC_API_KEY=$NGC_API_KEY \
    nvcr.io/nim/nvidia/corrdiff:1.1.0

# Health check
curl http://localhost:8000/v1/health/ready
```

## Critical Pitfalls

1. **FCN v1 has polar artifacts** → use FCNv2/FCN3 for high-latitude work
2. **Autoregressive drift** → error compounds each step; verify spectral energy
3. **CorrDiff needs 50K+ training samples** → don't try with small datasets
4. **ERA5 normalization is per-variable** → never skip standardization
5. **CorrDiff trained on present climate ≠ future climate** → use CMIP6 variant
6. **`earth2studio[all]` fails on macOS** → use `pip install earth2studio` instead
7. **Models predict in Kelvin** → always convert to °C for visualization
