"""
Exercise 1.1: Your First AI Weather Forecast
=============================================
Level: Beginner | Time: 15 min | GPU: Optional (runs on CPU, faster on GPU)

GOAL: Run a global 5-day weather forecast using FourCastNet3 in under 10 lines
      of Python. Understand the model-data-io-run pattern that Earth2Studio uses
      for ALL models.

CONCEPTS:
  - Earth2Studio's 4-component pattern: Model + Data + IO + Run
  - Prognostic models (px): models that step forward in time autoregressively
  - GFS data source: pulls real-time NOAA Global Forecast System initial conditions
  - ZarrBackend: saves forecast output in chunked, cloud-friendly Zarr format

WHAT YOU'LL LEARN:
  1. How Earth2Studio auto-downloads pretrained model weights
  2. How forecast time steps work (each step = 6 hours for most models)
  3. What variables FCN3 predicts (73 atmospheric channels)
"""

import sys
import os

# Add templates to path for shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'templates'))

# ===========================================================================
# PART 1: The Minimal Forecast (do this first)
# ===========================================================================

from earth2studio.models.px import FCN3
from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio import run

# Step 1: Load the pretrained FourCastNet3 model
# This auto-downloads weights from NGC/HuggingFace on first run (~500MB)
print("Loading FCN3 model...")
package = FCN3.load_default_package()
model = FCN3.load_model(package)

# Step 2: Set up GFS as the initial condition source
# GFS provides real-time atmospheric analysis from NOAA (updated every 6 hours)
data = GFS()

# Step 3: Configure output storage
# Zarr is a chunked array format — great for large gridded data
output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'ex01_forecast.zarr')
io = ZarrBackend(output_dir)

# Step 4: Run a deterministic forecast
# - time: initialization date (model reads real atmosphere from this date)
# - nsteps: number of 6-hour steps (20 steps = 5 days)
print("Running 5-day forecast (20 steps × 6hr = 120hr)...")
io = run.deterministic(
    time=["2025-06-01T00:00:00"],  # Try changing this to a recent date!
    nsteps=20,
    model=model,
    data=data,
    io=io
)

print(f"\nForecast complete!")
print(f"Output saved to: {output_dir}")

# ===========================================================================
# PART 2: Explore the output (do this after Part 1 succeeds)
# ===========================================================================

import xarray as xr

ds = xr.open_zarr(output_dir)

print("\n" + "=" * 60)
print("FORECAST OUTPUT EXPLORATION")
print("=" * 60)

# What variables did the model predict?
print(f"\nVariables ({len(ds.coords['variable'])} total):")
for v in sorted(ds.coords['variable'].values):
    print(f"  - {v}")

# What's the spatial resolution?
print(f"\nSpatial grid:")
print(f"  Latitude:  {len(ds.coords['lat'])} points ({float(ds.coords['lat'].min()):.1f}° to {float(ds.coords['lat'].max()):.1f}°)")
print(f"  Longitude: {len(ds.coords['lon'])} points ({float(ds.coords['lon'].min()):.1f}° to {float(ds.coords['lon'].max()):.1f}°)")

# What's the temporal resolution?
print(f"\nTime steps: {len(ds.coords['lead_time'])} steps")
print(f"  Lead times: {ds.coords['lead_time'].values}")

# ===========================================================================
# EXERCISES (try these modifications)
# ===========================================================================
"""
✅ Exercise 1.1a: Change the initialization date to today. Does the forecast
   change? (It should — different initial conditions = different forecast.)

✅ Exercise 1.1b: Increase nsteps to 40 (10-day forecast). How much longer
   does it take? Note: each step is independent computation, so time scales
   linearly.

✅ Exercise 1.1c: Look at the variable list. Can you identify:
   - Surface variables (2m temperature, 10m wind, sea level pressure)?
   - Upper-air variables (temperature, wind, geopotential at pressure levels)?
   - What pressure levels are included?

✅ Exercise 1.1d: What is the spatial resolution in kilometers? (Hint: 0.25°
   at the equator ≈ 28 km. How does this compare to your local TV weather
   forecast resolution?)

🔥 Challenge: Read the Earth2Studio docs and try loading a different model
   (e.g., DLWP or Pangu) with the same data/io/run pattern. Does the code
   structure change?
"""
