"""
Exercise 1.0: A Real Forecast on THIS Mac (DLWP on CPU)
========================================================
Level: Beginner | Time: ~10 min | GPU: Not required (built for Apple Silicon CPU)

GOAL: Run an actual global AI weather model end to end on this machine:
      pull a real initial condition from NOAA GFS, roll a 5-day forecast
      with the pretrained DLWP model, then verify against what the
      atmosphere actually did.

WHY DLWP AND NOT FCN3?
  This Mac (M2, 8 GB RAM, no NVIDIA GPU) cannot run the 0.25 degree
  heavyweights. FCN3 and SFNO need the makani package (CUDA-oriented) and
  several GB of weights. Pangu needs onnxruntime and more RAM than we have.
  DLWP (Deep Learning Weather Prediction, Weyn et al. 2020, NVIDIA's
  physicsnemo checkpoint) is a genuine published global model on a
  cubed-sphere grid, small enough that a full 5-day rollout takes seconds
  on CPU. Same Earth2Studio Model-Data-IO-Run pattern as the big models,
  so everything you learn here transfers directly to a GPU box or Colab.

WHAT IT PREDICTS (7 channels, 6-hour steps):
  t850, z1000, z700, z500, z300, tcwv, t2m

PIPELINE:
  1. Load pretrained DLWP (auto-downloads ~200 MB from NGC, cached after)
  2. Pull the real GFS analysis for INIT_TIME (and INIT_TIME - 6h; DLWP
     needs two input times) from NOAA's open S3 bucket. No account needed.
  3. Roll NSTEPS x 6h forward, save to outputs/dlwp_forecast.zarr
  4. Pull GFS verifying analyses every 24h and compare:
     maps of z500 and t2m (forecast vs analysis) plus an RMSE growth
     curve with a persistence baseline.

Run:
  source .venv/bin/activate
  python exercises/level1_beginner/ex00_local_dlwp_forecast.py
"""

import os
import shutil
import sys
import time as walltime
from datetime import datetime, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "templates"))
from earth2_utils import rmse, geopotential_to_dam, kelvin_to_celsius

# ===========================================================================
# Configuration
# ===========================================================================

# A past date, so we can check the forecast against reality.
INIT_TIME = datetime(2026, 8, 8, 0)   # forecast start (UTC)
NSTEPS = 20                            # 20 x 6h = 120h = 5 days
VERIFY_EVERY_H = 24                    # verify once per day

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_ZARR = os.path.join(ROOT, "outputs", "dlwp_forecast.zarr")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ===========================================================================
# Part 1: Run the forecast
# ===========================================================================

from earth2studio.models.px import DLWP
from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio import run

print("=" * 64)
print("DLWP local forecast")
print(f"  init:  {INIT_TIME:%Y-%m-%d %HZ}   steps: {NSTEPS} x 6h")
print("=" * 64)

t0 = walltime.perf_counter()
print("\n[1/4] Loading pretrained DLWP (downloads on first run, then cached)")
package = DLWP.load_default_package()
model = DLWP.load_model(package)
t_load = walltime.perf_counter() - t0
print(f"      model ready in {t_load:.1f}s")

data = GFS()
if os.path.exists(OUT_ZARR):
    shutil.rmtree(OUT_ZARR)  # rerun-safe: replace any previous forecast
io = ZarrBackend(OUT_ZARR)

print("\n[2/4] Fetching GFS initial condition and rolling the forecast")
t0 = walltime.perf_counter()
io = run.deterministic(
    time=[INIT_TIME.isoformat()],
    nsteps=NSTEPS,
    prognostic=model,
    data=data,
    io=io,
)
t_fcst = walltime.perf_counter() - t0
print(f"      forecast done in {t_fcst:.1f}s, saved to {OUT_ZARR}")

# ===========================================================================
# Part 2: Load forecast output (robust to both ZarrBackend layouts)
# ===========================================================================

ds = xr.open_zarr(OUT_ZARR)

def get_forecast_field(var, lead_hours):
    """Return a 2D (lat, lon) numpy field for one variable and lead time."""
    lead = np.timedelta64(lead_hours, "h")
    if var in ds.data_vars:
        da = ds[var].sel(lead_time=lead).squeeze()
    else:
        da = ds["fields"].sel(variable=var, lead_time=lead).squeeze()
    return np.asarray(da.values)

lats = np.asarray(ds["lat"].values)
lons = np.asarray(ds["lon"].values)

# ===========================================================================
# Part 3: Fetch verifying GFS analyses
# ===========================================================================

print("\n[3/4] Fetching GFS verifying analyses (every 24h out to 120h)")
verify_leads = list(range(0, NSTEPS * 6 + 1, VERIFY_EVERY_H))
verify_times = [INIT_TIME + timedelta(hours=h) for h in verify_leads]
verify_vars = ["z500", "t850", "t2m"]

t0 = walltime.perf_counter()
truth = data(verify_times, verify_vars)  # dims: (time, variable, lat, lon)
t_verify = walltime.perf_counter() - t0
print(f"      analyses fetched in {t_verify:.1f}s")

def get_truth_field(var, lead_hours):
    t = np.datetime64(INIT_TIME + timedelta(hours=lead_hours))
    return np.asarray(truth.sel(time=t, variable=var).values)

# ===========================================================================
# Part 4: Plots and skill numbers
# ===========================================================================

print("\n[4/4] Plotting")

import cartopy.crs as ccrs
import cartopy.feature as cfeature

def map_panel(ax, field, title, cmap, vmin, vmax):
    im = ax.pcolormesh(
        lons, lats, field, transform=ccrs.PlateCarree(),
        cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
    )
    ax.coastlines(linewidth=0.5)
    ax.set_title(title, fontsize=10)
    ax.set_global()
    return im

FINAL_LEAD = NSTEPS * 6  # 120h

# --- Figure 1: z500, forecast vs analysis vs error at final lead ---
fc_z = geopotential_to_dam(get_forecast_field("z500", FINAL_LEAD))
an_z = geopotential_to_dam(get_truth_field("z500", FINAL_LEAD))
vmin, vmax = np.percentile(an_z, [1, 99])

fig, axes = plt.subplots(
    3, 1, figsize=(11, 13), subplot_kw={"projection": ccrs.PlateCarree()}
)
im0 = map_panel(axes[0], fc_z, f"DLWP forecast z500 (dam), +{FINAL_LEAD}h "
                f"(init {INIT_TIME:%Y-%m-%d %HZ})", "viridis", vmin, vmax)
im1 = map_panel(axes[1], an_z, f"GFS analysis z500 (dam), valid "
                f"{INIT_TIME + timedelta(hours=FINAL_LEAD):%Y-%m-%d %HZ}",
                "viridis", vmin, vmax)
err = fc_z - an_z
im2 = map_panel(axes[2], err, "Forecast minus analysis (dam)",
                "RdBu_r", -15, 15)
fig.colorbar(im0, ax=axes[0], shrink=0.8)
fig.colorbar(im1, ax=axes[1], shrink=0.8)
fig.colorbar(im2, ax=axes[2], shrink=0.8)
p1 = os.path.join(OUT_DIR, "dlwp_z500_vs_analysis_120h.png")
fig.savefig(p1, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"      saved {p1}")

# --- Figure 2: t2m, forecast vs analysis at final lead ---
fc_t = kelvin_to_celsius(get_forecast_field("t2m", FINAL_LEAD))
an_t = kelvin_to_celsius(get_truth_field("t2m", FINAL_LEAD))

fig, axes = plt.subplots(
    3, 1, figsize=(11, 13), subplot_kw={"projection": ccrs.PlateCarree()}
)
im0 = map_panel(axes[0], fc_t, f"DLWP forecast 2m temperature (C), +{FINAL_LEAD}h",
                "coolwarm", -40, 40)
im1 = map_panel(axes[1], an_t, "GFS analysis 2m temperature (C), same valid time",
                "coolwarm", -40, 40)
im2 = map_panel(axes[2], fc_t - an_t, "Forecast minus analysis (C)",
                "RdBu_r", -8, 8)
for im, ax in zip([im0, im1, im2], axes):
    fig.colorbar(im, ax=ax, shrink=0.8)
p2 = os.path.join(OUT_DIR, "dlwp_t2m_vs_analysis_120h.png")
fig.savefig(p2, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"      saved {p2}")

# --- Figure 3: RMSE growth vs persistence baseline ---
# Persistence forecast: "tomorrow looks like today", i.e. the initial
# analysis carried forward unchanged. Any real model must beat this.
weights = np.cos(np.deg2rad(lats))[:, None]  # area weighting

def wrmse(a, b):
    return float(np.sqrt(np.average((a - b) ** 2, weights=np.broadcast_to(weights, a.shape))))

curves = {}
for var, scale in [("z500", 1 / 9.81), ("t850", 1.0)]:
    init_field = get_truth_field(var, 0) * scale
    model_rmse, persist_rmse = [], []
    for h in verify_leads:
        an = get_truth_field(var, h) * scale
        fc = get_forecast_field(var, h) * scale
        model_rmse.append(wrmse(fc, an))
        persist_rmse.append(wrmse(init_field, an))
    curves[var] = (model_rmse, persist_rmse)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
units = {"z500": "m (geopotential height)", "t850": "K"}
for ax, var in zip(axes, ["z500", "t850"]):
    m, p = curves[var]
    ax.plot(verify_leads, m, "-o", label="DLWP forecast")
    ax.plot(verify_leads, p, "--s", label="Persistence baseline")
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel(f"Global area-weighted RMSE, {units[var]}")
    ax.set_title(var)
    ax.grid(alpha=0.3)
    ax.legend()
fig.suptitle(f"DLWP skill vs persistence, init {INIT_TIME:%Y-%m-%d %HZ}")
fig.tight_layout()
p3 = os.path.join(OUT_DIR, "dlwp_rmse_vs_persistence.png")
fig.savefig(p3, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"      saved {p3}")

# --- Console summary ---
print("\n" + "=" * 64)
print("SKILL SUMMARY (global, area-weighted RMSE vs GFS analysis)")
print("=" * 64)
print(f"{'lead':>6} | {'z500 model (m)':>15} | {'z500 persist (m)':>16} | "
      f"{'t850 model (K)':>14} | {'t850 persist (K)':>16}")
for i, h in enumerate(verify_leads):
    print(f"{h:>5}h | {curves['z500'][0][i]:>15.1f} | {curves['z500'][1][i]:>16.1f} | "
          f"{curves['t850'][0][i]:>14.2f} | {curves['t850'][1][i]:>16.2f}")

print(f"\nTimings: model load {t_load:.1f}s | forecast {t_fcst:.1f}s | "
      f"verification fetch {t_verify:.1f}s")
print("Outputs:")
for p in [OUT_ZARR, p1, p2, p3]:
    print(f"  {p}")
