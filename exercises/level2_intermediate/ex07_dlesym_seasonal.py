"""
Exercise 2.4: DLESyM Multi-Week Seasonal Forecast
===================================================
Level: Intermediate | Time: 45 min | GPU: Required (V100+ recommended)

GOAL: Run a subseasonal-to-seasonal (S2S) forecast using DLESyM — NVIDIA's
      coupled atmosphere-ocean model. Predict beyond the 2-week "predictability
      barrier" where traditional weather forecasting breaks down.

CONCEPTS:
  - S2S gap: the 2-week to 2-month range where deterministic forecasts are
    useless but statistical patterns (e.g., MJO, ENSO) still provide skill
  - Coupled atmosphere-ocean: DLESyM jointly predicts atmospheric state AND
    sea surface temperatures — critical for S2S because the ocean drives
    predictability at these timescales
  - HEALPix grid: hierarchical equal-area pixelization that gives uniform
    coverage without polar singularities
  - Tercile probabilities: P(above normal), P(near normal), P(below normal)
    — the standard way to express S2S forecasts

WHY S2S MATTERS:
  - Agriculture: planting decisions need 4-8 week outlooks
  - Energy: heating/cooling demand planning
  - Water management: reservoir operations need seasonal inflow forecasts
  - Insurance: catastrophe modeling needs multi-month perspectives
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')

# ===========================================================================
# PART 1: Run DLESyM forecast
# ===========================================================================

from earth2studio.models.px import DLESyMLatLon
from earth2studio.data import ARCO  # ERA5 reanalysis data
from earth2studio.io import ZarrBackend
from earth2studio import run

print("=" * 60)
print("  DLESyM Subseasonal-to-Seasonal Forecast")
print("=" * 60)

# DLESyM uses ERA5 initial conditions (not GFS)
# ARCO provides ERA5 via Google Cloud in zarr format
print("\nLoading DLESyM model...")
package = DLESyMLatLon.load_default_package()
model = DLESyMLatLon.load_model(package)

data = ARCO()  # ERA5 reanalysis

output_path = os.path.join(OUTPUT_DIR, 'ex07_dlesym_90day.zarr')
print(f"Running 90-day forecast...")

# 90 steps = 90 days (~13 weeks)
# DLESyM uses daily timesteps (unlike FCN3's 6-hourly steps)
io = run.deterministic(
    time=["2025-01-01T00:00:00"],
    nsteps=90,
    model=model,
    data=data,
    io=ZarrBackend(output_path)
)

print(f"Forecast complete → {output_path}")

# ===========================================================================
# PART 2: Analyze seasonal evolution
# ===========================================================================

ds = xr.open_zarr(output_path)
print(f"\nDataset: {dict(ds.dims)}")
print(f"Variables: {list(ds.coords['variable'].values)}")

lats = ds.coords['lat'].values
lons = ds.coords['lon'].values

# ===========================================================================
# PART 3: Weekly mean temperature evolution
# ===========================================================================

def plot_weekly_evolution(ds, variable="t2m", save_path=None):
    """Plot forecast evolution in weekly averages."""

    # Group into weeks (7 days per week)
    n_steps = len(ds.coords['lead_time'])
    n_weeks = n_steps // 7

    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    axes = axes.flatten()

    for week in range(min(n_weeks, 12)):
        step_start = week * 7
        step_end = min(step_start + 7, n_steps)

        # Weekly mean
        weekly_data = ds.sel(variable=variable).isel(
            time=0, lead_time=slice(step_start, step_end)
        ).mean(dim="lead_time").values.squeeze()

        if variable == "t2m":
            weekly_data -= 273.15
            unit = "°C"
            cmap = "RdYlBu_r"
            levels = np.arange(-40, 45, 5)
        else:
            unit = ""
            cmap = "viridis"
            levels = 30

        ax = axes[week]
        cf = ax.contourf(lons, lats, weekly_data, levels=levels, cmap=cmap)
        ax.set_title(f'Week {week + 1}\n(Days {step_start + 1}–{step_end})')

        if week >= 8:
            ax.set_xlabel('Longitude')
        if week % 4 == 0:
            ax.set_ylabel('Latitude')

    for j in range(min(n_weeks, 12), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'DLESyM {variable} Weekly Mean Evolution (°C)\n'
                 f'90-Day Forecast from {str(ds.coords["time"].values[0])[:10]}',
                 fontsize=16, y=1.02)
    fig.colorbar(cf, ax=axes[:min(n_weeks, 12)], orientation='horizontal',
                 pad=0.05, label=f'{variable} ({unit})', shrink=0.8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

print("\nPlotting weekly temperature evolution...")
plot_weekly_evolution(ds, save_path=os.path.join(OUTPUT_DIR, 'ex07_weekly_evolution.png'))


# ===========================================================================
# PART 4: Regional time series
# ===========================================================================

def plot_regional_timeseries(ds, variable="t2m",
                              regions=None, save_path=None):
    """Plot temperature evolution for multiple regions.

    Great for comparing how different parts of the world evolve
    over the S2S forecast horizon.
    """
    if regions is None:
        regions = {
            "Tropics (10°S–10°N)": ((-10, 10), (0, 360)),
            "NH Midlats (30°N–60°N)": ((30, 60), (0, 360)),
            "SH Midlats (30°S–60°S)": ((-60, -30), (0, 360)),
            "Arctic (>60°N)": ((60, 90), (0, 360)),
        }

    lats = ds.coords['lat'].values
    lons = ds.coords['lon'].values

    fig, ax = plt.subplots(figsize=(12, 6))

    for name, (lat_range, lon_range) in regions.items():
        lat_mask = (lats >= lat_range[0]) & (lats <= lat_range[1])
        lon_mask = (lons >= lon_range[0]) & (lons <= lon_range[1])

        field = ds.sel(variable=variable).isel(time=0).values.squeeze()
        # field shape: (lead_time, lat, lon)

        regional_mean = []
        for step in range(len(ds.coords['lead_time'])):
            step_data = field[step]
            regional = step_data[np.ix_(lat_mask, lon_mask)]
            regional_mean.append(np.nanmean(regional))

        regional_mean = np.array(regional_mean)
        if variable == "t2m":
            regional_mean -= 273.15

        days = np.arange(len(regional_mean))
        ax.plot(days, regional_mean, linewidth=2, label=name)

    ax.set_xlabel('Forecast Day')
    ax.set_ylabel(f'{variable} (°C)')
    ax.set_title(f'DLESyM Regional Temperature Evolution\n'
                 f'90-Day Forecast', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark key S2S timescales
    ax.axvline(x=14, color='gray', linestyle='--', alpha=0.4, label='_')
    ax.text(14, ax.get_ylim()[1], ' Week 2', va='top', fontsize=9, color='gray')
    ax.axvline(x=28, color='gray', linestyle=':', alpha=0.4, label='_')
    ax.text(28, ax.get_ylim()[1], ' Week 4', va='top', fontsize=9, color='gray')
    ax.axvline(x=56, color='gray', linestyle='-.', alpha=0.4, label='_')
    ax.text(56, ax.get_ylim()[1], ' Week 8', va='top', fontsize=9, color='gray')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

print("\nPlotting regional time series...")
plot_regional_timeseries(ds,
    save_path=os.path.join(OUTPUT_DIR, 'ex07_regional_timeseries.png'))


# ===========================================================================
# PART 5: Tercile anomaly map (Week 3-4 outlook)
# ===========================================================================

def plot_anomaly_outlook(ds, variable="t2m", week_start=3, week_end=4,
                         save_path=None):
    """Create a tercile anomaly map for weeks 3-4.

    In operational S2S forecasting, outlooks are expressed as:
    - "Above normal" = upper tercile of climatology
    - "Near normal" = middle tercile
    - "Below normal" = lower tercile

    Without actual climatology, we show the anomaly from the forecast's
    own Day 1 field (a rough proxy).
    """
    step_start = (week_start - 1) * 7
    step_end = week_end * 7

    # Week 3-4 mean
    future = ds.sel(variable=variable).isel(
        time=0, lead_time=slice(step_start, step_end)
    ).mean(dim="lead_time").values.squeeze()

    # Day 1 as baseline (proxy for "climatology")
    baseline = ds.sel(variable=variable).isel(
        time=0, lead_time=0
    ).values.squeeze()

    anomaly = future - baseline

    if variable == "t2m":
        unit = "°C"  # anomaly in K = anomaly in °C

    lats = ds.coords['lat'].values
    lons = ds.coords['lon'].values

    fig, ax = plt.subplots(figsize=(14, 6))
    levels = np.arange(-10, 11, 1)
    cf = ax.contourf(lons, lats, anomaly, levels=levels, cmap='RdBu_r')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Temperature Anomaly (vs Day 1) — Weeks {week_start}–{week_end}\n'
                 f'DLESyM Forecast | Red = Warmer, Blue = Cooler', fontsize=14)
    plt.colorbar(cf, ax=ax, label=f'Anomaly ({unit})')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

print("\nPlotting Week 3-4 temperature anomaly outlook...")
plot_anomaly_outlook(ds, week_start=3, week_end=4,
    save_path=os.path.join(OUTPUT_DIR, 'ex07_anomaly_wk3_4.png'))


# ===========================================================================
# EXERCISES
# ===========================================================================
"""
✅ Exercise 2.4a: Compare the Week 1-2 anomaly map vs Week 5-6. Does the
   pattern change, or does it stay similar? (If it stays similar, the
   model may be "locking onto" a persistent mode like an ENSO phase.)

✅ Exercise 2.4b: Run a DLESyM ensemble (20 members) using the same
   ensemble pattern from Exercise 2.1. Does the ensemble spread grow
   faster or slower than FCN3 over the S2S timescale?

✅ Exercise 2.4c: If DLESyM outputs SST (sea surface temperature), plot
   its evolution over 90 days. The ocean changes much more slowly than
   the atmosphere — this is why it's a source of S2S predictability.

✅ Exercise 2.4d: Pick a region prone to drought or flooding (e.g., East
   Africa, India monsoon, US Great Plains). Plot the week-by-week
   evolution. Can you identify the onset or withdrawal of a rainy season?

🔥 Challenge: Compare DLESyM's Week 3-4 outlook against CPC's operational
   outlook (https://www.cpc.ncep.noaa.gov/). How does the AI model compare
   to the operational statistical-dynamical blend?
"""
