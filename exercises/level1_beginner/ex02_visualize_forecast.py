"""
Exercise 1.2: Visualize Your Forecast
======================================
Level: Beginner | Time: 20 min | GPU: Not needed (reads saved data)

GOAL: Create publication-quality weather maps from your Exercise 1.1 forecast.
      Learn to work with xarray + matplotlib + cartopy for geospatial plotting.

CONCEPTS:
  - Zarr → xarray workflow for reading Earth2Studio output
  - Cartopy projections for weather map plotting
  - Selecting specific variables, pressure levels, and lead times
  - Creating multi-panel forecast evolution plots

PREREQUISITES: Run ex01_first_forecast.py first to generate output data.
"""

import os
import sys
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Try cartopy for map projections — falls back to basic plot if not installed
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Note: Install cartopy for proper map projections: pip install cartopy")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')
FORECAST_PATH = os.path.join(OUTPUT_DIR, 'ex01_forecast.zarr')

# ===========================================================================
# PART 1: Load and inspect the forecast data
# ===========================================================================

print("Loading forecast data...")
ds = xr.open_zarr(FORECAST_PATH)
print(f"Dataset dimensions: {dict(ds.dims)}")
print(f"Available variables: {list(ds.coords['variable'].values)[:10]}...")

# ===========================================================================
# PART 2: Plot 500 hPa Geopotential Height at T+48h
# ===========================================================================
# This is the classic "weather map" — z500 shows the large-scale flow pattern.
# Troughs (low values) = stormy, Ridges (high values) = fair weather

def plot_z500(ds, step_idx=8, save_path=None):
    """Plot 500 hPa geopotential height.

    Args:
        ds: xarray Dataset from Earth2Studio
        step_idx: forecast step index (8 = T+48h for 6hr steps)
        save_path: optional file path to save figure
    """
    # Select z500 at the desired lead time
    # Earth2Studio stores data as (time, lead_time, variable, lat, lon)
    z500 = ds.sel(variable="z500").isel(time=0, lead_time=step_idx)

    # Get the actual field values
    field = z500.values.squeeze()
    lats = ds.coords['lat'].values
    lons = ds.coords['lon'].values

    # Convert geopotential to decameters (standard meteorological convention)
    field_dam = field / (9.81 * 10)  # m²/s² → dam

    if HAS_CARTOPY:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8),
                               subplot_kw={'projection': ccrs.Robinson()})

        # Filled contours
        cf = ax.contourf(lons, lats, field_dam,
                         levels=np.arange(480, 600, 4),
                         cmap='RdYlBu_r',
                         transform=ccrs.PlateCarree())

        # Contour lines
        cs = ax.contour(lons, lats, field_dam,
                        levels=np.arange(480, 600, 8),
                        colors='black', linewidths=0.5,
                        transform=ccrs.PlateCarree())
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.0f')

        # Map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.set_global()

        plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05,
                     label='500 hPa Geopotential Height (dam)')
    else:
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        cf = ax.contourf(lons, lats, field_dam,
                         levels=np.arange(480, 600, 4),
                         cmap='RdYlBu_r')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.colorbar(cf, ax=ax, label='500 hPa Geopotential Height (dam)')

    lead_hours = step_idx * 6
    ax.set_title(f'500 hPa Geopotential Height — T+{lead_hours}h Forecast\n'
                 f'FourCastNet3 | Init: {str(ds.coords["time"].values[0])[:10]}',
                 fontsize=14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

print("\nPlotting 500 hPa height at T+48h...")
plot_z500(ds, step_idx=8,
          save_path=os.path.join(OUTPUT_DIR, 'ex02_z500_48h.png'))


# ===========================================================================
# PART 3: Plot 2m Temperature evolution (T+0, T+24, T+48, T+72, T+96, T+120)
# ===========================================================================

def plot_temperature_evolution(ds, save_path=None):
    """Plot 2m temperature at multiple lead times to show forecast evolution."""

    steps = [0, 4, 8, 12, 16, 20]  # Every 24 hours (4 × 6hr = 24hr)
    # Clamp to available steps
    max_step = len(ds.coords['lead_time']) - 1
    steps = [s for s in steps if s <= max_step]

    n_panels = len(steps)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, step_idx in enumerate(steps):
        if i >= len(axes):
            break

        t2m = ds.sel(variable="t2m").isel(time=0, lead_time=step_idx)
        field = t2m.values.squeeze()

        # Convert Kelvin to Celsius
        field_c = field - 273.15

        lats = ds.coords['lat'].values
        lons = ds.coords['lon'].values

        ax = axes[i]
        cf = ax.contourf(lons, lats, field_c,
                         levels=np.arange(-40, 45, 5),
                         cmap='RdYlBu_r')
        ax.set_title(f'T+{step_idx * 6}h ({step_idx * 6 // 24} days)')

        if i >= 3:
            ax.set_xlabel('Longitude')
        if i % 3 == 0:
            ax.set_ylabel('Latitude')

    # Hide unused panels
    for j in range(len(steps), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('2m Temperature Evolution (°C) — FourCastNet3 Forecast',
                 fontsize=16, y=1.02)

    # Shared colorbar
    fig.colorbar(cf, ax=axes[:len(steps)], orientation='horizontal',
                 pad=0.08, label='Temperature (°C)', shrink=0.8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

print("\nPlotting temperature evolution...")
plot_temperature_evolution(ds,
    save_path=os.path.join(OUTPUT_DIR, 'ex02_t2m_evolution.png'))


# ===========================================================================
# PART 4: Regional Zoom — Your City
# ===========================================================================

def plot_regional(ds, variable="t2m", step_idx=8,
                  lat_range=(25, 50), lon_range=(235, 295),
                  title_suffix="CONUS", save_path=None):
    """Plot a regional subset of the forecast.

    Note: Earth2Studio uses 0-360° longitude convention.
    To convert from standard (-180 to 180): lon_360 = lon % 360
    Example: New York at -74°W → 360 - 74 = 286°
    """
    field = ds.sel(variable=variable).isel(time=0, lead_time=step_idx)

    # Subset to region
    lats = ds.coords['lat'].values
    lons = ds.coords['lon'].values

    lat_mask = (lats >= lat_range[0]) & (lats <= lat_range[1])
    lon_mask = (lons >= lon_range[0]) & (lons <= lon_range[1])

    field_regional = field.values.squeeze()[np.ix_(lat_mask, lon_mask)]
    lats_r = lats[lat_mask]
    lons_r = lons[lon_mask]

    # Convert units based on variable
    if variable == "t2m":
        field_regional -= 273.15
        unit = "°C"
        cmap = "RdYlBu_r"
    elif variable == "z500":
        field_regional /= (9.81 * 10)
        unit = "dam"
        cmap = "RdYlBu_r"
    elif variable in ("u10m", "v10m"):
        unit = "m/s"
        cmap = "coolwarm"
    else:
        unit = ""
        cmap = "viridis"

    fig, ax = plt.subplots(figsize=(12, 8))
    cf = ax.contourf(lons_r, lats_r, field_regional, levels=30, cmap=cmap)
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title(f'{variable} — {title_suffix} — T+{step_idx * 6}h\n'
                 f'FourCastNet3 Forecast', fontsize=14)
    plt.colorbar(cf, ax=ax, label=f'{variable} ({unit})')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

print("\nPlotting regional CONUS temperature...")
plot_regional(ds, variable="t2m", step_idx=8,
              lat_range=(25, 50), lon_range=(235, 295),
              title_suffix="Continental US",
              save_path=os.path.join(OUTPUT_DIR, 'ex02_t2m_conus.png'))


# ===========================================================================
# EXERCISES
# ===========================================================================
"""
✅ Exercise 1.2a: Plot sea-level pressure (msl) instead of z500. What
   patterns do you see? (Look for highs/lows — they drive surface weather.)

✅ Exercise 1.2b: Create a regional plot centered on your city. You'll need
   to convert your longitude: if your city is at 74°W, use 360 - 74 = 286°.

✅ Exercise 1.2c: Plot the difference between T+0 and T+120h for t2m.
   Where does the model predict the biggest temperature changes over 5 days?

✅ Exercise 1.2d: Plot 10m wind speed (compute from u10m and v10m:
   speed = sqrt(u10m² + v10m²)). Where are the strongest winds?

🔥 Challenge: Create an animated GIF showing the forecast evolution
   (hint: use matplotlib.animation or save individual frames and combine
   with imageio).
"""
