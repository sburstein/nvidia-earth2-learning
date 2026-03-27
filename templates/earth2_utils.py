"""
Earth2Studio Utility Templates
===============================
Reusable functions for common tasks across all exercises.
Import with: from templates.earth2_utils import *
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ===========================================================================
# Data Loading & Conversion
# ===========================================================================

def load_forecast(zarr_path):
    """Load an Earth2Studio forecast output.

    Returns:
        xarray.Dataset with dimensions (time, lead_time, variable, lat, lon)
        or (time, ensemble, lead_time, variable, lat, lon) for ensembles
    """
    import xarray as xr
    ds = xr.open_zarr(zarr_path)
    print(f"Loaded: {zarr_path}")
    print(f"  Dims: {dict(ds.dims)}")
    print(f"  Variables: {len(ds.coords.get('variable', []))}")
    return ds


def kelvin_to_celsius(field):
    """Convert temperature from Kelvin to Celsius."""
    return field - 273.15


def kelvin_to_fahrenheit(field):
    """Convert temperature from Kelvin to Fahrenheit."""
    return (field - 273.15) * 9/5 + 32


def geopotential_to_dam(field):
    """Convert geopotential (m²/s²) to decameters (standard for z500 maps)."""
    return field / (9.81 * 10)


def wind_speed(u, v):
    """Compute wind speed from u and v components."""
    return np.sqrt(u**2 + v**2)


def wind_direction(u, v):
    """Compute meteorological wind direction (degrees, where wind comes FROM)."""
    return (np.arctan2(-u, -v) * 180 / np.pi + 360) % 360


def lon_west_to_east(lon_west):
    """Convert western longitude (e.g., 74°W) to 0-360 convention.

    Earth2Studio uses 0-360° longitude.
    Example: lon_west_to_east(74) → 286 (for New York City)
    """
    return 360 - lon_west


# ===========================================================================
# Grid Operations
# ===========================================================================

def find_nearest_gridpoint(lats, lons, target_lat, target_lon):
    """Find the nearest grid point indices for a given lat/lon.

    Args:
        lats: 1D array of latitudes
        lons: 1D array of longitudes (0-360 convention)
        target_lat: desired latitude
        target_lon: desired longitude (0-360 convention)

    Returns:
        (lat_idx, lon_idx) tuple
    """
    lat_idx = np.argmin(np.abs(lats - target_lat))
    lon_idx = np.argmin(np.abs(lons - target_lon))
    return lat_idx, lon_idx


def extract_region(data, lats, lons, lat_range, lon_range):
    """Extract a regional subset from global data.

    Args:
        data: 2D array (lat, lon)
        lats: 1D latitude array
        lons: 1D longitude array
        lat_range: (min_lat, max_lat)
        lon_range: (min_lon, max_lon) in 0-360 convention

    Returns:
        (regional_data, regional_lats, regional_lons)
    """
    lat_mask = (lats >= lat_range[0]) & (lats <= lat_range[1])
    lon_mask = (lons >= lon_range[0]) & (lons <= lon_range[1])
    return (data[np.ix_(lat_mask, lon_mask)],
            lats[lat_mask], lons[lon_mask])


# Common regions (lon in 0-360 convention)
REGIONS = {
    "conus": {"lat": (25, 50), "lon": (235, 295), "name": "Continental US"},
    "europe": {"lat": (35, 72), "lon": (350, 40), "name": "Europe"},
    "east_asia": {"lat": (15, 55), "lon": (100, 150), "name": "East Asia"},
    "tropics": {"lat": (-23, 23), "lon": (0, 360), "name": "Tropics"},
    "arctic": {"lat": (60, 90), "lon": (0, 360), "name": "Arctic"},
    "southeast_asia": {"lat": (-10, 25), "lon": (95, 140), "name": "SE Asia"},
}

# Common cities (lat, lon_360)
CITIES = {
    "new_york": (40.7, 286.0),
    "london": (51.5, 359.9),
    "tokyo": (35.7, 139.7),
    "sydney": (-33.9, 151.2),
    "beijing": (39.9, 116.4),
    "mumbai": (19.1, 72.9),
    "sao_paulo": (-23.5, 313.4),
    "nairobi": (-1.3, 36.8),
    "los_angeles": (34.1, 241.8),
    "chicago": (41.9, 272.4),
}


# ===========================================================================
# Verification Metrics
# ===========================================================================

def rmse(forecast, observation):
    """Root Mean Square Error."""
    return np.sqrt(np.nanmean((forecast - observation)**2))


def bias(forecast, observation):
    """Mean bias (forecast - observation)."""
    return np.nanmean(forecast - observation)


def anomaly_correlation(forecast, observation, climatology):
    """Anomaly Correlation Coefficient (ACC).

    The standard skill metric for medium-range weather forecasting.
    ACC > 0.6 is considered "useful", ACC > 0.8 is "good".
    """
    f_anom = forecast - climatology
    o_anom = observation - climatology
    num = np.nanmean(f_anom * o_anom)
    den = np.sqrt(np.nanmean(f_anom**2) * np.nanmean(o_anom**2))
    return num / den if den > 0 else 0.0


def crps_ensemble(ensemble, observation):
    """Continuous Ranked Probability Score for an ensemble forecast.

    CRPS is the gold-standard metric for probabilistic forecasts.
    Lower is better. CRPS = 0 for a perfect forecast.

    Args:
        ensemble: array of shape (n_members,) at a single point/time
        observation: scalar observation value
    """
    n = len(ensemble)
    sorted_ens = np.sort(ensemble)

    # CRPS = E|X-y| - 0.5 * E|X-X'|
    abs_diff = np.mean(np.abs(sorted_ens - observation))
    pairwise = np.mean(np.abs(sorted_ens[:, None] - sorted_ens[None, :]))
    return abs_diff - 0.5 * pairwise


# ===========================================================================
# Plotting Helpers
# ===========================================================================

def quick_map(data, lats, lons, title="", cmap="viridis",
              vmin=None, vmax=None, save_path=None):
    """Quick global map plot."""
    fig, ax = plt.subplots(figsize=(14, 6))
    cf = ax.contourf(lons, lats, data, levels=30, cmap=cmap,
                      vmin=vmin, vmax=vmax)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title, fontsize=14)
    plt.colorbar(cf, ax=ax)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig, ax


def quick_timeseries(times, values, labels=None, title="",
                     xlabel="Time", ylabel="Value", save_path=None):
    """Quick time series plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    if labels is None:
        ax.plot(times, values, 'b-o', markersize=3)
    else:
        for v, l in zip(values, labels):
            ax.plot(times, v, '-o', markersize=3, label=l)
        ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig, ax


# ===========================================================================
# Model Loading Helpers
# ===========================================================================

def list_available_models():
    """Print all available Earth2Studio models."""
    models = {
        "Prognostic (px) — step forward in time": [
            "FCN3     — FourCastNet 3 (wavelet, fastest, 0.25°)",
            "SFNO     — Spherical Fourier Neural Operator (stable rollouts)",
            "DLWP     — Deep Learning Weather Prediction (~2°)",
            "Pangu    — Pangu-Weather (transformer, 0.25°)",
            "Atlas    — Latent diffusion transformer (2026, best accuracy)",
            "DLESyMLatLon — Subseasonal-to-Seasonal (daily steps, 90 days)",
        ],
        "Diagnostic (dx) — derived quantities": [
            "ClimateNet  — Extreme weather detection",
            "PrecipNet   — Precipitation classification",
        ],
        "Data sources": [
            "GFS     — NOAA Global Forecast System (real-time)",
            "ARCO    — ERA5 reanalysis (Google Cloud zarr)",
            "CDS     — ECMWF Climate Data Store (ERA5, official)",
            "HRRR    — NOAA High-Resolution Rapid Refresh (CONUS 3km)",
        ],
    }

    for category, items in models.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  • {item}")


if __name__ == "__main__":
    print("Earth2Studio Utility Templates")
    print("=" * 40)
    list_available_models()
