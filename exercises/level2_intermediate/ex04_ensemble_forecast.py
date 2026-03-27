"""
Exercise 2.1: Generate a 50-Member Ensemble Forecast
=====================================================
Level: Intermediate | Time: 45 min | GPU: Required (A100 recommended)

GOAL: Create a probabilistic weather forecast by running 50 ensemble members
      with perturbed initial conditions. Learn why ensembles matter more than
      single deterministic forecasts for real-world decision-making.

CONCEPTS:
  - Ensemble forecasting: running the same model many times with slightly
    different starting conditions to quantify uncertainty
  - SphericalGaussian perturbation: adds noise that respects Earth's geometry
  - Ensemble spread: where members disagree = high uncertainty
  - Ensemble mean: averages out noise, often more skillful than any single member
  - Probability of exceedance: P(temperature > 35°C) from ensemble distribution

WHY ENSEMBLES MATTER:
  A single forecast says "it will be 30°C." An ensemble says "there's a 70%
  chance it'll be 28-32°C, but a 15% chance of >35°C." The second is far more
  useful for energy grid operators, emergency managers, and agriculture.

  NVIDIA's FCN3 can generate 50 ensemble members in the time traditional NWP
  takes for ONE deterministic run — this is the revolution.
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')

# ===========================================================================
# PART 1: Run the ensemble
# ===========================================================================

from earth2studio.models.px import FCN3
from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.perturbation import SphericalGaussian
from earth2studio import run

print("Loading FCN3 model...")
package = FCN3.load_default_package()
model = FCN3.load_model(package)

# SphericalGaussian adds perturbations that respect the spherical geometry
# of Earth. The noise_amplitude controls how much uncertainty we inject.
# 0.05 is a reasonable starting point — tune this for your use case.
perturbation = SphericalGaussian(noise_amplitude=0.05)

ensemble_path = os.path.join(OUTPUT_DIR, 'ex04_ensemble.zarr')
print(f"Running 50-member ensemble (this takes a few minutes on GPU)...")

io = run.ensemble(
    time=["2025-06-01T00:00:00"],
    nsteps=40,        # 10 days
    nensemble=50,      # 50 members
    model=model,
    data=GFS(),
    io=ZarrBackend(ensemble_path),
    perturbation=perturbation
)

print(f"Ensemble complete → {ensemble_path}")

# ===========================================================================
# PART 2: Analyze ensemble statistics
# ===========================================================================

ds = xr.open_zarr(ensemble_path)
print(f"\nDataset shape: {dict(ds.dims)}")
# Expected: (time=1, ensemble=50, lead_time=41, variable=73, lat=721, lon=1440)

# Select 2m temperature
t2m = ds.sel(variable="t2m").isel(time=0)  # (ensemble, lead_time, lat, lon)
t2m_celsius = t2m - 273.15

# Compute ensemble statistics at T+120h (5 days)
step = 20  # T+120h

ensemble_mean = t2m_celsius.isel(lead_time=step).mean(dim="ensemble").values.squeeze()
ensemble_std = t2m_celsius.isel(lead_time=step).std(dim="ensemble").values.squeeze()
ensemble_min = t2m_celsius.isel(lead_time=step).min(dim="ensemble").values.squeeze()
ensemble_max = t2m_celsius.isel(lead_time=step).max(dim="ensemble").values.squeeze()

lats = ds.coords['lat'].values
lons = ds.coords['lon'].values

print(f"\nEnsemble statistics at T+120h:")
print(f"  Global mean temperature: {np.nanmean(ensemble_mean):.1f}°C")
print(f"  Mean ensemble spread (std): {np.nanmean(ensemble_std):.2f}°C")
print(f"  Max ensemble spread: {np.nanmax(ensemble_std):.2f}°C")
print(f"  Min-Max range: {np.nanmax(ensemble_max - ensemble_min):.1f}°C")


# ===========================================================================
# PART 3: Plot ensemble spread map
# ===========================================================================

def plot_ensemble_spread(mean_field, std_field, lats, lons,
                         lead_hours, save_path=None):
    """Plot ensemble mean and spread (standard deviation)."""

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Ensemble mean
    cf1 = axes[0].contourf(lons, lats, mean_field,
                            levels=np.arange(-40, 45, 5),
                            cmap='RdYlBu_r')
    axes[0].set_title(f'Ensemble Mean T2m (°C) — T+{lead_hours}h')
    plt.colorbar(cf1, ax=axes[0], label='Temperature (°C)')

    # Ensemble spread (standard deviation)
    cf2 = axes[1].contourf(lons, lats, std_field,
                            levels=np.arange(0, 8, 0.5),
                            cmap='YlOrRd')
    axes[1].set_title(f'Ensemble Spread (σ) — T+{lead_hours}h\n'
                      f'High spread = High uncertainty')
    plt.colorbar(cf2, ax=axes[1], label='Standard Deviation (°C)')

    for ax in axes:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    fig.suptitle(f'50-Member FCN3 Ensemble — T+{lead_hours}h Forecast',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

plot_ensemble_spread(ensemble_mean, ensemble_std, lats, lons, 120,
    save_path=os.path.join(OUTPUT_DIR, 'ex04_ensemble_spread_120h.png'))


# ===========================================================================
# PART 4: Spread growth over time
# ===========================================================================

def plot_spread_growth(ds, variable="t2m", save_path=None):
    """Plot how ensemble spread grows with forecast lead time.

    In a well-calibrated ensemble, spread should grow to match the
    actual forecast error. Too little spread = overconfident.
    """
    field = ds.sel(variable=variable).isel(time=0)

    spreads = []
    lead_hours = []

    for step in range(len(ds.coords['lead_time'])):
        std = field.isel(lead_time=step).std(dim="ensemble").values.squeeze()
        spreads.append(np.nanmean(std))
        lead_hours.append(step * 6)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lead_hours, spreads, 'r-o', markersize=3, linewidth=2)
    ax.set_xlabel('Forecast Lead Time (hours)')
    ax.set_ylabel(f'Mean Global Ensemble Spread (σ)')
    ax.set_title(f'Ensemble Spread Growth — {variable}\n'
                 f'50-member FCN3 Ensemble')
    ax.grid(True, alpha=0.3)

    # Annotate key thresholds
    ax.axvline(x=72, color='gray', linestyle='--', alpha=0.5, label='Day 3')
    ax.axvline(x=120, color='gray', linestyle=':', alpha=0.5, label='Day 5')
    ax.axvline(x=240, color='gray', linestyle='-.', alpha=0.5, label='Day 10')
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

print("\nPlotting spread growth...")
plot_spread_growth(ds, save_path=os.path.join(OUTPUT_DIR, 'ex04_spread_growth.png'))


# ===========================================================================
# PART 5: Probability of exceedance
# ===========================================================================

def plot_probability_exceedance(ds, variable="t2m", threshold_c=35.0,
                                 step_idx=20, save_path=None):
    """Compute P(variable > threshold) from ensemble distribution.

    This is the real power of ensembles — probabilistic forecasting.
    Example: "What's the probability of temperature > 35°C in 5 days?"
    """
    field = ds.sel(variable=variable).isel(time=0, lead_time=step_idx)

    if variable == "t2m":
        field_c = field - 273.15
    else:
        field_c = field

    # Count members exceeding threshold at each grid point
    exceed_count = (field_c > threshold_c).sum(dim="ensemble").values.squeeze()
    n_members = len(ds.coords['ensemble'])
    prob = exceed_count / n_members * 100  # Percentage

    lats = ds.coords['lat'].values
    lons = ds.coords['lon'].values

    fig, ax = plt.subplots(figsize=(14, 6))
    cf = ax.contourf(lons, lats, prob,
                      levels=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                      cmap='YlOrRd')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Probability of T2m > {threshold_c}°C — T+{step_idx * 6}h\n'
                 f'{n_members}-member FCN3 Ensemble', fontsize=14)
    plt.colorbar(cf, ax=ax, label='Probability (%)')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

    print(f"\nProbability of T > {threshold_c}°C at T+{step_idx * 6}h:")
    print(f"  Max probability: {np.nanmax(prob):.0f}%")
    print(f"  Grid points with P > 50%: {np.sum(prob > 50)}")

print("\nComputing probability of extreme heat...")
plot_probability_exceedance(ds, threshold_c=35.0, step_idx=20,
    save_path=os.path.join(OUTPUT_DIR, 'ex04_prob_heat.png'))


# ===========================================================================
# PART 6: Spaghetti plot for a single location
# ===========================================================================

def plot_spaghetti(ds, variable="t2m", lat_target=40.7, lon_target=286.0,
                   save_path=None):
    """Spaghetti plot: all ensemble members for one location over time.

    This shows the "cone of uncertainty" — members diverge as the
    forecast extends further into the future.
    """
    # Find nearest grid point
    lats = ds.coords['lat'].values
    lons = ds.coords['lon'].values
    lat_idx = np.argmin(np.abs(lats - lat_target))
    lon_idx = np.argmin(np.abs(lons - lon_target))

    field = ds.sel(variable=variable).isel(time=0)
    point_data = field.values[:, :, lat_idx, lon_idx]  # (ensemble, lead_time)

    if variable == "t2m":
        point_data -= 273.15
        unit = "°C"
    else:
        unit = ""

    n_ensemble, n_steps = point_data.shape
    lead_hours = np.arange(n_steps) * 6

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each ensemble member as a thin line
    for m in range(n_ensemble):
        ax.plot(lead_hours, point_data[m], color='steelblue',
                alpha=0.15, linewidth=0.8)

    # Ensemble mean and spread
    mean = np.mean(point_data, axis=0)
    p10 = np.percentile(point_data, 10, axis=0)
    p90 = np.percentile(point_data, 90, axis=0)

    ax.plot(lead_hours, mean, 'r-', linewidth=2.5, label='Ensemble Mean')
    ax.fill_between(lead_hours, p10, p90, color='red', alpha=0.15,
                    label='10th-90th percentile')

    ax.set_xlabel('Forecast Lead Time (hours)')
    ax.set_ylabel(f'{variable} ({unit})')
    ax.set_title(f'Spaghetti Plot — {variable} at ({lats[lat_idx]:.1f}°N, '
                 f'{lons[lon_idx] % 360 - 360 if lons[lon_idx] > 180 else lons[lon_idx]:.1f}°)\n'
                 f'{n_ensemble}-member FCN3 Ensemble', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

# New York City (40.7°N, 74°W → 286°E)
print("\nPlotting spaghetti diagram for NYC...")
plot_spaghetti(ds, lat_target=40.7, lon_target=286.0,
    save_path=os.path.join(OUTPUT_DIR, 'ex04_spaghetti_nyc.png'))


# ===========================================================================
# EXERCISES
# ===========================================================================
"""
✅ Exercise 2.1a: Change noise_amplitude from 0.05 to 0.01 and 0.20.
   How does this affect spread? Too small = underconfident ensembles.
   Too large = unrealistic initial conditions.

✅ Exercise 2.1b: Where is the spread largest at T+120h? (Hint: look at
   midlatitude storm tracks and tropical convection regions.) Why do you
   think uncertainty concentrates there?

✅ Exercise 2.1c: Create a probability map for a different threshold:
   P(T2m < 0°C) at T+72h — where might frost be likely?

✅ Exercise 2.1d: Make spaghetti plots for 3 different cities on different
   continents. Do all locations show the same spread growth rate?

🔥 Challenge: Compute the Continuous Ranked Probability Score (CRPS) by
   comparing your ensemble against ERA5 verification data. CRPS is the
   gold-standard metric for probabilistic forecasts.
   Hint: from scipy.stats import rankdata
"""
