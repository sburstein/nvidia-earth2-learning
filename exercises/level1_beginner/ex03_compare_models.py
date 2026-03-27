"""
Exercise 1.3: Compare Two AI Weather Models
============================================
Level: Beginner | Time: 30 min | GPU: Optional (CPU works, just slower)

GOAL: Run the same forecast with FCN3 and DLWP, then compare their predictions.
      This teaches you that different architectures make different tradeoffs.

CONCEPTS:
  - Model intercomparison: same initial conditions, different models
  - RMSE (Root Mean Square Error): standard weather forecast metric
  - Bias: systematic over/under-prediction
  - Spectral analysis: do models preserve small-scale weather features?

KEY INSIGHT:
  FCN3 (wavelet convolutions, 0.25°) vs DLWP (cubesphere conv, ~2°) have very
  different resolutions and architectures. FCN3 will be sharper but DLWP might
  be more stable for very long rollouts.
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime

# ===========================================================================
# PART 1: Run forecasts with both models
# ===========================================================================

from earth2studio.models.px import FCN3, DLWP
from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio import run

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')
INIT_TIME = "2025-06-01T00:00:00"
NSTEPS = 20  # 5 days

# --- Run FCN3 ---
print("=" * 60)
print("Running FCN3 forecast...")
print("=" * 60)
fcn3_path = os.path.join(OUTPUT_DIR, 'ex03_fcn3.zarr')
package_fcn3 = FCN3.load_default_package()
model_fcn3 = FCN3.load_model(package_fcn3)

io_fcn3 = run.deterministic(
    time=[INIT_TIME],
    nsteps=NSTEPS,
    model=model_fcn3,
    data=GFS(),
    io=ZarrBackend(fcn3_path)
)
print(f"FCN3 complete → {fcn3_path}")

# --- Run DLWP ---
print("\n" + "=" * 60)
print("Running DLWP forecast...")
print("=" * 60)
dlwp_path = os.path.join(OUTPUT_DIR, 'ex03_dlwp.zarr')
package_dlwp = DLWP.load_default_package()
model_dlwp = DLWP.load_model(package_dlwp)

io_dlwp = run.deterministic(
    time=[INIT_TIME],
    nsteps=NSTEPS,
    model=model_dlwp,
    data=GFS(),
    io=ZarrBackend(dlwp_path)
)
print(f"DLWP complete → {dlwp_path}")

# ===========================================================================
# PART 2: Load and compare
# ===========================================================================

ds_fcn3 = xr.open_zarr(fcn3_path)
ds_dlwp = xr.open_zarr(dlwp_path)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(f"\nFCN3 grid: {len(ds_fcn3.coords['lat'])} × {len(ds_fcn3.coords['lon'])}")
print(f"DLWP grid: {len(ds_dlwp.coords['lat'])} × {len(ds_dlwp.coords['lon'])}")
print(f"FCN3 variables: {len(ds_fcn3.coords['variable'])}")
print(f"DLWP variables: {len(ds_dlwp.coords['variable'])}")

# Find common variables
fcn3_vars = set(ds_fcn3.coords['variable'].values)
dlwp_vars = set(ds_dlwp.coords['variable'].values)
common_vars = sorted(fcn3_vars & dlwp_vars)
print(f"\nCommon variables ({len(common_vars)}): {common_vars[:10]}...")


# ===========================================================================
# PART 3: Compute difference maps
# ===========================================================================

def compute_and_plot_difference(ds1, ds2, variable, step_idx,
                                 label1="FCN3", label2="DLWP",
                                 save_path=None):
    """Compare two model forecasts for the same variable and time step.

    Note: If models have different grids, we interpolate DLWP to FCN3's grid.
    """
    field1 = ds1.sel(variable=variable).isel(time=0, lead_time=step_idx).values.squeeze()
    field2_raw = ds2.sel(variable=variable).isel(time=0, lead_time=step_idx).values.squeeze()

    lats1 = ds1.coords['lat'].values
    lons1 = ds1.coords['lon'].values

    # Interpolate field2 to field1's grid if different resolution
    if field1.shape != field2_raw.shape:
        from scipy.interpolate import RegularGridInterpolator
        lats2 = ds2.coords['lat'].values
        lons2 = ds2.coords['lon'].values
        interp = RegularGridInterpolator((lats2, lons2), field2_raw,
                                          method='linear', bounds_error=False,
                                          fill_value=np.nan)
        lat_grid, lon_grid = np.meshgrid(lats1, lons1, indexing='ij')
        field2 = interp((lat_grid, lon_grid))
    else:
        field2 = field2_raw

    diff = field1 - field2

    # Unit conversion
    if variable == "t2m":
        field1 -= 273.15
        field2 -= 273.15
        unit = "°C"
        diff_unit = "°C"
    elif "z" in variable:
        field1 /= (9.81 * 10)
        field2 /= (9.81 * 10)
        diff /= (9.81 * 10)
        unit = "dam"
        diff_unit = "dam"
    else:
        unit = ""
        diff_unit = ""

    # Stats
    rmse = np.sqrt(np.nanmean(diff**2))
    bias = np.nanmean(diff)
    max_diff = np.nanmax(np.abs(diff))

    print(f"\n{variable} at T+{step_idx * 6}h:")
    print(f"  RMSE({label1} - {label2}): {rmse:.3f} {diff_unit}")
    print(f"  Bias({label1} - {label2}): {bias:.3f} {diff_unit}")
    print(f"  Max absolute diff: {max_diff:.3f} {diff_unit}")

    # Plot: Model 1 | Model 2 | Difference
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    vmin = min(np.nanpercentile(field1, 2), np.nanpercentile(field2, 2))
    vmax = max(np.nanpercentile(field1, 98), np.nanpercentile(field2, 98))

    cf1 = axes[0].contourf(lons1, lats1, field1, levels=30,
                            cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'{label1}: {variable} ({unit})')
    plt.colorbar(cf1, ax=axes[0])

    cf2 = axes[1].contourf(lons1, lats1, field2, levels=30,
                            cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'{label2}: {variable} ({unit})')
    plt.colorbar(cf2, ax=axes[1])

    # Difference with symmetric colorbar
    diff_max = np.nanpercentile(np.abs(diff), 95)
    cf3 = axes[2].contourf(lons1, lats1, diff, levels=30,
                            cmap='coolwarm', vmin=-diff_max, vmax=diff_max)
    axes[2].set_title(f'Difference ({label1} − {label2})\n'
                      f'RMSE={rmse:.2f} {diff_unit}')
    plt.colorbar(cf3, ax=axes[2])

    for ax in axes:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    fig.suptitle(f'{variable} Model Comparison — T+{step_idx * 6}h',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()

    return rmse, bias


# Compare t2m at multiple lead times
print("\n" + "=" * 60)
print("COMPARING 2m Temperature")
print("=" * 60)

if "t2m" in common_vars:
    for step in [4, 8, 12, 16, 20]:
        if step < len(ds_fcn3.coords['lead_time']) and step < len(ds_dlwp.coords['lead_time']):
            compute_and_plot_difference(
                ds_fcn3, ds_dlwp, "t2m", step,
                save_path=os.path.join(OUTPUT_DIR, f'ex03_t2m_diff_T{step*6}h.png')
            )

# ===========================================================================
# PART 4: RMSE growth over time
# ===========================================================================

def plot_rmse_growth(ds1, ds2, variable, label1="FCN3", label2="DLWP",
                     save_path=None):
    """Plot how model differences grow with forecast lead time.

    This is key to understanding error growth — in chaotic systems,
    small initial differences amplify exponentially.
    """
    max_steps = min(len(ds1.coords['lead_time']), len(ds2.coords['lead_time']))
    rmses = []
    lead_hours = []

    for step in range(max_steps):
        f1 = ds1.sel(variable=variable).isel(time=0, lead_time=step).values.squeeze()
        f2_raw = ds2.sel(variable=variable).isel(time=0, lead_time=step).values.squeeze()

        # Interpolate if needed
        if f1.shape != f2_raw.shape:
            from scipy.interpolate import RegularGridInterpolator
            lats2 = ds2.coords['lat'].values
            lons2 = ds2.coords['lon'].values
            lats1 = ds1.coords['lat'].values
            lons1 = ds1.coords['lon'].values
            interp = RegularGridInterpolator((lats2, lons2), f2_raw,
                                              method='linear', bounds_error=False,
                                              fill_value=np.nan)
            lat_grid, lon_grid = np.meshgrid(lats1, lons1, indexing='ij')
            f2 = interp((lat_grid, lon_grid))
        else:
            f2 = f2_raw

        rmse = np.sqrt(np.nanmean((f1 - f2)**2))
        rmses.append(rmse)
        lead_hours.append(step * 6)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lead_hours, rmses, 'b-o', markersize=4, linewidth=2)
    ax.set_xlabel('Forecast Lead Time (hours)')
    ax.set_ylabel(f'RMSE ({variable})')
    ax.set_title(f'Model Divergence Over Time: {label1} vs {label2}\n'
                 f'Variable: {variable}')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=rmses[0], color='gray', linestyle='--', alpha=0.5,
               label=f'Initial diff: {rmses[0]:.3f}')
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

if "t2m" in common_vars:
    print("\n" + "=" * 60)
    print("RMSE GROWTH OVER LEAD TIME")
    print("=" * 60)
    plot_rmse_growth(ds_fcn3, ds_dlwp, "t2m",
                     save_path=os.path.join(OUTPUT_DIR, 'ex03_rmse_growth.png'))


# ===========================================================================
# EXERCISES
# ===========================================================================
"""
✅ Exercise 1.3a: Which model produces smoother fields? Look at the
   difference map — if DLWP is smoother, the difference map will show
   FCN3 having more small-scale detail.

✅ Exercise 1.3b: Does the RMSE growth look exponential? This is expected
   for chaotic systems — it's called the "Lorenz curve" of predictability.

✅ Exercise 1.3c: Compare z500 instead of t2m. Is the divergence faster
   or slower for upper-level fields vs surface?

✅ Exercise 1.3d: Look at the bias (mean difference). Does one model
   consistently predict warmer or cooler than the other?

🔥 Challenge: Add a third model (Pangu-Weather or FCNv2/SFNO) and create
   a 3-way comparison. Which model is the "odd one out"?
"""
