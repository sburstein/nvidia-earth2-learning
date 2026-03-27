"""
Exercise 2.3: Build a Custom Diagnostic Model
===============================================
Level: Intermediate | Time: 30 min | GPU: Optional

GOAL: Create custom diagnostics that compute derived weather variables from
      Earth2Studio model output. Learn the extension pattern that lets you
      plug your own calculations into any Earth2Studio workflow.

CONCEPTS:
  - Diagnostic models (dx): transform model output WITHOUT stepping forward
    in time — they compute derived quantities from existing fields
  - Prognostic (px) vs Diagnostic (dx): px predicts future, dx computes now
  - The input_coords / output_coords contract: how Earth2Studio knows what
    your diagnostic needs and produces
  - Chaining: you can stack diagnostics after any prognostic model

EXAMPLES OF USEFUL DIAGNOSTICS:
  - Wind speed from u/v components
  - Wind chill / heat index from temperature + humidity + wind
  - Precipitation type (rain vs snow) from temperature profile
  - Convective Available Potential Energy (CAPE) from soundings
  - Daily max/min temperature from 6-hourly output
"""

import os
import numpy as np
import torch
import xarray as xr
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')

# ===========================================================================
# PART 1: Wind Speed Diagnostic (simplest possible example)
# ===========================================================================

class WindSpeed:
    """Compute wind speed from u and v components.

    This is the simplest possible diagnostic — it takes two input fields
    and produces one output field: speed = sqrt(u² + v²).

    Earth2Studio diagnostics follow a simple contract:
    1. Declare what you need (input_coords)
    2. Declare what you produce (output_coords)
    3. Implement __call__ to do the computation
    """

    def __init__(self, level="10m"):
        """
        Args:
            level: "10m" for surface wind, or a pressure level like "850"
        """
        self.level = level
        if level == "10m":
            self.u_var = "u10m"
            self.v_var = "v10m"
            self.out_var = "wspd10m"
        else:
            self.u_var = f"u{level}"
            self.v_var = f"v{level}"
            self.out_var = f"wspd{level}"

    @property
    def input_coords(self):
        """What variables this diagnostic needs as input."""
        return {"variable": np.array([self.u_var, self.v_var])}

    @property
    def output_coords(self):
        """What variables this diagnostic produces."""
        return {"variable": np.array([self.out_var])}

    def __call__(self, x, coords):
        """Compute wind speed.

        Args:
            x: tensor of shape (..., 2, lat, lon) — the two input variables
            coords: coordinate metadata dict

        Returns:
            (output_tensor, output_coords)
        """
        # x[..., 0, :, :] = u component, x[..., 1, :, :] = v component
        if isinstance(x, torch.Tensor):
            speed = torch.sqrt(x[..., 0:1, :, :] ** 2 + x[..., 1:2, :, :] ** 2)
        else:
            speed = np.sqrt(x[..., 0:1, :, :] ** 2 + x[..., 1:2, :, :] ** 2)

        out_coords = dict(coords)
        out_coords["variable"] = np.array([self.out_var])
        return speed, out_coords


# ===========================================================================
# PART 2: Heat Index Diagnostic (more complex)
# ===========================================================================

class HeatIndex:
    """Compute heat index from temperature and relative humidity.

    The heat index (or "feels like" temperature) accounts for how humidity
    makes hot temperatures feel even hotter. This is critical for public
    health heat warnings.

    Uses the Rothfusz regression equation (same as US NWS).
    """

    @property
    def input_coords(self):
        return {"variable": np.array(["t2m", "r2"])}  # temperature + relative humidity

    @property
    def output_coords(self):
        return {"variable": np.array(["heat_index"])}

    def __call__(self, x, coords):
        if isinstance(x, torch.Tensor):
            t_k = x[..., 0:1, :, :]
            rh = x[..., 1:2, :, :]
            # Convert Kelvin to Fahrenheit for the Rothfusz equation
            t_f = (t_k - 273.15) * 9/5 + 32
        else:
            t_k = x[..., 0:1, :, :]
            rh = x[..., 1:2, :, :]
            t_f = (t_k - 273.15) * 9/5 + 32

        # Rothfusz regression
        hi = (-42.379
              + 2.04901523 * t_f
              + 10.14333127 * rh
              - 0.22475541 * t_f * rh
              - 6.83783e-3 * t_f**2
              - 5.481717e-2 * rh**2
              + 1.22874e-3 * t_f**2 * rh
              + 8.5282e-4 * t_f * rh**2
              - 1.99e-6 * t_f**2 * rh**2)

        # Convert back to Celsius
        hi_c = (hi - 32) * 5/9

        # Only apply heat index when T > 27°C (80°F), else use actual temp
        t_c = t_k - 273.15
        if isinstance(x, torch.Tensor):
            result = torch.where(t_c > 27, hi_c, t_c)
        else:
            result = np.where(t_c > 27, hi_c, t_c)

        out_coords = dict(coords)
        out_coords["variable"] = np.array(["heat_index"])
        return result, out_coords


# ===========================================================================
# PART 3: Daily Max/Min Temperature Diagnostic
# ===========================================================================

class DailyExtremes:
    """Compute daily max and min temperature from 6-hourly model output.

    Most weather models output every 6 hours, but many users need daily
    max/min values (agriculture, energy, health).

    This diagnostic takes 4 consecutive time steps (= 24 hours) and
    computes the extremes.
    """

    @property
    def input_coords(self):
        return {"variable": np.array(["t2m"])}

    @property
    def output_coords(self):
        return {"variable": np.array(["t2m_max", "t2m_min", "t2m_range"])}

    def compute_from_steps(self, data_4steps):
        """
        Args:
            data_4steps: array of shape (4, lat, lon) — four 6-hourly temps

        Returns:
            dict with max, min, and diurnal range
        """
        t_max = np.max(data_4steps, axis=0)
        t_min = np.min(data_4steps, axis=0)
        t_range = t_max - t_min

        return {
            "t2m_max": t_max - 273.15,
            "t2m_min": t_min - 273.15,
            "t2m_range": t_range  # Already a difference, units are K = °C
        }


# ===========================================================================
# PART 4: Apply diagnostics to a forecast
# ===========================================================================

def apply_diagnostics_to_forecast(forecast_path):
    """Load a saved forecast and compute derived variables."""

    ds = xr.open_zarr(forecast_path)
    print(f"Loaded forecast: {dict(ds.dims)}")

    lats = ds.coords['lat'].values
    lons = ds.coords['lon'].values

    # --- Wind Speed ---
    print("\nComputing 10m wind speed...")
    ws_diag = WindSpeed(level="10m")

    step = 8  # T+48h
    u = ds.sel(variable="u10m").isel(time=0, lead_time=step).values.squeeze()
    v = ds.sel(variable="v10m").isel(time=0, lead_time=step).values.squeeze()

    speed = np.sqrt(u**2 + v**2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    cf0 = axes[0].contourf(lons, lats, u, levels=30, cmap='coolwarm')
    axes[0].set_title('U-wind (m/s)')
    plt.colorbar(cf0, ax=axes[0])

    cf1 = axes[1].contourf(lons, lats, v, levels=30, cmap='coolwarm')
    axes[1].set_title('V-wind (m/s)')
    plt.colorbar(cf1, ax=axes[1])

    cf2 = axes[2].contourf(lons, lats, speed, levels=np.arange(0, 25, 1),
                            cmap='YlOrRd')
    axes[2].set_title('Wind Speed (m/s)\nDerived: √(u² + v²)')
    plt.colorbar(cf2, ax=axes[2])

    fig.suptitle(f'Wind Speed Diagnostic — T+{step*6}h', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ex06_wind_speed.png'),
                dpi=150, bbox_inches='tight')
    print(f"Saved: ex06_wind_speed.png")
    plt.show()

    # --- Daily Max/Min ---
    if len(ds.coords['lead_time']) >= 5:
        print("\nComputing daily temperature extremes (Day 2)...")
        daily_diag = DailyExtremes()

        # Day 2 = steps 4-7 (T+24h to T+42h)
        t2m_4steps = np.array([
            ds.sel(variable="t2m").isel(time=0, lead_time=i).values.squeeze()
            for i in range(4, 8)
        ])

        extremes = daily_diag.compute_from_steps(t2m_4steps)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        cf0 = axes[0].contourf(lons, lats, extremes["t2m_max"],
                                levels=np.arange(-30, 50, 5), cmap='RdYlBu_r')
        axes[0].set_title('Daily Max Temperature (°C)')
        plt.colorbar(cf0, ax=axes[0])

        cf1 = axes[1].contourf(lons, lats, extremes["t2m_min"],
                                levels=np.arange(-30, 50, 5), cmap='RdYlBu_r')
        axes[1].set_title('Daily Min Temperature (°C)')
        plt.colorbar(cf1, ax=axes[1])

        cf2 = axes[2].contourf(lons, lats, extremes["t2m_range"],
                                levels=np.arange(0, 20, 1), cmap='YlOrRd')
        axes[2].set_title('Diurnal Range (°C)\n(Max − Min)')
        plt.colorbar(cf2, ax=axes[2])

        fig.suptitle('Daily Temperature Extremes — Day 2 Forecast', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'ex06_daily_extremes.png'),
                    dpi=150, bbox_inches='tight')
        print(f"Saved: ex06_daily_extremes.png")
        plt.show()


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    forecast_path = os.path.join(OUTPUT_DIR, 'ex01_forecast.zarr')
    if os.path.exists(forecast_path):
        apply_diagnostics_to_forecast(forecast_path)
    else:
        print(f"Forecast data not found at: {forecast_path}")
        print("Run ex01_first_forecast.py first!")


# ===========================================================================
# EXERCISES
# ===========================================================================
"""
✅ Exercise 2.3a: Implement a wind direction diagnostic. Wind direction is
   computed as: direction = atan2(-u, -v) * 180/π + 180 (meteorological
   convention: direction wind is coming FROM).

✅ Exercise 2.3b: Create a "frost risk" diagnostic that outputs 1 where
   t2m < 273.15K and 0 elsewhere. Plot the frost line for a winter case.

✅ Exercise 2.3c: Implement a relative vorticity diagnostic at 500 hPa.
   Vorticity ≈ ∂v/∂x − ∂u/∂y. Use np.gradient() for the derivatives.
   Where do you see the strongest vorticity? (Hint: cyclone centers.)

✅ Exercise 2.3d: Create a "precipitation type" diagnostic:
   If T850 > 0°C → rain, if T850 < -5°C → snow, else → mixed.
   (This is a crude approximation — real precipitation type depends on
   the entire temperature profile, not just one level.)

🔥 Challenge: Register your diagnostic as a proper Earth2Studio DiagnosticModel
   subclass so it can be used in the run.diagnostic() workflow alongside
   prognostic models. Check the Earth2Studio docs for the DiagnosticModel ABC.
"""
