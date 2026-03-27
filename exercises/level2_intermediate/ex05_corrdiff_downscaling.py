"""
Exercise 2.2: CorrDiff Regional Downscaling Pipeline
=====================================================
Level: Intermediate | Time: 60 min | GPU: Required (8GB+ VRAM)

GOAL: Take a coarse global forecast (25km) and downscale it to 2-3km regional
      resolution using CorrDiff — NVIDIA's generative diffusion downscaling model.
      This is the "last-mile" problem in weather forecasting.

CONCEPTS:
  - Downscaling: converting coarse global model output to actionable local detail
  - CorrDiff's two-stage architecture:
    1. Regression UNet: predicts conditional mean (deterministic baseline)
    2. Diffusion model: predicts residual corrections (captures uncertainty)
  - Stochastic samples: each CorrDiff run produces a different plausible
    fine-resolution realization — this IS the uncertainty quantification
  - NIM deployment: containerized microservices for production use

WHY DOWNSCALING MATTERS:
  A 25km grid can't resolve individual thunderstorms, urban heat islands,
  or mountain valley winds. CorrDiff bridges this gap 22x faster and 1,300x
  more energy efficiently than traditional dynamical downscaling (WRF).

TWO APPROACHES SHOWN:
  A) Self-hosted Docker NIM (requires NVIDIA GPU + NGC API key)
  B) Cloud API via build.nvidia.com (no local GPU needed)
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')

# ===========================================================================
# APPROACH A: CorrDiff NIM via Docker (Self-Hosted)
# ===========================================================================
"""
SETUP (run in terminal before this script):

1. Get an NGC API key from https://ngc.nvidia.com/
   export NGC_API_KEY="your-key-here"

2. Pull and run the CorrDiff NIM container:
   docker pull nvcr.io/nim/nvidia/corrdiff:1.1.0
   docker run --rm --runtime=nvidia --gpus all --shm-size 4g \
       -p 8000:8000 \
       -e NGC_API_KEY=$NGC_API_KEY \
       nvcr.io/nim/nvidia/corrdiff:1.1.0

3. Wait for "Application startup complete" in the Docker logs.
"""

import requests

def check_nim_health(base_url="http://localhost:8000"):
    """Check if the CorrDiff NIM is running and ready."""
    try:
        resp = requests.get(f"{base_url}/v1/health/ready", timeout=5)
        if resp.status_code == 200 and resp.json().get("status") == "ready":
            print("CorrDiff NIM is ready!")
            return True
    except requests.ConnectionError:
        pass
    print("CorrDiff NIM not running. See SETUP instructions above.")
    print("Alternatively, use APPROACH B (cloud API) below.")
    return False


def run_corrdiff_nim(input_array_path, n_samples=4, n_steps=14, seed=42,
                     base_url="http://localhost:8000"):
    """Run CorrDiff inference via the local NIM container.

    Args:
        input_array_path: Path to .npy file with coarse input data
            Shape: (1, 38, lat, lon) — 38-channel GEFS input
        n_samples: Number of stochastic realizations to generate
        n_steps: Number of diffusion steps (more = higher quality, slower)
        seed: Random seed for reproducibility

    Returns:
        List of numpy arrays, one per stochastic sample
    """
    url = f"{base_url}/v1/infer"

    with open(input_array_path, 'rb') as f:
        files = {"input_array": ("input_array.npy", f)}
        params = {"samples": n_samples, "steps": n_steps, "seed": seed}
        response = requests.post(url, data=params, files=files, timeout=600)

    if response.status_code != 200:
        raise RuntimeError(f"CorrDiff NIM error: {response.status_code} — {response.text}")

    # Response is a .tar archive of numpy arrays
    import tarfile
    import io

    tar_buffer = io.BytesIO(response.content)
    samples = []
    with tarfile.open(fileobj=tar_buffer) as tar:
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f and member.name.endswith('.npy'):
                arr = np.load(io.BytesIO(f.read()))
                samples.append(arr)

    print(f"Received {len(samples)} downscaled samples")
    return samples


# ===========================================================================
# APPROACH B: CorrDiff via build.nvidia.com Cloud API
# ===========================================================================

def run_corrdiff_cloud(input_data, api_key=None):
    """Run CorrDiff via NVIDIA's cloud API (no local GPU needed).

    Get an API key from: https://build.nvidia.com/nvidia/corrdiff

    This endpoint handles everything — you just send the coarse data
    and get back fine-resolution output.
    """
    if api_key is None:
        api_key = os.environ.get("NGC_API_KEY", "")
        if not api_key:
            print("Set NGC_API_KEY environment variable or pass api_key parameter.")
            print("Get a free key at: https://build.nvidia.com/nvidia/corrdiff")
            return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # The cloud API may have a different interface — check current docs at
    # https://docs.nvidia.com/nim/earth-2/corrdiff/latest/
    print("Note: Cloud API interface may vary. Check NVIDIA docs for current format.")
    return None


# ===========================================================================
# PART 2: Prepare input data using Earth2Studio
# ===========================================================================

def prepare_corrdiff_input(init_time="2025-06-01T00:00:00",
                           save_path=None):
    """Fetch and prepare coarse-resolution input data for CorrDiff.

    CorrDiff expects input from a coarse global model (GEFS or GFS).
    Earth2Studio can fetch this data automatically.
    """
    from earth2studio.data import GFS

    data = GFS()

    # CorrDiff input channels vary by variant:
    # - CONUS variant: 38 channels from GEFS (sfc + pressure levels)
    # - Taiwan variant: different channel configuration
    # Check the NIM docs for your specific variant's requirements.

    print(f"Fetching GFS data for {init_time}...")

    # For demonstration, fetch standard variables that CorrDiff uses
    variables = [
        "t2m", "u10m", "v10m", "msl",        # Surface variables
        "t850", "u850", "v850", "z850",        # 850 hPa
        "t500", "u500", "v500", "z500",        # 500 hPa
        "t250", "u250", "v250", "z250",        # 250 hPa
    ]

    print(f"Input variables: {len(variables)} channels")
    print("Note: Actual CorrDiff NIM expects specific channel ordering.")
    print("      See the NIM quickstart guide for exact input format.")

    if save_path:
        print(f"Would save prepared input to: {save_path}")

    return variables


# ===========================================================================
# PART 3: Visualize downscaled output
# ===========================================================================

def visualize_downscaling(coarse, fine_samples, lats_c, lons_c,
                          lats_f, lons_f, variable="precipitation",
                          save_path=None):
    """Compare coarse input vs fine-resolution CorrDiff output.

    Shows the dramatic resolution improvement from 25km → 2-3km.
    Multiple stochastic samples show the range of plausible outcomes.
    """
    n_samples = len(fine_samples)
    fig, axes = plt.subplots(1, n_samples + 1, figsize=(5 * (n_samples + 1), 6))

    # Coarse input
    cf0 = axes[0].contourf(lons_c, lats_c, coarse, levels=30, cmap='Blues')
    axes[0].set_title(f'Coarse Input (~25km)\n{variable}')
    plt.colorbar(cf0, ax=axes[0])

    # Fine samples
    for i, sample in enumerate(fine_samples):
        cf = axes[i+1].contourf(lons_f, lats_f, sample, levels=30, cmap='Blues')
        axes[i+1].set_title(f'CorrDiff Sample {i+1} (~3km)')
        plt.colorbar(cf, ax=axes[i+1])

    for ax in axes:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    fig.suptitle(f'CorrDiff Downscaling: 25km → 3km Resolution\n'
                 f'{n_samples} Stochastic Realizations', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


# ===========================================================================
# PART 4: Synthetic demo (works without NIM running)
# ===========================================================================

def synthetic_downscaling_demo():
    """Demonstrate the concept of CorrDiff with synthetic data.

    This runs without the actual NIM container — useful for understanding
    the two-stage regression + diffusion architecture.
    """
    print("\n" + "=" * 60)
    print("SYNTHETIC CORRDIFF DEMONSTRATION")
    print("=" * 60)
    print("(Using synthetic data — actual CorrDiff requires NIM container)")

    np.random.seed(42)

    # Create synthetic coarse field (25km, ~40x40 grid)
    n_coarse = 40
    x_c = np.linspace(0, 10, n_coarse)
    y_c = np.linspace(0, 10, n_coarse)
    X_c, Y_c = np.meshgrid(x_c, y_c)
    coarse = (np.sin(X_c) * np.cos(Y_c) * 10 +
              np.random.normal(0, 0.5, (n_coarse, n_coarse)))

    # Create synthetic fine field (3km, ~333x333 grid)
    scale_factor = 8  # 25km / 3km ≈ 8x
    n_fine = n_coarse * scale_factor
    x_f = np.linspace(0, 10, n_fine)
    y_f = np.linspace(0, 10, n_fine)
    X_f, Y_f = np.meshgrid(x_f, y_f)

    # Stage 1: Regression UNet output (smooth interpolation)
    from scipy.ndimage import zoom
    regression_output = zoom(coarse, scale_factor, order=3)

    # Stage 2: Diffusion residual (adds fine-scale detail)
    # In real CorrDiff, this comes from a trained diffusion model
    n_samples = 4
    fine_samples = []
    for i in range(n_samples):
        # Each sample has the same large-scale structure but different details
        noise = np.random.normal(0, 1, (n_fine, n_fine))
        # High-frequency detail from diffusion
        fine_detail = np.sin(X_f * 5) * np.cos(Y_f * 7) * 0.3 + noise * 0.1
        fine_samples.append(regression_output + fine_detail)

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Architecture
    cf0 = axes[0, 0].contourf(X_c, Y_c, coarse, levels=30, cmap='viridis')
    axes[0, 0].set_title('Input: Coarse (~25km)')
    plt.colorbar(cf0, ax=axes[0, 0])

    cf1 = axes[0, 1].contourf(X_f, Y_f, regression_output, levels=30, cmap='viridis')
    axes[0, 1].set_title('Stage 1: Regression UNet\n(smooth upscaling)')
    plt.colorbar(cf1, ax=axes[0, 1])

    residual = fine_samples[0] - regression_output
    cf2 = axes[0, 2].contourf(X_f, Y_f, residual, levels=30, cmap='coolwarm')
    axes[0, 2].set_title('Stage 2: Diffusion Residual\n(fine-scale corrections)')
    plt.colorbar(cf2, ax=axes[0, 2])

    # Row 2: Multiple stochastic samples
    for i in range(3):
        cf = axes[1, i].contourf(X_f, Y_f, fine_samples[i], levels=30, cmap='viridis')
        axes[1, i].set_title(f'Final Output: Sample {i+1} (~3km)\n'
                             f'(regression + diffusion)')
        plt.colorbar(cf, ax=axes[1, i])

    fig.suptitle('CorrDiff Architecture: Two-Stage Downscaling\n'
                 'Regression (deterministic) + Diffusion (stochastic)',
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ex05_corrdiff_demo.png'),
                dpi=150, bbox_inches='tight')
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'ex05_corrdiff_demo.png')}")
    plt.show()

    # Show how samples differ
    fig, ax = plt.subplots(figsize=(10, 6))
    sample_std = np.std(fine_samples, axis=0)
    cf = ax.contourf(X_f, Y_f, sample_std, levels=30, cmap='YlOrRd')
    ax.set_title('Inter-Sample Variability (std across 4 samples)\n'
                 'High values = regions where fine-scale outcome is uncertain')
    plt.colorbar(cf, ax=ax, label='Standard Deviation')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ex05_corrdiff_uncertainty.png'),
                dpi=150, bbox_inches='tight')
    plt.show()

    print("\nKey takeaway: Each stochastic sample is equally plausible.")
    print("The regression UNet provides the 'best guess', and the diffusion")
    print("model adds realistic fine-scale detail that's different each time.")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    # Try actual NIM first, fall back to synthetic demo
    if check_nim_health():
        print("\nNIM is running! Preparing input data...")
        prepare_corrdiff_input()
        # If you have prepared input data:
        # samples = run_corrdiff_nim("corrdiff_inputs.npy", n_samples=4)
    else:
        print("\nRunning synthetic demo instead...")

    synthetic_downscaling_demo()


# ===========================================================================
# EXERCISES
# ===========================================================================
"""
✅ Exercise 2.2a: Run the synthetic demo and study the difference between
   Stage 1 (regression) and Stage 2 (diffusion). Which stage adds the
   fine-scale structure?

✅ Exercise 2.2b: If you have Docker + GPU, pull the CorrDiff NIM and
   run it end-to-end:
   docker pull nvcr.io/nim/nvidia/corrdiff:1.1.0
   docker run --rm --runtime=nvidia --gpus all --shm-size 4g \
       -p 8000:8000 -e NGC_API_KEY=$NGC_API_KEY \
       nvcr.io/nim/nvidia/corrdiff:1.1.0

✅ Exercise 2.2c: Experiment with the number of diffusion steps (n_steps).
   Default is 14. Try 7 vs 28 vs 56. How does quality vs speed trade off?

✅ Exercise 2.2d: Generate 20 stochastic samples and compute the
   pixel-wise standard deviation map. This IS CorrDiff's uncertainty
   estimate — where does it disagree most with itself?

🔥 Challenge: Chain a global FCN3 forecast (Exercise 2.1) → CorrDiff
   downscaling to create a full end-to-end pipeline: global AI forecast
   → regional km-scale detail. This is the production architecture.
"""
