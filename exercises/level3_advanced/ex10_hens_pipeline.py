"""
Exercise 3.3: Build a HENS (Huge Ensemble) Pipeline
=====================================================
Level: Advanced | Time: 2-6 hours | GPU: Multi-GPU (4+ A100s recommended)

GOAL: Generate 1,000+ member ensembles using SFNO with bred vectors.
      HENS (Huge Ensembles) is NVIDIA's approach to generating orders of
      magnitude more ensemble members than traditional NWP can produce.

CONCEPTS:
  - Bred vectors: a technique to generate initial condition perturbations
    that grow along the most unstable modes of the atmosphere
  - Multi-checkpoint ensembles: different model checkpoints act as different
    "physics" — combining them increases ensemble diversity
  - Spread-skill relationship: a well-calibrated ensemble should have spread
    proportional to actual forecast error
  - Tail risk: rare but high-impact events (the reason we need 1000+ members)

WHY 1,000+ MEMBERS:
  Traditional NWP runs 50-member ensembles (expensive). But for tail risk
  estimation (e.g., P(wind > 150 mph)), 50 members can't reliably estimate
  probabilities below ~2%. With 1,000+ members, you can estimate events
  with 0.1% probability — critical for insurance, infrastructure planning,
  and emergency management.

  AXA uses HENS to generate thousands of hypothetical hurricane scenarios.

REFERENCES:
  - "Huge Ensembles Part I" (EGUsphere preprint, 2024)
  - "Huge Ensembles Part II" (GMD, 2025)
  - Earth2Studio HENS recipe: earth2studio/recipes/hens/
"""

import os
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')

# ===========================================================================
# STEP 1: Understanding Bred Vectors
# ===========================================================================

BRED_VECTORS_EXPLANATION = """
============================================================
BRED VECTORS: SMARTER PERTURBATIONS
============================================================

Why not just add random noise to initial conditions?

Random noise → most energy dissipates quickly (not dynamically relevant)
Bred vectors → perturbations grow along the atmosphere's most unstable modes

The breeding process:
1. Run the model forward from a control analysis
2. Add a small random perturbation
3. Run the model forward from the perturbed state
4. After N steps, compute the difference (perturbation growth)
5. Rescale the difference to the original perturbation amplitude
6. Repeat from step 3

After several breeding cycles, the perturbation has "bred" onto the
fastest-growing instabilities — these are the directions where the
atmosphere is most sensitive to small changes.

This is much more physically meaningful than random Gaussian noise.
"""

print(BRED_VECTORS_EXPLANATION)


# ===========================================================================
# STEP 2: HENS Pipeline Commands
# ===========================================================================

HENS_COMMANDS = """
============================================================
HENS PIPELINE (Earth2Studio)
============================================================

# Step 1: Install Earth2Studio with HENS recipe
pip install "earth2studio[all]"

# Step 2: Navigate to HENS recipe
cd earth2studio/recipes/hens/

# Step 3: Run HENS (the recipe handles everything)
python hens_notebook.py

# The recipe generates:
# - Multiple SFNO checkpoints (different training seeds)
# - Bred vector initial conditions
# - 1,000+ member ensemble forecasts
# - Statistical analysis (spread, calibration)

============================================================
CUSTOM HENS PIPELINE
============================================================

# If you want to build your own HENS pipeline:

from earth2studio.models.px import SFNO
from earth2studio.data import GFS, ARCO
from earth2studio.io import ZarrBackend
from earth2studio.perturbation import BredVector  # If available
from earth2studio import run

# Load multiple SFNO checkpoints for multi-physics diversity
checkpoints = [
    "sfno_checkpoint_seed0.pth",
    "sfno_checkpoint_seed1.pth",
    "sfno_checkpoint_seed2.pth",
    "sfno_checkpoint_seed3.pth",
]

# Strategy: 250 members per checkpoint × 4 checkpoints = 1,000 members
members_per_checkpoint = 250
total_members = members_per_checkpoint * len(checkpoints)

print(f"Generating {total_members}-member HENS ensemble...")

for ckpt_idx, ckpt_path in enumerate(checkpoints):
    print(f"  Checkpoint {ckpt_idx + 1}/{len(checkpoints)}: {ckpt_path}")

    # Load model with this checkpoint
    # model = SFNO.load_model(ckpt_path)

    # Run ensemble batch
    # io = run.ensemble(
    #     time=["2025-09-15T00:00:00"],  # Hurricane season
    #     nsteps=40,  # 10 days
    #     nensemble=members_per_checkpoint,
    #     model=model,
    #     data=GFS(),
    #     io=ZarrBackend(f"hens_ckpt{ckpt_idx}.zarr"),
    #     perturbation=BredVector(breeding_cycles=6)
    # )
"""

print(HENS_COMMANDS)


# ===========================================================================
# STEP 3: Analyzing HENS output
# ===========================================================================

def demonstrate_hens_analysis():
    """Show how to analyze a HENS ensemble for tail risk."""

    print("\n" + "=" * 60)
    print("HENS ANALYSIS DEMONSTRATION (Synthetic)")
    print("=" * 60)

    import matplotlib.pyplot as plt

    np.random.seed(42)
    n_members = 1000

    # Simulate 1000-member wind speed ensemble at a coastal location
    # True distribution: GEV (generalized extreme value) — appropriate for maxima
    from scipy.stats import genextreme
    wind_samples = genextreme.rvs(c=-0.1, loc=30, scale=10, size=n_members)

    # Compare: what you'd see with 50 members vs 1000
    wind_50 = wind_samples[:50]
    wind_1000 = wind_samples

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Histogram comparison
    bins = np.arange(0, 80, 2)
    axes[0].hist(wind_50, bins=bins, alpha=0.7, density=True, label='50 members')
    axes[0].hist(wind_1000, bins=bins, alpha=0.5, density=True, label='1000 members')
    axes[0].set_xlabel('10m Wind Speed (m/s)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Distribution: 50 vs 1000 Members')
    axes[0].legend()
    axes[0].axvline(x=50, color='red', linestyle='--', label='Severe threshold')

    # Tail probability estimation
    thresholds = np.arange(30, 70, 1)
    prob_50 = [np.mean(wind_50 > t) * 100 for t in thresholds]
    prob_1000 = [np.mean(wind_1000 > t) * 100 for t in thresholds]

    axes[1].semilogy(thresholds, prob_1000, 'b-', linewidth=2, label='1000 members')
    axes[1].semilogy(thresholds, [max(p, 0.01) for p in prob_50],
                     'r--', linewidth=2, label='50 members')
    axes[1].set_xlabel('Wind Speed Threshold (m/s)')
    axes[1].set_ylabel('Probability of Exceedance (%)')
    axes[1].set_title('Tail Risk: Why 1000+ Members Matter')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.01, 100)
    axes[1].axvline(x=50, color='gray', linestyle=':', alpha=0.5)
    axes[1].text(51, 50, 'Category 3+', fontsize=9, color='gray')

    # Convergence: how does the 95th percentile estimate improve with N?
    member_counts = [10, 20, 50, 100, 200, 500, 1000]
    p95_estimates = []
    p99_estimates = []
    for n in member_counts:
        trials = [np.percentile(np.random.choice(wind_1000, n), 95) for _ in range(100)]
        p95_estimates.append((np.mean(trials), np.std(trials)))
        trials_99 = [np.percentile(np.random.choice(wind_1000, n), 99) for _ in range(100)]
        p99_estimates.append((np.mean(trials_99), np.std(trials_99)))

    means_95 = [e[0] for e in p95_estimates]
    stds_95 = [e[1] for e in p95_estimates]
    means_99 = [e[0] for e in p99_estimates]
    stds_99 = [e[1] for e in p99_estimates]

    axes[2].errorbar(member_counts, means_95, yerr=stds_95, fmt='bo-',
                     label='95th percentile')
    axes[2].errorbar(member_counts, means_99, yerr=stds_99, fmt='rs-',
                     label='99th percentile')
    axes[2].set_xlabel('Ensemble Size')
    axes[2].set_ylabel('Estimated Percentile (m/s)')
    axes[2].set_title('Percentile Estimation Convergence')
    axes[2].set_xscale('log')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('HENS: Why Huge Ensembles Change Everything', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ex10_hens_analysis.png'),
                dpi=150, bbox_inches='tight')
    print(f"Saved: ex10_hens_analysis.png")
    plt.show()

    # Key statistics
    print(f"\nKey Results:")
    print(f"  P(wind > 50 m/s) from 50 members:  {np.mean(wind_50 > 50)*100:.1f}%")
    print(f"  P(wind > 50 m/s) from 1000 members: {np.mean(wind_1000 > 50)*100:.1f}%")
    print(f"  99th percentile (50 members):  {np.percentile(wind_50, 99):.1f} m/s")
    print(f"  99th percentile (1000 members): {np.percentile(wind_1000, 99):.1f} m/s")
    print(f"\n  With 50 members, you CANNOT reliably estimate P < 2%.")
    print(f"  With 1000 members, you can estimate P down to ~0.1%.")

demonstrate_hens_analysis()


# ===========================================================================
# EXERCISES
# ===========================================================================
"""
✅ Exercise 3.3a: Run the Earth2Studio HENS recipe for a historical
   hurricane case. Compare ensemble track spread to the NHC cone.

✅ Exercise 3.3b: Generate ensembles with different perturbation methods:
   SphericalGaussian vs bred vectors. Compare spread-skill ratios.
   Which produces better-calibrated ensembles?

✅ Exercise 3.3c: Use HENS output to estimate return levels (e.g., the
   1-in-100-year wind speed). Fit a GEV distribution to the ensemble
   maxima. How does the return level estimate change with ensemble size?

✅ Exercise 3.3d: Implement multi-checkpoint ensembles by loading
   3-4 SFNO checkpoints trained with different seeds. Does this increase
   ensemble diversity beyond just adding more perturbation members?

🔥 Challenge: Build a full hurricane risk pipeline:
   1. HENS ensemble → track + intensity forecasts
   2. For each member, compute maximum wind at coastal grid points
   3. Estimate P(Category 3+) at each coastal city
   4. Create a risk map with confidence intervals
"""
