#!/usr/bin/env python3
"""
Simulate rBergomi variance and price paths, then compute the implied-vol
surface and print a summary.

Usage
-----
    python scripts/run_simulation.py [OPTIONS]

Options
-------
  --model      {rbergomi,rfheston}  Model to simulate  [default: rbergomi]
  --H          Hurst exponent       [default: 0.1]
  --eta        Vol-of-vol           [default: 1.9]
  --rho        Correlation          [default: -0.9]
  --xi0        Initial variance     [default: 0.04]
  --n-paths    Monte Carlo paths    [default: 5000]
  --n-steps    Steps per year       [default: 52]
  --T          Maturity in years    [default: 1.0]
  --seed       Random seed          [default: 42]
  --plot       Show matplotlib plot [default: False]
  --out        Output CSV path      [default: None]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from rough_vol.models.rbergomi import RBergomiParams, simulate_paths as rbergomi_sim
from rough_vol.models.rfheston import RHestonParams
from rough_vol.variance import (
    annualized_integrated_variance,
    annualized_realized_variance,
    variance_path_stats,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simulate rough-volatility paths and summarise variance statistics."
    )
    p.add_argument("--model", choices=["rbergomi", "rfheston"], default="rbergomi")
    p.add_argument("--H", type=float, default=0.1, help="Hurst exponent")
    p.add_argument("--eta", type=float, default=1.9, help="Vol-of-vol (rBergomi)")
    p.add_argument("--rho", type=float, default=-0.9, help="Spot-vol correlation")
    p.add_argument("--xi0", type=float, default=0.04, help="Initial fwd variance")
    p.add_argument("--n-paths", type=int, default=5_000, dest="n_paths")
    p.add_argument("--n-steps", type=int, default=52, dest="n_steps")
    p.add_argument("--T", type=float, default=1.0, help="Maturity in years")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot", action="store_true", help="Show matplotlib plot")
    p.add_argument("--out", type=str, default=None, help="Save path summary CSV")
    return p.parse_args()


def run(args: argparse.Namespace) -> dict:
    if args.model == "rbergomi":
        params = RBergomiParams(
            H=args.H, eta=args.eta, rho=args.rho, xi0=args.xi0
        )
        n_steps = max(int(np.ceil(args.T * args.n_steps)), 1)
        dt = args.T / n_steps
        print(f"Simulating rBergomi: H={params.H}, η={params.eta}, "
              f"ρ={params.rho}, ξ₀={params.xi0}")
        print(f"  n_paths={args.n_paths}, n_steps={n_steps}, T={args.T}, seed={args.seed}")
        V, S = rbergomi_sim(params, args.n_paths, n_steps, T=args.T, rng=args.seed)
    else:
        raise NotImplementedError("rfHeston path simulation not yet exposed in CLI")

    # --- Variance statistics ---
    stats = variance_path_stats(V, dt)
    ann_iv = annualized_integrated_variance(V, dt)
    ann_rv = annualized_realized_variance(S, args.T)

    print("\n--- Variance path statistics ---")
    print(f"  E[ann. integrated var]  = {stats['mean_iv']:.6f}  "
          f"(xi0 = {params.xi0:.6f})")
    print(f"  Std[ann. integrated var]= {stats['std_iv']:.6f}")
    print(f"  Skew[ann. IV]           = {stats['skew_iv']:.4f}")
    print(f"  ExcessKurt[ann. IV]     = {stats['excess_kurtosis_iv']:.4f}")
    print(f"  E[ann. realized var]    = {ann_rv.mean():.6f}")
    print(f"  VRP (RV - IV)           = {(ann_rv - ann_iv).mean():.6f}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            n_show = min(50, args.n_paths)
            t = np.linspace(0, args.T, V.shape[1])

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(
                f"rBergomi Simulation  H={params.H}  η={params.eta}  "
                f"ρ={params.rho}  ξ₀={params.xi0}",
                fontsize=13,
            )

            # Variance paths
            axes[0, 0].plot(t, V[:n_show].T, lw=0.4, alpha=0.5)
            axes[0, 0].set_title("Variance paths V(t)")
            axes[0, 0].set_xlabel("Time"); axes[0, 0].set_ylabel("V(t)")

            # Price paths
            axes[0, 1].plot(t, S[:n_show].T, lw=0.4, alpha=0.5)
            axes[0, 1].set_title("Asset price paths S(t)")
            axes[0, 1].set_xlabel("Time"); axes[0, 1].set_ylabel("S(t)")

            # Distribution of ann. IV
            axes[1, 0].hist(ann_iv, bins=50, density=True, alpha=0.7, color="steelblue")
            axes[1, 0].axvline(params.xi0, color="red", ls="--", label="ξ₀")
            axes[1, 0].set_title("Distribution of ann. integrated variance")
            axes[1, 0].legend()

            # Distribution of ann. RV
            axes[1, 1].hist(ann_rv, bins=50, density=True, alpha=0.7, color="coral")
            axes[1, 1].axvline(params.xi0, color="red", ls="--", label="ξ₀")
            axes[1, 1].set_title("Distribution of ann. realized variance")
            axes[1, 1].legend()

            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib not available; skipping plot")

    results = {
        "mean_iv": stats["mean_iv"],
        "std_iv": stats["std_iv"],
        "mean_rv": float(ann_rv.mean()),
        "vrp": float((ann_rv - ann_iv).mean()),
        "params": str(params),
    }

    if args.out:
        import pandas as pd
        df = pd.DataFrame([results])
        df.to_csv(args.out, index=False)
        print(f"\nSaved summary to {args.out}")

    return results


if __name__ == "__main__":
    run(parse_args())
