#!/usr/bin/env python3
"""
Variance swap analysis: fair strike, P&L, and forward variance curve.

Demonstrates:
  1. MC variance swap fair strike for rBergomi across maturities
  2. Model-free variance swap strike from the IV surface (ATM approximation)
  3. Forward variance curve extracted from the rfHeston surface
  4. Variance risk premium distribution from simulated paths

Usage
-----
    python scripts/variance_analysis.py [OPTIONS]

Options
-------
  --H          Hurst exponent  [default: 0.1]
  --xi0        Initial forward variance (rBergomi)  [default: 0.04]
  --eta        Vol-of-vol  [default: 1.9]
  --rho        Correlation  [default: -0.9]
  --n-paths    MC paths  [default: 5000]
  --n-steps    Steps per year  [default: 52]
  --seed       Random seed  [default: 42]
  --plot       Show plots  [default: False]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.synthetic import make_rfheston_surface, standard_maturity_grid
from rough_vol.models.rbergomi import RBergomiParams, simulate_paths
from rough_vol.models.rfheston import RHestonParams
from rough_vol.variance import (
    annualized_integrated_variance,
    annualized_realized_variance,
    atm_total_variance,
    forward_variance_curve,
    variance_swap_pnl,
    variance_swap_strike_atm_approx,
    variance_swap_strike_mc,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Variance swap and forward variance analysis.")
    p.add_argument("--H", type=float, default=0.1)
    p.add_argument("--xi0", type=float, default=0.04)
    p.add_argument("--eta", type=float, default=1.9)
    p.add_argument("--rho", type=float, default=-0.9)
    p.add_argument("--n-paths", type=int, default=5_000, dest="n_paths")
    p.add_argument("--n-steps", type=int, default=52, dest="n_steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    params = RBergomiParams(H=args.H, eta=args.eta, rho=args.rho, xi0=args.xi0)
    print(f"rBergomi params: {params}")
    print(f"Theoretical variance swap strike = ξ₀ = {params.xi0:.4f}\n")

    # --- MC variance swap strikes across maturities ---
    mats = np.array([0.25, 0.5, 1.0, 2.0])
    print("--- MC variance swap fair strikes ---")
    print(f"{'Maturity':>10}  {'E[RV]':>10}  {'Std err':>10}  {'Diff from ξ₀':>14}")
    mc_strikes = {}
    for T in mats:
        k_mc, se = variance_swap_strike_mc(
            params, T, n_paths=args.n_paths, n_steps_per_year=args.n_steps, rng=args.seed
        )
        diff = k_mc - params.xi0
        print(f"  T={T:>5.2f}  {k_mc:>10.6f}  {se:>10.6f}  {diff:>+14.6f}")
        mc_strikes[T] = k_mc

    # --- Forward variance curve from rfHeston surface ---
    print("\n--- Forward variance curve (rfHeston surface) ---")
    rh_params = RHestonParams(H=args.H, lambda_=1.5, theta=args.xi0, rho=-0.5, nu=0.4, V0=args.xi0)
    rh_surf = make_rfheston_surface(
        rh_params, maturities=mats, n_steps=50, n_quad=32
    )

    mats_atm, w_atm = atm_total_variance(rh_surf)
    t_mid, fv = forward_variance_curve(rh_surf)

    print(f"{'Maturity':>10}  {'ATM total var':>15}  {'ATM var rate':>14}")
    for i, T in enumerate(mats_atm):
        w = w_atm[i]
        var_rate = w / T if T > 0 else float("nan")
        print(f"  T={T:>5.2f}  {w:>15.6f}  {var_rate:>14.6f}")

    print(f"\n{'Interval':>18}  {'Forward var':>14}")
    for i, (tm, fv_i) in enumerate(zip(t_mid, fv)):
        T1 = mats[i]; T2 = mats[i + 1]
        print(f"  [{T1:.2f}, {T2:.2f}]  {fv_i:>14.6f}")

    # --- Model-free vs MC comparison at T=1 ---
    print("\n--- Variance swap strike comparison at T=1 ---")
    T_ref = 1.0
    k_mc, se_mc = variance_swap_strike_mc(
        params, T_ref, n_paths=args.n_paths, n_steps_per_year=args.n_steps, rng=args.seed
    )
    k_atm = variance_swap_strike_atm_approx(rh_surf, T_ref)
    print(f"  MC (rBergomi)  : {k_mc:.6f} ± {se_mc:.6f}")
    print(f"  ATM approx     : {k_atm:.6f}  (rfHeston surface)")
    print(f"  Theory (ξ₀)    : {params.xi0:.6f}")

    # --- Variance risk premium distribution ---
    print("\n--- Variance risk premium distribution (T=1) ---")
    n_steps = max(int(np.ceil(T_ref * args.n_steps)), 1)
    dt = T_ref / n_steps
    V, S = simulate_paths(params, args.n_paths, n_steps, T=T_ref, rng=args.seed)
    ann_iv = annualized_integrated_variance(V, dt)
    ann_rv = annualized_realized_variance(S, T_ref)
    vrp = ann_rv - ann_iv

    print(f"  E[IV]  = {ann_iv.mean():.6f}  (ξ₀ = {params.xi0:.6f})")
    print(f"  E[RV]  = {ann_rv.mean():.6f}")
    print(f"  VRP    = {vrp.mean():.6f} ± {vrp.std()/np.sqrt(args.n_paths):.6f}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(
                f"Variance Analysis  rBergomi H={params.H}  ξ₀={params.xi0}",
                fontsize=13,
            )

            # MC variance swap strikes
            mc_ks = [mc_strikes[T] for T in mats]
            axes[0].plot(mats, mc_ks, "o-", label="E[RV] (MC)", color="steelblue")
            axes[0].axhline(params.xi0, color="red", ls="--", label="ξ₀")
            axes[0].set_title("Variance swap fair strike"); axes[0].set_xlabel("T")
            axes[0].set_ylabel("K_var"); axes[0].legend(); axes[0].grid(alpha=0.3)

            # Forward variance curve
            axes[1].step(
                np.concatenate([[mats[0]], t_mid, [mats[-1]]]),
                np.concatenate([[fv[0]], fv, [fv[-1]]]),
                where="mid", color="coral", lw=2, label="Forward variance",
            )
            axes[1].axhline(args.xi0, color="red", ls="--", label="ξ₀")
            axes[1].set_title("Forward variance curve (rfHeston)")
            axes[1].set_xlabel("Time"); axes[1].set_ylabel("Forward var")
            axes[1].legend(); axes[1].grid(alpha=0.3)

            # VRP distribution
            axes[2].hist(vrp, bins=50, density=True, alpha=0.7, color="mediumseagreen")
            axes[2].axvline(0, color="black", ls="-", lw=1)
            axes[2].axvline(vrp.mean(), color="red", ls="--", label=f"mean={vrp.mean():.4f}")
            axes[2].set_title("Variance risk premium (RV − IV)  T=1")
            axes[2].set_xlabel("VRP"); axes[2].legend(); axes[2].grid(alpha=0.3)

            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib not available; skipping plot")


if __name__ == "__main__":
    run(parse_args())
