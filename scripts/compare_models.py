#!/usr/bin/env python3
"""
Compare rBergomi and rough Heston implied-vol surfaces for equivalent
roughness (same H) and matched ATM vol.

For each model:
  - Generate the IV surface with matched ATM vol at T=1
  - Print ATM vols, ATM skews, and no-arbitrage diagnostics
  - (Optionally) plot smile cross-sections

Usage
-----
    python scripts/compare_models.py [OPTIONS]

Options
-------
  --H          Hurst exponent shared by both models  [default: 0.1]
  --sigma-atm  Target ATM vol at T=1 (used to set ξ₀/V₀)  [default: 0.20]
  --mats       Comma-separated maturities  [default: 0.25,0.5,1.0,2.0]
  --n-paths    MC paths for rBergomi  [default: 10000]
  --n-steps    Solver steps for rfHeston  [default: 100]
  --n-quad     Quadrature nodes for rfHeston  [default: 64]
  --seed       Random seed  [default: 42]
  --plot       Show comparison plot  [default: False]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.synthetic import log_strike_grid, make_rbergomi_surface, make_rfheston_surface
from rough_vol.models.rbergomi import RBergomiParams
from rough_vol.models.rfheston import RHestonParams
from rough_vol.calibration.optimizer import RoughVolCalibrator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare rBergomi and rough Heston surfaces.")
    p.add_argument("--H", type=float, default=0.1)
    p.add_argument("--sigma-atm", type=float, default=0.20, dest="sigma_atm")
    p.add_argument("--mats", type=str, default="0.25,0.5,1.0,2.0")
    p.add_argument("--n-paths", type=int, default=10_000, dest="n_paths")
    p.add_argument("--n-steps", type=int, default=100, dest="n_steps")
    p.add_argument("--n-quad", type=int, default=64, dest="n_quad")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def print_surface_summary(name: str, surface) -> None:
    atm = surface.atm_vols()
    skews = surface.atm_skews(dk=0.05)
    arb = surface.check_no_arbitrage()

    print(f"\n{'─'*50}")
    print(f"Model: {name}")
    print(f"{'─'*50}")
    print(f"{'Maturity':>10}  {'ATM vol':>10}  {'ATM skew':>12}  {'Ann. skew':>12}")
    for i, T in enumerate(surface.maturities):
        atm_v = f"{atm[i]:.4f}" if np.isfinite(atm[i]) else "  NaN"
        skew_v = f"{skews[i]:.4f}" if np.isfinite(skews[i]) else "  NaN"
        ann_skew = f"{skews[i] * np.sqrt(T):.4f}" if np.isfinite(skews[i]) else "  NaN"
        print(f"  T={T:>6.2f}  {atm_v:>10}  {skew_v:>12}  {ann_skew:>12}")

    cal_ok = "✓" if arb["no_calendar_arbitrage"] else "✗"
    fly_ok = "✓" if arb["no_butterfly_arbitrage"] else "✗"
    print(f"\n  Calendar arbitrage-free: {cal_ok}")
    print(f"  Butterfly arbitrage-free: {fly_ok}")

    # H estimation
    H_est = RoughVolCalibrator.estimate_H_from_skew(surface)
    if H_est is not None:
        print(f"  Estimated H from skew: {H_est:.4f}")


def run(args: argparse.Namespace) -> None:
    mats = np.array([float(x) for x in args.mats.split(",")])
    strikes = log_strike_grid(k_lo=-0.20, k_hi=0.20, n_strikes=11)

    # ATM vol target implies: xi0 ≈ sigma_atm^2 for rBergomi
    xi0 = args.sigma_atm ** 2
    V0 = args.sigma_atm ** 2

    rb_params = RBergomiParams(H=args.H, eta=1.9, rho=-0.9, xi0=xi0)
    rh_params = RHestonParams(
        H=args.H, lambda_=1.5, theta=V0, rho=-0.7, nu=0.5, V0=V0
    )

    print(f"rBergomi params: {rb_params}")
    print(f"rfHeston params: {rh_params}")

    print("\nGenerating rBergomi surface …")
    rb_surf = make_rbergomi_surface(
        rb_params, strikes=strikes, maturities=mats,
        n_paths=args.n_paths, n_steps_per_year=52, rng=args.seed,
    )

    print("Generating rough Heston surface …")
    rh_surf = make_rfheston_surface(
        rh_params, strikes=strikes, maturities=mats,
        n_steps=args.n_steps, n_quad=args.n_quad,
    )

    print_surface_summary("Rough Bergomi (rBergomi)", rb_surf)
    print_surface_summary("Rough Heston (rfHeston)", rh_surf)

    # Skew power-law comparison
    print("\n\n--- ATM skew power-law log-log slopes ---")
    for name, surf in [("rBergomi", rb_surf), ("rfHeston", rh_surf)]:
        skews = surf.atm_skews(dk=0.05)
        valid = np.isfinite(skews) & (np.abs(skews) > 1e-12)
        if valid.sum() >= 2:
            slope, _ = np.polyfit(np.log(mats[valid]), np.log(np.abs(skews[valid])), 1)
            print(f"  {name}: slope={slope:.4f}  →  H_eff={slope+0.5:.4f}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, len(mats), figsize=(4 * len(mats), 5))
            if len(mats) == 1:
                axes = [axes]

            colors = {"rBergomi": "steelblue", "rfHeston": "coral"}
            for i, T in enumerate(mats):
                ax = axes[i]
                for name, surf in [("rBergomi", rb_surf), ("rfHeston", rh_surf)]:
                    k = surf.to_log_moneyness()[i]
                    iv = surf.implied_vols[i]
                    valid = np.isfinite(iv)
                    ax.plot(
                        k[valid], iv[valid],
                        label=name, color=colors[name], lw=2,
                    )
                ax.set_title(f"T = {T:.2f}")
                ax.set_xlabel("Log-moneyness k")
                ax.set_ylabel("Implied vol")
                ax.legend(); ax.grid(alpha=0.3)

            fig.suptitle(
                f"rBergomi vs rfHeston  H={args.H}  σ_ATM≈{args.sigma_atm:.2f}",
                fontsize=13,
            )
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib not available; skipping plot")


if __name__ == "__main__":
    run(parse_args())
