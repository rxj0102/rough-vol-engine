#!/usr/bin/env python3
"""
Calibrate a rough-volatility model to a synthetic or CSV-loaded surface.

Usage
-----
    # Calibrate rfHeston to a synthetic rfHeston surface (round-trip test)
    python scripts/run_calibration.py --model rfheston --method two_stage

    # Calibrate to a CSV surface
    python scripts/run_calibration.py --surface data/my_surface.csv --model rfheston

    # Fast NM-only run with custom H
    python scripts/run_calibration.py --method nelder_mead --H-init 0.15

Options
-------
  --model       {rbergomi,rfheston}  Model to calibrate  [default: rfheston]
  --method      {differential_evolution,nelder_mead,two_stage}  [default: two_stage]
  --surface     Path to CSV surface  (optional; uses synthetic if absent)
  --H-init      Initial H guess for nelder_mead  [default: estimate from skew]
  --de-popsize  DE population multiplier  [default: 8]
  --de-maxiter  DE max generations  [default: 200]
  --nm-maxiter  Nelder-Mead max iter  [default: 500]
  --n-paths     MC paths (rBergomi)  [default: 5000]
  --n-steps     Solver steps (rfHeston)  [default: 100]
  --n-quad      Quadrature nodes (rfHeston)  [default: 64]
  --seed        Random seed  [default: 42]
  --plot        Show convergence plot  [default: False]
  --out         Output JSON path  [default: None]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.synthetic import make_rfheston_surface, make_rbergomi_surface
from rough_vol.calibration import RoughVolCalibrator
from rough_vol.calibration.objective import RBergomiObjective, RHestonObjective
from rough_vol.calibration.surface import ImpliedVolSurface
from rough_vol.models.rfheston import RHestonParams


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate a rough-vol model to an IV surface.")
    p.add_argument("--model", choices=["rbergomi", "rfheston"], default="rfheston")
    p.add_argument(
        "--method",
        choices=["differential_evolution", "nelder_mead", "two_stage"],
        default="two_stage",
    )
    p.add_argument("--surface", type=str, default=None, help="CSV surface path")
    p.add_argument("--H-init", type=float, default=None, dest="H_init",
                   help="Initial H for nelder_mead (overrides skew estimate)")
    p.add_argument("--de-popsize", type=int, default=8, dest="de_popsize")
    p.add_argument("--de-maxiter", type=int, default=200, dest="de_maxiter")
    p.add_argument("--nm-maxiter", type=int, default=500, dest="nm_maxiter")
    p.add_argument("--n-paths", type=int, default=5_000, dest="n_paths")
    p.add_argument("--n-steps", type=int, default=100, dest="n_steps")
    p.add_argument("--n-quad", type=int, default=64, dest="n_quad")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def load_or_generate_surface(args: argparse.Namespace) -> ImpliedVolSurface:
    if args.surface:
        print(f"Loading surface from {args.surface}")
        return ImpliedVolSurface.from_csv(args.surface)

    print("Generating synthetic surface …")
    if args.model == "rfheston":
        true_params = RHestonParams(
            H=0.10, lambda_=1.5, theta=0.04, rho=-0.60, nu=0.35, V0=0.04
        )
        surf = make_rfheston_surface(
            true_params, n_steps=args.n_steps, n_quad=args.n_quad
        )
        print(f"  True params: {true_params}")
    else:
        from rough_vol.models.rbergomi import RBergomiParams
        true_params = RBergomiParams(H=0.10, eta=1.9, rho=-0.9, xi0=0.04)
        surf = make_rbergomi_surface(
            true_params,
            n_paths=args.n_paths,
            n_steps_per_year=52,
            rng=args.seed,
        )
        print(f"  True params: {true_params}")
    return surf


def build_x0(args: argparse.Namespace, surface: ImpliedVolSurface) -> np.ndarray | None:
    if args.method != "nelder_mead":
        return None

    H_init = args.H_init
    if H_init is None:
        H_init = RoughVolCalibrator.estimate_H_from_skew(surface)
        if H_init is not None:
            print(f"  Estimated H from skew: {H_init:.4f}")
        else:
            H_init = 0.10
            print(f"  Skew estimation failed; using H={H_init}")

    if args.model == "rfheston":
        # [H, lambda_, theta, rho, nu, V0]
        return np.array([H_init, 1.5, 0.04, -0.5, 0.3, 0.04])
    else:
        # [H, eta, rho, xi0]
        return np.array([H_init, 1.9, -0.7, 0.04])


def run(args: argparse.Namespace) -> dict:
    surface = load_or_generate_surface(args)

    print(f"\nSurface: {surface}")
    H_est = RoughVolCalibrator.estimate_H_from_skew(surface)
    if H_est is not None:
        print(f"Estimated H from ATM skew: {H_est:.4f}")

    cal = RoughVolCalibrator(
        surface,
        model=args.model,
        n_paths=args.n_paths,
        n_steps_per_year=52,
        rng_seed=args.seed,
        n_steps=args.n_steps,
        n_quad=args.n_quad,
    )

    x0 = build_x0(args, surface)

    print(f"\nRunning calibration: method={args.method}")
    result = cal.calibrate(
        method=args.method,
        x0=x0,
        de_popsize=args.de_popsize,
        de_maxiter=args.de_maxiter,
        nm_maxiter=args.nm_maxiter,
    )

    print(f"\n{'='*50}")
    print(f"Calibration complete")
    print(f"  Method    : {result.method}")
    print(f"  Loss      : {result.loss:.8f}")
    print(f"  n_evals   : {result.n_evals}")
    print(f"  Success   : {result.success}")
    print(f"  Params    : {result.params}")
    print(f"{'='*50}")

    if args.plot and len(result.history) > 0:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Convergence curve
            axes[0].semilogy(result.history)
            axes[0].set_title("Calibration convergence (running min WSSE)")
            axes[0].set_xlabel("Evaluation"); axes[0].set_ylabel("WSSE")
            axes[0].grid(True, alpha=0.3)

            # Fitted vs market smiles (first maturity)
            mats = surface.maturities
            k = surface.to_log_moneyness()
            for i in range(min(3, len(mats))):
                market_iv = surface.implied_vols[i]
                valid = np.isfinite(market_iv)
                label = f"Market T={mats[i]:.2f}"
                axes[1].plot(k[i][valid], market_iv[valid], "o--", label=label, ms=4)

            axes[1].set_title("Market IV smiles")
            axes[1].set_xlabel("Log-moneyness k"); axes[1].set_ylabel("Implied vol")
            axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib not available; skipping plot")

    output = {
        "method": result.method,
        "loss": result.loss,
        "n_evals": result.n_evals,
        "success": result.success,
        "params": str(result.params),
    }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved results to {args.out}")

    return output


if __name__ == "__main__":
    run(parse_args())
