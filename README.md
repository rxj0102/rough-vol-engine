# rough-vol-engine

A Python library for pricing and calibrating rough volatility models.

## Overview

`rough-vol-engine` implements a suite of rough stochastic volatility models,
Monte Carlo simulation infrastructure, option pricing routines, and parameter
calibration tools built on fractional Brownian motion (fBm).

Supported model families:

- **Rough Bergomi** (rBergomi) — Bayer, Friz & Gatheral (2016)
- **Rough Heston** — El Euch & Rosenbaum (2019)

## Repository Structure

```
rough-vol-engine/
├── rough_vol/           # Main library package
│   ├── models/          # Stochastic volatility model definitions
│   ├── simulation/      # fBm and Monte Carlo simulation engine
│   ├── pricing/         # Option pricing (European, path-dependent)
│   ├── calibration/     # Parameter optimisation routines
│   ├── analytics/       # Greeks and volatility surface construction
│   └── utils/           # Configuration loading and plotting helpers
├── tests/               # Unit and integration tests (pytest)
├── configs/             # YAML configuration files
├── notebooks/           # Example Jupyter notebooks
└── scripts/             # Stand-alone helper scripts
```

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
# TODO: example usage once models are implemented
```

## Running Tests

```bash
pytest
```

## License

MIT — see [LICENSE](LICENSE).
