[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/OWI-Lab/torque2026-ahmed-mujtaba-et-al)

# owilab-torque2026

Analysis of as-built dimensional tolerances on monopile fatigue life.

## Overview

> You can run this project in a **[GitHub Codespace](https://codespaces.new/OWI-Lab/torque2026-ahmed-mujtaba-et-al)** with the provided configuration. Just click the "Open in Codespaces" button at the top of this README.
>
> From there, you can open the `1.0.torque-2026.ipynb` notebook to see the calculations in action.

### Package summary

This package factors the core logic from the notebook into reusable modules:

- `geometry`: section properties for tubulars (moment of inertia, section modulus).
- `stress_factors`: DNV RP-C203-based stress concentration factor (SCF), effective thickness based scale effect, and stress multiplication factors (SMF).
- `fatigue`: two-slope S-N curve representation, Weibull stress range distribution with deterministic Palmgren–Miner damage, and helpers to solve the Weibull scale for a target design damage.
- `plot`: plotting utilities for Weibull distributions, and fatigue life evolution.
- `__main__`: command-line interface (CLI) to run the main calculation from terminal.

The included examples reproduce the figures and calculations from the original notebook using the package API.

## Local Installation 

### Via devcontainer

This project is configured to work with [devcontainer](https://code.visualstudio.com/docs/remote/devcontainer-overview). If you don't have `devcontainer` yet:

1. Install [Docker](https://docs.docker.com/get-docker/).
2. Install [Visual Studio Code](https://code.visualstudio.com/).
3. Install the [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

Then, open the project folder in Visual Studio Code and select "Reopen in Container" when prompted. This will set up the development environment with all dependencies installed.

### Via uv

This project is configured for [uv](https://github.com/astral-sh/uv). If you don't have `uv` yet:

```bash
# install uv (Linux/macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

From the project root (`pyproject.toml` present):

```bash
# create and sync a virtual environment
uv venv --python 3.11
uv sync --all-packages --all-extras --all-groups 

# activate the environment (Windows - PowerShell)
.venv\Scripts\Activate.ps1
```

```bash
# activate the environment (zsh)
source .venv/bin/activate
```

## Usage


### CLI

After `uv sync` and activation, you can run:

```bash
uv run python -m owilab_torque2026 --D 7200 --t 68 --T 73 --mis 3 --years 20 --FDF 3
```

or

```bash
uv run python scripts/main.py --D 7200 --t 68 --T 73 --mis 3
```

To run tests (which are taken from docstrings):

```bash
uv run pytest
```

### Python API

```python
import numpy as np
from owilab_torque2026 import (
    SNCurve, smf_inner_outer, solve_weibull_scale_for_damage,
)

# Nominal dimensions (mm)
D_nom, t_nom, T_nom = 7200, 68, 73
misalignment = 3

# SN curve (Water with CP)
sn = SNCurve(slope=[3, 5], intercept=[11.764, 15.606],
             environment='Water with cathodic protection',
             curve='DNV-D-C', norm='DNVGL-RP-C203/2016')
S = sn.get_knee_stress()[0]

# Effective thickness scale parameters
t_ref, t_eff_allowance, t_corr_exponent = 25, 6, 0.2
weld_width = 0.64 * t_nom

SMF_IN, SMF_OUT = smf_inner_outer(
    D_nom, t_nom, T_nom, misalignment,
    t_ref, t_eff_allowance, t_corr_exponent, weld_width,
    material_factor=1.25, section_modulus_reference="inner",
)

# Weibull parameters
h = 0.8                     # shape
n_per_year = 0.16*3600*24*365
FDF = 3
T_years = 20

q_mean = solve_weibull_scale_for_damage(
    *sn.slope, *sn.intercept, S,
    h, T_years, n_per_year, FDF,
    SMF_IN,
)
print("q_mean:", q_mean)
```

## Theory summary

- Section properties follow classic hollow circular formulas: moment of inertia `I = (π/64)(D^4 - d^4)` with `d = D - 2t`, section moduli `Z_out = I/(D/2)`, `Z_in = I/((D-2t)/2)`.
- Stress Concentration Factor (SCF) uses the simplified DNV RP-C203 3.3.7.3 approach consistent with the original notebook implementation, with misalignment and thickness transition contributions combined and exponentially attenuated by transition length.
- Scale effect per DNV-RP-C-203: an effective thickness `t_eff = min(14 + 0.66(weld_width + allowance), t)` alters the curve via `(t_eff/t_ref)^{k}` with exponent `k` (here `0.2`).
- Stress Multiplication Factor (SMF) aggregates SCF, scale effect, and structural extrapolation factor (SEF) as ratio of a reference section modulus to inner/outer section modulus. A material factor (e.g., 1.25) can be included.
- Fatigue: We assume a Weibull stress range distribution with shape `h` and unknown scale `q`. Deterministic damage with a two-slope S-N curve is computed via gamma/incomplete-gamma expressions. The target is meeting design damage `D_cr = 1/FDF` over `T` years with cycles per year `n`. We solve for `q` so that the damage equals `D_cr`.

## Reproduction notebook

A new notebook `Reproduce-Torque2026.ipynb` is included that imports the package and produces the same outputs as the original analysis.
