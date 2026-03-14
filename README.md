# higgs-mechanism

A SymPy-based Python package for computing the Higgs mechanism in generic Beyond Standard Model (BSM) gauge theories. All computations are fully symbolic — no numerical approximations.

## Features

- Define arbitrary gauge groups G = G₁ × G₂ × ... with generators and covariant derivatives
- Register scalar multiplets, assign VEVs, and decompose into real fluctuations
- Build the kinetic Lagrangian and extract gauge boson mass matrices
- Construct scalar potentials, apply tadpole conditions, and extract scalar mass matrices
- Rotate to the physical mass basis and extract mixing angles
- Extract n-point Feynman rule vertex coefficients
- Automated consistency checks: Goldstone theorem, Weinberg angle, custodial symmetry, unitarity

## Layer Stack

| Layer | Module | Purpose |
|-------|--------|---------|
| 0 | `gauge_group.py` | Define gauge group with generators and covariant derivative |
| 1 | `scalar_sector.py` | Register scalar multiplets, assign VEVs, decompose into real fields |
| 2 | `covariant_kinetic.py` | Build L_kin = (D_μΦ)†(D_μΦ), extract gauge boson mass matrix |
| 3 | `scalar_potential.py` | Build V(Φ), apply tadpole conditions, extract scalar mass matrix |
| 4 | `diagonalization.py` | Rotate to physical mass basis, extract mixing angles |
| 5 | `feynman_rules.py` | Extract n-point vertex coupling coefficients |
| 6 | `validation.py` | Automated consistency checks |

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd HiggsMechanism

# Create and activate a virtual environment
python3 -m virtualenv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

```python
from higgs_mechanism import (
    make_sm_gauge_group,
    make_sm_higgs_doublet,
    build_sm_kinetic,
    diagonalize_sm_gauge,
    run_sm_validation,
    print_validation_report,
)

# Layer 0 — gauge group SU(2)_L × U(1)_Y
group = make_sm_gauge_group()

# Layer 1 — Higgs doublet with VEV ⟨φ⁰⟩ = v/√2
higgs = make_sm_higgs_doublet()

# Layer 2 — kinetic term and gauge boson mass matrix
L_kin, M_gauge = build_sm_kinetic(group, higgs)

# Layer 4 — rotate to physical basis (W±, Z, γ)
masses, angles = diagonalize_sm_gauge(M_gauge)

# Layer 6 — run all consistency checks
results = run_sm_validation()
print_validation_report(results)
```

## Running the SM Reference Model

```bash
python higgs_mechanism/sm.py
```

This runs the full Standard Model computation end-to-end and prints a validation report.

## Interactive Examples

```bash
jupyter notebook examples/sm_higgs_mechanism.ipynb
```

## Running Tests

```bash
pytest tests/ -v
```

## BSM Extensions

New models go in `models/` as standalone scripts following the same layer-by-layer pattern as `sm.py`. Use `make_sm_gauge_group()` and `run_sm_validation()` as a baseline to validate BSM results against.

Key generic utilities for BSM use:

```python
from higgs_mechanism import build_mass_matrix, extract_interaction_coefficients
```

## Requirements

- Python >= 3.9
- sympy >= 1.12

## License

MIT
