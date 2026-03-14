# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run a single test file
pytest tests/test_imports.py

# Run the SM model (executes all 6 layers and prints a validation report)
python models/sm.py

# Launch Jupyter notebook for interactive exploration
jupyter notebook examples/sm_higgs_mechanism.ipynb
```

## Architecture

This is a SymPy-based Python package for computing the Higgs mechanism in generic BSM gauge theories. It is organized as a **six-layer computational stack**, each building on the previous. All computations are symbolic (SymPy), not numeric.

### Layer Stack

| Layer | Module | Purpose |
|-------|--------|---------|
| 0 | `gauge_group.py` | Define gauge group G = G₁ × G₂ × ... with generators and covariant derivatives |
| 1 | `scalar_sector.py` | Register scalar multiplets, assign VEVs, decompose into real fluctuations |
| 2 | `covariant_kinetic.py` | Build L_kin = (D_μΦ)†(D_μΦ), extract gauge boson mass matrix |
| 3 | `scalar_potential.py` | Build V(Φ), apply tadpole conditions, extract scalar mass matrix |
| 4 | `diagonalization.py` | Rotate to physical mass basis, extract mixing angles |
| 5 | `feynman_rules.py` | Extract n-point vertex coupling coefficients from the Lagrangian |
| 6 | `validation.py` | Automated consistency checks (Goldstone theorem, Weinberg relation, etc.) |

**Shared utilities:** `symbolic_tools.py` — mass matrix builder, vertex coefficient extractor, LaTeX table generator.

**Reference implementation:** `models/sm.py` — runs the full SM computation end-to-end; serves as the canonical example for any BSM extension. New models go in `models/` as standalone scripts following the same layer-by-layer pattern.

### Key Design Conventions

- **SM as baseline:** All new gauge theories should validate against SM results in `sm.py`.
- **SymPy-native:** All quantities are symbolic expressions. Avoid introducing numeric (numpy/scipy) computation unless explicitly requested.
- **Explicit basis tracking:** Mass matrices must be labeled with their field basis (e.g., `[W1, W2, W3, B]` vs `[Wp, Wm, Z, A]`).
- **Real-field decomposition:** Complex scalars are systematically decomposed into physical Higgs (h) and Goldstone (π) real components via `decompose_complex_scalar()`.
- **Tadpole conditions:** Applied before extracting scalar masses — `apply_tadpole_sm()` enforces ∂V/∂φ = 0 at the VEV.
- **Feynman rule convention:** Vertex = i × (coefficient extracted from −L).
- **Validation built-in:** Every major computation layer has a corresponding check in `validation.py`.

### Public API

The 22 public symbols exported from `higgs_mechanism/__init__.py` cover all six layers. Key entry points:

- `make_sm_gauge_group()` — builds SU(2)_L × U(1)_Y
- `make_sm_higgs_doublet()` — creates H = (φ⁺, φ⁰) with VEV ⟨φ⁰⟩ = v/√2
- `run_sm_validation()` / `print_validation_report()` — run and display all consistency checks
- `build_mass_matrix()` / `extract_interaction_coefficients()` — generic utilities for BSM use
