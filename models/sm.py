"""
models/sm.py  —  Standard Model test for the higgs_mechanism package
====================================================================
Runs every layer in sequence and prints a full validation report.

Expected results (tree-level):
  m_W^2 = g^2 v^2 / 4
  m_Z^2 = (g^2 + g'^2) v^2 / 4
  m_A   = 0
  m_h^2 = 2 λ v^2
  3 Goldstone bosons: G^+, G^-, G^0
  g_HWW = g^2 v / 2        = 2 m_W^2 / v
  g_HZZ = g^2 v/(2cos^2θ)  = 2 m_Z^2 / v
  g_HWW / g_HZZ = cos^2 θ_W
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import (
    symbols, sqrt, factor, simplify, pprint, latex,
    cos, sin, Rational, expand
)
from sympy.physics.quantum import Dagger

from higgs_mechanism.gauge_group      import make_sm_gauge_group
from higgs_mechanism.scalar_sector    import make_sm_higgs_doublet
from higgs_mechanism.covariant_kinetic import build_sm_kinetic
from higgs_mechanism.scalar_potential  import (build_sm_potential,
                                               apply_tadpole_sm,
                                               build_sm_scalar_mass_matrix,
                                               identify_goldstones)
from higgs_mechanism.diagonalization   import (diagonalize_sm_gauge,
                                               verify_sm_gauge_masses,
                                               diagonalize_sm_scalar)
from higgs_mechanism.feynman_rules     import (build_sm_physical_lagrangian,
                                               extract_sm_vertices,
                                               build_sm_scalar_vertices)
from higgs_mechanism.validation        import run_sm_validation, print_validation_report

SEP = "─" * 60

def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ===========================================================================
# LAYER 0: Gauge Group
# ===========================================================================
section("Layer 0 — Gauge group: SU(2)_L × U(1)_Y")

group, gf = make_sm_gauge_group()
print(f"  Group: {group}")
print(f"  Gauge fields: {group.all_gauge_fields}")
print(f"  Couplings:    g = {gf['g']},  g' = {gf['g_prime']}")

for fac in group.factors:
    print(f"  {fac.name}: {fac.dim} generators")


# ===========================================================================
# LAYER 1: Scalar Sector
# ===========================================================================
section("Layer 1 — Scalar sector: SM Higgs doublet")

doublet, sf = make_sm_higgs_doublet(gf)
print(f"  Multiplet: {doublet}")
print(f"  Components: {doublet.components}")
print(f"  VEV: {doublet.vev}")
print(f"  Real fields: h={sf['h']}, G0={sf['G0']}, G+/- = {sf['G1']}±i{sf['G2']}")

# Count: 4 real d.o.f. in the doublet
print(f"  d.o.f. in doublet: {2 * doublet.dim} real")


# ===========================================================================
# LAYER 2: Covariant Kinetic Term → Gauge Boson Mass Matrix
# ===========================================================================
section("Layer 2 — Gauge boson mass matrix")

L_mass, M2_gauge, vev_subs = build_sm_kinetic(gf, sf)
print("  Gauge boson mass matrix M^2 in [W1, W2, W3, B] basis:")
pprint(M2_gauge)

# Factor each entry
print("\n  Diagonal entries (mass-squared):")
g, gp, v = gf['g'], gf['g_prime'], sf['v']
for i, label in enumerate(['W1', 'W2', 'W3', 'B']):
    print(f"    M^2[{label},{label}] = {factor(M2_gauge[i,i])}")


# ===========================================================================
# LAYER 3: Scalar Potential → Scalar Mass Matrix
# ===========================================================================
section("Layer 3 — Scalar potential and mass matrix")

V, pot_params = build_sm_potential(sf)
print(f"  V = {V}")

tadpole_sol = apply_tadpole_sm(V, sf, pot_params)
print(f"  Tadpole solution: {tadpole_sol}")

M2_scalar, real_fields, V_real = build_sm_scalar_mass_matrix(
    V, sf, pot_params, tadpole_sol
)

print("\n  Scalar mass matrix M^2_S in [h, G0, G1, G2] basis:")
pprint(M2_scalar)

goldstones, physical, scalar_masses = identify_goldstones(M2_scalar, real_fields)
print(f"\n  Physical Higgs fields:  {physical}")
print(f"  Goldstone bosons:       {goldstones}")
print("\n  Scalar masses:")
for field, m2 in scalar_masses.items():
    print(f"    m^2[{field}] = {m2}")


# ===========================================================================
# LAYER 4: Diagonalization
# ===========================================================================
section("Layer 4 — Rotation to mass basis")

phys_fields, change_basis, gauge_mass_sq, weinberg = diagonalize_sm_gauge(
    gf, M2_gauge
)
print("  Physical gauge fields:", list(phys_fields.keys()))
print("  Change of basis (weak → physical):")
for k, v_expr in change_basis.items():
    print(f"    {k} → {v_expr}")

print("\n  Gauge boson masses squared:")
for name, m2 in gauge_mass_sq.items():
    print(f"    m^2({name}) = {factor(m2)}")

print("\n  Weinberg angle:")
for k, expr in weinberg.items():
    print(f"    {k} = {factor(expr)}")

# Verify the mass matrix is diagonal in the physical basis
M2_phys, checks = verify_sm_gauge_masses(M2_gauge, gf, change_basis)
print("\n  Gauge mass matrix diagonality check:")
print(f"    Diagonal: {checks['diagonal']}")
print(f"    m_W^2 = {checks['mW_sq']}")
print(f"    m_Z^2 = {checks['mZ_sq']}")
print(f"    m_A^2 = {checks['mA_sq']}")


# ===========================================================================
# LAYER 5: Feynman Rules
# ===========================================================================
section("Layer 5 — Feynman rule vertices")

L_int = build_sm_physical_lagrangian(gf, sf, weinberg)
print(f"  L_int = {expand(L_int)}")

vertices = extract_sm_vertices(L_int, sf, gf, weinberg)
print("\n  Gauge-scalar vertices:")
for name in ['HWW', 'HZZ', 'HHWW', 'HHZZ']:
    computed = factor(vertices[name])
    expected = factor(vertices[f'{name}_expected'])
    match = simplify(computed - expected) == 0
    status = "✓" if match else "✗"
    print(f"    [{status}] g_{name}:")
    print(f"         computed  = {computed}")
    print(f"         expected  = {expected}")

scalar_vertices = build_sm_scalar_vertices(sf, pot_params, tadpole_sol)
print("\n  Scalar self-coupling vertices:")
for name in ['HHH', 'HHHH']:
    computed = factor(scalar_vertices[name])
    expected = factor(scalar_vertices[f'{name}_expected'])
    match = simplify(computed - expected) == 0
    status = "✓" if match else "✗"
    print(f"    [{status}] g_{name}:")
    print(f"         computed  = {computed}")
    print(f"         expected  = {expected}")

print(f"\n  Higgs mass: m_h^2 = {scalar_vertices['mh_sq']}")
print(f"  Expected:   m_h^2 = {scalar_vertices['mh_sq_expected']}")


# ===========================================================================
# LAYER 6: Validation
# ===========================================================================
section("Layer 6 — Validation")

results = run_sm_validation(
    goldstones      = goldstones,
    scalar_masses   = scalar_masses,
    gauge_mass_sq   = gauge_mass_sq,
    vertices        = vertices,
    params          = pot_params,
    gauge_fields    = gf,
    weinberg        = weinberg,
)

print_validation_report(results)

# ===========================================================================
# Summary table
# ===========================================================================
section("Summary — Physical spectrum")
print(f"""
  ┌─────────────────────────────────────────────────────┐
  │  Field    │  Mass²                                  │
  ├─────────────────────────────────────────────────────┤
  │  W±       │  g²v²/4                                 │
  │  Z        │  (g²+g'²)v²/4  =  m_W²/cos²θ_W         │
  │  A        │  0  (photon, U(1)_em unbroken)          │
  │  h        │  2λv²                                   │
  │  G±, G0   │  0  (Goldstones, eaten by W±, Z)        │
  └─────────────────────────────────────────────────────┘
""")

if results['all_passed']:
    print("  All SM checks PASSED — module is ready for BSM extensions.\n")
else:
    print("  WARNING: Some checks failed — review output above.\n")
