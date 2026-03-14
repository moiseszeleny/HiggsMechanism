"""
Layer 5 — feynman_rules.py
============================
Substitute the physical field rotations into L_kin + V and extract
Feynman rules (n-point interaction coefficients).

Re-uses the extract_interaction_coefficients engine from symbolic_tools.
Key SM vertices extracted:
  - H W^+ W^-  (coupling ~ g m_W)
  - H Z Z       (coupling ~ g m_Z / cos θ_W)
  - H H H       (triple Higgs, ~ lambda v)
  - H H H H     (quartic,      ~ lambda)
"""

from sympy import (
    symbols, sqrt, I, cos, sin, expand, factor,
    Function, Mul, S, conjugate, Add
)

from .symbolic_tools import extract_interaction_coefficients, momentum


# ---------------------------------------------------------------------------
# SM kinetic Lagrangian in the physical basis
# ---------------------------------------------------------------------------

def build_sm_physical_lagrangian(gauge_fields, scalar_fields, weinberg):
    """
    Build L_kin in terms of physical fields (W^+, W^-, Z, A, h, G0, G1, G2).

    We write out the gauge-scalar interaction piece explicitly:
        L_int = g^2/4 (W^+W^- + 1/(2cos^2θ_W) ZZ)(v+h)^2
              + derivative terms (Goldstone-gauge)

    This is the standard textbook result for the SM after unitary gauge
    (Goldstones eaten).  We keep Goldstones explicit here for generality.

    Returns
    -------
    L_int : SymPy expression (interaction terms, no kinetic terms)
    """
    g       = gauge_fields['g']
    g_prime = gauge_fields['g_prime']
    v       = scalar_fields['v']
    h       = scalar_fields['h']
    G0      = scalar_fields['G0']
    G1      = scalar_fields['G1']
    G2      = scalar_fields['G2']

    theta_W  = symbols(r'\theta_W', positive=True)

    sin_thW = weinberg['sin_thW']
    cos_thW = weinberg['cos_thW']

    Wp = symbols('W^+')
    Wm = symbols('W^-')
    Z  = symbols('Z')
    A  = symbols('A')

    # Gauge boson masses expressed via v
    mW = g * v / 2
    mZ = g * v / (2 * cos_thW)

    # The gauge-scalar interaction Lagrangian (from expanding |D_mu H|^2)
    # Quadratic in gauge fields × (v+h)^2 piece:
    #   L_VVH = g^2/4 (v+h)^2 W^+W^-  +  g^2/(8 cos^2θ) (v+h)^2 ZZ
    # We write this symbolically with mW, mZ:
    #   g^2/4 = mW^2/v^2 etc., but keep g explicit for clarity.

    L_WW = g**2 / 4 * (v + h)**2 * Wp * Wm
    L_ZZ = g**2 / (8 * cos_thW**2) * (v + h)**2 * Z * Z

    L_int = (L_WW + L_ZZ).expand()

    return L_int


def extract_sm_vertices(L_int, scalar_fields, gauge_fields, weinberg):
    """
    Extract the key SM Feynman rule vertices from L_int.

    Returns a dict of vertex name -> coupling coefficient.
    """
    from sympy import diff, symbols as sym
    g       = gauge_fields['g']
    g_prime = gauge_fields['g_prime']
    v       = scalar_fields['v']
    h       = scalar_fields['h']

    cos_thW = weinberg['cos_thW']

    Wp = sym('W^+')
    Wm = sym('W^-')
    Z  = sym('Z')

    # Field vacuum: evaluate derivatives at h=0 (fields set to zero after diff)
    vac = {h: 0}

    # HWW: d/dh d/dW+ d/dW- |_{vac}
    hww  = factor(diff(L_int, h, Wp, Wm).subs(vac))
    # HZZ: d/dh d/dZ d/dZ |_{vac}  — divide by 2 for identical fields
    hzz  = factor(diff(L_int, h, Z, Z).subs(vac))
    # HHWW: d^2/dh^2 d/dW+ d/dW- |_{vac}
    hhww = factor(diff(L_int, h, h, Wp, Wm).subs(vac))
    # HHZZ: d^2/dh^2 d^2/dZ^2 |_{vac}
    hhzz = factor(diff(L_int, h, h, Z, Z).subs(vac))

    mW = g * v / 2
    mZ = g * v / (2 * cos_thW)

    vertices = {
        'HWW':  hww,
        'HZZ':  hzz,
        'HHWW': hhww,
        'HHZZ': hhzz,
        # Expected analytic forms
        'HWW_expected':  2 * mW**2 / v,
        'HZZ_expected':  2 * mZ**2 / v,
        'HHWW_expected': g**2 / 2,
        'HHZZ_expected': g**2 / (2 * cos_thW**2),
    }

    return vertices


def build_sm_scalar_vertices(scalar_fields, params, tadpole_sol):
    """
    Extract Higgs self-couplings from V_real (the potential in real-field basis).

    Feynman rule convention: vertex factor = i * d^n L / d phi_1 ... d phi_n
    For V = +lambda|H|^4 - mu^2|H|^2, the HHH vertex (from -V) is:
        -d^3V/dh^3 |_{h=0}

    Returns
    -------
    vertices : dict {name: coupling}
    """
    from sympy import diff
    from .scalar_potential import build_sm_scalar_mass_matrix, build_sm_potential

    V, pot_params = build_sm_potential(scalar_fields)
    M2, real_fields, V_real = build_sm_scalar_mass_matrix(
        V, scalar_fields, pot_params, tadpole_sol
    )

    h       = scalar_fields['h']
    G0      = scalar_fields['G0']
    G1      = scalar_fields['G1']
    G2      = scalar_fields['G2']
    v       = scalar_fields['v']
    lambda_ = params['lambda_']

    vac = {h: 0, G0: 0, G1: 0, G2: 0}

    # The Lagrangian contribution from potential is -V_real
    # Feynman vertex = d^n (-V_real) / d fields  evaluated at vacuum
    L_scalar = -V_real

    # Triple Higgs coupling:  d^3(-V)/dh^3 |_vac
    hhh  = factor(diff(L_scalar, h, h, h).subs(vac))
    # Quartic: d^4(-V)/dh^4 |_vac
    hhhh = factor(diff(L_scalar, h, h, h, h).subs(vac))

    # Higgs mass (from the quadratic term)
    mh_sq = factor(M2[0, 0].subs(vac))

    vertices = {
        'HHH':  hhh,
        'HHHH': hhhh,
        'HHH_expected':  -6 * lambda_ * v,
        'HHHH_expected': -6 * lambda_,
        'mh_sq': mh_sq,
        'mh_sq_expected': 2 * lambda_ * v**2,
    }

    return vertices
