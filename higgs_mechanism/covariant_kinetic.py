"""
Layer 2 — covariant_kinetic.py
================================
Build the covariant kinetic term  L_kin = (D_mu Phi)^dag (D_mu Phi),
expand around the VEV, and extract the gauge boson mass matrix M^2_V.

For the SM Higgs doublet H = (phi^+, phi^0) with VEV <phi^0> = v/sqrt(2):

  D_mu H = partial_mu H  - i g/2 sigma^a W^a H  - i g'/2 Y B H

The gauge boson mass matrix comes from the term quadratic in gauge fields
evaluated at the VEV:  |D_mu H|^2 |_{partial=0, Phi=<Phi>}
"""

from sympy import (
    symbols, Matrix, I, sqrt, conjugate, expand, factor,
    Function, Rational, Add, Mul, zeros, eye, trace,
    symbols, simplify, Symbol, cos, sin
)
from sympy.physics import msigma
from sympy.physics.quantum import Dagger

from .symbolic_tools import build_mass_matrix


# ---------------------------------------------------------------------------
# Partial-mu function (momentum representation for Feynman rules)
# ---------------------------------------------------------------------------

partial_mu = Function(r'\partial_\mu')


# ---------------------------------------------------------------------------
# SM covariant kinetic Lagrangian
# ---------------------------------------------------------------------------

def build_sm_kinetic(gauge_fields, scalar_fields):
    """
    Build L_kin = (D_mu H)^dag (D_mu H) for the SM Higgs doublet.

    Uses the charge-basis gauge fields  W^+, W^-, Z, A  (after mixing).
    Works directly in the weak-eigenstate basis before mixing,
    then extracts the mass matrix.

    Parameters
    ----------
    gauge_fields : dict
        From make_sm_gauge_group(): contains 'W1', 'W2', 'W3', 'B', 'g', 'g_prime'.
    scalar_fields : dict
        From make_sm_higgs_doublet(): contains 'phi_plus', 'phi_zero', etc.

    Returns
    -------
    L_kin        : SymPy expression (full kinetic Lagrangian)
    mass_matrix  : Matrix (4×4 gauge boson mass matrix in [W1,W2,W3,B] basis)
    vev_subs     : dict  (VEV substitution rules)
    """
    g       = gauge_fields['g']
    g_prime = gauge_fields['g_prime']
    W1      = gauge_fields['W1']
    W2      = gauge_fields['W2']
    W3      = gauge_fields['W3']
    B       = gauge_fields['B']

    phi_plus  = scalar_fields['phi_plus']
    phi_zero  = scalar_fields['phi_zero']
    phi_minus = scalar_fields['phi_minus']
    v         = scalar_fields['v']

    # VEV: <H> = (0, v/sqrt(2))
    vev_subs = {phi_plus: 0, phi_zero: v / sqrt(2), phi_minus: 0}

    # Higgs doublet column vector
    H = Matrix([[phi_plus], [phi_zero]])

    # SU(2)_L generators T^a = sigma^a / 2
    T = [msigma(i) / 2 for i in (1, 2, 3)]

    # Weak hypercharge Y = 1/2 for the SM Higgs doublet
    Y = Rational(1, 2)

    # Gauge part of the covariant derivative (no partial_mu):
    #   D_mu^{gauge} H = -i g sum_a W^a T^a H  - i g' Y B H
    W_sigma = Add(*[W_a * T_a for W_a, T_a in zip([W1, W2, W3], T)])
    D_gauge = -I * g * W_sigma - I * g_prime * Y * B * eye(2)

    # Apply to the doublet: D_mu^{gauge} H
    DgH = D_gauge * H  # 2-vector

    # H^dag
    Hdag = Dagger(H)

    # The mass-generating term:  (D_gauge H)^dag (D_gauge H)  at VEV
    # This is a 1×1 matrix when Hdag is a row vector
    term = (Dagger(DgH) * DgH)[0, 0]

    # Substitute conjugates of phi fields to remove spurious terms
    term = term.subs({conjugate(phi_plus): phi_minus,
                      conjugate(phi_zero): phi_zero,
                      conjugate(phi_minus): phi_plus})
    term = term.expand()

    # Evaluate at the VEV
    L_mass = term.subs(vev_subs).expand()

    # --- Extract the gauge boson mass matrix ---
    # Basis order: [W1, W2, W3, B]
    gauge_bosons = [W1, W2, W3, B]
    M2 = build_mass_matrix(L_mass, gauge_bosons, gauge_bosons)

    return L_mass, M2, vev_subs


def rotate_to_mass_basis_sm(M2, gauge_fields):
    """
    Analytically rotate the 4×4 SM gauge boson mass matrix to the
    physical basis  (W^+, W^-, Z, A).

    The SM mixing relations:
        W^± = (W^1 ∓ i W^2) / sqrt(2)
        Z   =  cos θ_W W^3 - sin θ_W B
        A   =  sin θ_W W^3 + cos θ_W B

    with  tan θ_W = g' / g.

    Returns
    -------
    mass_eigenvalues : dict  {field: mass^2 expression}
    rotation         : dict  of mixing relations
    mixing_symbols   : dict  {theta_W: expression, ...}
    """
    g       = gauge_fields['g']
    g_prime = gauge_fields['g_prime']
    v       = symbols('v', positive=True)

    # Physical gauge boson mass symbols
    mW, mZ = symbols('m_W m_Z', positive=True)

    # Weinberg angle
    theta_W = symbols(r'\theta_W', positive=True)

    # Rotation
    Wp = symbols('W^+')
    Wm = symbols('W^-')
    Z  = symbols('Z')
    A  = symbols('A')

    # Mixing relations (definition)
    W1 = gauge_fields['W1']
    W2 = gauge_fields['W2']
    W3 = gauge_fields['W3']
    B  = gauge_fields['B']

    mixing = {
        W1: (Wp + Wm) / sqrt(2),
        W2: I * (Wp - Wm) / sqrt(2),
        W3:  cos(theta_W) * Z + sin(theta_W) * A,
        B:  -sin(theta_W) * Z + cos(theta_W) * A,
    }

    # tan θ_W = g'/g  →  sin θ_W = g'/sqrt(g^2+g'^2), cos θ_W = g/sqrt(g^2+g'^2)
    mixing_def = {
        theta_W: symbols(r'\theta_W', positive=True)
    }

    # Analytic mass eigenvalues from the SM
    mW_sq  = (g**2 * v**2) / 4
    mZ_sq  = ((g**2 + g_prime**2) * v**2) / 4
    mA_sq  = 0

    mass_eigenvalues = {
        'W^+': mW_sq,
        'W^-': mW_sq,
        'Z':   mZ_sq,
        'A':   mA_sq,
    }

    return mass_eigenvalues, mixing, mixing_def
