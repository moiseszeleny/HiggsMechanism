"""
Layer 3 — scalar_potential.py
================================
Write the scalar potential V(Phi), apply tadpole conditions to fix
parameters, build the scalar mass matrix M^2_S, and identify Goldstone
bosons as zero eigenvectors.

For the SM:  V = -mu^2 |H|^2 + lambda |H|^4
             where |H|^2 = phi^+*phi^- + |phi^0|^2

After SSB: mu^2 = lambda v^2  (tadpole condition)
  Physical Higgs mass: m_h^2 = 2 lambda v^2 = 2 mu^2
  Three Goldstone bosons: G^+, G^-, G^0  (massless)
"""

from sympy import (
    symbols, conjugate, expand, factor, solve,
    sqrt, I, Rational, Matrix, diff, simplify, zeros,
    Eq, Symbol
)

from .symbolic_tools import build_mass_matrix


# ---------------------------------------------------------------------------
# SM scalar potential
# ---------------------------------------------------------------------------

def build_sm_potential(scalar_fields):
    """
    V_SM = -mu^2 (phi^+*phi^- + phi^0*phi^0) + lambda (phi^+*phi^- + phi^0*phi^0)^2

    Parameters
    ----------
    scalar_fields : dict
        From make_sm_higgs_doublet().

    Returns
    -------
    V : SymPy expression (symbolic potential in weak eigenstates)
    params : dict  {mu2, lambda_}
    """
    phi_plus  = scalar_fields['phi_plus']
    phi_zero  = scalar_fields['phi_zero']
    phi_minus = scalar_fields['phi_minus']

    mu2     = symbols(r'\mu^2',     positive=True)   # mu^2 > 0 for SSB
    lambda_ = symbols(r'\lambda',   positive=True)

    # |H|^2 = phi^+ phi^- + phi^0* phi^0
    # (treating phi^0 as real neutral field for mass matrix purposes;
    #  the imaginary part is the Goldstone)
    H2 = phi_plus * phi_minus + conjugate(phi_zero) * phi_zero

    V = -mu2 * H2 + lambda_ * H2**2

    params = {'mu2': mu2, 'lambda_': lambda_}
    return V, params


def apply_tadpole_sm(V, scalar_fields, params):
    """
    Apply the tadpole condition  dV/d(phi^0) |_VEV = 0  to fix mu^2 = lambda v^2.

    Returns
    -------
    tadpole_sol : dict  {mu2: lambda v^2}
    """
    phi_zero  = scalar_fields['phi_zero']
    phi_plus  = scalar_fields['phi_plus']
    phi_minus = scalar_fields['phi_minus']
    v         = scalar_fields['v']
    mu2       = params['mu2']
    lambda_   = params['lambda_']

    # Substitute phi^0 -> v/sqrt(2) + h/sqrt(2),  phi^± -> 0
    # and differentiate with respect to h then set h=0
    h = symbols('h', real=True)
    V_expanded = V.subs({phi_zero: v/sqrt(2) + h/sqrt(2),
                         conjugate(phi_zero): v/sqrt(2) + h/sqrt(2),
                         phi_plus: 0, phi_minus: 0})
    V_expanded = expand(V_expanded)

    # Tadpole: coefficient of h^1 must vanish
    tadpole = diff(V_expanded, h).subs(h, 0)
    sol = solve(tadpole, mu2)
    assert len(sol) == 1, f"Unexpected tadpole solutions: {sol}"

    return {mu2: sol[0]}


def build_sm_scalar_mass_matrix(V, scalar_fields, params, tadpole_sol):
    """
    Build the scalar mass matrix in the real-field basis.

    Real fields: h, G0, G1, G2  (where phi^0 = (v+h+iG0)/sqrt(2),
                                        phi^+ = (G1+iG2)/sqrt(2))

    Parameters
    ----------
    V            : symbolic potential
    scalar_fields: dict from make_sm_higgs_doublet()
    params       : dict {mu2, lambda_}
    tadpole_sol  : dict from apply_tadpole_sm()

    Returns
    -------
    M2_scalar  : Matrix  (4×4, in basis [h, G0, G1, G2])
    real_fields: list of symbols
    V_real     : potential in terms of real fields
    """
    phi_plus  = scalar_fields['phi_plus']
    phi_zero  = scalar_fields['phi_zero']
    phi_minus = scalar_fields['phi_minus']
    v         = scalar_fields['v']

    h   = scalar_fields['h']
    G0  = scalar_fields['G0']
    G1  = scalar_fields['G1']
    G2  = scalar_fields['G2']

    # Substitute real-field decompositions
    real_subs = {
        phi_zero:           (v + h + I * G0) / sqrt(2),
        conjugate(phi_zero):(v + h - I * G0) / sqrt(2),
        phi_plus:           (G1 + I * G2) / sqrt(2),
        phi_minus:          (G1 - I * G2) / sqrt(2),
    }

    V_real = V.subs(real_subs).expand()

    # Apply tadpole (mu^2 = lambda v^2)
    V_real = V_real.subs(tadpole_sol).expand()

    real_fields = [h, G0, G1, G2]
    M2 = build_mass_matrix(V_real, real_fields, real_fields)

    return M2, real_fields, V_real


def identify_goldstones(M2, real_fields):
    """
    Find zero eigenvalues of M2 and identify the corresponding Goldstone fields.
    Evaluates the mass matrix at the vacuum (all fluctuation fields = 0).

    Returns
    -------
    goldstones : list of field labels (str)
    physical   : list of field labels (str)
    masses_sq  : dict  {field: mass^2}
    """
    from sympy import factor
    goldstones = []
    physical   = []
    masses_sq  = {}

    # Evaluate at the vacuum point (all fluctuation fields = 0)
    vacuum_subs = {f: 0 for f in real_fields}

    for i, f in enumerate(real_fields):
        m2 = M2[i, i].subs(vacuum_subs)
        m2_simplified = factor(m2)
        masses_sq[str(f)] = m2_simplified
        if m2_simplified == 0:
            goldstones.append(str(f))
        else:
            physical.append(str(f))

    return goldstones, physical, masses_sq
