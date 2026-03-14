"""
Layer 4 — diagonalization.py
==============================
Rotate gauge and scalar sectors to the physical mass basis.

For the SM:
  Gauge sector:  W^± = (W^1 ∓ i W^2)/sqrt(2),
                 Z = cos θ_W W^3 - sin θ_W B,
                 A = sin θ_W W^3 + cos θ_W B
                 with tan θ_W = g'/g

  Scalar sector: physical Higgs h (massive), three Goldstones (massless)
"""

from sympy import (
    symbols, sqrt, I, cos, sin, atan2, trigsimp,
    Matrix, Rational, factor, simplify, expand,
    Symbol, Eq, solve, conjugate, tan
)


# ---------------------------------------------------------------------------
# SM gauge sector diagonalization
# ---------------------------------------------------------------------------

def diagonalize_sm_gauge(gauge_fields, mass_matrix):
    """
    Diagonalize the SM 4×4 gauge boson mass matrix and return:
      - physical field substitutions
      - mass eigenvalues
      - the Weinberg angle definition

    The mass matrix in the basis [W1, W2, W3, B] has the block structure:
        diag(g^2 v^2/4, g^2 v^2/4, [2x2 neutral sector])

    Parameters
    ----------
    gauge_fields : dict  (from make_sm_gauge_group)
    mass_matrix  : Matrix (4×4, from build_sm_kinetic)

    Returns
    -------
    phys_fields  : dict  of new physical field symbols
    change_basis : dict  (weak eigenstate -> physical expression)
    mass_sq      : dict  {field_name: mass^2}
    weinberg     : dict  {sin_thW: expr, cos_thW: expr, tan_thW: expr}
    """
    g       = gauge_fields['g']
    g_prime = gauge_fields['g_prime']
    W1      = gauge_fields['W1']
    W2      = gauge_fields['W2']
    W3      = gauge_fields['W3']
    B       = gauge_fields['B']

    v        = symbols('v', positive=True)
    theta_W  = symbols(r'\theta_W', positive=True)

    # Physical fields
    Wp = symbols('W^+')
    Wm = symbols('W^-')
    Z  = symbols('Z')
    A  = symbols('A')

    phys_fields = {'W^+': Wp, 'W^-': Wm, 'Z': Z, 'A': A}

    # Charge basis for W
    # W^+ = (W1 - i W2)/sqrt(2),  W^- = (W1 + i W2)/sqrt(2)
    # Inverse: W1 = (W^+ + W^-)/sqrt(2),  W2 = i(W^+ - W^-)/sqrt(2)  [Note: W2 = i(...)]
    # Neutral sector: Z = cos W3 - sin B,  A = sin W3 + cos B
    change_basis = {
        W1: (Wp + Wm) / sqrt(2),
        W2: I * (Wp - Wm) / sqrt(2),
        W3:  cos(theta_W) * Z + sin(theta_W) * A,
        B:  -sin(theta_W) * Z + cos(theta_W) * A,
    }

    # Mass eigenvalues (analytic for SM)
    mW_sq = g**2 * v**2 / 4
    mZ_sq = (g**2 + g_prime**2) * v**2 / 4

    mass_sq = {
        'W^+': mW_sq,
        'W^-': mW_sq,
        'Z':   mZ_sq,
        'A':   0,
    }

    # Weinberg angle: tan θ_W = g'/g
    sin_thW = g_prime / sqrt(g**2 + g_prime**2)
    cos_thW = g        / sqrt(g**2 + g_prime**2)

    weinberg = {
        'sin_thW': sin_thW,
        'cos_thW': cos_thW,
        'tan_thW': g_prime / g,
        'mZ_over_mW': sqrt(mZ_sq / mW_sq),
    }

    return phys_fields, change_basis, mass_sq, weinberg


def verify_sm_gauge_masses(mass_matrix, gauge_fields, change_basis, theta_W_value=None):
    """
    Rotate M^2 into the physical (W^+, W^-, Z, A) basis using explicit
    g, g' expressions for sin/cos θ_W — no trig symbols needed.

    Returns
    -------
    M2_phys : 4×4 Matrix (should be diagonal)
    checks  : dict with 'diagonal', 'mW_sq', 'mZ_sq', 'mA_sq'
    """
    from sympy import S as Sz

    g       = gauge_fields['g']
    g_prime = gauge_fields['g_prime']

    W1 = gauge_fields['W1']
    W2 = gauge_fields['W2']
    W3 = gauge_fields['W3']
    B  = gauge_fields['B']

    Wp = symbols('W^+')
    Wm = symbols('W^-')
    Z  = symbols('Z')
    A  = symbols('A')

    # Explicit Weinberg expressions (no trig symbols)
    sin_thW = g_prime / sqrt(g**2 + g_prime**2)
    cos_thW = g        / sqrt(g**2 + g_prime**2)

    # Physical → weak eigenstate map (inverse of change_basis)
    # W1 = (W^+ + W^-)/√2,  W2 = i(W^+ - W^-)/√2
    # W3 = cos_thW Z + sin_thW A,  B = -sin_thW Z + cos_thW A
    phys_to_weak = {
        W1: (Wp + Wm) / sqrt(2),
        W2: I * (Wp - Wm) / sqrt(2),
        W3:  cos_thW * Z + sin_thW * A,
        B:  -sin_thW * Z + cos_thW * A,
    }

    orig_fields = [W1, W2, W3, B]
    phys_fields_list = [Wp, Wm, Z, A]

    # Extract transformation coefficients T_{a,i}: orig[a] = sum_i T_{a,i} phys[i]
    T = Matrix(4, 4, lambda a, i:
        phys_to_weak[orig_fields[a]].coeff(phys_fields_list[i])
    )

    # M2_phys = T^† M2 T
    from sympy.physics.quantum import Dagger
    M2_phys = (Dagger(T) * mass_matrix * T).applyfunc(
        lambda x: factor(expand(x))
    )

    is_diagonal = all(
        M2_phys[i, j] == 0
        for i in range(4) for j in range(4) if i != j
    )

    checks = {
        'diagonal': is_diagonal,
        'mW_sq':    factor(M2_phys[0, 0]),
        'mZ_sq':    factor(M2_phys[2, 2]),
        'mA_sq':    factor(M2_phys[3, 3]),
    }

    return M2_phys, checks


# ---------------------------------------------------------------------------
# SM scalar sector diagonalization (trivial — already diagonal)
# ---------------------------------------------------------------------------

def diagonalize_sm_scalar(M2_scalar, real_fields):
    """
    The SM scalar mass matrix is already diagonal in the [h, G0, G1, G2] basis.
    Return the physical field assignments.

    Returns
    -------
    physical_higgs : dict  {field: mass^2}
    goldstones     : list of field names that are massless
    """
    from sympy import factor
    result = {}
    goldstones = []
    for i, f in enumerate(real_fields):
        m2 = factor(M2_scalar[i, i])
        result[str(f)] = m2
        if m2 == 0:
            goldstones.append(str(f))

    return result, goldstones
