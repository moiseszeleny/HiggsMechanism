"""
Layer 1 — scalar_sector.py
===========================
Register scalar multiplets, assign VEVs, and parameterise fluctuations
as  Phi_i = vev_i + (real + i*imag) / sqrt(2).

Each ScalarMultiplet carries:
  - its component symbols (complex or real fields)
  - the VEV substitution dict
  - the real-component dict (h_k, pi_k  around the VEV)
  - charge quantum numbers used by covariant_kinetic
"""

from sympy import (
    symbols, Matrix, sqrt, conjugate, I, zeros, Rational, Symbol
)


# ---------------------------------------------------------------------------
# ScalarMultiplet
# ---------------------------------------------------------------------------

class ScalarMultiplet:
    """
    One scalar multiplet in a given representation of the gauge group.

    Parameters
    ----------
    name : str
        Label, e.g. 'Higgs_doublet'.
    components : list[Symbol]
        Complex field symbols, length = dim of representation.
    vev : list
        VEV for each component (0 for most; the neutral component gets v).
    gauge_reps : list
        Representation data passed to GaugeGroup.covariant_derivative,
        one entry per gauge factor.  Use None for singlet factors.
    real_components : dict, optional
        Map  complex_symbol -> real_expression  (after decomposing into
        real + imaginary fluctuations around the VEV).
        If None it is built automatically for the neutral component.
    """

    def __init__(self, name, components, vev, gauge_reps,
                 real_components=None):
        self.name = name
        self.components = components          # list of SymPy symbols
        self.vev = vev                        # list, same length as components
        self.gauge_reps = gauge_reps          # list, one per gauge factor
        self.dim = len(components)

        assert len(vev) == self.dim

        # Build vev substitution dict
        self.vev_subs = {c: v for c, v in zip(components, vev)}

        # Real-component decomposition (provided or empty by default)
        self.real_components = real_components if real_components else {}

    @property
    def vev_matrix(self):
        return Matrix([v for v in self.vev])

    def __repr__(self):
        return f"ScalarMultiplet({self.name}, dim={self.dim})"


# ---------------------------------------------------------------------------
# Helpers for decomposing complex scalars around VEVs
# ---------------------------------------------------------------------------

def decompose_complex_scalar(symbol, vev_value, h_name, pi_name):
    """
    phi = vev + (h + i*pi) / sqrt(2)

    Returns
    -------
    h, pi : symbols
    subs  : dict  {symbol: decomposed_expression}
    """
    h   = symbols(h_name,  real=True)
    pi  = symbols(pi_name, real=True)
    expr = vev_value + (h + I * pi) / sqrt(2)
    return h, pi, {symbol: expr}


def decompose_charged_scalar(symbol_plus, symbol_minus,
                              h_plus_name, h_minus_name):
    """
    For a charged pair phi^+, phi^-:  treat as independent symbols
    (no VEV).  Return the complex conjugate identification.

    phi^- = (phi^+)* 
    """
    hp = symbols(h_plus_name)
    hm = symbols(h_minus_name)
    return hp, hm, {symbol_plus: hp, symbol_minus: hm,
                    conjugate(hp): hm, conjugate(hm): hp}


# ---------------------------------------------------------------------------
# SM Higgs doublet builder
# ---------------------------------------------------------------------------

def make_sm_higgs_doublet(gauge_group_fields):
    """
    Build the SM Higgs doublet  H = (phi^+, phi^0)  with VEV  <phi^0> = v/sqrt(2).

    The full decomposition around the VEV:
        phi^+ = (phi1 + i*phi2) / sqrt(2)        (Goldstones G^+/G^-)
        phi^0 = (v + h + i*G^0) / sqrt(2)         (Higgs h, Goldstone G^0)

    where phi1, phi2 are real.  In the standard convention the charged
    Goldstone is  G^+ = phi^+ itself (single complex symbol).

    Parameters
    ----------
    gauge_group_fields : dict
        Output of make_sm_gauge_group(), contains 'g', 'g_prime'.

    Returns
    -------
    doublet : ScalarMultiplet
    field_dict : dict  of all symbols (h, G0, Gp, Gm, phi^+, phi^0, v)
    """
    g       = gauge_group_fields['g']
    g_prime = gauge_group_fields['g_prime']

    # SM Higgs VEV
    v = symbols('v', positive=True)

    # Weak eigenstates (complex fields)
    phi_plus  = symbols(r'\phi^+')
    phi_zero  = symbols(r'\phi^0')
    phi_minus = symbols(r'\phi^-')   # conjugate of phi_plus

    # Physical / Goldstone fields (real)
    h   = symbols('h',  real=True)     # physical Higgs
    G0  = symbols('G0', real=True)     # neutral Goldstone
    G1  = symbols('G1', real=True)     # real part of charged Goldstone
    G2  = symbols('G2', real=True)     # imag part of charged Goldstone

    # Convenience: complex charged Goldstone symbols
    Gp  = symbols('G^+')
    Gm  = symbols('G^-')

    # Decompositions around the VEV
    #   phi^0 = (v + h + i G0) / sqrt(2)
    #   phi^+ = (G1 + i G2) / sqrt(2)   ≡ G^+ / sqrt(2) ... but we keep
    #            the symbol phi^+ and parameterise it directly.
    real_components = {
        phi_zero:  (v + h + I * G0) / sqrt(2),
        phi_plus:  (G1 + I * G2) / sqrt(2),
        phi_minus: (G1 - I * G2) / sqrt(2),
    }

    # VEV substitution  (apply to get mass matrices etc.)
    vev_subs = {
        phi_zero:  v / sqrt(2),
        phi_plus:  0,
        phi_minus: 0,
    }

    # Gauge representations for each factor [SU2_L, U1_Y]:
    #   Doublet H = (phi^+, phi^0)  transforms as  (2, +1/2)
    #   SU(2)_L representation: T^a (fundamental), handled via gauge_matrix
    #   U(1)_Y representation:  hypercharge Y = +1/2

    # We will build the SU(2)_L gauge matrix in covariant_kinetic using
    # the group's su2_generators. Here we just record the hypercharge.
    gauge_reps = [
        'fundamental',           # SU2_L: use fundamental generators
        {'hypercharge': Rational(1, 2)},  # U1_Y: Y = 1/2
    ]

    doublet = ScalarMultiplet(
        name='SM_Higgs_doublet',
        components=[phi_plus, phi_zero],
        vev=[0, v / sqrt(2)],
        gauge_reps=gauge_reps,
        real_components=real_components,
    )

    field_dict = {
        'phi_plus':  phi_plus,
        'phi_zero':  phi_zero,
        'phi_minus': phi_minus,
        'h':  h,
        'G0': G0,
        'G1': G1,
        'G2': G2,
        'Gp': Gp,
        'Gm': Gm,
        'v':  v,
    }

    return doublet, field_dict
