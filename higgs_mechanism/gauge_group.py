"""
Layer 0 — gauge_group.py
========================
Define a gauge group G = G1 × G2 × ... with generators, gauge fields,
couplings, and the covariant derivative operator.

Each factor is a GaugeFactor (U1, SU2, SU3).
A GaugeGroup is a product of GaugeFactor objects.
"""

from sympy import (
    symbols, Matrix, I, sqrt, zeros, eye,
    Function, Symbol, Add, Mul, conjugate
)
from sympy.physics import msigma


# ---------------------------------------------------------------------------
# SU(N) / U(1) generator libraries
# ---------------------------------------------------------------------------

def u1_generator():
    """U(1): the single generator is just the identity on C^1 (normalization 1)."""
    return [Matrix([[1]])]


def su2_generators():
    """
    SU(2) generators in the fundamental (doublet) representation: T^a = sigma^a / 2.
    Returns list [T1, T2, T3].
    """
    return [msigma(i) / 2 for i in (1, 2, 3)]


def su2_generators_adjoint():
    """
    SU(2) generators in the adjoint (triplet) representation.
    (T^a)_bc = -i epsilon_abc.
    """
    eps = [
        [[0, 0, 0], [0, 0, -1], [0, 1, 0]],
        [[0, 0, 1], [0, 0, 0], [-1, 0, 0]],
        [[0, -1, 0], [1, 0, 0], [0, 0, 0]],
    ]
    return [Matrix(e) * (-I) for e in eps]


def su3_generators():
    """SU(3) Gell-Mann matrices / 2 (fundamental)."""
    from sympy import Rational
    lam = [None] * 8
    lam[0] = Matrix([[0,1,0],[1,0,0],[0,0,0]])
    lam[1] = Matrix([[0,-I,0],[I,0,0],[0,0,0]])
    lam[2] = Matrix([[1,0,0],[0,-1,0],[0,0,0]])
    lam[3] = Matrix([[0,0,1],[0,0,0],[1,0,0]])
    lam[4] = Matrix([[0,0,-I],[0,0,0],[I,0,0]])
    lam[5] = Matrix([[0,0,0],[0,0,1],[0,1,0]])
    lam[6] = Matrix([[0,0,0],[0,0,-I],[0,I,0]])
    lam[7] = Matrix([[1,0,0],[0,1,0],[0,0,-2]]) / sqrt(3)
    return [l / 2 for l in lam]


# ---------------------------------------------------------------------------
# GaugeFactor
# ---------------------------------------------------------------------------

class GaugeFactor:
    """
    Represents one simple or abelian factor G_i in the gauge group.

    Parameters
    ----------
    name : str
        Label, e.g. 'SU2_L', 'U1_Y'.
    group_type : str
        One of 'U1', 'SU2', 'SU3'.
    coupling_symbol : Symbol
        SymPy symbol for the gauge coupling g_i.
    field_symbols : list[Symbol]
        Gauge boson symbols A^a_mu (length = dim of adjoint).
    hypercharge : expr, optional
        For U(1) factors the hypercharge Y is carried by the scalar, not by
        the generator here. Set to None; pass Y per-multiplet.
    """

    _generators = {
        'U1':  (lambda: [Matrix([[1]])],),
        'SU2': (su2_generators,),
        'SU3': (su3_generators,),
    }

    def __init__(self, name, group_type, coupling_symbol, field_symbols):
        self.name = name
        self.group_type = group_type
        self.g = coupling_symbol
        self.fields = field_symbols          # gauge boson symbols

        if group_type == 'U1':
            self.generators_fundamental = [Matrix([[1]])]
            self.dim = 1
        elif group_type == 'SU2':
            self.generators_fundamental = su2_generators()
            self.dim = 3
        elif group_type == 'SU3':
            self.generators_fundamental = su3_generators()
            self.dim = 8
        else:
            raise ValueError(f"Unknown group type: {group_type}")

        assert len(field_symbols) == self.dim, (
            f"{name}: expected {self.dim} gauge fields, got {len(field_symbols)}"
        )

    def gauge_matrix(self, representation='fundamental', hypercharge=None):
        """
        Return the matrix  g * sum_a  A^a T^a_R  for this factor.

        For U(1): g * Y * A_mu  (hypercharge Y must be supplied).
        For SU(2)/SU(3): g * sum_a A^a * T^a  in the fundamental rep.
        """
        if self.group_type == 'U1':
            if hypercharge is None:
                raise ValueError("U(1) factor requires a hypercharge value.")
            return self.g * hypercharge * self.fields[0] * Matrix([[1]])
        else:
            gens = self.generators_fundamental
            return self.g * Add(*[self.fields[a] * gens[a]
                                  for a in range(self.dim)])

    def __repr__(self):
        return f"GaugeFactor({self.name}, {self.group_type}, g={self.g})"


# ---------------------------------------------------------------------------
# GaugeGroup  (product of factors)
# ---------------------------------------------------------------------------

class GaugeGroup:
    """
    Product gauge group G = G_1 × G_2 × ...

    Parameters
    ----------
    factors : list[GaugeFactor]
    """

    def __init__(self, factors):
        self.factors = factors

    @property
    def all_gauge_fields(self):
        fields = []
        for f in self.factors:
            fields.extend(f.fields)
        return fields

    def covariant_derivative(self, field_matrix, partial_mu_matrix,
                             representations):
        """
        Build D_mu * Phi  = partial_mu Phi - i * sum_factors g_i A^a T^a_R Phi
        for a scalar multiplet Phi.

        Parameters
        ----------
        field_matrix : Matrix
            The scalar field column vector (or matrix for bidoublets).
        partial_mu_matrix : Matrix
            partial_mu applied to each component of field_matrix.
        representations : list
            One entry per gauge factor; each entry is either:
              - a SymPy matrix  g_i * sum A^a T^a  (pre-built), or
              - a dict with key 'hypercharge' for U(1) factors.
            If entry is None the factor acts trivially (singlet).

        Returns
        -------
        Matrix
            D_mu Phi
        """
        assert len(representations) == len(self.factors)

        result = partial_mu_matrix
        for factor, rep in zip(self.factors, representations):
            if rep is None:
                continue
            if isinstance(rep, dict):
                Y = rep['hypercharge']
                G_mat = factor.gauge_matrix(hypercharge=Y)
            else:
                G_mat = rep

            # D_mu Phi += -i G_mat Phi  (left-action)
            result = result - I * G_mat * field_matrix

        return result

    def covariant_derivative_rhs(self, field_matrix, partial_mu_matrix,
                                  representations, right_action=None):
        """
        For bidoublets: D_mu Phi = partial_mu Phi - i G_L Phi + i Phi G_R
        Pass right_action as the right-hand gauge matrix.
        """
        result = self.covariant_derivative(field_matrix, partial_mu_matrix,
                                           representations)
        if right_action is not None:
            result = result + I * field_matrix * right_action
        return result

    def __repr__(self):
        return "GaugeGroup([" + ", ".join(str(f) for f in self.factors) + "])"


# ---------------------------------------------------------------------------
# Convenience builder: Standard Model gauge group
# ---------------------------------------------------------------------------

def make_sm_gauge_group():
    """
    Build the SM gauge group SU(2)_L × U(1)_Y.
    (We omit SU(3)_C since it does not break.)

    Returns
    -------
    GaugeGroup
        With factors [SU2_L, U1_Y].
    dict
        Named gauge field symbols for convenience.
    """
    g, g_prime = symbols('g g_prime', positive=True)

    # SU(2)_L: W^1, W^2, W^3
    W1, W2, W3 = symbols('W1 W2 W3', real=True)
    su2L = GaugeFactor('SU2_L', 'SU2', g, [W1, W2, W3])

    # U(1)_Y: B
    B = symbols('B', real=True)
    u1Y = GaugeFactor('U1_Y', 'U1', g_prime, [B])

    group = GaugeGroup([su2L, u1Y])

    fields = {
        'W1': W1, 'W2': W2, 'W3': W3, 'B': B,
        'g': g, 'g_prime': g_prime,
    }
    return group, fields
