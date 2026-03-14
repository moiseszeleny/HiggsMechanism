"""
Shared symbolic tools — adapted from DLRSM1/symbolic_tools.py
=============================================================
Contains: build_mass_matrix, extract_interaction_coefficients,
          PartialMu, momentum, generate_latex_table
"""

from sympy import (
    diff, S, factorial, Mul, Function, I, Symbol, expand,
    Pow, cancel, Poly, Add, Integer, Basic, latex, simplify
)
from sympy.utilities.iterables import multiset_permutations
from sympy import derive_by_array


# ---------------------------------------------------------------------------
# Momentum / partial-mu helpers
# ---------------------------------------------------------------------------

momentum = Function('p')


class PartialMu(Function):
    """Emulate the differential operator ∂_μ (momentum-space rule)."""

    @classmethod
    def eval(cls, arg):
        return None

    def doit(self, **hints):
        field = self.args[0]
        return I * momentum(field) * field


# ---------------------------------------------------------------------------
# build_mass_matrix
# ---------------------------------------------------------------------------

def build_mass_matrix(potential, fields1, fields2):
    """
    Build the mass matrix  M_{ij} = d^2 V / (d phi_i d phi_j).

    Parameters
    ----------
    potential : SymPy expression
    fields1   : list of symbols (rows)
    fields2   : list of symbols (columns)

    Returns
    -------
    Matrix
    """
    M = derive_by_array(derive_by_array(potential, fields1), fields2)
    return M.tomatrix()


# ---------------------------------------------------------------------------
# extract_interaction_coefficients  (fast Poly path + fallback)
# ---------------------------------------------------------------------------

def _extract_fallback(L, fields_set, parameters):
    """Term-by-term fallback extractor."""
    L_expanded = expand(L)
    interaction_dict = {}

    for term in L_expanded.as_ordered_terms():
        if term == S.Zero:
            continue

        detected = []
        for f in fields_set:
            if f not in term.free_symbols:
                continue
            exp = 0
            if term.is_Mul:
                for factor in term.args:
                    if factor == f:
                        exp += 1
                    elif factor.is_Pow and factor.base == f:
                        if isinstance(factor.exp, (int, Integer)):
                            exp += int(factor.exp)
            elif term.is_Pow:
                if term.base == f and isinstance(term.exp, (int, Integer)):
                    exp = int(term.exp)
            elif term == f:
                exp = 1
            if exp > 0:
                detected.extend([f] * exp)

        key = tuple(sorted(detected, key=lambda s: s.sort_key()))
        n   = len(key)
        prod = Mul(*key) if key else S.One
        coef = cancel(term / prod)

        if n not in interaction_dict:
            interaction_dict[n] = {}
        current = interaction_dict[n].get(key, S.Zero)
        interaction_dict[n][key] = current + coef

    return interaction_dict


def extract_interaction_coefficients(L, fields, parameters=None):
    """
    Extract all n-point vertex coefficients from a Lagrangian.

    Returns
    -------
    dict  {n_fields: {(field_tuple): coefficient}}
    """
    if parameters is None:
        parameters = set()

    fields_set = {f for f in fields if isinstance(f, Symbol)}
    if not fields_set:
        return {}

    fields_tuple = tuple(sorted(fields_set, key=lambda s: s.sort_key()))
    poly_L = None
    try:
        poly_L = L.as_poly(*fields_tuple, domain='EX')
    except Exception:
        poly_L = None

    if poly_L is not None:
        result = {}
        for monom, coeff in poly_L.terms():
            if coeff == S.Zero:
                continue
            detected = []
            for sym, exp in zip(fields_tuple, monom):
                detected.extend([sym] * exp)
            key = tuple(sorted(detected, key=lambda s: s.sort_key()))
            n = len(key)
            if n not in result:
                result[n] = {}
            current = result[n].get(key, S.Zero)
            result[n][key] = current + coeff
        return result
    else:
        return _extract_fallback(L, fields_set, parameters)


# ---------------------------------------------------------------------------
# test_feynman_coefficients
# ---------------------------------------------------------------------------

def test_feynman_coefficients(Lagrangian, fields, parameters=None):
    """Verify extraction by reconstructing the Lagrangian."""
    interactions = extract_interaction_coefficients(Lagrangian, fields, parameters)
    reconstructed = sum(
        coeff * Mul(*fs)
        for n, terms in interactions.items()
        for fs, coeff in terms.items()
    )
    diff = (expand(Lagrangian) - reconstructed).expand().factor()
    passed = diff == 0 or not diff.free_symbols.intersection(set(fields))
    return passed, diff


# ---------------------------------------------------------------------------
# LaTeX table generator
# ---------------------------------------------------------------------------

def generate_latex_table(interactions, simplification_coeff=None):
    """Generate a LaTeX array for Feynman rule vertices."""
    table = r"\begin{array}{|c|c|}" + "\n"
    table += r"\hline" + "\n"
    table += r"\textbf{Vertex} & \textbf{Coefficient} \\" + "\n"
    table += r"\hline" + "\n"
    for fields_key, coeff in interactions.items():
        vertex = " ".join(latex(f) for f in fields_key)
        c = simplification_coeff(coeff) if simplification_coeff else coeff
        table += f"${vertex}$ & ${latex(c)}$ \\\\ \n"
        table += r"\hline" + "\n"
    table += r"\end{array}"
    return table
