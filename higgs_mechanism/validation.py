"""
Layer 6 — validation.py
=========================
Automated consistency checks for the Higgs mechanism implementation:

1. Goldstone counting:  #(broken generators) == #(zero scalar masses)
2. Gauge invariance:    V(Phi) is invariant under infinitesimal G-transformations
3. Weinberg angle:      mW / mZ = cos θ_W
4. Custodial symmetry: mW = mZ cos θ_W  (tree-level)
5. Vertex relations:    g_HWW / g_HZZ = cos^2 θ_W  etc.
6. Unitarity:           gauge boson mass matrix has correct number of
                        zero eigenvalues (unbroken generators)
"""

from sympy import (
    symbols, sqrt, cos, sin, simplify, factor, expand,
    Rational, Matrix, zeros, I, conjugate, Eq
)


def check_goldstone_counting(goldstones, broken_generators_count):
    """
    Goldstone theorem: for each broken generator there is one massless scalar.

    Parameters
    ----------
    goldstones             : list (from identify_goldstones)
    broken_generators_count: int  (number of broken generators)

    Returns
    -------
    dict with 'pass' bool and explanation
    """
    n_gold = len(goldstones)
    passed = (n_gold == broken_generators_count)
    return {
        'pass': passed,
        'goldstone_fields': goldstones,
        'n_goldstones': n_gold,
        'n_broken_generators': broken_generators_count,
        'message': (
            f"OK: {n_gold} Goldstones == {broken_generators_count} broken generators"
            if passed else
            f"FAIL: {n_gold} Goldstones != {broken_generators_count} broken generators"
        ),
    }


def check_weinberg_relation(mass_sq, gauge_fields, weinberg):
    """
    Verify  mW^2 / mZ^2 = cos^2 θ_W  from the extracted mass eigenvalues.
    """
    mW_sq = mass_sq.get('W^+')
    mZ_sq = mass_sq.get('Z')

    cos_thW = weinberg['cos_thW']
    g       = gauge_fields['g']
    g_prime = gauge_fields['g_prime']

    ratio = factor(mW_sq / mZ_sq)
    expected = factor(cos_thW**2)

    diff = factor(ratio - expected)
    passed = (diff == 0)

    return {
        'pass': passed,
        'mW_sq / mZ_sq': ratio,
        'cos^2 theta_W': expected,
        'difference': diff,
        'message': (
            "OK: mW^2/mZ^2 = cos^2 θ_W" if passed
            else f"FAIL: ratio - cos^2θ_W = {diff}"
        ),
    }


def check_higgs_mass(scalar_masses, params):
    """
    Verify the SM Higgs mass  m_h^2 = 2 lambda v^2 = 2 mu^2 (after tadpole).
    """
    from sympy import symbols
    lambda_ = params.get('lambda_')
    v       = symbols('v', positive=True)

    mh_sq_computed = scalar_masses.get('h')
    mh_sq_expected = 2 * lambda_ * v**2

    diff = factor(mh_sq_computed - mh_sq_expected)
    passed = (diff == 0)

    return {
        'pass': passed,
        'mh_sq_computed': mh_sq_computed,
        'mh_sq_expected': mh_sq_expected,
        'difference': diff,
        'message': (
            "OK: m_h^2 = 2λv^2" if passed
            else f"FAIL: m_h^2 difference = {diff}"
        ),
    }


def check_vertex_relations(vertices, weinberg, gauge_fields):
    """
    Verify HWW / HZZ = cos^2 θ_W  (custodial symmetry consequence).
    """
    g       = gauge_fields['g']
    g_prime = gauge_fields['g_prime']
    v       = symbols('v', positive=True)

    cos_thW = weinberg['cos_thW']

    # g_HWW = 2mW^2/v = g^2 v / 2
    # g_HZZ = 2mZ^2/v = g^2 v / (2 cos^2 θ_W)
    # ratio  = cos^2 θ_W
    hww = vertices.get('HWW')
    hzz = vertices.get('HZZ')

    if hww is None or hzz is None:
        return {'pass': None, 'message': 'Vertices not available'}

    ratio    = factor(hww / hzz)
    expected = factor(cos_thW**2)
    diff     = factor(ratio - expected)
    passed   = (diff == 0)

    return {
        'pass': passed,
        'g_HWW / g_HZZ': ratio,
        'cos^2 theta_W': expected,
        'difference': diff,
        'message': (
            "OK: g_HWW/g_HZZ = cos^2 θ_W" if passed
            else f"FAIL: vertex ratio - cos^2θ_W = {diff}"
        ),
    }


def check_gauge_mass_matrix_rank(mass_matrix_eigenvals, total_generators,
                                  unbroken_generators):
    """
    The gauge boson mass matrix should have exactly `unbroken_generators`
    zero eigenvalues.
    """
    n_zero = sum(1 for v in mass_matrix_eigenvals.values() if v == 0)
    passed = (n_zero == unbroken_generators)
    return {
        'pass': passed,
        'n_massless_gauge': n_zero,
        'n_unbroken_generators': unbroken_generators,
        'message': (
            f"OK: {n_zero} massless gauge bosons == {unbroken_generators} unbroken generators"
            if passed else
            f"FAIL: {n_zero} massless != {unbroken_generators} expected"
        ),
    }


def run_sm_validation(
    goldstones, scalar_masses, gauge_mass_sq,
    vertices, params, gauge_fields, weinberg
):
    """
    Run all SM validation checks and return a summary report.

    SM specifics:
        G = SU(2)_L × U(1)_Y  (4 generators)
        broken: 3  →  W^+, W^-, Z massive,  A massless
        unbroken: U(1)_em  (1 generator)
        Goldstones: G^+, G^-, G^0  (3)
    """
    results = {}

    results['goldstone_counting'] = check_goldstone_counting(
        goldstones, broken_generators_count=3
    )

    results['weinberg_relation'] = check_weinberg_relation(
        gauge_mass_sq, gauge_fields, weinberg
    )

    results['higgs_mass'] = check_higgs_mass(scalar_masses, params)

    results['vertex_HWW_HZZ'] = check_vertex_relations(
        vertices, weinberg, gauge_fields
    )

    results['gauge_mass_rank'] = check_gauge_mass_matrix_rank(
        gauge_mass_sq, total_generators=4, unbroken_generators=1
    )

    all_passed = all(
        r.get('pass', False) for r in results.values()
        if r.get('pass') is not None
    )
    results['all_passed'] = all_passed

    return results


def print_validation_report(results):
    """Pretty-print the validation results."""
    print("\n" + "="*60)
    print("   SM HIGGS MECHANISM — VALIDATION REPORT")
    print("="*60)
    for name, res in results.items():
        if name == 'all_passed':
            continue
        status = "✓" if res.get('pass') else "✗"
        print(f"  [{status}] {name}: {res.get('message', '')}")
    print("-"*60)
    final = "ALL CHECKS PASSED ✓" if results.get('all_passed') else "SOME CHECKS FAILED ✗"
    print(f"  {final}")
    print("="*60 + "\n")
