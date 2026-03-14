"""Smoke tests: verify that all public symbols can be imported."""

import higgs_mechanism as hm


def test_public_api():
    expected = [
        "GaugeFactor", "GaugeGroup", "make_sm_gauge_group",
        "ScalarMultiplet", "make_sm_higgs_doublet",
        "build_sm_kinetic", "rotate_to_mass_basis_sm",
        "build_sm_potential", "apply_tadpole_sm",
        "build_sm_scalar_mass_matrix", "identify_goldstones",
        "diagonalize_sm_gauge", "verify_sm_gauge_masses", "diagonalize_sm_scalar",
        "build_sm_physical_lagrangian", "extract_sm_vertices", "build_sm_scalar_vertices",
        "run_sm_validation", "print_validation_report",
        "build_mass_matrix", "extract_interaction_coefficients", "test_feynman_coefficients",
    ]
    for name in expected:
        assert hasattr(hm, name), f"Missing symbol: {name}"
