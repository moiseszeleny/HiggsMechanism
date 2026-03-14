"""
higgs_mechanism — a SymPy-based package for computing the Higgs mechanism
in generic BSM gauge theories.

Layers:
  0  gauge_group      — GaugeGroup / GaugeFactor, covariant derivative
  1  scalar_sector    — ScalarMultiplet, VEVs, real-field decomposition
  2  covariant_kinetic — kinetic Lagrangian, gauge boson mass matrix
  3  scalar_potential  — potential, tadpole conditions, scalar mass matrix
  4  diagonalization   — rotation to mass basis, mixing angles
  5  feynman_rules     — extraction of n-point vertices
  6  validation        — Goldstone counting, Ward identity, SM limit checks
"""

from .gauge_group       import GaugeFactor, GaugeGroup, make_sm_gauge_group
from .scalar_sector     import ScalarMultiplet, make_sm_higgs_doublet
from .covariant_kinetic import build_sm_kinetic, rotate_to_mass_basis_sm
from .scalar_potential  import (build_sm_potential, apply_tadpole_sm,
                                build_sm_scalar_mass_matrix, identify_goldstones)
from .diagonalization   import (diagonalize_sm_gauge, verify_sm_gauge_masses,
                                diagonalize_sm_scalar)
from .feynman_rules     import (build_sm_physical_lagrangian, extract_sm_vertices,
                                build_sm_scalar_vertices)
from .validation        import run_sm_validation, print_validation_report
from .symbolic_tools    import (build_mass_matrix, extract_interaction_coefficients,
                                test_feynman_coefficients)

__all__ = [
    'GaugeFactor', 'GaugeGroup', 'make_sm_gauge_group',
    'ScalarMultiplet', 'make_sm_higgs_doublet',
    'build_sm_kinetic', 'rotate_to_mass_basis_sm',
    'build_sm_potential', 'apply_tadpole_sm',
    'build_sm_scalar_mass_matrix', 'identify_goldstones',
    'diagonalize_sm_gauge', 'verify_sm_gauge_masses', 'diagonalize_sm_scalar',
    'build_sm_physical_lagrangian', 'extract_sm_vertices', 'build_sm_scalar_vertices',
    'run_sm_validation', 'print_validation_report',
    'build_mass_matrix', 'extract_interaction_coefficients', 'test_feynman_coefficients',
]
