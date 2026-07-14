import pytest
import numpy as np
from adb import atomic_block_minimal_basis
from molutil import basis_functions_per_atom


# ╭─────────────────────────────────────────────────────────────────────────╮
# │ Module-level fixtures                                                   │
# ╰─────────────────────────────────────────────────────────────────────────╯

@pytest.fixture(scope="module")
def h2o_fock_init(h2o_def2tzvp):
    """Initial-guess Fock and overlap matrices for H2O/def2-tzvp.

    Uses the atomic density guess, so no SCF cycles are run. The functions
    under test only need a physically reasonable Fock matrix, not a
    converged one.
    """
    mf = h2o_def2tzvp.HF()
    mf.verbose = 0
    dm0 = mf.get_init_guess(key='atom')
    F = mf.get_fock(dm=dm0)
    S = mf.get_ovlp()
    return h2o_def2tzvp, F, S

# ╭─────────────────────────────────────────────────────────────────────────╮
# │ molutil.basis_functions_per_atom                                        │
# ╰─────────────────────────────────────────────────────────────────────────╯

class TestBasisFunctionsPerAtom:

    def test_sum_equals_mol_nao(self, h2o_def2tzvp):
        """Sum of per-atom function counts must equal mol.nao."""
        mol = h2o_def2tzvp
        result = basis_functions_per_atom(mol)
        assert np.sum(result) == mol.nao

    def test_returns_one_entry_per_atom(self, h2o_def2tzvp):
        """Length of result equals number of atoms."""
        mol = h2o_def2tzvp
        result = basis_functions_per_atom(mol)
        assert len(result) == mol.natm

    def test_h2_sto3g_one_function_per_hydrogen(self, h2_sto3g):
        """H2/STO-3G has exactly 1 basis function per hydrogen atom."""
        result = basis_functions_per_atom(h2_sto3g)
        np.testing.assert_array_equal(result, [1, 1])

    def test_oxygen_has_more_functions_than_hydrogen(self, h2o_def2tzvp):
        """In H2O, oxygen (atom 0) has more functions than each hydrogen."""
        result = basis_functions_per_atom(h2o_def2tzvp)
        assert result[0] > result[1]   # O > H1
        assert result[0] > result[2]   # O > H2


# ╭─────────────────────────────────────────────────────────────────────────╮
# │ adb.atomic_block_minimal_basis                                          │
# ╰─────────────────────────────────────────────────────────────────────────╯

class TestAtomicBlockMinimalBasis:

    def test_returns_boolean_mask(self, h2o_fock_init):
        """Return value is a 1D boolean array."""
        mol, F, S = h2o_fock_init
        mask = atomic_block_minimal_basis(mol, F, S)
        assert mask.dtype == bool
        assert mask.ndim == 1

    def test_mask_length_equals_nao(self, h2o_fock_init):
        """Mask length matches the number of AOs in the molecule."""
        mol, F, S = h2o_fock_init
        mask = atomic_block_minimal_basis(mol, F, S)
        assert len(mask) == mol.nao

    def test_at_least_minimal_number_of_functions_selected(self, h2o_fock_init):
        """At least as many functions are selected as the minimal basis requires.

        For H2O (no ECPs):
            O:  ceil(8 / 2) = 4 functions
            H1: ceil(1 / 2) = 1 function
            H2: ceil(1 / 2) = 1 function
            Total minimum: 6
        """
        mol, F, S = h2o_fock_init
        mask = atomic_block_minimal_basis(mol, F, S)
        assert np.sum(mask) >= 6

    def test_strict_subset_of_full_basis(self, h2o_fock_init):
        """The minimal basis must be a strict subset — not every function selected."""
        mol, F, S = h2o_fock_init
        mask = atomic_block_minimal_basis(mol, F, S)
        assert np.sum(mask) < mol.nao

    def test_mask_history_returned_when_requested(self, h2o_fock_init):
        """With get_mask_history=True, a tuple (mask, history) is returned."""
        mol, F, S = h2o_fock_init
        result = atomic_block_minimal_basis(mol, F, S, get_mask_history=True)
        assert isinstance(result, tuple) and len(result) == 2
        mask, history = result
        assert mask.dtype == bool
        assert isinstance(history, list) and len(history) >= 1
