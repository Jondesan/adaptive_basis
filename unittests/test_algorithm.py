import pytest
import numpy as np
import copy
import adb


# ---------------------------------------------------------------------------
# Fixtures
#
# All tests here use H2O/STO-3G.  STO-3G has one contraction per shell, so
# mol.nao == create_shell_separated_mol(mol).nao.  This keeps the Fock/overlap
# matrices, the masks, and the hcore all consistently sized throughout — no
# contraction-expansion mismatch to reason about.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def h2o_sto3g_mf(h2o_sto3g):
    """Converged RHF for H2O/STO-3G."""
    mf = h2o_sto3g.HF()
    mf.verbose = 0
    mf.kernel()
    return mf


@pytest.fixture(scope="module")
def scf_data(h2o_sto3g_mf):
    """Unpack the data most tests need: mol, F, S, hcore, and the mf object."""
    mf = h2o_sto3g_mf
    return {
        "mol":   mf.mol,
        "F":     mf.get_fock(),
        "S":     mf.get_ovlp(),
        "hcore": mf.get_hcore(),
        "Cfull": mf.mo_coeff,
        "nocc":  mf.mol.nelec,
        "mf":    mf,
    }


@pytest.fixture(scope="module")
def initial_mask(scf_data):
    """A starting mask with exactly max(nocc) functions selected.

    Picks the max(nocc) orbitals with the lowest Fock diagonal values —
    the same heuristic find_subspace uses when abd_initialization=False
    and initialize_by_projection=False.

    H2O/STO-3G: nelec=(5,5) so we start with 5 True values out of 7.
    """
    F, nocc = scf_data["F"], scf_data["nocc"]
    n_select = max(nocc)
    fock_diag = np.diag(F)
    mask = np.zeros(len(fock_diag), dtype=bool)
    mask[np.argsort(fock_diag)[:n_select]] = True
    return mask


# ---------------------------------------------------------------------------
# adb.expand_mask
# ---------------------------------------------------------------------------

class TestExpandMask:

    def test_returns_five_tuple(self, scf_data, initial_mask):
        """expand_mask must return exactly five values (mask, difference,
        criteria_val, nfuncs_in_trial, smask)."""
        d = scf_data
        result = adb.expand_mask(
            d["F"], d["S"], d["nocc"], copy.deepcopy(initial_mask),
            hcore=d["hcore"], Cfull=d["Cfull"], variant="enocc",
        )
        assert len(result) == 5

    def test_new_mask_is_superset_of_old(self, scf_data, initial_mask):
        """Every function that was True before must still be True after expansion."""
        d = scf_data
        old_mask = copy.deepcopy(initial_mask)
        new_mask, _, _, _, _ = adb.expand_mask(
            d["F"], d["S"], d["nocc"], copy.deepcopy(old_mask),
            hcore=d["hcore"], Cfull=d["Cfull"], variant="enocc",
        )
        assert np.all(new_mask[old_mask])

    def test_mask_grows_by_at_least_one(self, scf_data, initial_mask):
        """Each call must add at least one function to the mask."""
        d = scf_data
        old_count = np.sum(initial_mask)
        new_mask, _, _, _, _ = adb.expand_mask(
            d["F"], d["S"], d["nocc"], copy.deepcopy(initial_mask),
            hcore=d["hcore"], Cfull=d["Cfull"], variant="enocc",
        )
        assert np.sum(new_mask) > old_count

    def test_smask_is_none_when_not_provided(self, scf_data, initial_mask):
        """When called without a smask argument, the returned smask must be None."""
        d = scf_data
        _, _, _, _, smask_out = adb.expand_mask(
            d["F"], d["S"], d["nocc"], copy.deepcopy(initial_mask),
            hcore=d["hcore"], Cfull=d["Cfull"],
        )
        assert smask_out is None

    def test_difference_is_negative_for_enocc(self, scf_data, initial_mask):
        """For enocc, adding an orbital lowers the sum, so difference should be ≤ 0."""
        d = scf_data
        _, diff, _, _, _ = adb.expand_mask(
            d["F"], d["S"], d["nocc"], copy.deepcopy(initial_mask),
            hcore=d["hcore"], Cfull=d["Cfull"], variant="enocc",
        )
        assert diff <= 0.0

    def test_sequential_calls_grow_mask(self, scf_data, initial_mask):
        """Repeated calls should keep adding functions until the full basis is reached."""
        d = scf_data
        mask = copy.deepcopy(initial_mask)
        prev_count = np.sum(mask)

        for _ in range(d["mol"].nao - np.sum(initial_mask)):
            mask, _, _, _, _ = adb.expand_mask(
                d["F"], d["S"], d["nocc"], mask,
                hcore=d["hcore"], Cfull=d["Cfull"], variant="enocc",
            )
            assert np.sum(mask) > prev_count
            prev_count = np.sum(mask)

        assert np.all(mask)


# ---------------------------------------------------------------------------
# adb.find_subspace
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestFindSubspace:
    """Integration tests for find_subspace.

    Marked slow because each call runs find_subspace to convergence.
    Run with: pytest -m slow
    Skip with: pytest -m "not slow"
    """

    def test_returns_boolean_mask(self, h2o_sto3g, h2o_sto3g_mf, scf_data):
        d = scf_data
        mask = adb.find_subspace(
            d["F"], d["S"], h2o_sto3g, h2o_sto3g_mf,
            conv_tol=1e-2, verbose=False,
        )
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool

    def test_mask_length_equals_nao(self, h2o_sto3g, h2o_sto3g_mf, scf_data):
        d = scf_data
        mask = adb.find_subspace(
            d["F"], d["S"], h2o_sto3g, h2o_sto3g_mf,
            conv_tol=1e-2, verbose=False,
        )
        assert len(mask) == h2o_sto3g.nao

    def test_at_least_nocc_functions_selected(self, h2o_sto3g, h2o_sto3g_mf, scf_data):
        """The subspace must span at least as many functions as occupied orbitals."""
        d = scf_data
        mask = adb.find_subspace(
            d["F"], d["S"], h2o_sto3g, h2o_sto3g_mf,
            conv_tol=1e-2, verbose=False,
        )
        assert np.sum(mask) >= max(d["nocc"])

    def test_history_mode_returns_list(self, h2o_sto3g, h2o_sto3g_mf, scf_data):
        """With return_mask_history=True, a list of (mask, val, diff, ...) tuples is returned."""
        d = scf_data
        history = adb.find_subspace(
            d["F"], d["S"], h2o_sto3g, h2o_sto3g_mf,
            conv_tol=0.5, verbose=False, return_mask_history=True,
        )
        assert isinstance(history, list)
        assert len(history) >= 1
        # Each entry is a tuple whose first element is a mask
        first_mask = history[0][0]
        assert len(first_mask) == h2o_sto3g.nao

    def test_tight_tolerance_selects_more_functions(
        self, h2o_sto3g, h2o_sto3g_mf, scf_data
    ):
        """A tighter convergence tolerance should select at least as many functions."""
        d = scf_data
        loose = adb.find_subspace(
            d["F"], d["S"], h2o_sto3g, h2o_sto3g_mf,
            conv_tol=0.5, verbose=False,
        )
        tight = adb.find_subspace(
            d["F"], d["S"], h2o_sto3g, h2o_sto3g_mf,
            conv_tol=1e-3, verbose=False,
        )
        assert np.sum(tight) >= np.sum(loose)
