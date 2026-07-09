import copy

import numpy as np
import pytest

import adb


# ---------------------------------------------------------------------------
# adb.symmetrized_eig
# ---------------------------------------------------------------------------

class TestSymmetrizedEig:

    def test_reproduces_scf_orbital_energies(self, h2o_sto3g_c2v):
        """On the full (unmasked) Fock matrix, symmetrized_eig must give the
        same physics as pyscf's own symmetric diagonalization -- it just
        solves each irrep block with adb's canonical_orth instead of
        scipy.linalg.eigh."""
        mf = h2o_sto3g_c2v.RHF()
        mf.verbose = 0
        mf.kernel()
        F = mf.get_fock()
        S = mf.get_ovlp()

        e, c, orbsym = adb.symmetrized_eig(
            F, S, h2o_sto3g_c2v.symm_orb, h2o_sto3g_c2v.irrep_id)

        np.testing.assert_allclose(
            np.sort(np.real(e)), np.sort(mf.mo_energy), atol=1e-6)
        assert set(np.unique(orbsym)) <= set(h2o_sto3g_c2v.irrep_id)
        assert c.shape == (h2o_sto3g_c2v.nao_nr(), e.size)


# ---------------------------------------------------------------------------
# adb.get_iteration_criteria_value (irrep-aware 'enocc' branch)
# ---------------------------------------------------------------------------

class TestEnoccByIrrep:

    def test_matches_manual_sum_restricted(self):
        """With enough orbitals in every target irrep, the criterion is
        2x the sum of the lowest target-count eigenvalues per irrep (each
        restricted spatial orbital holds 2 electrons) -- no penalty
        involved."""
        epsilon_i = np.array([-2.0, -1.0, -0.5, 0.1, 0.2])
        orbsym = np.array(['A1', 'A1', 'B1', 'A1', 'B1'])
        irrep_nelec = {'A1': 4, 'B1': 2}  # 2 A1 orbitals + 1 B1 orbital

        val = adb.get_iteration_criteria_value(
            'enocc', epsilon_i=epsilon_i, nocc=(3, 3),
            irrep_nelec=irrep_nelec, orbsym=orbsym)

        expected = 2 * ((-2.0 + -1.0) + (-0.5))
        assert val == pytest.approx(expected)

    def test_shortfall_adds_penalty(self):
        """If a target irrep doesn't have enough orbitals yet, each missing
        slot adds SYMMETRY_SHORTFALL_PENALTY (x2 for restricted, since a
        missing restricted orbital represents 2 missing electrons),
        dominating the real energies."""
        epsilon_i = np.array([-2.0, -1.0, 0.1])
        orbsym = np.array(['A1', 'A1', 'B2'])
        irrep_nelec = {'A1': 4, 'B1': 2}  # B1 has zero available orbitals

        val = adb.get_iteration_criteria_value(
            'enocc', epsilon_i=epsilon_i, nocc=(3, 3),
            irrep_nelec=irrep_nelec, orbsym=orbsym)

        expected = 2 * (-2.0 + -1.0) + 2 * adb.SYMMETRY_SHORTFALL_PENALTY
        assert val == pytest.approx(expected)

    def test_unrestricted_uses_per_spin_targets(self):
        epsilon_i = np.array([
            [-2.0, -1.0, 0.1],   # alpha
            [-1.8, -0.9, 0.2],   # beta
        ])
        orbsym = np.array(['A1', 'A1', 'B1'])
        irrep_nelec = {'A1': (2, 1)}  # 2 alpha, 1 beta in A1; nothing in B1

        val = adb.get_iteration_criteria_value(
            'enocc', epsilon_i=epsilon_i, nocc=(2, 1),
            irrep_nelec=irrep_nelec, orbsym=orbsym)

        expected = (-2.0 + -1.0) + (-1.8)
        assert val == pytest.approx(expected)

    def test_orbsym_required_with_irrep_nelec(self):
        with pytest.raises(ValueError):
            adb.get_iteration_criteria_value(
                'enocc', epsilon_i=np.array([-1.0]), nocc=(1, 1),
                irrep_nelec={'A1': 2})

    def test_default_behaviour_unchanged_without_irrep_nelec(self):
        """Sanity check: omitting irrep_nelec/orbsym must reproduce exactly
        the pre-existing, symmetry-blind lowest-N-by-energy criterion (2x
        for restricted -- each spatial orbital holds 2 electrons)."""
        epsilon_i = np.array([-2.0, -1.0, -0.5, 0.1, 0.2])
        val = adb.get_iteration_criteria_value(
            'enocc', epsilon_i=epsilon_i, nocc=(2, 2))
        assert val == pytest.approx(2 * (-2.0 + -1.0))


# ---------------------------------------------------------------------------
# adb.expand_mask (symmetry-aware mode)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def h2o_c2v_scf_data(h2o_sto3g_c2v):
    mf = h2o_sto3g_c2v.RHF()
    mf.verbose = 0
    mf.kernel()
    shellsep_mol = adb.create_shell_separated_mol(h2o_sto3g_c2v)
    return {
        "mol": shellsep_mol,
        "F": mf.get_fock(),
        "S": mf.get_ovlp(),
        "hcore": mf.get_hcore(),
        "Cfull": mf.mo_coeff,
        "nocc": h2o_sto3g_c2v.nelec,
    }


@pytest.fixture
def start_smask(h2o_c2v_scf_data):
    """O-1s + H-1s(linked) toggled -- every atom already has >=1 shell (a
    subbasis with a whole element missing zero basis functions is not
    buildable), and both the O-2s and O-2p shells are still untoggled, so
    the very next expand_mask call has a real choice to make between them.
    """
    mol = h2o_c2v_scf_data["mol"]
    smask = adb.init_smask(mol, mol.cart)
    smask[0][0] = True  # O-1s
    smask[3][0] = True  # H-1s (atom 1)
    smask = adb.set_linked_shells(smask, True)  # also flips the other H-1s
    return smask


class TestExpandMaskSymmetryAware:
    """H2O/STO-3G/C2v ground state occupies {A1: 6, B1: 2, B2: 2} electrons
    (3 A1 + 1 B1 + 1 B2 orbitals) -- O-2p is the only source of B1 orbitals
    in this minimal basis (see conftest fixture docstring), so a purely
    energy-based (symmetry-blind) search has no reason to prefer it over
    the lower-lying O-2s shell. A target occupation that needs B1 filled
    forces the symmetry-aware search to add O-2p first regardless of its
    energy, which is exactly the "guess predicts the wrong symmetry"
    failure mode described in the paper draft this feature addresses.
    """

    ALT_TARGET = {'A1': 4, 'B1': 2, 'B2': 4}  # 2 A1 + 1 B1 + 2 B2 orbitals

    def test_symmetry_aware_prioritises_missing_irrep(
            self, h2o_c2v_scf_data, start_smask):
        d = h2o_c2v_scf_data
        smask = copy.deepcopy(start_smask)
        mask = adb.smask_to_mask(smask, d["mol"].cart)

        mask, _, _, n_added, smask = adb.expand_mask(
            d["F"], d["S"], d["nocc"], mask, smask=smask,
            hcore=d["hcore"], Cfull=d["Cfull"],
            mol=d["mol"], irrep_nelec=self.ALT_TARGET,
        )

        # O-2p (3 functions) must be the shell picked, since it is the only
        # way to stop B1's target occupation from being permanently
        # unreachable.
        assert n_added == 3
        assert smask[2][0] == True  # (0, 'O', '2p') per get_all_shell_labels

    def test_symmetry_blind_default_prefers_lower_energy_shell(
            self, h2o_c2v_scf_data, start_smask):
        """Same starting point, no irrep_nelec/mol given: must reproduce
        today's plain, symmetry-blind choice -- which picks O-2s (the
        lowest-energy remaining shell) instead, ignoring that B1 is empty.
        """
        d = h2o_c2v_scf_data
        smask = copy.deepcopy(start_smask)
        mask = adb.smask_to_mask(smask, d["mol"].cart)

        mask, _, _, n_added, smask = adb.expand_mask(
            d["F"], d["S"], d["nocc"], mask, smask=smask,
            hcore=d["hcore"], Cfull=d["Cfull"],
        )

        assert n_added == 1
        assert smask[1][0] == True  # (0, 'O', '2s')
        assert smask[2][0] == False  # O-2p NOT picked -- unlike the aware case

    def test_requires_smask(self, h2o_c2v_scf_data):
        d = h2o_c2v_scf_data
        mask = np.zeros(d["mol"].nao_nr(), dtype=bool)
        mask[:2] = True
        with pytest.raises(RuntimeError):
            adb.expand_mask(
                d["F"], d["S"], d["nocc"], mask,
                hcore=d["hcore"], Cfull=d["Cfull"],
                mol=d["mol"], irrep_nelec=self.ALT_TARGET,
            )


# ---------------------------------------------------------------------------
# adb.find_subspace (symmetry-aware mode) -- validation only; the search
# itself is exercised (slowly) via TestExpandMaskSymmetryAware above and via
# adaptive_basis/vsap_symmetry_check/ for the real FeF3/aug-pc-2 case.
# ---------------------------------------------------------------------------

class TestFindSubspaceSymmetryAware:

    def test_requires_irrep_nelec(self, h2o_sto3g_c2v):
        mf = h2o_sto3g_c2v.RHF()
        mf.verbose = 0
        mf.kernel()
        with pytest.raises(RuntimeError):
            adb.find_subspace(
                mf.get_fock(), mf.get_ovlp(), h2o_sto3g_c2v, mf,
                symmetry_aware=True, verbose=False,
            )

    def test_requires_link_shells(self, h2o_sto3g_c2v):
        mf = h2o_sto3g_c2v.RHF()
        mf.verbose = 0
        mf.kernel()
        with pytest.raises(RuntimeError):
            adb.find_subspace(
                mf.get_fock(), mf.get_ovlp(), h2o_sto3g_c2v, mf,
                symmetry_aware=True,
                irrep_nelec={'A1': 6, 'B1': 2, 'B2': 2},
                get_smask=True, link_shells=False, verbose=False,
            )

    def test_requires_get_smask(self, h2o_sto3g_c2v):
        mf = h2o_sto3g_c2v.RHF()
        mf.verbose = 0
        mf.kernel()
        with pytest.raises(RuntimeError):
            adb.find_subspace(
                mf.get_fock(), mf.get_ovlp(), h2o_sto3g_c2v, mf,
                symmetry_aware=True,
                irrep_nelec={'A1': 6, 'B1': 2, 'B2': 2},
                get_smask=False, verbose=False,
            )

    def test_requires_non_c1_symmetry(self, h2o_sto3g):
        """h2o_sto3g (no fixture-level symmetry=True) has mol.symmetry
        falsy, so symmetry_aware must be rejected outright."""
        mf = h2o_sto3g.RHF()
        mf.verbose = 0
        mf.kernel()
        with pytest.raises(RuntimeError):
            adb.find_subspace(
                mf.get_fock(), mf.get_ovlp(), h2o_sto3g, mf,
                symmetry_aware=True,
                irrep_nelec={'A1': 6, 'B1': 2, 'B2': 2},
                get_smask=True, verbose=False,
            )
