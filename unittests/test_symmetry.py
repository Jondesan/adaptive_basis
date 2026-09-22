import copy

import numpy as np
import pytest

from adb import \
    find_subspace, expand_mask, get_occupied_orbitals, \
    get_occupied_orbitals_from_scf, mask_analysis, \
    eig, diagonalize_masked, symmetrized_eig, \
    get_iteration_criteria_value, \
    create_shell_separated_mol, \
    init_smask, set_linked_shells, smask_to_mask, \
    write_orbital_history, CONSTANTS, \
    open_run_log, write_cycle_report
import pyscf


# ╭─────────────────────────────────────────────────────────────────────────╮
# │ calculations.symmetrized_eig                                            │
# ╰─────────────────────────────────────────────────────────────────────────╯

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

        e, c, orbsym = symmetrized_eig(
            F, S, h2o_sto3g_c2v.symm_orb, h2o_sto3g_c2v.irrep_id)

        np.testing.assert_allclose(
            np.sort(np.real(e)), np.sort(mf.mo_energy), atol=1e-6)
        assert set(np.unique(orbsym)) <= set(h2o_sto3g_c2v.irrep_id)
        assert c.shape == (h2o_sto3g_c2v.nao_nr(), e.size)


# ╭─────────────────────────────────────────────────────────────────────────╮
# │ calculations.get_iteration_criteria_value (irrep-aware 'enocc' branch)  │
# ╰─────────────────────────────────────────────────────────────────────────╯

class TestEnoccByIrrep:

    def test_matches_manual_sum_restricted(self):
        """With enough orbitals in every target irrep, the criterion is
        2x the sum of the lowest target-count eigenvalues per irrep (each
        restricted spatial orbital holds 2 electrons) -- no penalty
        involved."""
        epsilon_i = np.array([-2.0, -1.0, -0.5, 0.1, 0.2])
        orbsym = np.array(['A1', 'A1', 'B1', 'A1', 'B1'])
        irrep_nelec = {'A1': 4, 'B1': 2}  # 2 A1 orbitals + 1 B1 orbital

        val = get_iteration_criteria_value(
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

        val = get_iteration_criteria_value(
            'enocc', epsilon_i=epsilon_i, nocc=(3, 3),
            irrep_nelec=irrep_nelec, orbsym=orbsym)

        expected = 2 * (-2.0 + -1.0) + 2 * CONSTANTS.SYMMETRY_SHORTFALL_PENALTY
        assert val == pytest.approx(expected)

    def test_unrestricted_uses_per_spin_targets(self):
        epsilon_i = np.array([
            [-2.0, -1.0, 0.1],   # alpha
            [-1.8, -0.9, 0.2],   # beta
        ])
        orbsym = np.array(['A1', 'A1', 'B1'])
        irrep_nelec = {'A1': (2, 1)}  # 2 alpha, 1 beta in A1; nothing in B1

        val = get_iteration_criteria_value(
            'enocc', epsilon_i=epsilon_i, nocc=(2, 1),
            irrep_nelec=irrep_nelec, orbsym=orbsym)

        expected = (-2.0 + -1.0) + (-1.8)
        assert val == pytest.approx(expected)

    def test_orbsym_required_with_irrep_nelec(self):
        with pytest.raises(ValueError):
            get_iteration_criteria_value(
                'enocc', epsilon_i=np.array([-1.0]), nocc=(1, 1),
                irrep_nelec={'A1': 2})

    def test_default_behaviour_unchanged_without_irrep_nelec(self):
        """Sanity check: omitting irrep_nelec/orbsym must reproduce exactly
        the pre-existing, symmetry-blind lowest-N-by-energy criterion (2x
        for restricted -- each spatial orbital holds 2 electrons)."""
        epsilon_i = np.array([-2.0, -1.0, -0.5, 0.1, 0.2])
        val = get_iteration_criteria_value(
            'enocc', epsilon_i=epsilon_i, nocc=(2, 2))
        assert val == pytest.approx(2 * (-2.0 + -1.0))


# ╭─────────────────────────────────────────────────────────────────────────╮
# │ maskutil.expand_mask (symmetry-aware mode)                              │
# ╰─────────────────────────────────────────────────────────────────────────╯

@pytest.fixture(scope="module")
def h2o_c2v_scf_data(h2o_sto3g_c2v):
    mf = h2o_sto3g_c2v.RHF()
    mf.verbose = 0
    mf.kernel()
    shellsep_mol = create_shell_separated_mol(h2o_sto3g_c2v)
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
    smask = init_smask(mol, mol.cart)
    smask[0][0] = True  # O-1s
    smask[3][0] = True  # H-1s (atom 1)
    smask = set_linked_shells(smask)  # also flips the other H-1s
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
        mask = smask_to_mask(smask, d["mol"].cart)

        mask, _, _, n_added, smask = expand_mask(
            d["F"], d["S"], d["nocc"], mask, smask=smask, Cfull=d["Cfull"],
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
        mask = smask_to_mask(smask, d["mol"].cart)

        mask, _, _, n_added, smask = expand_mask(
            d["F"], d["S"], d["nocc"], mask, smask=smask, Cfull=d["Cfull"],
            )

        assert n_added == 1
        assert smask[1][0] == True  # (0, 'O', '2s')
        assert smask[2][0] == False  # O-2p NOT picked -- unlike the aware case

    def test_requires_smask(self, h2o_c2v_scf_data):
        d = h2o_c2v_scf_data
        mask = np.zeros(d["mol"].nao_nr(), dtype=bool)
        mask[:2] = True
        with pytest.raises(RuntimeError):
            expand_mask(
                d["F"], d["S"], d["nocc"], mask, Cfull=d["Cfull"],
                mol=d["mol"], irrep_nelec=self.ALT_TARGET,
            )


# ╭─────────────────────────────────────────────────────────────────────────╮
# │ adb.find_subspace (symmetry-aware mode) -- validation only; the search  │
# │ itself is exercised (slowly) via TestExpandMaskSymmetryAware above and  │
# │ via adaptive_basis/vsap_symmetry_check/ for the real FeF3/aug-pc-2 case.│
# ╰─────────────────────────────────────────────────────────────────────────╯

class TestFindSubspaceSymmetryAware:

    def test_requires_irrep_nelec(self, h2o_sto3g_c2v):
        mf = h2o_sto3g_c2v.RHF()
        mf.verbose = 0
        mf.kernel()
        with pytest.raises(RuntimeError):
            find_subspace(
                mf.get_fock(), mf.get_ovlp(), h2o_sto3g_c2v, mf,
                symmetry_aware=True, verbose=False,
            )

    def test_requires_link_shells(self, h2o_sto3g_c2v):
        mf = h2o_sto3g_c2v.RHF()
        mf.verbose = 0
        mf.kernel()
        with pytest.raises(RuntimeError):
            find_subspace(
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
            find_subspace(
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
            find_subspace(
                mf.get_fock(), mf.get_ovlp(), h2o_sto3g, mf,
                symmetry_aware=True,
                irrep_nelec={'A1': 6, 'B1': 2, 'B2': 2},
                get_smask=True, verbose=False,
            )


# ╭─────────────────────────────────────────────────────────────────────────╮
# │ calculations.diagonalize_masked                                         │
# ╰─────────────────────────────────────────────────────────────────────────╯

class TestDiagonalizeMasked:

    def test_mol_none_matches_plain_eig(self):
        """mol=None must reproduce adb.eig exactly (it's the same code
        path -- diagonalize_masked is just expand_mask/find_subspace's
        shared symmetric-or-plain dispatcher, pulled out to a top-level
        function)."""
        h = np.array([[2.0, 0.3], [0.3, 1.0]])
        s = np.eye(2)
        e1, c1 = eig(h, s)
        e2, c2, orbsym = diagonalize_masked(h, s, mol=None)
        np.testing.assert_allclose(np.real(e1), np.real(e2))
        np.testing.assert_allclose(np.real(c1), np.real(c2))
        assert orbsym is None

    def test_mol_given_matches_symmetrized_eig(self, h2o_sto3g_c2v):
        """mol given (with the identity/full smask) must reproduce
        symmetrized_eig's result on the full basis."""
        mf = h2o_sto3g_c2v.RHF()
        mf.verbose = 0
        mf.kernel()
        F, S = mf.get_fock(), mf.get_ovlp()

        shellsep_mol = create_shell_separated_mol(h2o_sto3g_c2v)
        full_smask = init_smask(shellsep_mol, shellsep_mol.cart)
        for row in full_smask:
            row[0] = True

        e1, _, orbsym1 = symmetrized_eig(
            F, S, h2o_sto3g_c2v.symm_orb, h2o_sto3g_c2v.irrep_id)
        e2, _, orbsym2 = diagonalize_masked(
            F, S, mol=shellsep_mol, smask=full_smask)

        np.testing.assert_allclose(np.sort(np.real(e1)), np.sort(np.real(e2)))
        # symmetrized_eig returns raw irrep ids; diagonalize_masked
        # translates them to names (what get_occupied_orbitals/irrep_nelec
        # actually key on).
        assert set(orbsym1.tolist()) <= set(shellsep_mol.irrep_id)
        assert set(orbsym2.tolist()) <= set(shellsep_mol.irrep_name)


# ╭─────────────────────────────────────────────────────────────────────────╮
# │ adb.get_occupied_orbitals                                               │
# ╰─────────────────────────────────────────────────────────────────────────╯

class TestGetOccupiedOrbitals:

    def test_symmetry_blind_restricted(self):
        epsilon_i = np.array([-2.0, -1.0, -0.5, 0.1, 0.2])
        occ = get_occupied_orbitals(epsilon_i, nocc=(2, 2))
        assert occ == [(-2.0, None), (-1.0, None)]

    def test_symmetry_blind_unrestricted(self):
        epsilon_i = np.array([[-2.0, -1.0, 0.1], [-1.8, -0.9, 0.2]])
        occ = get_occupied_orbitals(
            epsilon_i, nocc=(2, 1), restricted=False)
        assert occ == [(-2.0, None), (-1.0, None), (-1.8, None)]

    def test_symmetry_aware_restricted(self):
        epsilon_i = np.array([-2.0, -1.0, -0.5, 0.1, 0.2])
        orbsym = np.array(['A1', 'A1', 'B1', 'A1', 'B1'])
        irrep_nelec = {'A1': 4, 'B1': 2}
        occ = get_occupied_orbitals(
            epsilon_i, nocc=(3, 3), irrep_nelec=irrep_nelec, orbsym=orbsym)
        assert sorted(occ) == sorted([(-2.0, 'A1'), (-1.0, 'A1'), (-0.5, 'B1')])

    def test_symmetry_aware_unrestricted(self):
        epsilon_i = np.array([
            [-2.0, -1.0, 0.1],
            [-1.8, -0.9, 0.2],
        ])
        orbsym = np.array(['A1', 'A1', 'B1'])
        irrep_nelec = {'A1': (2, 1)}
        occ = get_occupied_orbitals(
            epsilon_i, nocc=(2, 1), irrep_nelec=irrep_nelec, orbsym=orbsym,
            restricted=False)
        assert sorted(occ) == sorted([(-2.0, 'A1'), (-1.0, 'A1'), (-1.8, 'A1')])

    def test_orbsym_required_with_irrep_nelec(self):
        with pytest.raises(ValueError):
            get_occupied_orbitals(
                np.array([-1.0]), nocc=(1, 1), irrep_nelec={'A1': 2})


# ╭─────────────────────────────────────────────────────────────────────────╮
# │ find_subspace(track_orbitals=True)                                      │
# ╰─────────────────────────────────────────────────────────────────────────╯

class TestTrackOrbitals:

    def test_default_returns_plain_result(self, h2o_sto3g):
        """track_orbitals=False (default) must return exactly what
        find_subspace returned before this option existed -- no tuple."""
        mf = h2o_sto3g.RHF()
        mf.verbose = 0
        mf.kernel()
        result = find_subspace(
            mf.get_fock(), mf.get_ovlp(), h2o_sto3g, mf,
            conv_tol=1e-2, verbose=False,
        )
        assert isinstance(result, np.ndarray)

    def test_returns_tuple_when_enabled(self, h2o_sto3g):
        mf = h2o_sto3g.RHF()
        mf.verbose = 0
        mf.kernel()
        result = find_subspace(
            mf.get_fock(), mf.get_ovlp(), h2o_sto3g, mf,
            conv_tol=1e-2, verbose=False, track_orbitals=True,
        )
        assert isinstance(result, tuple) and len(result) == 2
        mask, orbital_history = result
        assert isinstance(mask, np.ndarray)
        assert isinstance(orbital_history, list) and len(orbital_history) >= 1

    def test_entries_have_expected_shape_and_occupied_count(self, h2o_sto3g):
        """Every recorded cycle must list exactly nocc[0] occupied spatial
        orbitals (H2O/STO-3G RHF: 5), symmetry-blind (irrep=None)."""
        mf = h2o_sto3g.RHF()
        mf.verbose = 0
        mf.kernel()
        _, orbital_history = find_subspace(
            mf.get_fock(), mf.get_ovlp(), h2o_sto3g, mf,
            conv_tol=1e-2, verbose=False, track_orbitals=True,
        )
        for entry in orbital_history:
            assert set(entry.keys()) == {"nfunc", "orbitals"}
            assert len(entry["orbitals"]) == h2o_sto3g.nelec[0]
            assert all(irrep is None for _, irrep in entry["orbitals"])

    def test_symmetry_aware_labels_present(self, h2o_sto3g_c2v):
        mf = h2o_sto3g_c2v.RHF()
        mf.verbose = 0
        mf.kernel()
        irrep_nelec = mf.get_irrep_nelec()
        _, orbital_history = find_subspace(
            mf.get_fock(), mf.get_ovlp(), h2o_sto3g_c2v, mf,
            conv_tol=1e-2, verbose=False, get_smask=True,
            symmetry_aware=True, irrep_nelec=irrep_nelec, track_orbitals=True,
        )
        assert len(orbital_history) >= 1
        for entry in orbital_history:
            assert len(entry["orbitals"]) == h2o_sto3g_c2v.nelec[0]
            assert all(irrep in h2o_sto3g_c2v.irrep_name
                       for _, irrep in entry["orbitals"])


# ╭─────────────────────────────────────────────────────────────────────────╮
# │ ioutil.write_orbital_history                                            │
# ╰─────────────────────────────────────────────────────────────────────────╯

class TestWriteOrbitalHistory:

    def test_writes_expected_csv(self, tmp_path):
        orbital_history = [
            {"nfunc": 7, "orbitals": [(-1.5, "A1"), (-0.5, None)]},
            {"nfunc": 8, "orbitals": [(-1.6, "A1")]},
        ]
        fn = str(tmp_path / "hist")
        write_orbital_history(orbital_history, fn, molname="h2o", basisname="sto-3g")

        content = (tmp_path / "hist.csv").read_text().splitlines()
        assert content[0] == "# molecule=h2o basis=sto-3g"
        assert content[1] == "nfunc,energy,irrep"
        assert content[2] == "7,-1.500000000000,A1"
        assert content[3] == "7,-0.500000000000,"
        assert content[4] == "8,-1.600000000000,A1"


# ╭─────────────────────────────────────────────────────────────────────────╮
# │ adb.get_occupied_orbitals_from_scf                                      │
# ╰─────────────────────────────────────────────────────────────────────────╯

class TestGetOccupiedOrbitalsFromSCF:

    def test_symmetry_blind_restricted(self, h2o_sto3g):
        mf = h2o_sto3g.RHF()
        mf.verbose = 0
        mf.kernel()
        occ = get_occupied_orbitals_from_scf(mf)
        assert len(occ) == h2o_sto3g.nelec[0]
        assert all(irrep is None for _, irrep in occ)
        np.testing.assert_allclose(
            sorted(e for e, _ in occ),
            sorted(mf.mo_energy[mf.mo_occ > 0]))

    def test_symmetry_aware_restricted(self, h2o_sto3g_c2v):
        mf = h2o_sto3g_c2v.RHF()
        mf.verbose = 0
        mf.kernel()
        occ = get_occupied_orbitals_from_scf(mf)
        assert len(occ) == h2o_sto3g_c2v.nelec[0]
        assert all(irrep in h2o_sto3g_c2v.irrep_name for _, irrep in occ)

        counts = {}
        for _, irrep in occ:
            counts[irrep] = counts.get(irrep, 0) + 1
        expected = {k: v // 2 for k, v in mf.get_irrep_nelec().items() if v > 0}
        assert counts == expected

    def test_unrestricted(self):
        mol = pyscf.M(atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
                       basis="sto-3g", charge=1, spin=1, verbose=0)
        mf = mol.UHF()
        mf.verbose = 0
        mf.kernel()
        occ = get_occupied_orbitals_from_scf(mf)
        assert len(occ) == mol.nelec[0] + mol.nelec[1]
        assert all(irrep is None for _, irrep in occ)


# ╭─────────────────────────────────────────────────────────────────────────╮
# │ adb.mask_analysis(track_orbitals=True)                                  │
# ╰─────────────────────────────────────────────────────────────────────────╯

@pytest.mark.slow
class TestMaskAnalysisTrackOrbitals:
    """Integration tests -- each runs a real (if tiny) SCF for every
    subbasis in mask_history, so they're marked slow like
    TestFindSubspace in test_algorithm.py. Run with: pytest -m slow
    """

    def _run_find_subspace(self, mol):
        mf = mol.RHF()
        mf.verbose = 0
        mf.kernel()
        F, S = mf.get_fock(), mf.get_ovlp()
        shellsep_mol = create_shell_separated_mol(mol)
        mask_history = find_subspace(
            F, S, mol, mf, conv_tol=0.5, verbose=False,
            get_smask=True, return_mask_history=True,
        )
        return mf, F, S, shellsep_mol, mask_history

    def test_default_returns_plain_dataframe(self, h2o_sto3g):
        mf, F, S, shellsep_mol, mask_history = self._run_find_subspace(h2o_sto3g)
        result = mask_analysis(
            mask_history, shellsep_mol, mf, F, S, verbose=False,
            C_full=mf.mo_coeff, calculate_correction=False,
        )
        assert isinstance(result, list)

    def test_returns_tuple_when_enabled(self, h2o_sto3g):
        mf, F, S, shellsep_mol, mask_history = self._run_find_subspace(h2o_sto3g)
        result = mask_analysis(
            mask_history, shellsep_mol, mf, F, S, verbose=False,
            C_full=mf.mo_coeff, calculate_correction=False, track_orbitals=True,
        )
        assert isinstance(result, tuple) and len(result) == 2
        dataframe, orbital_history = result
        assert isinstance(dataframe, list)
        assert isinstance(orbital_history, list) and len(orbital_history) >= 1
        for entry in orbital_history:
            assert set(entry.keys()) == {"nfunc", "orbitals"}
            assert len(entry["orbitals"]) == h2o_sto3g.nelec[0]
            assert all(irrep is None for _, irrep in entry["orbitals"])

    def test_symmetry_aware_labels_present(self, h2o_sto3g_c2v):
        mf, F, S, shellsep_mol, mask_history = self._run_find_subspace(h2o_sto3g_c2v)
        irrep_nelec = mf.get_irrep_nelec()
        _, orbital_history = mask_analysis(
            mask_history, shellsep_mol, mf, F, S, verbose=False,
            C_full=mf.mo_coeff, irrep_nelec=irrep_nelec,
            calculate_correction=False, track_orbitals=True,
        )
        assert len(orbital_history) >= 1
        for entry in orbital_history:
            assert len(entry["orbitals"]) == h2o_sto3g_c2v.nelec[0]
            assert all(irrep in h2o_sto3g_c2v.irrep_name
                       for _, irrep in entry["orbitals"])


# ╭─────────────────────────────────────────────────────────────────────────╮
# │ adb.mask_analysis(check_stability=True)                                 │
# ╰─────────────────────────────────────────────────────────────────────────╯

@pytest.mark.slow
class TestMaskAnalysisCheckStability:
    """Integration tests for the optional internal-stability check/re-
    optimization added to _run_subbasis_scf (see
    adaptive_basis/untracked/jagged_convergence_check/REPORT.md for the
    investigation this addresses). Real (if tiny) SCFs, marked slow like
    TestMaskAnalysisTrackOrbitals above.

    h2o_sto3g's RHF ground state is not expected to be internally unstable
    at any subbasis size, so these don't exercise the reconverge branch
    itself (that's validated against a real case -- FeO -- outside the
    unit suite); they confirm the option is wired correctly and inert by
    default.
    """

    def _run_find_subspace(self, mol):
        mf = mol.RHF()
        mf.verbose = 0
        mf.kernel()
        F, S = mf.get_fock(), mf.get_ovlp()
        shellsep_mol = create_shell_separated_mol(mol)
        mask_history = find_subspace(
            F, S, mol, mf, conv_tol=0.5, verbose=False,
            get_smask=True, return_mask_history=True,
        )
        return mf, F, S, shellsep_mol, mask_history

    def test_default_is_unaffected(self, h2o_sto3g):
        """check_stability defaults to False: calling mask_analysis with
        no mention of it must reproduce calling it with check_stability=False
        (regression safety for every existing call site). Compared with a
        tolerance, not exact equality: re-running the identical SCF twice
        already differs at the ~1e-13 level (BLAS-threading floating-point
        noise, confirmed present with no check_stability involved at all
        -- unrelated to this option)."""
        mf, F, S, shellsep_mol, mask_history = self._run_find_subspace(h2o_sto3g)
        kwargs = {'verbose': False, 'C_full': mf.mo_coeff, 'calculate_correction': False}
        result_default = mask_analysis(mask_history, shellsep_mol, mf, F, S, **kwargs)
        result_explicit_false = mask_analysis(
            mask_history, shellsep_mol, mf, F, S, check_stability=False, **kwargs)
        assert len(result_default) == len(result_explicit_false)
        for row_a, row_b in zip(result_default, result_explicit_false):
            nfunc_a, _cursum_a, _diff_a, e_scf_a, *_rest_a, conv_a = row_a
            nfunc_b, _cursum_b, _diff_b, e_scf_b, *_rest_b, conv_b = row_b
            assert nfunc_a == nfunc_b
            assert bool(conv_a) and bool(conv_b)
            assert e_scf_a == pytest.approx(e_scf_b, abs=1e-8)

    def test_enabled_runs_and_stays_converged(self, h2o_sto3g):
        mf, F, S, shellsep_mol, mask_history = self._run_find_subspace(h2o_sto3g)
        result = mask_analysis(
            mask_history, shellsep_mol, mf, F, S, verbose=False,
            C_full=mf.mo_coeff, calculate_correction=False, check_stability=True,
        )
        assert isinstance(result, list) and len(result) >= 1
        for row in result:
            assert bool(row[-1])  # conv_stat: subbasis SCF still converged


# ╭─────────────────────────────────────────────────────────────────────────╮
# │ adb.mask_analysis(log_stdout=...)                                       │
# ╰─────────────────────────────────────────────────────────────────────────╯

@pytest.mark.slow
class TestMaskAnalysisLogging:
    """Integration tests for the optional log_stdout/log_verbose passthrough
    added to find_subspace/mask_analysis/_run_subbasis_scf, so a whole
    run.py invocation's pyscf-level output can be captured to one external
    file (see run.py's --log_file/--log_verbose)."""

    def _run_find_subspace(self, mol, **kwargs):
        mf = mol.RHF()
        mf.verbose = 0
        mf.kernel()
        F, S = mf.get_fock(), mf.get_ovlp()
        shellsep_mol = create_shell_separated_mol(mol)
        mask_history = find_subspace(
            F, S, mol, mf, conv_tol=0.5, verbose=False,
            get_smask=True, return_mask_history=True, **kwargs,
        )
        return mf, F, S, shellsep_mol, mask_history

    def test_default_is_unaffected(self, h2o_sto3g):
        """Omitting log_stdout must not touch stdout/verbose anywhere --
        regression safety for every existing call site."""
        stdout_before = h2o_sto3g.stdout
        verbose_before = h2o_sto3g.verbose
        mf, F, S, shellsep_mol, mask_history = self._run_find_subspace(h2o_sto3g)
        mask_analysis(
            mask_history, shellsep_mol, mf, F, S, verbose=False,
            C_full=mf.mo_coeff, calculate_correction=False,
        )
        assert h2o_sto3g.stdout is stdout_before
        assert h2o_sto3g.verbose == verbose_before

    def test_enabled_writes_to_file(self, h2o_sto3g, tmp_path):
        log_path = tmp_path / "run.log"
        with open(log_path, 'w') as handle:
            mf, F, S, shellsep_mol, mask_history = self._run_find_subspace(
                h2o_sto3g, log_stdout=handle, log_verbose=5)
            mask_analysis(
                mask_history, shellsep_mol, mf, F, S, verbose=False,
                C_full=mf.mo_coeff, calculate_correction=False,
                log_stdout=handle, log_verbose=5,
            )
        content = log_path.read_text()
        assert content, "log file should not be empty once logging is enabled"
        assert "SCF" in content

    def test_open_run_log_is_reused_across_cycles(self, h2o_def2tzvp, tmp_path):
        """The whole point of using a shared handle instead of Mole.output:
        confirm content from an *earlier* cycle survives a *later* cycle's
        own Mole.build() (which would truncate a naively-reused filename).
        Needs a basis with room to grow past the minimal seed (unlike
        STO-3G, where the minimal basis already *is* the full basis) to
        get more than one cycle."""
        log_path = tmp_path / "run.log"
        handle = open_run_log(str(log_path))
        mf, F, S, shellsep_mol, mask_history = self._run_find_subspace(
            h2o_def2tzvp, log_stdout=handle, log_verbose=5)
        mask_analysis(
            mask_history, shellsep_mol, mf, F, S, verbose=False,
            C_full=mf.mo_coeff, calculate_correction=False,
            log_stdout=handle, log_verbose=5,
        )
        handle.close()
        assert len(mask_history) > 1, "need >1 cycle for this check to mean anything"
        content = log_path.read_text()
        # Every cycle builds its own subbasis Mole; if an earlier cycle's
        # output were being wiped by a later cycle's Mole.build(), the
        # file would only ever contain one cycle's worth of "SCF" mentions.
        assert content.count("SCF") > 1


# ╭─────────────────────────────────────────────────────────────────────────╮
# │ adb.mask_analysis(cycle_report=True) / adb.write_cycle_report           │
# ╰─────────────────────────────────────────────────────────────────────────╯

@pytest.mark.slow
class TestMaskAnalysisCycleReport:
    """Integration tests for the optional cycle_report passthrough -- an
    independent sibling of track_orbitals that records occupied *and*
    nearby virtual orbital energies per cycle, for the separate
    write_cycle_report human-readable formatter (see run.py's
    --cycle_report)."""

    def _run_find_subspace(self, mol):
        mf = mol.RHF()
        mf.verbose = 0
        mf.kernel()
        F, S = mf.get_fock(), mf.get_ovlp()
        shellsep_mol = create_shell_separated_mol(mol)
        mask_history = find_subspace(
            F, S, mol, mf, conv_tol=0.5, verbose=False,
            get_smask=True, return_mask_history=True,
        )
        return mf, F, S, shellsep_mol, mask_history

    def test_default_is_unaffected(self, h2o_sto3g):
        """cycle_report defaults to False: omitting it must reproduce the
        exact same return shape/values as explicitly passing False --
        regression safety for every existing call site (which never
        mentions cycle_report at all)."""
        mf, F, S, shellsep_mol, mask_history = self._run_find_subspace(h2o_sto3g)
        kwargs = {'verbose': False, 'C_full': mf.mo_coeff, 'calculate_correction': False}
        result_default = mask_analysis(mask_history, shellsep_mol, mf, F, S, **kwargs)
        result_explicit_false = mask_analysis(
            mask_history, shellsep_mol, mf, F, S, cycle_report=False, **kwargs)
        assert isinstance(result_default, list)
        assert len(result_default) == len(result_explicit_false)
        for row_a, row_b in zip(result_default, result_explicit_false):
            assert row_a[0] == row_b[0]  # nfunc
            assert row_a[3] == pytest.approx(row_b[3], abs=1e-8)  # E_scf

    def test_enabled_returns_extra_history(self, h2o_def2tzvp):
        mf, F, S, shellsep_mol, mask_history = self._run_find_subspace(h2o_def2tzvp)
        kwargs = {'verbose': False, 'C_full': mf.mo_coeff, 'calculate_correction': False}
        result = mask_analysis(
            mask_history, shellsep_mol, mf, F, S, cycle_report=True, **kwargs)
        assert isinstance(result, tuple) and len(result) == 2
        dataframe, cycle_report_history = result
        assert len(dataframe) == len(cycle_report_history)
        for row, entry in zip(dataframe, cycle_report_history):
            assert entry['nfunc'] == row[0]
            occ = [o for o in entry['orbitals'] if o[2]]
            virt = [o for o in entry['orbitals'] if not o[2]]
            assert len(occ) > 0
            assert len(virt) > 0

    def test_both_flags_together(self, h2o_def2tzvp):
        mf, F, S, shellsep_mol, mask_history = self._run_find_subspace(h2o_def2tzvp)
        result = mask_analysis(
            mask_history, shellsep_mol, mf, F, S, verbose=False,
            C_full=mf.mo_coeff, calculate_correction=False,
            track_orbitals=True, cycle_report=True,
        )
        assert isinstance(result, tuple) and len(result) == 3
        dataframe, orbital_history, cycle_report_history = result
        assert len(dataframe) == len(orbital_history) == len(cycle_report_history)


class TestWriteCycleReport:
    """Unit tests for the standalone formatter, fed synthetic data --
    doesn't need a real SCF run."""

    def _synthetic_row(self, nfunc, e_scf, mask):
        return [nfunc, 0.0, 0.0, e_scf, 0.0, 0.0, mask, 0.0, 0.0, True]

    def test_writes_expected_sections(self, h2o_sto3g, tmp_path):
        nao = h2o_sto3g.nao_nr()
        mask1 = np.zeros(nao, dtype=bool)
        mask1[:3] = True
        mask2 = mask1.copy()
        mask2[3] = True

        dataframe = [
            self._synthetic_row(3, -74.9, mask1),
            self._synthetic_row(4, -75.0, mask2),
        ]
        cycle_report_history = [
            {'nfunc': 3, 'orbitals': [(-20.0, None, True, None), (-1.0, None, False, None)]},
            {'nfunc': 4, 'orbitals': [(-20.0, None, True, None), (-0.9, None, False, None)]},
        ]

        path = tmp_path / "report.txt"
        write_cycle_report(str(path), dataframe, cycle_report_history, h2o_sto3g,
                            header_info={'molecule': 'h2o'})
        content = path.read_text()
        assert "molecule: h2o" in content
        assert "Cycle 0" in content and "Cycle 1" in content
        assert "nfunc=3" in content and "nfunc=4" in content
        assert "E_scf=-74.900000" in content
        assert "Added functions" in content
        assert "Frontier orbitals by irrep" in content

    def test_flags_occ_virt_inversion(self, h2o_sto3g, tmp_path):
        nao = h2o_sto3g.nao_nr()
        mask = np.zeros(nao, dtype=bool)
        mask[:3] = True
        dataframe = [self._synthetic_row(3, -74.9, mask)]
        # occ energy (-0.5) above virt energy (-0.6): a real inversion.
        cycle_report_history = [
            {'nfunc': 3, 'orbitals': [(-0.5, 'A1', True, None), (-0.6, 'A1', False, None)]},
        ]
        path = tmp_path / "report.txt"
        write_cycle_report(str(path), dataframe, cycle_report_history, h2o_sto3g)
        assert "INVERSION" in path.read_text()

    def test_mismatched_nfunc_raises(self, h2o_sto3g, tmp_path):
        nao = h2o_sto3g.nao_nr()
        mask = np.zeros(nao, dtype=bool)
        mask[:3] = True
        dataframe = [self._synthetic_row(3, -74.9, mask)]
        cycle_report_history = [{'nfunc': 4, 'orbitals': []}]
        with pytest.raises(ValueError):
            write_cycle_report(str(tmp_path / "report.txt"), dataframe,
                                cycle_report_history, h2o_sto3g)
