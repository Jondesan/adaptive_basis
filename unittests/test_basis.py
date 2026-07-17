import numpy as np
import pytest
import pyscf
from pyscf import gto
import adb.basisutil
from adb import (
    create_shell_separated_mol, create_subbasis_mol, init_smask, extract_basis,
    find_projected_minimal_basis_mask, get_array_of_angular_momenta_and_atom_id,
)


# ╭─────────────────────────────────────────────────────────────────────────╮
# │ molutil.create_shell_separated_mol                                      │
# ╰─────────────────────────────────────────────────────────────────────────╯

class TestCreateShellSeparatedMol:

    def test_same_natm(self, h2o_sto3g):
        shellsep = create_shell_separated_mol(h2o_sto3g)
        assert shellsep.natm == h2o_sto3g.natm

    def test_same_atom_symbols(self, h2o_sto3g):
        mol = h2o_sto3g
        shellsep = create_shell_separated_mol(mol)
        orig_symbols = [mol.atom_pure_symbol(i) for i in range(mol.natm)]
        sep_symbols  = [shellsep.atom_pure_symbol(i) for i in range(shellsep.natm)]
        assert orig_symbols == sep_symbols

    def test_same_charge_and_spin(self, h2o_sto3g):
        mol = h2o_sto3g
        shellsep = create_shell_separated_mol(mol)
        assert shellsep.charge == mol.charge
        assert shellsep.spin == mol.spin

    def test_sto3g_nao_unchanged(self, h2o_sto3g):
        """STO-3G has one contraction per shell: shell separation leaves nao unchanged."""
        shellsep = create_shell_separated_mol(h2o_sto3g)
        assert shellsep.nao == h2o_sto3g.nao


# ╭─────────────────────────────────────────────────────────────────────────╮
# │ mask.init_smask                                                         │
# ╰─────────────────────────────────────────────────────────────────────────╯

class TestInitSmask:

    def test_all_false_at_init(self, h2o_sto3g):
        shellsep = create_shell_separated_mol(h2o_sto3g)
        smask = init_smask(shellsep)
        assert all(not sm[0] for sm in smask)

    def test_length_matches_nbas(self, h2o_sto3g):
        shellsep = create_shell_separated_mol(h2o_sto3g)
        smask = init_smask(shellsep)
        assert len(smask) == shellsep.nbas

    def test_nfunc_sum_matches_nao(self, h2o_sto3g):
        """Sum of per-shell function counts must equal nao."""
        shellsep = create_shell_separated_mol(h2o_sto3g)
        smask = init_smask(shellsep)
        assert sum(int(sm[1]) for sm in smask) == shellsep.nao

    def test_each_entry_has_atom_index_and_symbol(self, h2o_sto3g):
        """Every smask entry's metadata should reference a valid atom."""
        mol = h2o_sto3g
        shellsep = create_shell_separated_mol(mol)
        smask = init_smask(shellsep)
        valid_indices = set(range(mol.natm))
        for sm in smask:
            atom_idx, atom_sym = sm[3][0], sm[3][1]
            assert atom_idx in valid_indices
            assert isinstance(atom_sym, str) and len(atom_sym) > 0


# ╭─────────────────────────────────────────────────────────────────────────╮
# │ basisutil.extract_basis                                                 │
# ╰─────────────────────────────────────────────────────────────────────────╯

class TestExtractBasis:

    def test_full_smask_rebuilds_with_same_nao(self, h2o_sto3g):
        """Extracting all shells produces a mol with the same nao as shellsep_mol."""
        mol = h2o_sto3g
        shellsep = create_shell_separated_mol(mol)
        smask = init_smask(shellsep)
        smask[:, 0] = True

        basis, ecp = extract_basis(smask, shellsep)
        rebuilt = gto.Mole(
            atom=mol.atom, basis=basis, charge=mol.charge,
            spin=mol.spin, unit=mol.unit, verbose=0,
        )
        rebuilt.build()
        assert rebuilt.nao == shellsep.nao

    def test_extracted_basis_covers_all_species(self, h2o_sto3g):
        """The returned dict has an entry for every atom species in the molecule."""
        mol = h2o_sto3g
        shellsep = create_shell_separated_mol(mol)
        smask = init_smask(shellsep)
        smask[:, 0] = True

        basis, _ = extract_basis(smask, shellsep)
        assert set(basis.keys()) == set(shellsep._basis.keys())

    def test_length_mismatch_raises(self, h2o_sto3g):
        """Passing a smask whose length doesn't match shellsep must raise ValueError."""
        shellsep = create_shell_separated_mol(h2o_sto3g)
        smask = init_smask(shellsep)
        bad_smask = smask[:-1]          # one entry too short
        with pytest.raises(ValueError):
            extract_basis(bad_smask, shellsep)

    def test_no_ecp_returns_none(self, h2o_sto3g):
        """For a molecule without ECPs, ecp_basis should be None."""
        shellsep = create_shell_separated_mol(h2o_sto3g)
        smask = init_smask(shellsep)
        smask[:, 0] = True
        _, ecp = extract_basis(smask, shellsep)
        assert ecp is None

    def test_noncontiguous_selection_on_standard_basis(self, h2o_def2tzvp):
        """Dropping an entire angular momentum from the mask (keep O's S and
        D shells, drop all of its P shells) must not crash and must extract
        the right shells -- this is the kind of non-contiguous *selection*
        the adaptive greedy search can produce day to day. (On a standard,
        angular-momentum-contiguous basis set like def2-tzvp this actually
        doesn't exercise the historical indexing bug below -- ogbas is built
        from the full, unmasked basis at every call site, so position
        happens to equal angular momentum regardless of what's selected.
        This test instead documents/guards the expected composition for
        ordinary sparse selections.)"""
        mol = h2o_def2tzvp
        shellsep = create_shell_separated_mol(mol)
        smask = init_smask(shellsep)
        for row in smask:
            row[0] = not (row[2] == 1 and row[3][1] == 'O')

        basis, _ = extract_basis(smask, shellsep)
        oxygen_angls = {shell[0] for shell in basis['O']}
        assert 1 not in oxygen_angls
        assert oxygen_angls == {0, 2, 3}

        rebuilt = gto.Mole(
            atom=mol.atom, basis=basis, charge=mol.charge,
            spin=mol.spin, unit=mol.unit, verbose=0,
        )
        rebuilt.build()
        expected_nao = sum(int(sm[1]) for sm in smask if sm[0])
        assert rebuilt.nao == expected_nao

    def test_gapped_basis_definition_does_not_crash(self):
        """Regression test for the `ogbas[i]` half of the historical
        positional-vs-angular-momentum indexing bug. `to_general_contraction`
        only emits one entry per angular momentum *actually present*, so
        position and angular momentum coincide only when the basis has no
        l-gaps -- true for essentially every named basis set (hence
        test_noncontiguous_selection_on_standard_basis above doesn't trigger
        it), but not guaranteed. This hand-built O basis (S and D shells
        only, no P) makes the *full* basis itself gapped, which reproduces
        the crash directly: confirmed against the pre-fix code, this raised
        `IndexError: list index out of range` from the `ogbas[i]` line."""
        o_gapped_basis = [
            [0, [130.70932, 0.15432897], [23.808861, 0.53532814], [6.4436083, 0.44463454]],
            [2, [1.0, 1.0]],
        ]
        mol = pyscf.M(
            atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
            basis={'O': o_gapped_basis, 'H': 'sto-3g'},
            verbose=0,
        )
        shellsep = create_shell_separated_mol(mol)
        smask = init_smask(shellsep)
        smask[:, 0] = True

        basis, _ = extract_basis(smask, shellsep)
        assert {shell[0] for shell in basis['O']} == {0, 2}

        rebuilt = gto.Mole(atom=mol.atom, basis=basis, verbose=0)
        rebuilt.build()
        assert rebuilt.nao == shellsep.nao

    def test_all_zero_contraction_drops_correct_shell_not_by_position(
            self, monkeypatch, h2o_def2tzvp):
        """Regression test for the `basis[key].pop(i)` half of the historical
        bug -- distinct from test_gapped_basis_definition_does_not_crash
        above, which only exercises the neighboring `ogbas[i]` line. Even
        when `ogbas[i]` is correct (a contiguous full basis), `pop(i)` used
        list *position* `i` (an angular momentum value) on `basis[key]`,
        which genuinely is angular-momentum-sparse under an ordinary mask
        selection. Forcing that path requires a selected shell's contraction
        column to filter down to all-zero, which real basis coefficients
        never do -- so `to_general_contraction` is monkeypatched to inject
        one all-zero D-shell column while leaving everything else real.
        Confirmed against the pre-fix code, this raises `IndexError: pop
        index out of range` (`basis['O'] = [[0], [2]]`, `.pop(2)`)."""
        mol = h2o_def2tzvp
        shellsep = create_shell_separated_mol(mol)
        smask = init_smask(shellsep)
        for sm in smask:
            if sm[3][1] == 'O':
                # O: keep all S shells, drop P and F entirely, keep only
                # the first of O's two D shells (n_remove_ecp == 3).
                sm[0] = (sm[2] == 0) or (sm[2] == 2 and sm[3][4] == 3)
            else:
                sm[0] = True

        original = adb.basisutil.to_general_contraction

        def fake_to_general_contraction(basis):
            real = original(basis)
            if [entry[0] for entry in real] == [0, 1, 2, 3]:
                # Replace O's D entry (index 2) with one whose sole
                # contraction column is entirely zero -- filtered_shell
                # ends up empty, so this shell must be dropped.
                zeroed_d = [2, [1.5, 0.0], [0.6, 0.0]]
                return [real[0], real[1], zeroed_d, real[3]]
            return real

        monkeypatch.setattr(
            adb.basisutil, "to_general_contraction", fake_to_general_contraction)

        basis, _ = extract_basis(smask, shellsep)
        assert {shell[0] for shell in basis['O']} == {0}


# ╭─────────────────────────────────────────────────────────────────────────╮
# │ molutil.create_subbasis_mol                                             │
# ╰─────────────────────────────────────────────────────────────────────────╯

class TestCreateSubbasisMol:

    def test_full_smask_builds_successfully(self, h2o_sto3g):
        """A full smask produces a molecule that builds without error."""
        mol = h2o_sto3g
        shellsep = create_shell_separated_mol(mol)
        smask = init_smask(shellsep)
        smask[:, 0] = True
        sub_mol = create_subbasis_mol(mol, smask)
        assert sub_mol.nao > 0

    def test_full_smask_nao_equals_shellsep(self, h2o_sto3g):
        """With all shells selected, the subbasis mol has the same nao as shellsep_mol."""
        mol = h2o_sto3g
        shellsep = create_shell_separated_mol(mol)
        smask = init_smask(shellsep)
        smask[:, 0] = True
        sub_mol = create_subbasis_mol(mol, smask)
        assert sub_mol.nao == shellsep.nao

    def test_preserves_charge_and_spin(self, h2o_sto3g):
        mol = h2o_sto3g
        shellsep = create_shell_separated_mol(mol)
        smask = init_smask(shellsep)
        smask[:, 0] = True
        sub_mol = create_subbasis_mol(mol, smask)
        assert sub_mol.charge == mol.charge
        assert sub_mol.spin == mol.spin


# ╭─────────────────────────────────────────────────────────────────────────╮
# │ initialization.find_projected_minimal_basis_mask                        │
# ╰─────────────────────────────────────────────────────────────────────────╯

class TestFindProjectedMinimalBasisMask:

    def test_returns_boolean_mask_of_correct_length(self, h2o_def2tzvp):
        mask = find_projected_minimal_basis_mask(h2o_def2tzvp)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == h2o_def2tzvp.nao_nr()

    def test_selected_count_matches_sto3g_size(self, h2o_def2tzvp):
        """The number of selected functions must equal the nao of STO-3G
        built on the same geometry, since that is the minimal basis being
        projected onto."""
        mask = find_projected_minimal_basis_mask(h2o_def2tzvp)
        mol_sto3g = pyscf.M(atom=h2o_def2tzvp.atom, basis='sto3g', verbose=0)
        assert np.sum(mask) == mol_sto3g.nao_nr()

    def test_identity_when_mol_is_already_sto3g(self, h2o_sto3g):
        """Projecting STO-3G onto itself should select every function."""
        mask = find_projected_minimal_basis_mask(h2o_sto3g)
        assert np.all(mask)

    def test_identity_for_h2_sto3g(self, h2_sto3g):
        mask = find_projected_minimal_basis_mask(h2_sto3g)
        assert np.all(mask)

    def test_selection_is_shell_complete(self, h2o_def2tzvp):
        """link_shells guarantees that within any given shell (fixed atom,
        angular momentum and contraction) the selection is all-or-nothing."""
        mol = h2o_def2tzvp
        mask = find_projected_minimal_basis_mask(mol)
        ao_loc = mol.ao_loc_nr()
        for start, end in zip(ao_loc[:-1], ao_loc[1:]):
            shell_mask = mask[start:end]
            assert shell_mask.all() or not shell_mask.any()

    def test_shell_composition_matches_sto3g(self, h2o_def2tzvp):
        """For water, the projected minimal basis should reproduce STO-3G's
        shell composition: 2 s-type + 3 p-type functions on O, 1 s-type on
        each H."""
        mol = h2o_def2tzvp
        mask = find_projected_minimal_basis_mask(mol)
        angls_aid = get_array_of_angular_momenta_and_atom_id(mol)
        selected = angls_aid[mask]

        oxygen = selected[selected[:, 1] == 0]
        assert np.sum(oxygen[:, 0] == 0) == 2
        assert np.sum(oxygen[:, 0] == 1) == 3

        for atom_id in (1, 2):
            hydrogen = selected[selected[:, 1] == atom_id]
            assert len(hydrogen) == 1
            assert hydrogen[0, 0] == 0

    def test_selected_count_matches_sto3g_size_general_contraction(self, h2o_augpc1):
        """Same invariant as test_selected_count_matches_sto3g_size, but for a
        generally-contracted basis (aug-pc-1), where several shells of the
        same atom/angular-momentum share the same set of primitives."""
        mask = find_projected_minimal_basis_mask(h2o_augpc1)
        mol_sto3g = pyscf.M(atom=h2o_augpc1.atom, basis='sto3g', verbose=0)
        assert np.sum(mask) == mol_sto3g.nao_nr()

    def test_shell_composition_matches_sto3g_general_contraction(self, h2o_augpc1):
        """Same invariant as test_shell_composition_matches_sto3g, but for a
        generally-contracted basis (aug-pc-1)."""
        mol = h2o_augpc1
        mask = find_projected_minimal_basis_mask(mol)
        angls_aid = get_array_of_angular_momenta_and_atom_id(mol)
        selected = angls_aid[mask]

        oxygen = selected[selected[:, 1] == 0]
        assert np.sum(oxygen[:, 0] == 0) == 2
        assert np.sum(oxygen[:, 0] == 1) == 3

        for atom_id in (1, 2):
            hydrogen = selected[selected[:, 1] == atom_id]
            assert len(hydrogen) == 1
            assert hydrogen[0, 0] == 0

