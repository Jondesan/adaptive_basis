import numpy as np
import pytest
import pyscf
from pyscf import gto
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

