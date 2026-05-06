import pytest
import numpy as np
from pyscf import gto
import adb


# ---------------------------------------------------------------------------
# adb.create_shell_separated_mol
# ---------------------------------------------------------------------------

class TestCreateShellSeparatedMol:

    def test_same_natm(self, h2o_sto3g):
        shellsep = adb.create_shell_separated_mol(h2o_sto3g)
        assert shellsep.natm == h2o_sto3g.natm

    def test_same_atom_symbols(self, h2o_sto3g):
        mol = h2o_sto3g
        shellsep = adb.create_shell_separated_mol(mol)
        orig_symbols = [mol.atom_pure_symbol(i) for i in range(mol.natm)]
        sep_symbols  = [shellsep.atom_pure_symbol(i) for i in range(shellsep.natm)]
        assert orig_symbols == sep_symbols

    def test_same_charge_and_spin(self, h2o_sto3g):
        mol = h2o_sto3g
        shellsep = adb.create_shell_separated_mol(mol)
        assert shellsep.charge == mol.charge
        assert shellsep.spin == mol.spin

    def test_sto3g_nao_unchanged(self, h2o_sto3g):
        """STO-3G has one contraction per shell: shell separation leaves nao unchanged."""
        shellsep = adb.create_shell_separated_mol(h2o_sto3g)
        assert shellsep.nao == h2o_sto3g.nao


# ---------------------------------------------------------------------------
# adb.init_smask
# ---------------------------------------------------------------------------

class TestInitSmask:

    def test_all_false_at_init(self, h2o_sto3g):
        shellsep = adb.create_shell_separated_mol(h2o_sto3g)
        smask = adb.init_smask(shellsep)
        assert all(not sm[0] for sm in smask)

    def test_length_matches_nbas(self, h2o_sto3g):
        shellsep = adb.create_shell_separated_mol(h2o_sto3g)
        smask = adb.init_smask(shellsep)
        assert len(smask) == shellsep.nbas

    def test_nfunc_sum_matches_nao(self, h2o_sto3g):
        """Sum of per-shell function counts must equal nao."""
        shellsep = adb.create_shell_separated_mol(h2o_sto3g)
        smask = adb.init_smask(shellsep)
        assert sum(int(sm[1]) for sm in smask) == shellsep.nao

    def test_each_entry_has_atom_index_and_symbol(self, h2o_sto3g):
        """Every smask entry's metadata should reference a valid atom."""
        mol = h2o_sto3g
        shellsep = adb.create_shell_separated_mol(mol)
        smask = adb.init_smask(shellsep)
        valid_indices = set(range(mol.natm))
        for sm in smask:
            atom_idx, atom_sym = sm[3][0], sm[3][1]
            assert atom_idx in valid_indices
            assert isinstance(atom_sym, str) and len(atom_sym) > 0


# ---------------------------------------------------------------------------
# adb.extract_basis
# ---------------------------------------------------------------------------

class TestExtractBasis:

    def test_full_smask_rebuilds_with_same_nao(self, h2o_sto3g):
        """Extracting all shells produces a mol with the same nao as shellsep_mol."""
        mol = h2o_sto3g
        shellsep = adb.create_shell_separated_mol(mol)
        smask = adb.init_smask(shellsep)
        smask[:, 0] = True

        basis, ecp = adb.extract_basis(smask, shellsep)
        rebuilt = gto.Mole(
            atom=mol.atom, basis=basis, charge=mol.charge,
            spin=mol.spin, unit=mol.unit, verbose=0,
        )
        rebuilt.build()
        assert rebuilt.nao == shellsep.nao

    def test_extracted_basis_covers_all_species(self, h2o_sto3g):
        """The returned dict has an entry for every atom species in the molecule."""
        mol = h2o_sto3g
        shellsep = adb.create_shell_separated_mol(mol)
        smask = adb.init_smask(shellsep)
        smask[:, 0] = True

        basis, _ = adb.extract_basis(smask, shellsep)
        assert set(basis.keys()) == set(shellsep._basis.keys())

    def test_length_mismatch_raises(self, h2o_sto3g):
        """Passing a smask whose length doesn't match shellsep must raise ValueError."""
        shellsep = adb.create_shell_separated_mol(h2o_sto3g)
        smask = adb.init_smask(shellsep)
        bad_smask = smask[:-1]          # one entry too short
        with pytest.raises(ValueError):
            adb.extract_basis(bad_smask, shellsep)

    def test_no_ecp_returns_none(self, h2o_sto3g):
        """For a molecule without ECPs, ecp_basis should be None."""
        shellsep = adb.create_shell_separated_mol(h2o_sto3g)
        smask = adb.init_smask(shellsep)
        smask[:, 0] = True
        _, ecp = adb.extract_basis(smask, shellsep)
        assert ecp is None


# ---------------------------------------------------------------------------
# adb.create_subbasis_mol
# ---------------------------------------------------------------------------

class TestCreateSubbasisMol:

    def test_full_smask_builds_successfully(self, h2o_sto3g):
        """A full smask produces a molecule that builds without error."""
        mol = h2o_sto3g
        shellsep = adb.create_shell_separated_mol(mol)
        smask = adb.init_smask(shellsep)
        smask[:, 0] = True
        sub_mol = adb.create_subbasis_mol(mol, smask)
        assert sub_mol.nao > 0

    def test_full_smask_nao_equals_shellsep(self, h2o_sto3g):
        """With all shells selected, the subbasis mol has the same nao as shellsep_mol."""
        mol = h2o_sto3g
        shellsep = adb.create_shell_separated_mol(mol)
        smask = adb.init_smask(shellsep)
        smask[:, 0] = True
        sub_mol = adb.create_subbasis_mol(mol, smask)
        assert sub_mol.nao == shellsep.nao

    def test_preserves_charge_and_spin(self, h2o_sto3g):
        mol = h2o_sto3g
        shellsep = adb.create_shell_separated_mol(mol)
        smask = adb.init_smask(shellsep)
        smask[:, 0] = True
        sub_mol = adb.create_subbasis_mol(mol, smask)
        assert sub_mol.charge == mol.charge
        assert sub_mol.spin == mol.spin

