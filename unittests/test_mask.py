import numpy as np
from maskutil import init_smask, smask_to_mask, mask_to_smask, mask_matrix

# ---------------------------------------------------------------------------
# maskutil.init_smask
# ---------------------------------------------------------------------------

class TestInitSmask:

    def test_smask_all_false_init(self, h2o_def2tzvp):
        init_sm = init_smask(h2o_def2tzvp, False)
        assert(not np.all([sm[0] for sm in init_sm]))

    def test_smask_minimal_basis_init(self, h2o_sto3g):
        init_sm = init_smask(h2o_sto3g, h2o_sto3g.cart)
        smask = np.array([[False, 1, 0, (0, 'O', 1, 'S', 1)],
                          [False, 1, 0, (0, 'O', 2, 'S', 2)],
                          [False, 3, 1, (0, 'O', 2, 'P', 2)],
                          [False, 1, 0, (1, 'H', 1, 'S', 1)],
                          [False, 1, 0, (2, 'H', 1, 'S', 1)]], dtype=object)
        np.testing.assert_array_equal(init_sm, smask)


# ---------------------------------------------------------------------------
# maskutil.smask_to_mask
# ---------------------------------------------------------------------------

class TestSmaskToMask:

    def test_smask_to_mask_roundtrip(self, h2o_def2tzvp):
        m = np.zeros(h2o_def2tzvp.nao_nr(), dtype=bool)
        # 0O2p
        m[5:8] = 1
        #1H1s     1H2p      2H1s     2H2p
        m[31] = m[34:37] = m[37] = m[40:43] = 1
        sm = init_smask(h2o_def2tzvp, h2o_def2tzvp.cart)
        np.testing.assert_array_equal(m, smask_to_mask(mask_to_smask(m, sm)))

    def test_smask_nfuncs_matches_mol_nao(self, h2o_def2tzvp):
        sm = init_smask(h2o_def2tzvp, h2o_def2tzvp.cart)
        nfunc_in_smask = np.sum(sm[:,1])
        assert(h2o_def2tzvp.nao_nr() == nfunc_in_smask)


# ---------------------------------------------------------------------------
# maskutil.mask_matrix
# ---------------------------------------------------------------------------

class TestMaskMatrix:

    def test_mask_matrix_rhf(self):
        mat = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
        ])
        mask = np.array([0, 0, 1, 1, 0, 1, 0, 0], dtype=bool)
        masked_mat = mask_matrix(mat, mask)
        test_mat = np.array([
            [3, 4, 6],
            [3, 4, 6],
            [3, 4, 6],
        ])
        np.testing.assert_array_equal(masked_mat, test_mat)

    def test_mask_matrix_uhf(self):
        mat = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
        ])
        mat = np.array([mat, mat])
        mask = np.array([0, 0, 1, 1, 0, 1, 0, 0], dtype=bool)
        masked_mat = mask_matrix(mat, mask)
        test_mat = np.array([
            [3, 4, 6],
            [3, 4, 6],
            [3, 4, 6],
        ])
        test_mat = np.array([test_mat, test_mat])
        np.testing.assert_array_equal(masked_mat, test_mat)
