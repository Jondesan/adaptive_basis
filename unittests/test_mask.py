import pytest
import numpy as np
import adb

# ---------------------------------------------------------------------------
# adb.init_smask
# ---------------------------------------------------------------------------

class TestInitSmask:

    def test_smask_all_false_init(self, h2o_def2tzvp):
        init_sm = adb.init_smask(h2o_def2tzvp, False)
        assert(not np.all([sm[0] for sm in init_sm]))

    def test_smask_minimal_basis_init(self, h2o_sto3g):
        init_sm = adb.init_smask(h2o_sto3g, h2o_sto3g.cart)
        smask = np.array([[False, 1, 0, (0, 'O', 1, 'S', 1)],
                          [False, 1, 0, (0, 'O', 2, 'S', 2)],
                          [False, 3, 1, (0, 'O', 2, 'P', 2)],
                          [False, 1, 0, (1, 'H', 1, 'S', 1)],
                          [False, 1, 0, (2, 'H', 1, 'S', 1)]], dtype=object)
        np.testing.assert_array_equal(init_sm, smask)


# ---------------------------------------------------------------------------
# adb.smask_to_mask
# ---------------------------------------------------------------------------

class TestSmaskToMask:

    def test_smask_to_mask_roundtrip(self, h2o_def2tzvp):
        m = np.zeros(h2o_def2tzvp.nao_nr(), dtype=bool)
        # 0O2p
        m[5:8] = 1
        #1H1s     1H2p      2H1s     2H2p
        m[31] = m[34:37] = m[37] = m[40:43] = 1
        sm = adb.init_smask(h2o_def2tzvp, h2o_def2tzvp.cart)
        np.testing.assert_array_equal(m, adb.smask_to_mask(adb.mask_to_smask(m, sm)))
