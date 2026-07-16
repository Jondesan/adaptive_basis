import sys
import os
import pytest
import pyscf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture(scope="session")
def h2_sto3g():
    return pyscf.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)


@pytest.fixture(scope="session")
def h2o_def2tzvp():
    return pyscf.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis="def2-tzvp",
        verbose=0,
    )

@pytest.fixture(scope="session")
def h2o_augpc1():
    return pyscf.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis="aug-pc-1",
        verbose=0,
    )

@pytest.fixture(scope="session")
def h2o_sto3g():
    return pyscf.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis="sto-3g",
        verbose=0,
    )

@pytest.fixture(scope="session")
def h2o_sto3g_c2v():
    """Same geometry/basis as h2o_sto3g, but with mol.symmetry enabled.

    Used by the optional symmetry-aware adaptive-basis-search tests; none
    of the other fixtures here set mol.symmetry.
    """
    return pyscf.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis="sto-3g",
        symmetry=True,
        verbose=0,
    )

@pytest.fixture(scope="session")
def h2o_mf(h2o_def2tzvp):
    mf = h2o_def2tzvp.HF()
    mf.verbose = 0
    mf.kernel()
    return mf

@pytest.fixture(scope="session")
def h2o_apc1_mf(h2o_augpc1):
    mf = h2o_augpc1.HF()
    mf.verbose = 0
    mf.kernel()
    return mf