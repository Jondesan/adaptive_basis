import numpy as np
import pytest
from adb import symmetry_safe_newton


# ╭─────────────────────────────────────────────────────────────────────────╮
# │ scf_fixes.symmetry_safe_newton                                          │
# ╰─────────────────────────────────────────────────────────────────────────╯

class TestSymmetrySafeNewton:
    """Regression tests for the pyscf bug documented in adb/scf_fixes.py:
    pyscf's own `mf.newton()` SOSCF solver crashes when `mol.symmetry` is
    enabled together with a mean-field object whose basis has been reduced
    relative to `mol.nao` (e.g. via scf.addons.remove_linear_dep_ on a
    near-linearly-dependent basis) -- its internal "keyframe" bookkeeping
    composes two (nmo, nmo) orbital-rotation matrices by reusing rotate_mo,
    which unconditionally tries to symmetry-label its first argument as if
    it were a real (nao, nmo) AO-basis MO coefficient matrix. That only
    raises once nao != nmo, which isn't practical to trigger through a
    real, fast SCF in a unit test -- so these tests exercise rotate_mo
    directly with hand-built shapes, which is exactly the mechanism that
    fails.
    """

    def test_matches_plain_newton_for_real_ao_basis_case(self, h2o_sto3g_c2v):
        """For a genuine (nao, nmo) AO-basis call (the normal case, no
        basis reduction), the patched rotate_mo must behave identically
        to pyscf's own -- same returned array, same orbsym labels."""
        mf = h2o_sto3g_c2v.RHF()
        mf.kernel()
        nao = h2o_sto3g_c2v.nao

        old_soscf = mf.newton()
        new_soscf = symmetry_safe_newton(mf)

        u = np.eye(nao)
        old_result = old_soscf.rotate_mo(mf.mo_coeff, u)
        new_result = new_soscf.rotate_mo(mf.mo_coeff, u)

        assert np.allclose(old_result, new_result)
        assert list(old_result.orbsym) == list(new_result.orbsym)

    def test_full_scf_energy_matches_plain_newton(self, h2o_sto3g_c2v):
        """End-to-end: running the wrapped solver to convergence on an
        ordinary (non-reduced-basis) symmetric molecule gives the same
        energy as pyscf's own .newton() -- the wrapper changes nothing
        about the actual optimization, only the one buggy bookkeeping call.
        """
        mf1 = h2o_sto3g_c2v.RHF()
        mf1.kernel()
        soscf1 = mf1.newton()
        soscf1.kernel()

        mf2 = h2o_sto3g_c2v.RHF()
        mf2.kernel()
        soscf2 = symmetry_safe_newton(mf2)
        soscf2.kernel()

        assert np.isclose(soscf1.e_tot, soscf2.e_tot)

    def test_old_rotate_mo_crashes_on_reduced_basis_rotation_matrix(self, h2o_sto3g_c2v):
        """Reproduces the actual pyscf bug directly: composing two (nmo,
        nmo) rotation matrices with nmo < nao -- exactly what happens
        internally whenever scf.addons.remove_linear_dep_ has actually
        dropped some AO combinations -- crashes pyscf's own rotate_mo with
        a shape mismatch between the (nao, nao) overlap and the (nmo, nmo)
        rotation."""
        mf = h2o_sto3g_c2v.RHF()
        mf.kernel()
        nao = h2o_sto3g_c2v.nao
        nmo = nao - 2   # simulate 2 linearly-dependent combinations removed

        rng = np.random.default_rng(0)
        ukf = rng.standard_normal((nmo, nmo))
        u = rng.standard_normal((nmo, nmo))

        old_soscf = mf.newton()
        with pytest.raises(ValueError, match="not aligned"):
            old_soscf.rotate_mo(ukf, u)

    def test_new_rotate_mo_handles_reduced_basis_rotation_matrix(self, h2o_sto3g_c2v):
        """The fixed version must not crash on the same input, and must
        return the mathematically correct composed rotation (a bare
        matrix product -- symmetry labeling a rotation matrix is
        meaningless and correctly skipped, not silently wrong)."""
        mf = h2o_sto3g_c2v.RHF()
        mf.kernel()
        nao = h2o_sto3g_c2v.nao
        nmo = nao - 2

        rng = np.random.default_rng(0)
        ukf = rng.standard_normal((nmo, nmo))
        u = rng.standard_normal((nmo, nmo))

        new_soscf = symmetry_safe_newton(mf)
        result = new_soscf.rotate_mo(ukf, u)

        assert result.shape == (nmo, nmo)
        assert np.allclose(result, np.dot(ukf, u))
        assert not hasattr(result, 'orbsym')


class TestSymmetrySafeNewtonUHF:
    """UHF counterpart of TestSymmetrySafeNewton (fix 2 in scf_fixes.py's
    module docstring): mo_coeff/u are spin-stacked (2, n, n) arrays for UHF,
    so the restricted branch's plain `numpy.dot(mo_coeff, u)` does not
    compute the intended per-spin product -- symmetry_safe_newton must
    dispatch UHF mean fields to a rotate_mo that composes each spin
    channel separately, as pyscf's own (otherwise unguarded)
    `_SecondOrderUHF.rotate_mo` does.
    """

    def test_matches_plain_newton_for_real_ao_basis_case(self, nh2_sto3g_c2v):
        """Genuine (2, nao, nmo) AO-basis call, no basis reduction: the
        patched rotate_mo must behave identically to pyscf's own."""
        mf = nh2_sto3g_c2v.UHF()
        mf.kernel()
        nao = nh2_sto3g_c2v.nao

        old_soscf = mf.newton()
        new_soscf = symmetry_safe_newton(mf)

        u = np.asarray([np.eye(nao), np.eye(nao)])
        old_result = old_soscf.rotate_mo(mf.mo_coeff, u)
        new_result = new_soscf.rotate_mo(mf.mo_coeff, u)

        assert np.allclose(old_result, new_result)
        assert list(old_result.orbsym) == list(new_result.orbsym)

    def test_full_scf_energy_matches_plain_newton(self, nh2_sto3g_c2v):
        """End-to-end: same converged energy as pyscf's own .newton() on an
        ordinary (non-reduced-basis) symmetric open-shell molecule."""
        mf1 = nh2_sto3g_c2v.UHF()
        mf1.kernel()
        soscf1 = mf1.newton()
        soscf1.kernel()

        mf2 = nh2_sto3g_c2v.UHF()
        mf2.kernel()
        soscf2 = symmetry_safe_newton(mf2)
        soscf2.kernel()

        assert np.isclose(soscf1.e_tot, soscf2.e_tot)

    def test_old_rotate_mo_crashes_on_reduced_basis_rotation_matrix(self, nh2_sto3g_c2v):
        """Same bug as the restricted case, reproduced for UHF: pyscf's own
        (unguarded) _SecondOrderUHF.rotate_mo also symmetry-labels its
        first argument unconditionally, so composing two spin-stacked
        (2, nmo, nmo) rotation matrices with nmo < nao crashes it too."""
        mf = nh2_sto3g_c2v.UHF()
        mf.kernel()
        nao = nh2_sto3g_c2v.nao
        nmo = nao - 2

        rng = np.random.default_rng(0)
        ukf = np.asarray([rng.standard_normal((nmo, nmo)) for _ in range(2)])
        u = np.asarray([rng.standard_normal((nmo, nmo)) for _ in range(2)])

        old_soscf = mf.newton()
        with pytest.raises(Exception):
            old_soscf.rotate_mo(ukf, u)

    def test_new_rotate_mo_handles_reduced_basis_rotation_matrix(self, nh2_sto3g_c2v):
        """The fixed version must not crash on the same input, and must
        return the mathematically correct per-spin composed rotation."""
        mf = nh2_sto3g_c2v.UHF()
        mf.kernel()
        nao = nh2_sto3g_c2v.nao
        nmo = nao - 2

        rng = np.random.default_rng(0)
        ukf = np.asarray([rng.standard_normal((nmo, nmo)) for _ in range(2)])
        u = np.asarray([rng.standard_normal((nmo, nmo)) for _ in range(2)])

        new_soscf = symmetry_safe_newton(mf)
        result = new_soscf.rotate_mo(ukf, u)

        assert result.shape == (2, nmo, nmo)
        assert np.allclose(result[0], np.dot(ukf[0], u[0]))
        assert np.allclose(result[1], np.dot(ukf[1], u[1]))
        assert not hasattr(result, 'orbsym')

    def test_does_not_perform_naive_dot(self, nh2_sto3g_c2v):
        """Regression guard for the actual reported bug: a naive
        numpy.dot(mo_coeff, u) on spin-stacked (2, n, n) arrays produces a
        (2, n, 2, n)-shaped non-answer instead of raising -- assert the
        patched rotate_mo never returns that shape."""
        mf = nh2_sto3g_c2v.UHF()
        mf.kernel()
        nao = nh2_sto3g_c2v.nao

        new_soscf = symmetry_safe_newton(mf)
        u = np.asarray([np.eye(nao), np.eye(nao)])
        result = new_soscf.rotate_mo(mf.mo_coeff, u)

        assert result.shape == (2, nao, nao)
