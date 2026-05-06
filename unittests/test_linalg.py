import pytest
import numpy as np
import adb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_symmetric(n, seed):
    rng = np.random.RandomState(seed)
    A = rng.randn(n, n)
    return A + A.T


def random_spd(n, seed):
    """Random symmetric positive-definite matrix."""
    A = random_symmetric(n, seed)
    return A + n * np.eye(n)  # diagonal dominance guarantees SPD


def orthonormal(n, seed):
    """Random orthonormal n-by-n matrix via QR decomposition."""
    rng = np.random.RandomState(seed)
    Q, _ = np.linalg.qr(rng.randn(n, n))
    return Q


# ---------------------------------------------------------------------------
# adb.eig / adb.canonical_orth
# ---------------------------------------------------------------------------

class TestEig:

    def test_identity_overlap_matches_numpy_eigh(self):
        """With identity overlap, eigenvalues match numpy's symmetric solver."""
        H = random_symmetric(4, seed=0)
        e, _ = adb.eig(H, np.eye(4))
        expected = np.linalg.eigh(H)[0]
        np.testing.assert_allclose(np.real(e), expected, atol=1e-10)

    def test_eigenvalues_ascending(self):
        """Returned eigenvalues are sorted in ascending order."""
        H = random_symmetric(5, seed=1)
        S = random_spd(5, seed=2)
        e, _ = adb.eig(H, S)
        assert np.all(np.diff(np.real(e)) >= 0), "eigenvalues not sorted"

    def test_rhf_output_shapes(self):
        """2D Fock -> 1D eigenvalues and 2D coefficient matrix."""
        n = 4
        e, C = adb.eig(random_symmetric(n, seed=3), np.eye(n))
        assert e.shape == (n,)
        assert C.shape == (n, n)

    def test_uhf_output_shapes(self):
        """3D Fock (UHF) -> shape (2, n) eigenvalues and (2, n, n) coefficients."""
        n = 3
        Ha = random_symmetric(n, seed=4)
        Hb = random_symmetric(n, seed=5)
        e, C = adb.eig(np.array([Ha, Hb]), np.eye(n))
        assert e.shape == (2, n)
        assert C.shape == (2, n, n)

    def test_uhf_both_spins_sorted(self):
        """Both alpha and beta eigenvalues are sorted ascending."""
        n = 4
        H = np.array([random_symmetric(n, seed=6), random_symmetric(n, seed=7)])
        e, _ = adb.eig(H, np.eye(n))
        assert np.all(np.diff(np.real(e[0])) >= 0), "alpha not sorted"
        assert np.all(np.diff(np.real(e[1])) >= 0), "beta not sorted"

    def test_coefficients_are_s_orthonormal(self):
        """C^T S C ≈ I (columns of C are S-orthonormal)."""
        n = 4
        H = random_symmetric(n, seed=8)
        S = random_spd(n, seed=9)
        _, C = adb.eig(H, S)
        CTC = C.conj().T @ S @ C
        # Ensure the diagonal assertion matrix has the same size as
        # CTC, canonical orthogonalization removes rows and columns if
        # too small eigenvalues are present
        n_assertion = CTC.shape[0]
        np.testing.assert_allclose(np.abs(CTC), np.eye(n_assertion), atol=1e-8)

    def test_generalized_eigenvalue_equation(self):
        """H C ≈ S C diag(e) — the Roothaan equation is satisfied."""
        n = 3
        H = random_symmetric(n, seed=10)
        S = random_spd(n, seed=11)
        e, C = adb.eig(H, S)
        lhs = H @ C
        rhs = S @ C @ np.diag(e)
        np.testing.assert_allclose(np.real(lhs), np.real(rhs), atol=1e-10)


# ---------------------------------------------------------------------------
# adb.get_q_sqrd
# ---------------------------------------------------------------------------

class TestGetQSquared:

    def test_rhf_full_projection_equals_twice_nocc(self):
        """RHF: identical orthonormal C and identity overlap → Q² = 2·nocc."""
        n, nocc = 4, (2, 2)
        C = orthonormal(n, seed=0)
        result = adb.get_q_sqrd(C, C, np.eye(n), nocc)
        np.testing.assert_allclose(result, 2.0 * nocc[0], atol=1e-10)

    def test_uhf_full_projection_equals_sum_nocc(self):
        """UHF: identical orthonormal C and identity overlap → Q² = nocc_a + nocc_b."""
        n, nocc = 4, (2, 1)
        Ca = orthonormal(n, seed=1)
        Cb = orthonormal(n, seed=2)
        C = np.array([Ca, Cb])
        result = adb.get_q_sqrd(C, C, np.eye(n), nocc)
        np.testing.assert_allclose(result, nocc[0] + nocc[1], atol=1e-10)

    def test_orthogonal_subspaces_gives_zero(self):
        """Occupied orbitals in orthogonal subspaces → Q² = 0."""
        # Cfull occupied = span{e0, e1}, Csub occupied = span{e2, e3}
        n, nocc = 4, (2, 2)
        Cfull = np.eye(n)
        Csub  = np.eye(n)[:, [2, 3, 0, 1]]   # reorder: "occupied" cols → e2, e3
        result = adb.get_q_sqrd(Cfull, Csub, np.eye(n), nocc)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_returns_real_scalar(self):
        """Return value is a real Python float, not complex."""
        n, nocc = 3, (1, 1)
        C = orthonormal(n, seed=3)
        result = adb.get_q_sqrd(C, C, np.eye(n), nocc)
        assert isinstance(result, float)
        assert np.imag(result) == 0.0

    def test_value_bounded_by_nocc(self):
        """Q² is non-negative and ≤ its maximum (2·nocc for RHF)."""
        n, nocc = 5, (2, 2)
        C_full = orthonormal(n, seed=4)
        C_sub  = orthonormal(n, seed=5)
        result = adb.get_q_sqrd(C_full, C_sub, np.eye(n), nocc)
        assert result >= -1e-10
        assert result <= 2.0 * nocc[0] + 1e-10


# ---------------------------------------------------------------------------
# adb.spherical_average / adb.sph_avg
# ---------------------------------------------------------------------------

class TestSphericalAverage:

    def test_s_shell_block_unchanged(self):
        """S-shells (ml=[1]) are not modified."""
        mat = np.array([[3.0, 1.0], [1.0, 2.0]])
        result = adb.spherical_average(mat, [1, 1])
        np.testing.assert_array_equal(result, mat)

    def test_p_shell_diagonal_becomes_uniform(self):
        """3x3 p-shell diagonal -> all elements equal to the mean."""
        diag_vals = np.array([3.0, 1.0, 2.0])
        mat = np.diag(diag_vals)
        result = adb.spherical_average(mat, [3])
        np.testing.assert_allclose(np.diag(result), [np.mean(diag_vals)] * 3)

    def test_mixed_s_and_p_only_p_changes(self):
        """Only the p-block changes; the s-block is left as-is."""
        # Layout: 1 s-function, then 3 p-functions
        mat = np.diag([5.0, 3.0, 1.0, 2.0])
        result = adb.spherical_average(mat, [1, 3])
        assert result[0, 0] == 5.0                                 # s unchanged
        np.testing.assert_allclose(
            np.diag(result)[1:], [np.mean([3.0, 1.0, 2.0])] * 3
        )

    def test_idempotent(self):
        """Applying twice gives the same result as applying once."""
        mat = np.diag([4.0, 2.0, 1.0, 3.0])
        ml = [1, 3]
        once  = adb.spherical_average(mat, ml)
        twice = adb.spherical_average(once, ml)
        np.testing.assert_allclose(twice, once, atol=1e-14)

    def test_does_not_mutate_input(self):
        """The input matrix must not be modified in-place."""
        mat = np.diag([1.0, 2.0, 3.0])
        original = mat.copy()
        adb.spherical_average(mat, [3])
        np.testing.assert_array_equal(mat, original)

    def test_uhf_both_spins_averaged(self):
        """3D (UHF) input: each spin component's p-diagonal is averaged."""
        alpha = np.diag([3.0, 1.0, 2.0])
        beta  = np.diag([6.0, 2.0, 4.0])
        result = adb.spherical_average(np.array([alpha, beta]), [3])
        np.testing.assert_allclose(np.diag(result[0]), [2.0] * 3, atol=1e-14)
        np.testing.assert_allclose(np.diag(result[1]), [4.0] * 3, atol=1e-14)


# ---------------------------------------------------------------------------
# adb.get_iteration_criteria_value
# ---------------------------------------------------------------------------

class TestIterationCriteria:

    def test_enocc_rhf_sums_occupied_eigenvalues(self):
        """enocc RHF: returns sum of the first nocc[0] eigenvalues."""
        epsilon = np.array([-1.5, -0.5, 0.3, 1.0])
        result = adb.get_iteration_criteria_value('enocc', epsilon_i=epsilon, nocc=(2, 2))
        np.testing.assert_allclose(result, -2.0)

    def test_enocc_uhf_sums_both_spins(self):
        """enocc UHF: returns sum of occupied alpha + occupied beta eigenvalues."""
        epsilon_a = np.array([-2.0, -0.5,  0.5])
        epsilon_b = np.array([-1.5, -0.3,  0.8])
        result = adb.get_iteration_criteria_value(
            'enocc',
            epsilon_i=np.array([epsilon_a, epsilon_b]),
            nocc=(2, 1),
        )
        # alpha: -2.0 + -0.5 = -2.5 ; beta: -1.5
        np.testing.assert_allclose(result, -4.0)

    def test_enocc_returns_float(self):
        """Return value is a Python float."""
        result = adb.get_iteration_criteria_value(
            'enocc', epsilon_i=np.array([-1.0, 0.5]), nocc=(1, 1))
        assert isinstance(result, float)

    def test_unknown_variant_raises_runtime_error(self):
        """Unrecognised variant string must raise RuntimeError."""
        with pytest.raises(RuntimeError):
            adb.get_iteration_criteria_value(
                'not_a_variant', epsilon_i=np.array([1.0]), nocc=(1, 1))

    def test_enocc_missing_epsilon_raises_value_error(self):
        """enocc without epsilon_i raises ValueError."""
        with pytest.raises(ValueError):
            adb.get_iteration_criteria_value('enocc', nocc=(1, 1))

    def test_enocc_missing_nocc_raises_value_error(self):
        """enocc without nocc raises ValueError."""
        with pytest.raises(ValueError):
            adb.get_iteration_criteria_value('enocc', epsilon_i=np.array([-1.0]))
