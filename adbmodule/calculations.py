import numpy
from scipy import linalg
from pyscf.scf.addons import canonical_orth_
from pyscf.symm import symmetrize_matrix

def eig(
        h: numpy.ndarray,
        s: numpy.ndarray
        ) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Wrapper for eigh, calculates orthogonalisation for RHF and UHF.
    """
    if len(numpy.asarray(h).shape) == 3:
        ea, ca = canonical_orth(h[0], s)
        eb, cb = canonical_orth(h[1], s)
        return numpy.asarray([ea, eb]), numpy.asarray([ca, cb])
    else:
        return canonical_orth(h, s)


def canonical_orth(
        h: numpy.ndarray | tuple[numpy.ndarray, numpy.ndarray],
        s: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Modified canonical orthogonalisation.

    Args:
        h : ndarray
            Fock matrix
        s : ndarray
            Overlap matrix
        get_idx : bool
            Whether to return the indices that sort the eigenvalues.
            Default is False

    Returns:
        Sorted eigenvalues (ascending) and coefficients, if get_idx is
        True the indices that sort the eigenvalues are also returned
    """
    x = canonical_orth_(s, 1e-8)
    xhx = x.conj().T @ h @ x
    e, c = linalg.eig(xhx)
    c = x @ c
    idx = numpy.argsort(e)
    return e[idx], c[:,idx]


def symmetrized_eig(
        h:          numpy.ndarray,
        s:          numpy.ndarray,
        symm_orb:   list,
        irrep_id:   list,
        ) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Block-diagonalize (h, s) by irrep and solve each block with
    canonical_orth.

    This mirrors the structure of pyscf.scf.hf_symm.eig (symmetrize h/s
    into per-irrep blocks via pyscf.symm.symmetrize_matrix, one
    generalized eigenproblem per irrep) but solves each block with adb's
    own canonical_orth instead of plain scipy.linalg.eigh, since the
    (masked) sub-Fock matrices this is used on are frequently
    near-singular -- the same reason adb.eig exists instead of calling
    scipy directly.

    Args:
        h : ndarray
            Fock/Hamiltonian matrix. 2D for RHF. For UHF, pass a stacked
            (2, nao, nao) array (alpha, beta) exactly like adb.eig -- the
            irrep block structure is spin-independent (it only depends on
            symm_orb/s), so both spins share one `orbsym`.
        s : ndarray
            Overlap matrix, same AO dimension as h (spin-independent).
        symm_orb : list of ndarray
            Symmetry-adapted basis coefficients, one (nao, n_ir) array per
            irrep, e.g. mol.symm_orb / subbasis_mol.symm_orb. Must be
            expressed in the same AO basis/ordering as h and s.
        irrep_id : list of int
            Irrep id for each entry of symm_orb, e.g. mol.irrep_id.

    Returns:
        e : ndarray
            Eigenvalues, grouped by irrep block (not globally sorted --
            matches pyscf's own symmetric-eig convention). Shape (2, nmo)
            for UHF input, (nmo,) for RHF input.
        c : ndarray
            Eigenvectors, back-transformed to the AO basis of h/s. Shape
            (2, nao, nmo) for UHF input, (nao, nmo) for RHF input.
        orbsym : ndarray
            Irrep id for each column of e/c along its last axis, shape
            (nmo,) -- shared between spins for UHF input.
    """
    def _block_eig(h2d):
        hs = symmetrize_matrix(h2d, symm_orb)
        ss = symmetrize_matrix(s, symm_orb)
        es, cs, osym = [], [], []
        for ir in range(len(symm_orb)):
            if symm_orb[ir].shape[1] == 0:
                continue
            e_ir, c_ir = canonical_orth(hs[ir], ss[ir])
            es.append(e_ir)
            cs.append(symm_orb[ir] @ c_ir)
            osym.append(numpy.full(e_ir.size, irrep_id[ir]))
        return numpy.hstack(es), numpy.hstack(cs), numpy.hstack(osym)

    if len(numpy.asarray(h).shape) == 3:
        ea, ca, orbsym = _block_eig(h[0])
        eb, cb, _ = _block_eig(h[1])
        return numpy.asarray([ea, eb]), numpy.asarray([ca, cb]), orbsym
    else:
        return _block_eig(h)