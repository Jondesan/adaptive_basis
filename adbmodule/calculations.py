import numpy
from scipy import linalg
from pyscf.scf.addons import canonical_orth_
from pyscf.symm import symmetrize_matrix
from CONSTANTS import VARIANTS, SYMMETRY_SHORTFALL_PENALTY

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
    

# ------------------------------------------------------------------------
# Criterion calculations
# ------------------------------------------------------------------------


def get_iteration_criteria_value(
    variant:    str,
    epsilon_i:  numpy.ndarray | None   = None,
    nocc:       tuple         | None   = None,
    Csub:       numpy.ndarray | None   = None,
    Cfull:      numpy.ndarray | None   = None,
    ovlp:       numpy.ndarray | None   = None,
    irrep_nelec: dict         | None   = None,
    orbsym:     numpy.ndarray | None   = None,
    ) -> float:
    """Calculates the value of the chosen variants criteria.

    Args:
        variant : string
            Which variant to calculate.
        The needed variables for calculating the different criteria.
        
        enocc:
        epislon_i : ndarray
            energy eigenvalues
        nocc : tuple
            number of occupations
        irrep_nelec : dict | None
            Optional, 'enocc' only. Target occupation per irrep name, in
            the same format as pyscf's mf.irrep_nelec (int for restricted,
            (n_alpha, n_beta) tuple for unrestricted). When given (together
            with `orbsym`), the criterion sums the lowest target-occupation
            eigenvalues *within each irrep* instead of the lowest N
            eigenvalues overall, penalising irreps whose current orbital
            count falls short of their target (see
            SYMMETRY_SHORTFALL_PENALTY). When None (default), behaviour is
            unchanged from before this option existed.
        orbsym : ndarray | None
            Optional, 'enocc' only. Irrep name (string) for each entry
            along epsilon_i's last axis, as returned by symmetrized_eig
            after translating irrep ids to names. Required whenever
            `irrep_nelec` is given.
        
        elden:
        Cfull : ndarray
            full basis coeff matrix
        Csub : ndarray
            subbasis coeff matrix
        ovlp : 2D array
            overlap matrix
        nocc : tuple
            number of occupations
        
    Returns:
        criteria : float
            Value of the criteria
    """
    criteria = 0.0
    if variant not in VARIANTS:
        raise RuntimeError(
            'The variant you are trying to use was not recognised!')
    match variant:
        case 'enocc':
            if epsilon_i is None or nocc is None:
                raise ValueError("Energies 'epsilon_i' or occupations 'nocc' not provided.")
            restricted = (len(numpy.asarray(epsilon_i).shape) == 1)
            if irrep_nelec is not None:
                if orbsym is None:
                    raise ValueError(
                        "'orbsym' must be provided together with 'irrep_nelec'.")
                criteria = _enocc_by_irrep(epsilon_i, orbsym, irrep_nelec, restricted)
            elif restricted:
                criteria = 2 * numpy.sum(epsilon_i[:nocc[0]])
            else:
                criteria  = numpy.sum(epsilon_i[0,:nocc[0]])
                criteria += numpy.sum(epsilon_i[1,:nocc[1]])
            return float(numpy.real(criteria))
        case 'elden':
            return get_q_sqrd(Cfull, Csub, ovlp, nocc)


def _enocc_by_irrep(
        epsilon_i:      numpy.ndarray,
        orbsym:         numpy.ndarray,
        irrep_nelec:    dict,
        restricted:     bool,
        ) -> float:
    """Irrep-resolved 'enocc' criterion used by the optional symmetry-aware
    search: sum the lowest target-occupation orbitals within each irrep
    (per pyscf's mf.irrep_nelec convention) instead of the lowest N
    eigenvalues overall. If the current (partially-grown) subbasis doesn't
    yet have enough orbitals of some irrep to hold its target occupation,
    each missing slot adds SYMMETRY_SHORTFALL_PENALTY -- this makes the
    greedy search prioritise adding shells of an under-represented irrep
    first, and degrades smoothly to the ordinary energy sum once every
    irrep has enough capacity.

    Each restricted (RHF-like) spatial orbital holds 2 electrons, so its
    energy and shortfall-penalty contributions are weighted by 2 -- the
    same convention as the plain (non-symmetry-aware) restricted 'enocc'
    branch above (`2 * numpy.sum(epsilon_i[:nocc[0]])`). Unrestricted spin
    channels are weighted by 1, one electron per occupied spin-orbital.
    """
    total = 0.0
    for irname, target in irrep_nelec.items():
        spin_targets = [(None, target // 2)] if restricted else \
            [(0, target[0]), (1, target[1])]
        for spin, n_need in spin_targets:
            if n_need == 0:
                continue
            weight = 2 if spin is None else 1
            e_ir = epsilon_i[orbsym == irname] if spin is None \
                else epsilon_i[spin][orbsym == irname]
            e_ir = numpy.sort(numpy.real(e_ir))
            n_avail = e_ir.size
            n_take = min(n_avail, n_need)
            if n_take > 0:
                total += weight * numpy.sum(e_ir[:n_take])
            if n_avail < n_need:
                total += weight * SYMMETRY_SHORTFALL_PENALTY * (n_need - n_avail)
    return total


def get_q_sqrd(
        Cfull: numpy.ndarray,
        Csub:  numpy.ndarray,
        ovlp:  numpy.ndarray,
        nocc:  numpy.ndarray  ) -> float:
    """Calculates the square of the projection Q"""
    restricted = (len(Cfull.shape) == 2)
    if restricted:
        Q = Cfull[:, :nocc[0]].T @ ovlp @ Csub[:, :nocc[0]]
        return 2.0 * numpy.real(numpy.sum(numpy.sum(Q**2)))
    else:
        Q = [
            Cfull[0, :, :nocc[0]].T @ ovlp @ Csub[0, :, :nocc[0]],
            Cfull[1, :, :nocc[1]].T @ ovlp @ Csub[1, :, :nocc[1]]
        ]
        return numpy.real((numpy.sum(numpy.sum(Q[0]**2)) + numpy.sum(numpy.sum(Q[1]**2))))