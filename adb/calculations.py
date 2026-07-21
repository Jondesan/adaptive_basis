import numpy as np
from scipy import linalg
from pyscf.scf.addons import canonical_orth_
from pyscf.gto import Mole
from pyscf.symm import symmetrize_matrix
from .molutil import create_subbasis_mol
from .CONSTANTS import VARIANTS, SYMMETRY_SHORTFALL_PENALTY


def eig(h: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve the generalized eigenproblem HC = SCE by canonical orthogonalization.

    Thin dispatch wrapper around `canonical_orth` that also handles the
    unrestricted (stacked alpha/beta) case.

    Parameters
    ----------
    h : ndarray, shape (nao, nao) or (2, nao, nao)
        Fock (or other Hermitian) matrix. A stacked ``(2, nao, nao)`` array
        is treated as unrestricted (alpha, beta); a plain 2D array as
        restricted.
    s : ndarray, shape (nao, nao)
        Overlap matrix, shared between spins.

    Returns
    -------
    e : ndarray, shape (nmo,) or (2, nmo)
        Eigenvalues, ascending within each spin channel.
    c : ndarray, shape (nao, nmo) or (2, nao, nmo)
        Eigenvectors, columns ordered to match `e`.
    """
    if np.asarray(h).ndim == 3:
        ea, ca = canonical_orth(h[0], s)
        eb, cb = canonical_orth(h[1], s)
        return np.asarray([ea, eb]), np.asarray([ca, cb])
    return canonical_orth(h, s)


def canonical_orth(h: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve a generalized eigenproblem by canonical orthogonalization.

    Robust to a near-singular (but not decorated with
    ``scf.addons.remove_linear_dep_``) overlap matrix `s`, which is common
    for the masked sub-Fock/sub-overlap matrices this is used on -- unlike
    a plain `scipy.linalg.eigh(h, s)` call, small overlap eigenvalues are
    dropped rather than amplified.

    Parameters
    ----------
    h : ndarray, shape (nao, nao)
        Fock (or other Hermitian) matrix.
    s : ndarray, shape (nao, nao)
        Overlap matrix.

    Returns
    -------
    e : ndarray, shape (nmo,)
        Eigenvalues, sorted ascending. ``nmo <= nao``: any near-singular
        directions of `s` are discarded by `canonical_orth_`.
    c : ndarray, shape (nao, nmo)
        Eigenvectors, columns ordered to match `e`.
    """
    x = canonical_orth_(s, 1e-8)
    xhx = x.conj().T @ h @ x
    e, c = linalg.eigh(xhx)
    c = x @ c
    idx = np.argsort(e)
    return e[idx], c[:, idx]


def symmetrized_eig(
        h:          np.ndarray,
        s:          np.ndarray,
        symm_orb:   list[np.ndarray],
        irrep_id:   list[int],
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Block-diagonalize (h, s) by irrep and solve each block with `canonical_orth`.

    Mirrors the structure of `pyscf.scf.hf_symm.eig` (symmetrize h/s into
    per-irrep blocks via `pyscf.symm.symmetrize_matrix`, one generalized
    eigenproblem per irrep) but solves each block with `canonical_orth`
    instead of plain `scipy.linalg.eigh`, since the (masked) sub-Fock
    matrices this is used on are frequently near-singular -- the same
    reason `eig` exists instead of calling `scipy` directly.

    Parameters
    ----------
    h : ndarray, shape (nao, nao) or (2, nao, nao)
        Fock/Hamiltonian matrix. 2D for restricted. For unrestricted, pass
        a stacked ``(2, nao, nao)`` array (alpha, beta) exactly like `eig`
        -- the irrep block structure is spin-independent (it only depends
        on `symm_orb`/`s`), so both spins share one `orbsym`.
    s : ndarray, shape (nao, nao)
        Overlap matrix, same AO dimension as `h` (spin-independent).
    symm_orb : list of ndarray
        Symmetry-adapted basis coefficients, one ``(nao, n_ir)`` array per
        irrep, e.g. ``mol.symm_orb`` / ``subbasis_mol.symm_orb``. Must be
        expressed in the same AO basis/ordering as `h` and `s`.
    irrep_id : list of int
        Irrep id for each entry of `symm_orb`, e.g. ``mol.irrep_id``.

    Returns
    -------
    e : ndarray, shape (nmo,) or (2, nmo)
        Eigenvalues, grouped by irrep block (not globally sorted -- matches
        pyscf's own symmetric-eig convention).
    c : ndarray, shape (nao, nmo) or (2, nao, nmo)
        Eigenvectors, back-transformed to the AO basis of `h`/`s`.
    orbsym : ndarray, shape (nmo,)
        Irrep id for each column of `e`/`c` along its last axis -- shared
        between spins for unrestricted input.
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
            osym.append(np.full(e_ir.size, irrep_id[ir]))
        return np.hstack(es), np.hstack(cs), np.hstack(osym)

    if np.asarray(h).ndim == 3:
        ea, ca, orbsym = _block_eig(h[0])
        eb, cb, _ = _block_eig(h[1])
        return np.asarray([ea, eb]), np.asarray([ca, cb]), orbsym
    return _block_eig(h)


# ------------------------------------------------------------------------
# Criterion calculations
# ------------------------------------------------------------------------


def get_iteration_criteria_value(
    variant:        str,
    epsilon_i:      np.ndarray | None  = None,
    nocc:           tuple      | None  = None,
    Csub:           np.ndarray | None  = None,
    Cfull:          np.ndarray | None  = None,
    ovlp:           np.ndarray | None  = None,
    irrep_nelec:    dict       | None  = None,
    orbsym:         np.ndarray | None  = None,
    ) -> float:
    """Evaluate the greedy-search criterion `find_subspace`/`expand_mask` minimize.

    Two variants are available (see `adb.CONSTANTS.VARIANTS`):

    - ``'enocc'``: :math:`\\sum_i^{nocc} \\epsilon_i`, the sum of the
      lowest `nocc` orbital energies `epsilon_i`. Uses `epsilon_i` and
      `nocc` (and, for the optional symmetry-aware search, `irrep_nelec`/
      `orbsym`).
    - ``'elden'``: the squared projection of the full-basis occupied
      orbitals onto the subbasis (see `get_q_sqrd`). Uses `Cfull`, `Csub`,
      `ovlp`, and `nocc`.

    Parameters
    ----------
    variant : {'enocc', 'elden'}
        Which criterion to evaluate.
    epsilon_i : ndarray, optional
        Orbital energies. Required for ``'enocc'``. Shape ``(nmo,)`` for
        restricted, ``(2, nmo)`` for unrestricted.
    nocc : tuple, optional
        ``(n_alpha, n_beta)`` occupied counts. Required for both variants.
    Csub : ndarray, optional
        Subbasis MO coefficients. Required for ``'elden'``.
    Cfull : ndarray, optional
        Full-basis MO coefficients. Required for ``'elden'``.
    ovlp : ndarray, optional
        Overlap matrix between the full and sub bases. Required for
        ``'elden'``.
    irrep_nelec : dict, optional
        ``'enocc'`` only. Target occupation per irrep name, in the same
        format as pyscf's ``mf.irrep_nelec`` (int for restricted,
        ``(n_alpha, n_beta)`` tuple for unrestricted). When given (together
        with `orbsym`), the criterion sums the lowest target-occupation
        eigenvalues *within each irrep* instead of the lowest N eigenvalues
        overall, penalizing irreps whose current orbital count falls short
        of their target (see `adb.CONSTANTS.SYMMETRY_SHORTFALL_PENALTY`).
        When `None` (default), the plain symmetry-blind sum is used.
    orbsym : ndarray, optional
        ``'enocc'`` only. Irrep name (string) for each entry along
        `epsilon_i`'s last axis, as returned by `symmetrized_eig` after
        translating irrep ids to names. Required whenever `irrep_nelec` is
        given.

    Returns
    -------
    float
        Value of the criterion.

    Examples
    --------
    >>> get_iteration_criteria_value('enocc', epsilon_i=e_sub, nocc=mol.nelec)
    >>> get_iteration_criteria_value('elden', Cfull=c_full, Csub=c_sub, ovlp=s, nocc=mol.nelec)
    """
    if variant not in VARIANTS:
        raise RuntimeError(
            'The variant you are trying to use was not recognised!')
    match variant:
        case 'enocc':
            if epsilon_i is None or nocc is None:
                raise ValueError("Energies 'epsilon_i' or occupations 'nocc' not provided.")
            restricted = (np.asarray(epsilon_i).ndim == 1)
            if irrep_nelec is not None:
                if orbsym is None:
                    raise ValueError(
                        "'orbsym' must be provided together with 'irrep_nelec'.")
                criteria = _enocc_by_irrep(epsilon_i, orbsym, irrep_nelec, restricted)
            elif restricted:
                criteria = 2 * np.sum(epsilon_i[:nocc[0]])
            else:
                criteria  = np.sum(epsilon_i[0, :nocc[0]])
                criteria += np.sum(epsilon_i[1, :nocc[1]])
            return float(np.real(criteria))
        case 'elden':
            return get_q_sqrd(Cfull, Csub, ovlp, nocc)


def _enocc_by_irrep(
        epsilon_i:      np.ndarray,
        orbsym:         np.ndarray,
        irrep_nelec:    dict,
        restricted:     bool,
        ) -> float:
    """Irrep-resolved 'enocc' criterion for the optional symmetry-aware search.

    Sums the lowest target-occupation orbitals within each irrep (per
    pyscf's ``mf.irrep_nelec`` convention) instead of the lowest N
    eigenvalues overall. If the current (partially-grown) subbasis doesn't
    yet have enough orbitals of some irrep to hold its target occupation,
    each missing slot adds `adb.CONSTANTS.SYMMETRY_SHORTFALL_PENALTY` --
    this makes the greedy search prioritize adding shells of an
    under-represented irrep first, and degrades smoothly to the ordinary
    energy sum once every irrep has enough capacity.

    Each restricted (RHF-like) spatial orbital holds 2 electrons, so its
    energy and shortfall-penalty contributions are weighted by 2 -- the
    same convention as the plain (non-symmetry-aware) restricted 'enocc'
    branch in `get_iteration_criteria_value`. Unrestricted spin channels
    are weighted by 1, one electron per occupied spin-orbital.

    Parameters
    ----------
    epsilon_i : ndarray, shape (nmo,) or (2, nmo)
        Orbital energies.
    orbsym : ndarray, shape (nmo,)
        Irrep name for each entry of `epsilon_i`'s last axis.
    irrep_nelec : dict
        Target occupation per irrep name (pyscf ``mf.irrep_nelec`` format).
    restricted : bool
        Whether `epsilon_i` is restricted-shaped.

    Returns
    -------
    float
        The irrep-resolved criterion value.
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
            e_ir = np.sort(np.real(e_ir))
            n_avail = e_ir.size
            n_take = min(n_avail, n_need)
            if n_take > 0:
                total += weight * np.sum(e_ir[:n_take])
            if n_avail < n_need:
                total += weight * SYMMETRY_SHORTFALL_PENALTY * (n_need - n_avail)
    return total


def get_q_sqrd(
        Cfull:  np.ndarray,
        Csub:   np.ndarray,
        ovlp:   np.ndarray,
        nocc:   tuple,
        ) -> float:
    """Squared projection of the full-basis occupied orbitals onto the subbasis.

    :math:`Q^2 = \\sum_{ij}^{nocc} \\lvert \\langle i^{full} \\vert
    j^{sub} \\rangle \\rvert^2`, the ``'elden'`` search criterion.

    Parameters
    ----------
    Cfull : ndarray, shape (nao, nmo) or (2, nao, nmo)
        Full-basis MO coefficients.
    Csub : ndarray, shape (nao, nmo) or (2, nao, nmo)
        Subbasis MO coefficients, same AO dimension as `Cfull`.
    ovlp : ndarray, shape (nao, nao)
        Overlap matrix between the full and sub bases.
    nocc : tuple
        ``(n_alpha, n_beta)`` occupied counts.

    Returns
    -------
    float
        :math:`Q^2`, doubled for restricted input (2 electrons/orbital).
    """
    restricted = (Cfull.ndim == 2)
    if restricted:
        Q = Cfull[:, :nocc[0]].T @ ovlp @ Csub[:, :nocc[0]]
        return 2.0 * np.real(np.sum(Q**2))
    Q = [
        Cfull[0, :, :nocc[0]].T @ ovlp @ Csub[0, :, :nocc[0]],
        Cfull[1, :, :nocc[1]].T @ ovlp @ Csub[1, :, :nocc[1]],
    ]
    return np.real(np.sum(Q[0]**2) + np.sum(Q[1]**2))


def diagonalize_masked(
        maskedF:    np.ndarray,
        maskedS:    np.ndarray,
        mol:        Mole | None       = None,
        smask:      np.ndarray | None = None,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Diagonalize an already-masked (Fock, overlap) pair.

    Plain `eig` (symmetry-blind) when `mol` is `None`. When `mol` is given
    (the shell-separated mol whose shells `smask` indexes into -- see
    `expand_mask`'s docstring), builds a fresh subbasis `Mole` via
    `create_subbasis_mol(mol, smask)` and diagonalizes block-by-irrep with
    `symmetrized_eig` instead, returning a name-tagged `orbsym` array.

    Shared by `expand_mask`'s symmetry-aware branch, `find_subspace`'s
    symmetry-aware pre-loop baseline, and `find_subspace`'s optional
    ``track_orbitals`` bookkeeping -- pulled out to a top-level function so
    those three call sites diagonalize a masked subbasis identically
    instead of maintaining three copies of the same branch.

    Parameters
    ----------
    maskedF : ndarray
        Masked Fock matrix.
    maskedS : ndarray
        Masked overlap matrix.
    mol : Mole, optional
        Shell-separated mol whose shells `smask` indexes into. When given,
        diagonalization is done block-by-irrep instead of symmetry-blind.
    smask : ndarray, optional
        Shell mask indexing into `mol`'s shells. Required when `mol` is
        given.

    Returns
    -------
    evals : ndarray
        Eigenvalues.
    coeffs : ndarray
        Eigenvectors.
    orbsym : ndarray or None
        Irrep name per eigenvector, or `None` when `mol` is `None`.
    """
    if mol is None:
        evals, coeffs = eig(maskedF, maskedS)
        return evals, coeffs, None
    sub_mol = create_subbasis_mol(mol, smask)
    evals, coeffs, orbsym_id = symmetrized_eig(
        maskedF, maskedS, sub_mol.symm_orb, sub_mol.irrep_id)
    id_to_name = dict(zip(sub_mol.irrep_id, sub_mol.irrep_name))
    orbsym = np.asarray([id_to_name[i] for i in orbsym_id])
    return evals, coeffs, orbsym


def spherical_average(mat: np.ndarray, ml: np.ndarray) -> np.ndarray:
    """Spherically average a (Fock) matrix's shell-diagonal blocks.

    Dispatches to `sph_avg` per spin channel for unrestricted input.

    Parameters
    ----------
    mat : ndarray, shape (nao, nao) or (2, nao, nao)
        The (Fock) matrix to spherically average.
    ml : array_like
        Number of functions per shell, in AO order.

    Returns
    -------
    ndarray
        The spherically-averaged matrix, same shape as `mat`.
    """
    mat_copy = mat.copy()
    restricted = (mat_copy.ndim == 2)
    if restricted:
        return sph_avg(mat_copy, ml)
    mat_out = np.ndarray(mat_copy.shape)
    mat_out[0] = sph_avg(mat_copy[0], ml)
    mat_out[1] = sph_avg(mat_copy[1], ml)
    return mat_out


def sph_avg(mat: np.ndarray, ml: np.ndarray) -> np.ndarray:
    """Spherically average one matrix's shell-diagonal blocks.

    For each shell (other than S, which is already spherically symmetric),
    replaces its diagonal block with a diagonal matrix holding the mean of
    that block's own diagonal -- i.e. zeroes out in-shell mixing and
    equalizes the diagonal, removing any spurious angular anisotropy a
    non-spherical (e.g. atom-in-molecule) environment would otherwise
    introduce.

    Parameters
    ----------
    mat : ndarray, shape (nao, nao)
        The (Fock) matrix to spherically average.
    ml : array_like
        Number of functions per shell, in AO order; must sum to `nao`.

    Returns
    -------
    ndarray, shape (nao, nao)
        The spherically-averaged matrix.
    """
    mat_copy = mat.copy()
    offset = 0
    for nfunc in ml:
        if nfunc == 1:
            offset += nfunc
            continue
        shell_mat = mat_copy[offset:offset + nfunc, offset:offset + nfunc]
        diag = np.diag(shell_mat)
        avg = np.mean(diag)
        shell_mat = np.diag([avg] * diag.shape[0])

        for i in range(nfunc):
            mat_copy[offset + i, offset:offset + nfunc] = shell_mat[i, :]

        offset += nfunc
    return mat_copy


def dual_basis_energy_correction(
        scf_obj,
        P_full_projected: np.ndarray,
        ) -> tuple[float, float]:
    """Dual-basis energy correction of Liang, Steele & Head-Gordon.

    Evaluates the large-basis Roothaan-step energy lowering obtained by
    projecting a converged subbasis density into the full basis and taking
    one Fock-diagonalization step there, without fully reconverging the
    full-basis SCF.

    Parameters
    ----------
    scf_obj : pyscf.scf.hf.SCF
        Full-basis SCF object (e.g. ``fullbasis_mol.HF()``), used to build
        the full-basis Fock matrix and take the trial step.
    P_full_projected : ndarray
        Subbasis density matrix projected into the full basis.

    Returns
    -------
    dE : float
        The Roothaan-step energy lowering, :math:`\\mathrm{Tr}[(P_{new} -
        P_{full,proj}) \\cdot F_{full}]`.
    e_tot : float
        `scf_obj.e_tot` at the point this was called (the reference total
        energy the correction is measured relative to).
    """
    F_full = scf_obj.get_fock(dm=P_full_projected)
    E_new, C_new = scf_obj.eig(F_full, scf_obj.get_ovlp())
    P_new = scf_obj.make_rdm1(
        mo_coeff=C_new,
        mo_occ=scf_obj.get_occ(mo_energy=E_new, mo_coeff=C_new))
    dP = P_new - P_full_projected
    if dP.ndim > 2:  # unrestricted
        dE = np.trace(dP[0] @ F_full[0]) + np.trace(dP[1] @ F_full[1])
    else:
        dE = np.trace(dP @ F_full)
    return np.real(dE), scf_obj.e_tot
