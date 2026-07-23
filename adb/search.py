"""The core ADB greedy search algorithm: find_subspace and its per-step worker expand_mask."""

import copy
from operator import itemgetter

import numpy as np
from pyscf import gto, scf

from .calculations import (
    diagonalize_masked,
    eig,
    get_iteration_criteria_value
)
from .CONSTANTS import EXPAND_MASK_EPS, NFUNCS
from .initialization import (
    atomic_block_minimal_basis,
    find_projected_minimal_basis_mask
)
from .ioutil import (
    print_find_subspace_start,
    print_projection_initialization_message,
    warn_conflicting_initialization,
)
from .maskutil import (
    init_smask,
    linked_shell_idx,
    mask_matrix,
    mask_to_smask,
    set_linked_shells,
    smask_to_mask,
)
from .molutil import create_shell_separated_mol
from .orbitalutil import get_occupied_orbitals


def _validate_symmetry_aware_args(
        mol:            gto.Mole,
        symmetry_aware: bool,
        irrep_nelec:    dict | None,
        get_smask:      bool,
        link_shells:    bool,
        ) -> None:
    """Check `find_subspace(symmetry_aware=True)`'s upfront preconditions.

    See that argument's docstring for why each is required.

    Raises
    ------
    RuntimeError
        If `symmetry_aware` is `True` and any precondition fails.
    """
    if not symmetry_aware:
        return
    if isinstance(mol.symmetry, bool) and not mol.symmetry and mol.groupname != 'C1':
        raise RuntimeError(
            "find_subspace(symmetry_aware=True) requires mol.symmetry "
            "to be enabled with a non-C1 point group.")
    if mol.groupname != 'C1' and irrep_nelec is None:
        raise RuntimeError(
            "find_subspace(symmetry_aware=True) requires irrep_nelec "
            "(the target per-irrep occupation) to be provided.")
    if not get_smask:
        raise RuntimeError(
            "find_subspace(symmetry_aware=True) requires get_smask=True: "
            "a per-function mask cannot be guaranteed symmetry-closed.")
    if not link_shells:
        raise RuntimeError(
            "find_subspace(symmetry_aware=True) requires link_shells=True: "
            "a mask that toggles individual symmetry-equivalent atoms' "
            "shells independently is not guaranteed to stay "
            "symmetry-closed, which block-by-irrep diagonalization "
            "depends on.")


def _initialize_mask(
        mol:                        gto.Mole,
        F:                          np.ndarray,
        S:                          np.ndarray,
        fullbasis_mol:              gto.Mole,
        verbose:                    bool,
        abd_initialization:         bool,
        initialize_by_projection:   bool,
        spherical_average:          bool,
        abd_Q_tol:                  float,
        ) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to one of `find_subspace`'s three initial-mask strategies.

    In priority order: atomic block decomposition, STO-3G overlap
    projection, or (fallback) the single AO with the smallest diagonal
    Fock element.

    Returns
    -------
    mask : ndarray
        The initial AO mask.
    mask_init_idx : ndarray
        Indices selected by the initial mask.
    """
    if abd_initialization and initialize_by_projection:
        warn_conflicting_initialization()

    is_restricted = F.ndim == 2
    if is_restricted:
        Fii = np.diag(F)
    else:
        Fii = .5 * np.sum(np.diagonal(F, axis1=1, axis2=2), axis=0)

    if abd_initialization:
        mask = atomic_block_minimal_basis(
            mol,
            F,
            S,
            Q_tol=abd_Q_tol,
            spherically_average_fock=spherical_average,
            verbose=verbose)
        mask_init_idx = np.where(mask)[0]
    elif initialize_by_projection:
        if verbose:
            print_projection_initialization_message()
        mask = find_projected_minimal_basis_mask(mol)
        mask_init_idx = np.where(mask)[0]
    else:
        mask_init_idx = [np.argmin(Fii)]
        mask = [False] * fullbasis_mol.nao_nr()
        mask[mask_init_idx[0]] = True

    return mask, mask_init_idx


def _orbital_history_entry(
        mask:           np.ndarray,
        e:              np.ndarray,
        orbsym:         np.ndarray | None,
        nocc:           tuple,
        irrep_nelec:    dict | None,
        symmetry_aware: bool,
        is_restricted:  bool,
        ) -> dict:
    """Build one `find_subspace(track_orbitals=True)` history entry."""
    return {
        'nfunc': int(np.sum(mask)),
        'orbitals': get_occupied_orbitals(
            e, nocc, irrep_nelec if symmetry_aware else None, orbsym,
            restricted=is_restricted),
    }


def find_subspace(
        F:                          np.ndarray,
        S:                          np.ndarray,
        mol:                        gto.Mole,
        scf_obj:                    scf.hf.SCF | scf.hf.RHF | scf.uhf.UHF | scf.rohf.ROHF | scf.ghf.GHF,
        conv_tol:                   float           = 1e-2,
        verbose:                    bool            = True,
        get_smask:                  bool            = False,
        variant:                    str             = 'enocc',
        link_shells:                bool            = True,
        nfunc_normalisation:        bool            = True,
        return_mask_history:        bool            = False,
        abd_initialization:         bool            = False,
        initialize_by_projection:   bool            = True,
        spherical_average:          bool            = False,
        abd_Q_tol:                  float           = .5,
        symmetry_aware:             bool            = False,
        irrep_nelec:                dict | None     = None,
        track_orbitals:             bool            = False,
        ) -> np.ndarray | list | tuple:
    r"""Greedily grow an AO subspace that approximately solves FC = SCE.

    Starting from a minimal-basis seed, repeatedly adds the single
    function or shell (see `get_smask`) that most improves `variant`'s
    criterion, stopping once the improvement per step falls below
    `conv_tol` or every function has been added.

    Parameters
    ----------
    F : ndarray
        The full Fock matrix that will be sampled.
    S : ndarray
        The overlap matrix.
    mol : pyscf.gto.Mole
        The molecule object.
    scf_obj : pyscf.scf.hf.SCF
        The SCF object corresponding to `mol`.
    conv_tol : float, default 1e-2
        Convergence criterion: the search stops once a step's criterion
        improvement, normalized by the number of functions it added (see
        `nfunc_normalisation`), falls below this.
    verbose : bool, default True
        Whether to print progress during the search.
    get_smask : bool, default False
        Whether to grow the subspace shell-by-shell (returning a shell
        mask) instead of function-by-function. Shell-by-shell may converge
        faster but can also add more functions overall.
    variant : {'enocc', 'elden'}, default 'enocc'
        Which criterion to minimize/maximize at each step -- see
        `adb.calculations.get_iteration_criteria_value`:

        - ``'enocc'``: :math:`\sum_i^{nocc} \epsilon_i`, the sum of the
          occupied diagonal Fock matrix elements.
        - ``'elden'``: :math:`\Delta Q`, the squared projection of the
          full-basis occupied orbitals onto the subbasis.
    link_shells : bool, default True
        Whether to toggle shells of symmetry-equivalent atoms together.
    nfunc_normalisation : bool, default True
        Whether to normalize each step's criterion improvement by the
        number of functions it added.
    return_mask_history : bool, default False
        Whether to return the mask/smask at every accepted step instead of
        only the final one.
    abd_initialization : bool, default False
        Whether to seed the search with an atomic block decomposition
        minimal basis (see `adb.atomic_block_minimal_basis`) instead of
        `initialize_by_projection`'s STO-3G projection.
    initialize_by_projection : bool, default True
        Whether to seed the search with an STO-3G overlap projection (see
        `adb.find_projected_minimal_basis_mask`). Ignored if
        `abd_initialization` is `True` (a warning is raised if both are
        requested).
    spherical_average : bool, default False
        Whether atomic block decomposition spherically averages the Fock
        matrix. Only used when `abd_initialization` is `True`.
    abd_Q_tol : float, default 0.5
        Atomic block decomposition's charge tolerance -- how much of each
        atom's charge the minimal basis is allowed to not account for.
        Only used when `abd_initialization` is `True`.
    symmetry_aware : bool, default False
        Optional feature, off by default. When `True`, the search
        diagonalizes trial Fock matrices block-by-irrep and targets
        `irrep_nelec`'s per-irrep occupation instead of the lowest N
        eigenvalues overall (see `expand_mask`/
        `adb.calculations.symmetrized_eig`). Requires `mol.symmetry`
        truthy, ``mol.groupname != 'C1'``, `irrep_nelec` given,
        `get_smask=True` and `link_shells=True` -- a partial, unlinked mask
        is not guaranteed to stay symmetry-closed, which block-by-irrep
        diagonalization depends on. Default `False`: behaviour is
        byte-identical to before this option existed.
    irrep_nelec : dict, optional
        Target occupation per irrep name (pyscf ``mf.irrep_nelec`` format,
        e.g. from ``scf_obj.get_irrep_nelec()`` on the converged reference
        full-basis SCF). Required when `symmetry_aware=True`.
    track_orbitals : bool, default False
        Optional feature, off by default. When `True`, records the
        occupied orbital energies and their symmetry labels (via
        `adb.get_occupied_orbitals`) at every accepted ADB cycle -- the
        initial mask and every subsequent growth step, mirroring
        `return_mask_history`'s per-step bookkeeping. Symmetry labels are
        only meaningful (non-`None`) when `symmetry_aware=True`; with it
        `False` they're still recorded, just with ``irrep=None``
        throughout. When `True`, the return value becomes a 2-tuple
        ``(result, orbital_history)`` instead of just `result` -- existing
        call sites that assign the return value to a single variable are
        unaffected as long as they leave this at its default.
        `orbital_history` is a list of ``{'nfunc': int, 'orbitals':
        [(energy, irrep_label), ...]}`` dicts, one per recorded cycle,
        saveable via `adb.write_orbital_history`.

    Returns
    -------
    ndarray or list
        A 1D boolean mask with selected function indices set to `True`
        (shell mask instead of function mask if `get_smask` is `True`), or
        (if `return_mask_history=True`) a list of ``(mask, criterion,
        difference)`` history entries. If `track_orbitals=True`, a
        ``(result, orbital_history)`` tuple is returned instead of just
        `result` -- see the `track_orbitals` parameter.
    """
    _validate_symmetry_aware_args(mol, symmetry_aware, irrep_nelec, get_smask, link_shells)

    if verbose:
        print_find_subspace_start(mol)
    scf_obj_copy = scf_obj.copy()
    fullbasis_mol = create_shell_separated_mol(mol)

    # mask or smask initialization
    is_restricted = F.ndim == 2
    mask, _ = _initialize_mask(
        mol, F, S, fullbasis_mol, verbose,
        abd_initialization, initialize_by_projection,
        spherical_average, abd_Q_tol)
    smask = None
    nocc = mol.nelec

    if get_smask:
        smask = init_smask(fullbasis_mol, fullbasis_mol.cart)
        smask = mask_to_smask(mask, smask, fullbasis_mol.cart)
        # If link_shells true, set same shells of same atoms to True
        smask = set_linked_shells(smask)

        mask = smask_to_mask(smask, fullbasis_mol.cart)

    Cfull = None
    if variant == 'elden':
        _, Cfull = eig(F, S)

    e_sub, Csub, orbsym = diagonalize_masked(
        mask_matrix(F, mask), mask_matrix(S, mask),
        fullbasis_mol if symmetry_aware else None, smask)

    previous_sum = get_iteration_criteria_value(
        variant, epsilon_i=e_sub, nocc=nocc,
        Csub=Csub, Cfull=Cfull, ovlp=S[:, mask],
        irrep_nelec=irrep_nelec if symmetry_aware else None, orbsym=orbsym)

    orbital_history = None
    if track_orbitals:
        orbital_history = [_orbital_history_entry(
            mask, e_sub, orbsym, nocc, irrep_nelec, symmetry_aware, is_restricted)]

    basis_initialized = False
    if return_mask_history:
        mask_history = []
        if abd_initialization:
            dual_basis_initialization = 'Atomic Block Decomposition'
        elif initialize_by_projection:
            dual_basis_initialization = 'Minimal basis projection'
        else:
            dual_basis_initialization = 'Max element of Fock matrix'
        mask_history.append((
            copy.deepcopy(smask) if get_smask else copy.deepcopy(mask),
            previous_sum,
            0.0,
            dual_basis_initialization))
        basis_initialized = True

    while not np.all(mask):
        mask, difference, current_criteria_val, n_added, smask = expand_mask(
            F, S, nocc, mask,
            Cfull=scf_obj_copy.mo_coeff,
            smask=smask, variant=variant, link_shells=link_shells,
            nfunc_normalisation=nfunc_normalisation,
            mol=fullbasis_mol if symmetry_aware else None,
            irrep_nelec=irrep_nelec if symmetry_aware else None,
        )

        if n_added == 0:
            # expand_mask found no remaining candidate that genuinely
            # improves the criterion (see its docstring/EXPAND_MASK_EPS) --
            # the mask above is unchanged, and calling again would return
            # the identical no-op forever. Stop instead of looping.
            break

        if track_orbitals:
            # Record every accepted cycle unconditionally (unlike
            # return_mask_history below, which skips both the
            # not-yet-basis_initialized bootstrap steps and the final,
            # conv_tol-satisfying step) -- track_orbitals is meant to give
            # a complete per-cycle record.
            e_step, _, orbsym_step = diagonalize_masked(
                mask_matrix(F, mask),
                mask_matrix(S, mask),
                fullbasis_mol if symmetry_aware else None, smask)
            orbital_history.append(_orbital_history_entry(
                mask, e_step, orbsym_step, nocc, irrep_nelec, symmetry_aware, is_restricted))

        # If the dual basis does not span the minimal basis, skip collecting
        # the mask, resulting orbitals would not be even qualitatively correct
        if not basis_initialized:
            basis_initialized = np.sum(mask) >= np.max(nocc)
            continue

        if abs(n_added * difference) < conv_tol or sum(mask) == len(mask):
            break

        if return_mask_history:
            mask_history.append(
                (copy.deepcopy(smask) if get_smask else copy.deepcopy(mask),
                 current_criteria_val,
                 difference))

        previous_sum = current_criteria_val

    if get_smask:
        mask = smask

    if return_mask_history:
        mask = mask_history

    if track_orbitals:
        return mask, orbital_history

    return mask


def expand_mask(
        F:                      np.ndarray,
        S:                      np.ndarray,
        nocc:                   tuple,
        mask:                   np.ndarray,
        smask:                  np.ndarray | None   = None,
        variant:                str                 = 'enocc',
        Cfull:                  np.ndarray | None   = None,
        link_shells:            bool                = True,
        nfunc_normalisation:    bool                = True,
        mol:                    gto.Mole | None     = None,
        irrep_nelec:            dict | None         = None,
        ) -> tuple[np.ndarray, float, float, int, np.ndarray | None]:
    r"""Expand `mask` by the single function or shell that most improves `variant`.

    Tests every remaining (not-yet-selected) function or shell in turn,
    diagonalizing the resulting trial masked (Fock, overlap) pair and
    evaluating `variant`'s criterion, then commits whichever trial improves
    the criterion the most (normalized by functions added, if
    `nfunc_normalisation`).

    Parameters
    ----------
    F : ndarray
        Full Fock matrix.
    S : ndarray
        Full overlap matrix.
    nocc : tuple
        ``(n_alpha, n_beta)`` occupied counts.
    mask : ndarray
        The current AO mask.
    smask : ndarray, optional
        If `None`, functions are tested individually. Otherwise shell-by-
        shell testing is used, with shells determined by `smask` (see
        `adb.maskutil.init_smask`).
    variant : {'enocc', 'elden'}, default 'enocc'
        Which criterion to evaluate for each trial -- see
        `adb.calculations.get_iteration_criteria_value`:

        - ``'enocc'``: :math:`\sum_i^{nocc} \epsilon_i`, the sum of the
          occupied diagonal Fock matrix elements.
        - ``'elden'``: :math:`\Delta Q`, the squared projection of the
          full-basis occupied orbitals onto the subbasis.
    Cfull : ndarray, optional
        Full-basis MO coefficients, required for `variant='elden'`
        (computed internally if not given).
    link_shells : bool, default True
        Whether to test shells of symmetry-equivalent atoms as one
        combined candidate rather than independently. Only used in shell
        mode (`smask` given).
    nfunc_normalisation : bool, default True
        Whether to normalize each trial's criterion improvement by the
        number of functions it would add.
    mol : pyscf.gto.Mole, optional
        When given together with `irrep_nelec`, every trial (and the
        current) masked Fock/overlap matrix is diagonalized block-by-irrep
        (`adb.calculations.symmetrized_eig`) instead of with the plain,
        symmetry-blind `adb.eig`, and the ``'enocc'`` criterion targets
        `irrep_nelec`'s per-irrep occupation instead of the lowest N
        eigenvalues overall. Requires `smask` (shell mode) and
        `mol.symmetry` truthy. Must be the *shell-separated* mol whose
        shells `smask`/`mask` index into (i.e. what `find_subspace` calls
        `fullbasis_mol`, not necessarily its own `mol` argument) -- it is
        passed straight to `adb.create_subbasis_mol` to build each trial's
        symmetry-adapted basis.
    irrep_nelec : dict, optional
        Target occupation per irrep name, pyscf ``mf.irrep_nelec`` format.
        Must be given together with `mol`.

    Returns
    -------
    mask : ndarray
        The new mask (unchanged if no candidate improved the criterion --
        see `n_added`).
    difference : float
        The current step's (possibly normalized) criterion improvement.
        ``0.0`` if `n_added` is ``0``.
    current_sum : float
        The criterion value after this step (unnormalized).
    n_added : int
        Number of functions added. ``0`` if no remaining candidate was a
        genuine improvement (see `adb.CONSTANTS.EXPAND_MASK_EPS`), in which
        case `mask`/`smask` are returned unchanged.
    smask : ndarray or None
        The new shell mask, if `smask` was given.

    Raises
    ------
    RuntimeError
        If `mol`/`irrep_nelec` (symmetry-aware mode) are given without
        `smask` (shell mode).
    """
    symmetry_aware = mol is not None and irrep_nelec is not None
    if symmetry_aware and smask is None:
        raise RuntimeError(
            "expand_mask's symmetry-aware mode (mol/irrep_nelec given) "
            "requires shell mode (smask must not be None); a per-function "
            "mask cannot be guaranteed symmetry-closed.")

    def _eig(maskedF, maskedS, test_smask):
        """Plain adb.eig, or symmetrized_eig + name-tagged orbsym when
        symmetry_aware."""
        evals, coeffs, orbsym = diagonalize_masked(
            maskedF, maskedS, mol if symmetry_aware else None, test_smask)
        return (evals, coeffs), orbsym

    maskedF = mask_matrix(F, mask)
    maskedS = mask_matrix(S, mask)
    (evals, coeffs), orbsym = _eig(maskedF, maskedS, smask)
    if Cfull is None and variant == 'elden':
        _, Cfull = eig(F, S)
    last_sum = get_iteration_criteria_value(
        variant, epsilon_i=evals, nocc=nocc,
        Csub=coeffs, Cfull=Cfull, ovlp=S[:, mask],
        irrep_nelec=irrep_nelec, orbsym=orbsym)

    test_sums = []
    if smask is None:
        for i, m in enumerate(mask):
            if m:
                continue

            test_mask = copy.deepcopy(mask)
            test_mask[i] = True
            maskedF = mask_matrix(F, test_mask)
            maskedS = mask_matrix(S, test_mask)
            evals, coeffs = eig(maskedF, maskedS)

            test_sums.append(
                (i,
                 get_iteration_criteria_value(
                     variant, epsilon_i=evals, nocc=nocc,
                     Csub=coeffs, Cfull=Cfull,
                     ovlp=S[:, test_mask]),
                 1))
    else:
        # Gather indices of duplicate shells if link_shells enabled
        # ( if system has more than 1 atom of same type, shells will be
        #   duplicated. )
        if link_shells:
            shl_indices = linked_shell_idx(smask)
        else:
            shl_indices = [[i] for i in range(len(smask))]

        for i, sidx in enumerate(shl_indices):
            if smask[sidx][0, 0]:
                continue
            test_smask = copy.deepcopy(smask)

            submask = test_smask[sidx]
            submask[:, 0] = True
            test_smask[sidx] = submask
            test_mask = smask_to_mask(test_smask)

            maskedF = mask_matrix(F, test_mask)
            maskedS = mask_matrix(S, test_mask)
            (evals, coeffs), test_orbsym = _eig(maskedF, maskedS, test_smask)

            func_keys = [shell[3] for shell in submask[:, 3]]
            nfuncs = np.sum(itemgetter(*func_keys)(NFUNCS))
            test_sums.append(
                (i,
                 get_iteration_criteria_value(
                     variant, epsilon_i=evals, nocc=nocc, Csub=coeffs,
                     Cfull=Cfull, ovlp=S[:, test_mask],
                     irrep_nelec=irrep_nelec, orbsym=test_orbsym),
                 nfuncs))

    if nfunc_normalisation:
        test_differences = [(test_sum[1] - last_sum) / test_sum[2] for test_sum in test_sums]
    else:
        test_differences = [(test_sum[1] - last_sum) for test_sum in test_sums]
    if variant == 'elden':
        array_index = np.argmax(test_differences)
        no_improvement = test_differences[array_index] <= EXPAND_MASK_EPS
    else:
        array_index = np.argmin(test_differences)
        no_improvement = test_differences[array_index] >= -EXPAND_MASK_EPS

    if no_improvement:
        # Every remaining candidate is tied with (or worse than) the
        # current mask -- none of them is a genuine improvement. This
        # happens once a target (e.g. a symmetry-aware irrep_nelec) is
        # already fully satisfied and no untried candidate can affect it:
        # np.argmin/argmax would otherwise just tie-break by array order
        # and commit an arbitrary, meaningless addition. Leave the mask
        # unchanged and signal "nothing to add" via n_added=0 instead --
        # find_subspace stops on seeing this rather than looping forever.
        return mask, 0.0, last_sum, 0, smask

    current_idx_to_flip = test_sums[array_index][0]

    if smask is None:
        mask[current_idx_to_flip] = True
    else:
        submask = smask[shl_indices[current_idx_to_flip]]
        submask[:, 0] = True
        smask[shl_indices[current_idx_to_flip]] = submask
        mask = smask_to_mask(smask)

    nfuncs_in_trial = test_sums[array_index][2]
    return mask, test_differences[array_index], test_sums[array_index][1], nfuncs_in_trial, smask
