"""Post-hoc SCF verification/reporting for a find_subspace mask_history."""

import copy

import numpy as np
from pyscf import gto, scf
from pyscf.gto import Mole
from pyscf.scf.addons import project_dm_nr2nr

from .basisutil import extract_basis
from .calculations import dual_basis_energy_correction, eig, get_q_sqrd
from .ioutil import (
    print_data,
    print_data_footer,
    print_data_header,
    print_initialization_footer,
    print_link_shells_notice,
    print_mask_analysis_init_header,
    print_mask_history_label,
    print_minimal_basis_summary,
    print_subbasis_scf_not_converged_warning,
)
from .maskutil import (
    get_atom_shell_label,
    init_smask,
    mask_matrix,
    smask_to_mask
)
from .molutil import create_shell_separated_mol
from .orbitalutil import get_occupied_orbitals_from_scf


def _split_init_and_growth_history(mask_history: list) -> tuple[list, list]:
    """Split a mask_history into initialization and growth-step entries.

    A `find_subspace(return_mask_history=True)` history mixes two kinds of
    entries: 4+-tuples for initialization steps, <=3-tuples for growth
    steps. This splits them apart, re-inserting the last initialization
    entry as the first growth entry -- the (fragile) tuple-length contract
    `find_subspace`/`mask_analysis` share for distinguishing the two.

    Parameters
    ----------
    mask_history : list
        A `find_subspace(return_mask_history=True)` history.

    Returns
    -------
    mask_history_init : list
        The 4+-tuple initialization entries.
    mask_history_growth : list
        The <=3-tuple growth entries, with the last initialization entry
        prepended.
    """
    mask_history_init = list(filter(lambda x: len(x) >= 4, mask_history))
    mask_history_growth = list(filter(lambda x: len(x) <= 3, mask_history))
    mask_history_growth.insert(0, mask_history_init[-1][:3])
    return mask_history_init, mask_history_growth


def _step_label(
        mol,
        fullbasis_mol,
        mask_i:         np.ndarray,
        last_mask:      np.ndarray,
        last_smask:     np.ndarray | None,
        is_smask:       bool,
        link_shells:    bool,
        initialized:    bool,
        empty_label:    str | None = None,
        ) -> str:
    """Compute the human-readable label for one step's added shell/functions.

    Diffs `mask_i` (this step's mask/smask) against `last_mask`/
    `last_smask` (the previous step's).

    Shared by the verbose init-history preamble (`empty_label=None` --
    `changes` is never empty there, since every init step adds at least
    one shell/function relative to the previous one) and the main per-step
    loop (`empty_label="Init"`, plus a `link_shells` ``'*'``-prefix strip
    the init preamble doesn't need).

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object.
    fullbasis_mol : pyscf.gto.Mole
        Shell-separated full-basis molecule.
    mask_i : ndarray
        This step's mask or shell mask.
    last_mask : ndarray
        The previous step's function mask.
    last_smask : ndarray, optional
        The previous step's shell mask, required when `is_smask`.
    is_smask : bool
        Whether `mask_i` is a shell mask (rather than a function mask).
    link_shells : bool
        Whether shells were linked across symmetry-equivalent atoms during
        the search.
    initialized : bool
        Whether the initialization phase has completed.
    empty_label : str, optional
        Label to return when there's no diff against the previous step.

    Returns
    -------
    str
        The label for this step.
    """
    if is_smask:
        smask = mask_i
        changes = [i for i in range(len(smask)) if smask[i][0] != last_smask[i][0]]
        if len(changes) == 0 and empty_label is not None:
            return empty_label
        label = get_atom_shell_label(mol, changes[0], link_shells=link_shells * initialized)
        if empty_label is not None and link_shells:
            label = ' '.join(label.split(' ')[1:])
            label = '*' + label
        return label
    else:
        mask = mask_i
        changes = [i for i in range(len(mask)) if mask[i] != last_mask[i]]
        if len(changes) == 0 and empty_label is not None:
            return empty_label
        aolabels = fullbasis_mol.ao_labels()
        aolabels = [aolabels[i] for i in changes]
        return ' '.join(aolabels)


def _print_verbose_init_summary(
        mol,
        fullbasis_mol,
        mask_history_init:  list,
        is_smask:           bool,
        link_shells:        bool,
        initialized:        bool,
        last_mask:          np.ndarray,
        last_smask:         np.ndarray | None,
        ) -> tuple[np.ndarray, np.ndarray | None]:
    """Print the verbose pre-loop summary of `mask_analysis`'s initialization history.

    Prints one labeled line per init step (`mask_history`'s 4+-tuple
    entries), followed by the minimal-basis/initialization-footer/
    link-shells-notice/data-header block.

    Returns
    -------
    last_mask : ndarray
        The final initialization step's function mask, which the main
        per-step loop diffs its first (re-inserted) entry against.
    last_smask : ndarray or None
        The final initialization step's shell mask.
    """
    init_method = mask_history_init[0][3]
    print_mask_analysis_init_header(init_method)
    for i, (mask_i, current_val, difference, *init) in enumerate(mask_history_init, start=1):
        label = _step_label(
            mol, fullbasis_mol, mask_i, last_mask, last_smask,
            is_smask, link_shells, initialized)
        if is_smask:
            last_smask = mask_i
            last_mask = smask_to_mask(last_smask, mol.cart)
        else:
            last_mask = mask_i
        print_mask_history_label(label, i)

    if is_smask:
        minimal_mask = smask_to_mask(mask_history_init[-1][0], mol.cart)
    else:
        minimal_mask = mask_history_init[-1][0]
    print_minimal_basis_summary(minimal_mask, mol)

    print_initialization_footer(np.sum(last_mask))

    if link_shells:
        print_link_shells_notice()

    print_data_header()

    return last_mask, last_smask


def _run_subbasis_scf(
        smask:              np.ndarray,
        fullbasis_mol,
        scf_obj,
        scf_obj_copy,
        mol,
        ovlp:               np.ndarray,
        dft:                bool,
        xc:                 str,
        grid_level:         int,
        irrep_nelec:        dict | None,
        debug:              bool,
        original_irre_nelec: dict | None,
        ) -> tuple:
    """Build and run a converged SCF calculation for the subbasis described by `smask`.

    RHF/UHF, or RKS/UKS if `dft`, using the full-basis density projected
    into the subbasis as the initial guess and a two-stage convergence
    (level-shift warmup, then second-order CIAH).

    Parameters
    ----------
    smask : ndarray
        Shell mask describing the subbasis.
    fullbasis_mol : pyscf.gto.Mole
        Shell-separated full-basis molecule.
    scf_obj : pyscf.scf.hf.SCF
        The converged full-basis SCF object.
    scf_obj_copy : pyscf.scf.hf.SCF
        A copy of `scf_obj` (used for irrep-occupation bookkeeping without
        mutating the original).
    mol : pyscf.gto.Mole
        The original molecule object.
    ovlp : ndarray
        The full-basis overlap matrix.
    dft : bool
        Whether to run RKS/UKS instead of RHF/UHF.
    xc : str
        XC functional string, used only when `dft`.
    grid_level : int
        DFT integration grid level, used only when `dft`.
    irrep_nelec : dict, optional
        Target per-irrep occupation. Falls back to
        `scf_obj_copy.get_irrep_nelec()` when `None` and `mol.symmetry` is
        on.
    debug : bool
        Whether to raise the subbasis SCF's verbosity.
    original_irre_nelec : dict, optional
        The full-basis irrep occupations, used only for the error message
        if the subbasis SCF's irrep occupations turn out inconsistent.

    Returns
    -------
    submf : pyscf.scf.hf.SCF
        The converged (or best-effort) subbasis mean-field object.
    subbasis_mol : pyscf.gto.Mole
        The subbasis molecule.
    mask : ndarray
        The AO-level boolean mask `smask` expands to.

    Raises
    ------
    RuntimeError
        If the masked overlap doesn't match the subbasis molecule's own
        overlap, or if the subbasis SCF's converged irrep occupations
        differ from those dictated by the full-basis solution.
    """
    extracted_basis, ecp_bas = extract_basis(smask, create_shell_separated_mol(fullbasis_mol))

    subbasis_mol = Mole(
        atom=fullbasis_mol.atom, basis=extracted_basis,
        charge=fullbasis_mol.charge, spin=fullbasis_mol.spin,
        verbose=fullbasis_mol.verbose, unit=fullbasis_mol.unit,
        ecp=ecp_bas, symmetry=fullbasis_mol.symmetry
    )
    subbasis_mol.build()
    is_restricted = subbasis_mol.spin == 0
    if is_restricted:
        submf = subbasis_mol.RHF()
    else:
        submf = subbasis_mol.UHF()

    mask = smask_to_mask(smask, fullbasis_mol.cart)
    maskedS = mask_matrix(ovlp, mask)
    if not np.allclose(maskedS, submf.get_ovlp()):
        raise RuntimeError('The masked overlap and the full overlap of masked molecule do not match!')

    if dft:
        if is_restricted:
            submf = subbasis_mol.RKS()
        else:
            submf = subbasis_mol.UKS()
        submf.xc = xc
        submf.grids.level = grid_level
        submf.grids.prune = None
    submf = submf.apply(scf.addons.remove_linear_dep_)

    # SCF initial guess by projecting the density from
    # full basis to current basis
    dm0_init = scf.addons.project_dm_nr2nr(
        scf_obj.mol,
        scf_obj.make_rdm1(scf_obj.mo_coeff, scf_obj.mo_occ),
        subbasis_mol)

    # Set the symmetry adapted occupations if present
    if not mol.symmetry or mol.groupname == 'C1':
        submf.irrep_nelec = None
    else:
        if irrep_nelec is not None:
            submf.irrep_nelec = irrep_nelec
        else:
            submf.irrep_nelec = scf_obj_copy.get_irrep_nelec()
        # Remove any irreps that are not present in mol
        irrep_list = copy.deepcopy(submf.irrep_nelec)
        for irname in submf.irrep_nelec:
            if irname not in subbasis_mol.irrep_name:
                if is_restricted:
                    assert submf.irrep_nelec[irname] == 0
                else:
                    assert submf.irrep_nelec[irname] == (0, 0)
                del irrep_list[irname]
        submf.irrep_nelec = irrep_list

    if debug:
        submf.verbose = 4

    # Using level_shift, do a few first order SCF cycles
    submf.level_shift = 1.0
    submf.max_cycle = 3
    submf.kernel(dm0=dm0_init)

    # Use the level shift calculation density as initial guess
    # Restore default parameters and switch to second order CIAH
    submf = submf.newton()
    submf.level_shift = 0.0
    submf.max_cycle = 50
    submf.kernel()

    # Double check that occupations of the irreps have not changed
    if mol.symmetry and mol.groupname != 'C1':
        if not all(elem == submf.get_irrep_nelec()[key] for key, elem in submf.get_irrep_nelec().items()):
            raise RuntimeError(
                'The irrep occupations have changed from the ones dictated by the full basis solution.\n'
                f'Original: {original_irre_nelec}\nThis cycle: {submf.get_irrep_nelec()}')

    if not submf.converged:
        print_subbasis_scf_not_converged_warning()

    return submf, subbasis_mol, mask


def _scf_orbital_energy(submf, is_restricted: bool) -> float:
    """Sum of the converged SCF's occupied-orbital energies.

    Spin-averaged (each spin channel weighted 1, restricted weighted 2 --
    same convention as `adb.calculations.get_iteration_criteria_value`'s
    ``'enocc'`` variant) for unrestricted.

    Parameters
    ----------
    submf : pyscf.scf.hf.SCF
        A converged mean-field object.
    is_restricted : bool
        Whether `submf` is restricted.

    Returns
    -------
    float
        The occupied-orbital energy sum.
    """
    if is_restricted:
        nocc_sb = np.sum(submf.mo_occ > 0)
        return 2 * sum(submf.mo_energy[:nocc_sb])
    nocc_sb = [np.sum(submf.mo_occ[0] > 0), np.sum(submf.mo_occ[1] > 0)]
    return (sum(submf.mo_energy[0][:nocc_sb[0]]) +
            sum(submf.mo_energy[1][:nocc_sb[1]]))


def _dual_basis_correction(submf, subbasis_mol, fullbasis_mol) -> tuple[float, float]:
    """Dual-basis energy correction of Liang, Steele & Head-Gordon.

    Projects the converged subbasis density into the full basis and
    evaluates the large-basis Roothaan-step energy lowering against it
    (see `adb.calculations.dual_basis_energy_correction`).

    Parameters
    ----------
    submf : pyscf.scf.hf.SCF
        Converged subbasis mean-field object.
    subbasis_mol : pyscf.gto.Mole
        The subbasis molecule.
    fullbasis_mol : pyscf.gto.Mole
        The shell-separated full-basis molecule.

    Returns
    -------
    dE : float
        The Roothaan-step energy lowering.
    e_tot : float
        The reference full-basis-HF total energy the correction is
        measured relative to.
    """
    P_sub = submf.make_rdm1()
    P_full_projected = project_dm_nr2nr(subbasis_mol, P_sub, fullbasis_mol)
    return dual_basis_energy_correction(fullbasis_mol.HF(), P_full_projected)


def _analysis_row(
        mask:                   np.ndarray,
        current_val:            float,
        difference:             float,
        scf_energy:             float | None,
        scf_orbital_energy:     float | None,
        Q_sqrd:                 float,
        smask_or_mask:          np.ndarray,
        dE:                     float,
        E_HF_largebasis:        float,
        subbasis_converged:     bool,
        ) -> list:
    """Build one `mask_analysis` dataframe row."""
    return [
        sum(mask),
        current_val,
        difference,
        scf_energy,
        scf_orbital_energy,
        Q_sqrd,
        copy.deepcopy(smask_or_mask),
        dE,
        E_HF_largebasis,
        subbasis_converged,
    ]


def mask_analysis(
        mask_history:           np.ndarray,
        mol:                    gto.Mole,
        scf_obj:                scf.hf.SCF | scf.hf.RHF | scf.uhf.UHF | scf.rohf.ROHF | scf.ghf.GHF,
        fock:                   np.ndarray,
        ovlp:                   np.ndarray,
        verbose:                bool                = True,
        link_shells:            bool                = True,
        dft:                    bool                = False,
        xc:                     str                 = 'pbe,pbe',
        grid_level:             int                 = 7,
        C_full:                 np.ndarray | None   = None,
        calculate_correction:   bool                = False,
        irrep_nelec:            dict | None         = None,
        debug:                  bool                = False,
        track_orbitals:         bool                = False,
        ) -> list | tuple[list, list]:
    """Run a converged SCF for every mask in a find_subspace history and tabulate the results.

    For each accepted step in `mask_history`, builds and converges the
    subbasis SCF it describes (see `_run_subbasis_scf`), then records the
    number of functions, the search criterion value, the converged SCF
    energy, the occupied-orbital energy sum, and the projection onto the
    converged full-basis wavefunction.

    Parameters
    ----------
    mask_history : ndarray
        The `find_subspace(return_mask_history=True)` history to analyze.
        Element `i` is a tuple ``(i-th mask/smask, i-th criterion value,
        i-th difference)``, or a 4+-tuple for initialization entries (see
        `_split_init_and_growth_history`).
    mol : pyscf.gto.Mole
        The molecule object.
    scf_obj : pyscf.scf.hf.SCF
        The converged full-basis SCF object.
    fock : ndarray
        The converged full-basis Fock matrix.
    ovlp : ndarray
        The full-basis overlap matrix.
    verbose : bool, default True
        Whether to print progress during analysis.
    link_shells : bool, default True
        Whether shells of same-element atoms were linked during the ADB
        search. Not strictly required even when they were, but omitting it
        will result in slightly incorrect data prints.
    dft : bool, default False
        Whether to run RKS/UKS instead of RHF/UHF for each subbasis.
    xc : str, default 'pbe,pbe'
        XC functional string accepted by pyscf, used only when `dft`.
    grid_level : int, default 7
        DFT integration grid level (0 very sparse, 9 very dense), used
        only when `dft`.
    C_full : ndarray, optional
        Full-basis MO coefficient matrix, used for projection
        calculations.
    calculate_correction : bool, default False
        Whether to calculate the dual-basis correction of Liang, Steele &
        Head-Gordon for every subbasis.
    irrep_nelec : dict, optional
        Target occupation per irrep name. Falls back to the full-basis
        SCF's own irrep occupations when `None`.
    debug : bool, default False
        Whether to raise verbosity on each subbasis SCF.
    track_orbitals : bool, default False
        Optional feature, off by default. When `True`, records the
        *self-consistent* occupied orbital energies and their symmetry
        labels (via `adb.get_occupied_orbitals_from_scf`) for every
        subbasis in `mask_history` that gets its own SCF run (i.e. every
        smask entry; the plain-mask/non-SCF branch has no self-consistent
        solution to record). This is the converged counterpart to
        `adb.find_subspace(track_orbitals=True)`, which records the
        unconverged guess-Fock-matrix spectrum seen *during* the ADB
        search rather than the actual SCF solution of each subbasis. When
        `True`, the return value becomes a 2-tuple ``(dataframe,
        orbital_history)`` instead of just `dataframe` -- existing call
        sites assigning the return value to a single variable are
        unaffected as long as this stays at its default.

    Returns
    -------
    dataframe : list
        One row per `mask_history` entry: number of functions, criterion
        value, difference, total SCF energy, occupied-orbital SCF energy,
        squared projection onto the converged full-basis wavefunction,
        the mask/smask, the dual-basis correction (if
        `calculate_correction`), the full-basis HF energy it's relative
        to, and whether the subbasis SCF converged. If `track_orbitals`,
        a ``(dataframe, orbital_history)`` tuple is returned instead --
        see the `track_orbitals` parameter.
    """
    scf_obj_copy = scf_obj.copy()
    original_irre_nelec = None
    if mol.symmetry and mol.groupname != 'C1':
        original_irre_nelec = scf_obj_copy.get_irrep_nelec()
    fullbasis_mol = create_shell_separated_mol(mol)
    is_restricted = fock.ndim == 2
    nocc = fullbasis_mol.nelec

    initialized = False
    dataframe = []
    orbital_history = [] if track_orbitals else None
    last_mask = [False] * fullbasis_mol.nao_nr()
    last_smask = None
    is_smask = isinstance(mask_history[0][0], np.ndarray)
    if is_smask:
        last_smask = init_smask(mol, mol.cart)
    scf_energy = None
    scf_orbital_energy = None

    # Filter mask_history of initialization and ABS
    mask_history_init, mask_history = _split_init_and_growth_history(mask_history)

    if verbose:
        last_mask, last_smask = _print_verbose_init_summary(
            mol, fullbasis_mol, mask_history_init, is_smask, link_shells,
            initialized, last_mask, last_smask)

    for mask_i, current_val, difference, *init in mask_history:
        dE = 0.0
        E_HF_largebasis = 0.0
        if is_smask:
            smask = mask_i
            submf, subbasis_mol, mask = _run_subbasis_scf(
                smask, fullbasis_mol, scf_obj, scf_obj_copy, mol, ovlp,
                dft, xc, grid_level, irrep_nelec, debug, original_irre_nelec)
            is_restricted = subbasis_mol.spin == 0

            subbasis_converged = submf.converged
            scf_energy = submf.e_tot

            if track_orbitals:
                orbital_history.append({
                    'nfunc': int(np.sum(mask)),
                    'orbitals': get_occupied_orbitals_from_scf(submf),
                })

            scf_orbital_energy = _scf_orbital_energy(submf, is_restricted)

            if is_restricted:
                Q_sqrd = get_q_sqrd(C_full, submf.mo_coeff, ovlp[:, mask], nocc)
            else:
                Q_sqrd = get_q_sqrd(
                    np.asarray(C_full), np.asarray(submf.mo_coeff), ovlp[:, mask], nocc)

            # Calculate dual basis correction as per Liang, Steele,
            # Head-Gordon et al.
            if calculate_correction:
                dE, E_HF_largebasis = _dual_basis_correction(submf, subbasis_mol, fullbasis_mol)
        else:
            mask = mask_i
            _, subbasis_coeffs = eig(mask_matrix(fock, mask), mask_matrix(ovlp, mask))
            Q_sqrd = get_q_sqrd(C_full, subbasis_coeffs, ovlp[:, mask], nocc)

        if verbose:
            label = _step_label(
                mol, fullbasis_mol, smask if is_smask else mask,
                last_mask, last_smask, is_smask, link_shells, initialized,
                empty_label="Init")
            print_data(
                mask, current_val, difference, label, scf_energy, Q_sqrd,
                print_header=False
            )
        dataframe.append(_analysis_row(
            mask, current_val, difference, scf_energy, scf_orbital_energy,
            Q_sqrd, smask if is_smask else mask, dE, E_HF_largebasis,
            subbasis_converged,
        ))
        last_mask = copy.deepcopy(mask)
        if is_smask:
            last_smask = copy.deepcopy(smask)

    if verbose:
        print_data_footer()

    if track_orbitals:
        return dataframe, orbital_history

    return dataframe
