"""Post-hoc SCF verification/reporting for a find_subspace mask_history"""

import copy
import numpy as np
from pyscf import gto, scf
from pyscf.gto import Mole
from pyscf.scf.addons import project_dm_nr2nr
from .calculations import eig, get_q_sqrd, dual_basis_energy_correction
from .maskutil import init_smask, get_atom_shell_label, smask_to_mask, mask_matrix
from .molutil import create_shell_separated_mol
from .basisutil import extract_basis
from .ioutil import (
    print_mask_analysis_init_header, print_mask_history_label,
    print_minimal_basis_summary, print_initialization_footer,
    print_link_shells_notice, print_data_header, print_data, print_data_footer,
    print_subbasis_scf_not_converged_warning,
)
from .orbitalutil import get_occupied_orbitals_from_scf


def _split_init_and_growth_history(mask_history):
    """Split a find_subspace(return_mask_history=True) mask_history into
    its initialization entries (4+-tuples) and growth-step entries
    (<=3-tuples), re-inserting the last initialization entry as the first
    growth entry -- the fragile tuple-length contract find_subspace/
    mask_analysis share for distinguishing the two record kinds.
    """
    mask_history_init = list(filter(lambda x: len(x) >= 4, mask_history))
    mask_history_growth = list(filter(lambda x: len(x) <= 3, mask_history))
    mask_history_growth.insert(0, mask_history_init[-1][:3])
    return mask_history_init, mask_history_growth


def _step_label(
        mol, fullbasis_mol, mask_i, last_mask, last_smask,
        is_smask, link_shells, initialized,
        empty_label: str | None = None,
        ) -> str:
    """Compute the human-readable label for one step's added shell/
    functions, diffing `mask_i` (this step's mask/smask) against
    `last_mask`/`last_smask` (the previous step's).

    Shared by the verbose init-history preamble (empty_label=None --
    `changes` is never empty there, since every init step adds at least
    one shell/function relative to the previous one) and the main
    per-step loop (empty_label="Init", plus a link_shells '*'-prefix
    strip the init preamble doesn't need).
    """
    if is_smask:
        smask = mask_i
        changes = [i for i in range(len(smask)) if smask[i][0] != last_smask[i][0]]
        if len(changes) == 0 and empty_label is not None:
            return empty_label
        label = get_atom_shell_label(mol, changes[0], link_shells=link_shells*initialized)
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
        mol, fullbasis_mol, mask_history_init, is_smask, link_shells,
        initialized, last_mask, last_smask,
        ):
    """Print the verbose pre-loop summary of mask_analysis's
    initialization history (mask_history's 4+-tuple entries): one
    labeled line per init step, followed by the minimal-basis/
    initialization-footer/link-shells-notice/data-header block.

    Returns the updated (last_mask, last_smask) reflecting the final
    initialization step, which the main per-step loop diffs its first
    (re-inserted) entry against.
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
        smask, fullbasis_mol, scf_obj, scf_obj_copy, mol, ovlp,
        dft, xc, grid_level, irrep_nelec, debug, original_irre_nelec,
        ):
    """Build and run a converged SCF calculation (RHF/UHF, or RKS/UKS if
    `dft`) for the subbasis described by `smask`, using the full-basis
    density projected into the subbasis as the initial guess and a
    two-stage convergence (level-shift warmup, then second-order CIAH).

    Returns (submf, subbasis_mol, mask) -- the converged mean-field
    object, the subbasis Mole, and the AO-level boolean mask `smask`
    expands to.
    """
    extracted_basis, ecp_bas = extract_basis(smask, create_shell_separated_mol(fullbasis_mol))

    subbasis_mol = Mole(
        atom = fullbasis_mol.atom, basis = extracted_basis,
        charge = fullbasis_mol.charge, spin = fullbasis_mol.spin,
        verbose = fullbasis_mol.verbose, unit = fullbasis_mol.unit,
        ecp = ecp_bas, symmetry = fullbasis_mol.symmetry
        )
    subbasis_mol.build()
    is_restricted = subbasis_mol.spin == 0
    if is_restricted:
        submf = subbasis_mol.RHF()#.newton()
    else:
        submf = subbasis_mol.UHF()#.newton()

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
    dm0_init = scf.addons.project_dm_nr2nr(scf_obj.mol,
                                           scf_obj.make_rdm1(scf_obj.mo_coeff,
                                                             scf_obj.mo_occ),
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

    # Use the level shift calulcation density as initial guess
    # Restore default parameters and switch to second order CIAH
    submf = submf.newton()
    submf.level_shift = 0.0
    submf.max_cycle = 50
    submf.kernel()

    # Double check that occupations of the irreps have not changed
    if mol.symmetry and mol.groupname != 'C1':
        if not all([elem == submf.get_irrep_nelec()[key] for key, elem in submf.get_irrep_nelec().items()]):
            raise RuntimeError(f'The irrep occupations have changed from the ones dictated by the full basis solution.\nOriginal: {original_irre_nelec}\nThis cycle: {submf.get_irrep_nelec()}')

    if not submf.converged:
        print_subbasis_scf_not_converged_warning()

    return submf, subbasis_mol, mask


def _scf_orbital_energy(submf, is_restricted):
    """Sum of the converged SCF's occupied-orbital energies (spin-averaged
    for unrestricted)."""
    if is_restricted:
        # nocc_sb = np.sum(submf.mo_occ > 0)
        nocc_sb = len(submf.mo_occ > 0)
        return sum(submf.mo_energy[:nocc_sb])
    else:
        # nocc_sb = [np.sum(submf.mo_occ[0] > 0), np.sum(submf.mo_occ[1] > 0)]
        nocc_sb = [len(submf.mo_occ[0] > 0), len(submf.mo_occ[1] > 0)]
        return .5 * sum(
            submf.mo_energy[0][:nocc_sb[0]] +
            submf.mo_energy[1][:nocc_sb[1]])


def _dual_basis_correction(submf, subbasis_mol, fullbasis_mol):
    """Calculate dual basis correction as per Liang, Steele, Head-Gordon
    et al.: project the converged subbasis density into the full basis
    and evaluate the large-basis Roothaan-step energy lowering against
    it."""
    P_sub = submf.make_rdm1()
    P_full_projected = project_dm_nr2nr(
        subbasis_mol,
        P_sub,
        fullbasis_mol)
    return dual_basis_energy_correction(
        fullbasis_mol.HF(),
        P_full_projected,
    )


def _analysis_row(
        mask, current_val, difference, scf_energy, scf_orbital_energy,
        Q_sqrd, smask_or_mask, dE, E_HF_largebasis, subbasis_converged,
        ):
    """Build one mask_analysis dataframe row."""
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
    xc:                     str                 = 'b3lyp',
    grid_level:             int                 = 7,
    C_full:                 np.ndarray | None   = None,
    calculate_correction:   bool                = False,
    irrep_nelec:            dict | None         = None,
    debug:                  bool                = False,
    track_orbitals:         bool                = False,
    ) -> list:
    """Run mask analysis.

    Args:
        mask_history : array
            The mask/smask history for which to run the analysis on. Elements
            will be tuples, with i:th tuple being
            (i:th mask/smask, i:th criteria value, i:th difference)
        mol : pyscf.gto.Mole object
            The molecule object
        scf_obj : pyscf.scf.(U/R/RO/D/-)HF object
            The self-consistent field object
        fock : numpy.ndarray
            The converged Fock matrix
        ovlp : numpy.ndarray
            The overlap matrix
        verbose : bool
            Whether to print data during analysis. Default is True.
        basis : str
            Name of the full basis set. Will be used in psi4 full
            basis calculations.
        link_shells : bool
            Whether the shells of same atoms have been linked during ABS
            calculation. Not strictly required even in the case of linked shell
            ABS calculation, but will result in slightly inccorrect data prints.
            Default is True.
        dft : bool
            Hartree-Fock or DFT.
            Optional, default is False
        xc : str
            XC functional string accepted by PySCF.
            Optional, default is 'b3lyp'.
        grid_level : int
            predefined integration grid levels, 0-9 (0 very sparse, 9 very dense).
            Optional, default is 3.
        use_psi4 : bool
            If True, psi4 will be used for SCF computation instead of PySCF.
            Optional, default is False.
        C_full : np.ndarray | None
            Full basis coefficient matrix, used for projection calculations.
        calculate_correction : bool
            Whether to calculate the dual basis correction of Liang, Steele,
            Head-Gordon et al. for every subbasis.
        track_orbitals : bool
            Optional feature, off by default. When True, records the
            *self-consistent* occupied orbital energies and their symmetry
            labels (via get_occupied_orbitals_from_scf) for every subbasis
            in mask_history that gets its own SCF run (i.e. every smask
            entry; the plain-mask/non-SCF branch has no self-consistent
            solution to record). This is the converged counterpart to
            find_subspace(track_orbitals=True), which records the
            unconverged guess-Fock-matrix spectrum seen *during* the ADB
            search rather than the actual SCF solution of each subbasis.
            When True, the return value becomes a 2-tuple
            `(dataframe, orbital_history)` instead of just `dataframe` --
            existing call sites assigning the return value to a single
            variable are unaffected as long as this stays at its default.
            Default False: behaviour/return shape identical to before this
            option existed.
    Return:
        dataframe : array
            A python array with number of functions,
            current_sum, difference, total SCF energy, SCF energy of
            occupied orbitals and the projection onto the converged
            full basis wave function will on every iteration. If
            `track_orbitals=True`, a `(dataframe, orbital_history)` tuple
            is returned instead -- see the `track_orbitals` Args entry.
    """
    scf_obj_copy = scf_obj.copy()
    original_irre_nelec = None
    if mol.symmetry and mol.groupname != 'C1':
        original_irre_nelec = scf_obj_copy.get_irrep_nelec()
    fullbasis_mol = create_shell_separated_mol(mol)
    is_restricted = len(fock.shape) == 2
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
                Q_sqrd = get_q_sqrd(
                    C_full, submf.mo_coeff,
                    ovlp[:,mask], nocc
                )
            else:
                Q_sqrd = get_q_sqrd(
                    np.asarray(C_full), np.asarray(submf.mo_coeff),
                    ovlp[:,mask], nocc
                )

            # Calculate dual basis correction as per Liang, Steele,
            # Head-Gordon et al.
            if calculate_correction:
                dE, E_HF_largebasis = _dual_basis_correction(submf, subbasis_mol, fullbasis_mol)
        else:
            mask = mask_i
            _, subbasis_coeffs = eig(mask_matrix(fock, mask, is_restricted=is_restricted), mask_matrix(ovlp, mask))
            Q_sqrd = get_q_sqrd(
                C_full, subbasis_coeffs,
                ovlp[:,mask], nocc
            )

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
