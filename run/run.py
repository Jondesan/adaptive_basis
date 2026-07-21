#!/bin/python3

import sys
import os
import copy
import argparse
import fcntl

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import adb
from adb.ioutil import get_molecules_in_dir
from pyscf import scf, gto
import numpy as np
import pandas as pd
import datetime
import pyscf
from time import time

AVAIL_INIT_METHODS = [
    'scf',
    'atom',
    'sap',
    'huckel',
    'vsap'
]


def add_initial_guesses(ig_list, mol_list):
    if mol_list is None:
        mol_list = ig_list
    else:
        mol_list.extend(ig_list)

    return mol_list


def get_initial_fock_matrix(init_method, scf_method_object, sap_basis=None, xc=None):
    fock = None
    match init_method.lower():
        case "scf":
            if not scf_method_object.converged:
                raise RuntimeError("Trying to initialize with a converged Fock matrix from an unconverged calculation!")
            dm0 = None
        case "vsap":
            if xc is None:
                raise RuntimeError("Exchange correlation functional is None!")
            dm0 = scf_method_object.mol.KS().set(xc=xc).get_init_guess(key=init_method)
        case "sap":
            if sap_basis is None:
                raise RuntimeError("SAP basis is None! Must be either Pyscf built-in or path to external file!")
            scf_method_object.sap_basis = sap_basis
            dm0 = scf_method_object.get_init_guess(key=init_method)
        case _:
            dm0 = scf_method_object.get_init_guess(key=init_method)

    fock = scf_method_object.get_fock(dm=dm0)
    return fock


def run_abs(
    mol_list,
    variant='enocc',
    lshells=True,
    conv_tol=1e-4,
    sap_basis_sets='sapgraspsmall',
    nfunc_normalisation=True,
    dft=False,
    xc='pbe,pbe',
    grid_level=7,
    abd_init=True,
    symmetry_occ_fname=None,
    q_tol=1.0,
    ODIR="output",
    debug=False,
    symmetry_aware_search=False,
    track_orbitals=False,
    ):
    """Run subbasis iteration for molecules in mol_list"""

    """
    dataframe structure:
        dtype : list

    dataframe = [
        dat1,
        dat2,
        .
        .
        .
        datN ]
    where
    dat_i = [
        0: molname,
        1: time taken for full basis SCF,
        2: time for find_subspace f_by_f,
        3: time for find_subspace s_by_s,
        4: f_by_f data,
        5: s_by_s data,
        6: shell mask,
        7: occupations,
        8: full basis SCF energy
        ]
    """
    datacols = ["nfunc", "cursum", "diff", "E_scf", "E_orb", "Qsqrd", "smask"]
    calculate_DB_correction = True
    if calculate_DB_correction:
        datacols.extend(['dE', 'E_HF_largebasis'])
    datacols.extend(['conv_stat'])

    for molfilename, mol, shellsep_mol, shells, init_guess, basisname in mol_list:
        # Open the output file
        bsname = basisname#mol.basis
        molname = molfilename.split(".")[0]
        charge = mol.charge
        spin = mol.spin
        irrep_nelec = None
        
        if len(bsname) > 25:
            bsname = "basis_NA"
        
        if init_guess is None:
            init_guess = ['atom']

        if not os.path.isdir('output'):
            os.mkdir('output')

        is_restricted = mol.spin == 0

        
        if symmetry_occ_fname is not None:
            irrep_nelec, _ = adb.ioutil.read_symmetry_occs_from_file(
                symmetry_occ_fname, molfname=molfilename)
        
        # Set up Hartree-Fock, remove linear dependencies from basis
        if is_restricted:
            scf_method_object = mol.RHF()
        else:
            scf_method_object = mol.UHF()
        if dft:
            if is_restricted:
                scf_method_object = mol.RKS()
            else:
                scf_method_object = mol.UKS()
            scf_method_object.xc = xc
            scf_method_object.grids.level = grid_level
            scf_method_object.grids.prune = None
        scf_method_object = scf_method_object.apply(scf.addons.remove_linear_dep_)

        start = time()
        # Set the symmetry occupations if present
        if irrep_nelec is not None:
            scf_method_object.irrep_nelec = {}
            for key in irrep_nelec:
                if irrep_nelec[key] != 0:
                    if key not in scf_method_object.mol.irrep_name:
                        raise RuntimeError(f'irrep {key} not found in subbasis:\n{scf_method_object.mol.irrep_name}')
                    scf_method_object.irrep_nelec[key] = irrep_nelec[key]
        else:
            print('Symmetry occupations not set explicitly for the full basis calculation!', file=sys.stderr)

        if debug: scf_method_object.verbose = 4

        # Using level_shift, do a few first order SCF cycles
        scf_method_object.init_guess = 'atom'
        scf_method_object.level_shift = 1.0
        scf_method_object.max_cycle = 3
        scf_method_object.kernel()

        # Restore default parameters and switch to second order CIAH.
        # The coarse level-shifted warmup above only approximately respects
        # mol's point-group symmetry -- Newton's orbital-rotation step
        # labels orbital symmetry at every micro-iteration with a strict
        # (1e-9) tolerance, so hand it an explicitly symmetrized MO space
        # rather than relying on the warmup having converged that cleanly.
        mo_coeff_sym = scf_method_object.mo_coeff
        if mol.symmetry and mol.groupname != 'C1':
            if is_restricted:
                mo_coeff_sym = pyscf.symm.symmetrize_space(mol, scf_method_object.mo_coeff)
            else:
                mo_coeff_sym = [
                    pyscf.symm.symmetrize_space(mol, scf_method_object.mo_coeff[0]),
                    pyscf.symm.symmetrize_space(mol, scf_method_object.mo_coeff[1]),
                ]
        scf_method_object = adb.symmetry_safe_newton(scf_method_object)
        scf_method_object.level_shift = 0.0
        scf_method_object.max_cycle = 50
        scf_method_object.kernel(mo_coeff_sym, scf_method_object.mo_occ)

        end = time()
        e_tot = scf_method_object.e_tot

        fullbasis_hf_time = end - start
        F_scf = scf_method_object.get_fock()

        if not mol.symmetry or mol.groupname == 'C1':
            irrep_nelec = None
        else:
            irrep_nelec = scf_method_object.get_irrep_nelec()

        # Save the SCF matrices
        mo_coeff_scf = copy.deepcopy(scf_method_object.mo_coeff)

        for ig in init_guess:
            print(f'Running calculation for mol {molfilename}, with basis {basisname} and init guess {ig}')
            if ig == 'sap':
                sapbases = np.asarray(sap_basis_sets)
            else:
                sapbases = [None]

            for sapbs in sapbases:
                if ig == 'sap':
                    sapbasisname = sapbs.strip().split('/')[-1].split('.')[0]
                    fname = ".".join([molname, f'charge{charge}', f'spin{spin}', bsname, ig, sapbasisname])
                else:
                    fname = ".".join([molname, f'charge{charge}', f'spin{spin}', bsname, ig])
                if not lshells:
                    fname = ".".join([fname, 'unlinked'])
                with open(f'{ODIR}/{fname}.out', "w") as f:
                    f.write("{:<15s} {:<15s} {:<15s} {:<15s} {:<15s}".format(
                        "molecule", "basis_set", "variant", "init_guess", "link_status"))
                    if ig == 'sap':
                        f.write(" {:<15s}".format("sap_basis"))
                    f.write("\n")
                    f.write(f"{molname:<15s} {bsname:<15s} {variant:<15s} {ig:<15s} {str(lshells):<15s}")
                    if ig == 'sap':
                        f.write(f" {sapbasisname:<15s}")
                    f.write("\n")
                    f.write("{:<15s} {:<15s}\n".format('charge', 'spin'))
                    f.write(f"{charge:<15d} {spin:<15d}\n")
                    f.write(f"Calculations done on {datetime.datetime.now()}\n\n")

                    start = time()
                    # Based on the initial guess, get the Fock matrix which
                    # is used as the initial guess in the subbasis
                    F = get_initial_fock_matrix(ig, scf_method_object, sap_basis=sapbs, xc=xc)
                    end = time()
                    S = scf_method_object.get_ovlp()

                    f.write("time stats [s]\n")
                    f.write("{:<17s}{:<17s}{:<17s}\n".format("t_HF", "t_fbyf", "t_sbys"))
                    f.write(f"{(fullbasis_hf_time + end - start):15.9e}  ")

                    if variant == 'enocc':
                        start = time()
                        # maskhistory = adb.find_subspace(
                        #     F, S, mol, scf_method_object,
                        #     conv_tol=conv_tol,
                        #     variant=variant,
                        #     return_mask_history=True,
                        #     nfunc_normalisation=nfunc_normalisation,
                        #     abd_initialization=abd_init
                        # )
                        # data_fbyf = adb.mask_analysis(
                        #     maskhistory, mol, scf_method_object,
                        #     F, S
                        # )
                        end = time()

                    if variant == 'enocc':
                        f.write(f"{end-start:15.9e}  ")
                    else:
                        f.write("{:<15s}".format("-"))

                    start = time()
                    symmetry_aware_kwargs = {}
                    if symmetry_aware_search:
                        symmetry_aware_kwargs = dict(
                            symmetry_aware=True,
                            irrep_nelec=irrep_nelec,
                        )
                    find_subspace_result = adb.find_subspace(
                        F, S, mol, scf_method_object,
                        conv_tol=conv_tol,
                        get_smask=True,
                        variant=variant,
                        link_shells=lshells,
                        nfunc_normalisation=nfunc_normalisation,
                        return_mask_history=True,
                        abd_initialization=abd_init,
                        abd_Q_tol=q_tol,
                        track_orbitals=track_orbitals,
                        **symmetry_aware_kwargs,
                    )
                    if track_orbitals:
                        smaskhistory, orbital_history = find_subspace_result
                        adb.write_orbital_history(
                            orbital_history, fn=f'{ODIR}/{fname}.orbitals',
                            molname=molname, basisname=bsname)
                    else:
                        smaskhistory = find_subspace_result
                    mask_analysis_result = adb.mask_analysis(
                        smaskhistory, shellsep_mol, scf_method_object,
                        F, S, dft = dft, xc = xc, grid_level = grid_level,
                        C_full = mo_coeff_scf,
                        calculate_correction = calculate_DB_correction,
                        irrep_nelec = irrep_nelec,
                        debug = debug,
                        track_orbitals = track_orbitals,
                    )
                    if track_orbitals:
                        data_sbys, scf_orbital_history = mask_analysis_result
                        adb.write_orbital_history(
                            scf_orbital_history, fn=f'{ODIR}/{fname}.scf_orbitals',
                            molname=molname, basisname=bsname)
                    else:
                        data_sbys = mask_analysis_result
                    end = time()

                    f.write(f"{end-start:15.9e}\n\n")

                    # if variant == 'enocc':
                    #     df_fbyf = pd.DataFrame(data_fbyf, columns=datacols)
                    df_sbys = pd.DataFrame(data_sbys, columns=datacols)
                    minimal_mol = pyscf.M(
                        atom = mol.atom,
                        basis = 'sto3g',
                        charge = mol.charge,
                        spin = mol.spin,
                        unit = mol.unit
                    )
                    f.write("{:<15s} {:<30s} {:<15s} {:<15s}\n".format("N_occ", "E_HF", "nfunc", "nfunc_minimal"))
                    f.write(
                        "{:<15d} {:<30.20f} {:<15d} {:<15d}\n\n".format(
                            np.sum(mol.nelec), e_tot, mol.nao_nr(), minimal_mol.nao_nr()
                        )
                    )

                    f.write("shell-by-shell iteration\n")
                    df_sbys.to_csv(f, index=False)
                    f.write("\n\n")


def _append_criterion_result(output_file, row):
    """Append one result row to output_file under an exclusive flock.

    Safe to call from concurrent processes on the same file: the lock
    prevents interleaved writes and ensures only one process writes the
    header (when the file is empty).
    """
    header = 'molname;basis;init_guess;variant;criterion_value\n'
    line = (
        f"{row['molname']};{row['basis']};{row['init_guess']};"
        f"{row['variant']};{row['criterion_value']:.12f}\n"
    )
    with open(output_file, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            if f.tell() == 0:
                f.write(header)
            f.write(line)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def compute_fullbasis_criterion(
    mol_list,
    variant='enocc',
    dft=False,
    xc='pbe,pbe',
    grid_level=7,
    sap_basis_sets='sapgraspsmall',
    output_file=None,
):
    """Compute the iteration criterion value at full basis for each molecule.

    Builds the Fock matrix from the initial guess density (no SCF), then
    evaluates the criterion defined by `variant` by diagonalizing in the
    full basis — equivalent to the value find_subspace converges toward.

    Args:
        mol_list : list
            List of molecules as returned by get_molecules_in_dir.
        variant : str
            Criterion variant: 'enocc' (sum of occupied orbital energies)
            or 'elden' (Q^2 projection quality). Default 'enocc'.
        dft : bool
            Use DFT instead of HF. Default False.
        xc : str
            XC functional string (PySCF format). Default 'pbe,pbe'.
        grid_level : int
            DFT integration grid level (0-9). Default 7.
        sap_basis_sets : str or list
            SAP basis set name(s), used when init_guess is 'sap'.
        output_file : str or None
            Path to the output file. Results are appended one line at a
            time under an exclusive lock, so concurrent script instances
            writing to the same file will not lose data. If None, no file
            is written.

    Returns:
        results : list of dict
            Each dict contains 'molname', 'basis', 'init_guess', 'variant',
            and 'criterion_value' for one (molecule, init_guess) pair.
    """
    results = []

    for molfilename, mol, shellsep_mol, shells, ig_list, basisname in mol_list:
        molname = molfilename.split(".")[0]
        nocc = mol.nelec

        if mol.spin == 0:
            scf_method_object = mol.RHF()
        else:
            scf_method_object = mol.UHF()
        if dft:
            if mol.spin == 0:
                scf_method_object = mol.RKS()
            else:
                scf_method_object = mol.UKS()
            scf_method_object.xc = xc
            scf_method_object.grids.level = grid_level
            scf_method_object.grids.prune = None
        scf_method_object.remove_overlap_zero_eigenvalue = True
        scf_method_object.overlap_zero_eigenvalue_threshold = 1e-6
        # scf_method_object = scf_method_object.apply(scf.addons.remove_linear_dep_)

        S = scf_method_object.get_ovlp()

        if ig_list is None:
            ig_list = ['atom']

        F_scf = None
        if 'scf' in ig_list:
            scf_method_object.init_guess = 'atom'
            scf_method_object.level_shift = 1.0
            scf_method_object.max_cycle = 3
            scf_method_object.kernel()
            # See adb/scf_fixes.py: plain .newton() crashes when
            # mol.symmetry is enabled and remove_linear_dep_ (above) has
            # actually reduced the basis.
            scf_method_object = adb.symmetry_safe_newton(scf_method_object)
            scf_method_object.level_shift = 0.0
            scf_method_object.max_cycle = 50
            scf_method_object.kernel()
            F_scf = scf_method_object.get_fock()

        for ig in ig_list:
            if ig == 'sap':
                sapbases = np.asarray(sap_basis_sets)
            else:
                sapbases = [None]

            for sapbs in sapbases:
                if ig == 'scf':
                    F = F_scf
                elif ig == 'vsap':
                    tempmf = mol.KS().set(xc=xc)
                    dm0 = tempmf.get_init_guess(key='vsap')
                    F = scf_method_object.get_fock(dm=dm0)
                else:
                    scf_method_object.sap_basis = sapbs
                    dm0 = scf_method_object.get_init_guess(key=ig)
                    F = scf_method_object.get_fock(dm=dm0)

                evals, evecs = adb.eig(F, S)
                criterion_value = adb.get_iteration_criteria_value(
                    variant,
                    epsilon_i=evals,
                    nocc=nocc,
                    Cfull=evecs,
                    Csub=evecs,
                    ovlp=S,
                )

                ig_label = ig if ig != 'sap' else \
                    f"sap({sapbs.strip().split('/')[-1].split('.')[0]})"
                print(f'{molname} ({basisname}, {ig_label}): full basis {variant} = {criterion_value:.15f}')
                row = {
                    'molname': molname,
                    'basis': basisname,
                    'init_guess': ig_label,
                    'variant': variant,
                    'criterion_value': criterion_value,
                }
                results.append(row)
                if output_file is not None:
                    _append_criterion_result(output_file, row)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run adaptive basis Hartree-Fock calculations."
    )
    parser.add_argument(
        "--mpath", type=str, required=True, help="path to molecule directory"
    )
    parser.add_argument(
        "--basis", type=str, required=True, nargs='+',
        help="path to basis input file or list of basis sets to use"
    )
    parser.add_argument(
        "-c", "--convtol", type=float, required=False, default=1e-4,
        help="convergence tolerance"
    )
    parser.add_argument(
        "--init_guesses", type=str, required=False, nargs='+',
        default='all', choices=['all', 'scf', 'sap', 'atom', 'vsap', 'huckel'],
        help="which initialization methods to use, 'all' will select all available methods"
    )
    parser.add_argument(
        "--sapbasis", type=str, required=False, nargs='+',
        default='sapgraspsmall',
        help="SAP basis, either path to file or basis name"
    )
    parser.add_argument(
        "--decontractions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="whether to run decontracted calculations too, optional",
    )
    parser.add_argument(
        "--var",
        type=str,
        required=False,
        default='enocc',
        choices=['enocc', 'elden'],
        help="which minimisation criteria to use, optional. Default is enocc.",
    )
    parser.add_argument(
        "--linkshells",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Turn duplicate shell linking on/off during shell by shell calculations.",
    )
    parser.add_argument(
        "-u", "--unit", type=str, required=True,
        choices=['Angstrom', 'Bohr'],
        help="coordinate units of xyz file"
    )
    parser.add_argument(
        "--nfunc_normalisation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalise criteria value w.r.t. the number of added functions, optional. Default is True.",
    )
    parser.add_argument(
        "--dft",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use DFT, optional. Default is False.",
    )
    parser.add_argument(
        "--abd",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Initialize with atomic block decomposition, optional. Default is True.",
    )
    parser.add_argument(
        "--q_tol", type=float, required=False, default=1.0,
        help="charge tolerance, default 1.0"
    )
    parser.add_argument(
        "--sym_occ_file", type=str, required=False, default=None,
        help="path to file with required symmetry occupations."
    )
    parser.add_argument(
        "--point_group_file", type=str, required=False, default=None,
        help="path to file with point group labels."
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        default='abs',
        choices=['abs', 'abd', 'full_crit'],
        help="Run mode, optional. Default is 'abs'.",
    )
    parser.add_argument(
        "--fn_output",
        type=str,
        default='output.dat',
        help="Ouput file name for 'abd' and 'full_crit' run modes. Default 'output.dat'"
    )
    parser.add_argument(
        "--sph_avg_fock",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Spherically average the Fock matrix when running atomic block decomposition, optional. Default is True.",
    )
    parser.add_argument(
        "--symmetry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether symmetry adapted orbitals are used, optional. Default is False.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='output',
        help="Ouput directory path for 'abs' run mode. Default 'output'"
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Turn on debugging. Default is False.",
    )
    parser.add_argument(
        "--symmetry_aware_search",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optional feature, off by default. Make the adaptive-basis "
             "shell search (find_subspace) target the reference SCF's "
             "per-irrep occupation instead of the plain, symmetry-blind "
             "lowest-N-by-energy criterion it uses otherwise. Requires "
             "--point_group_file/--symmetry (a non-C1 point group) and "
             "--linkshells (the default).",
    )
    parser.add_argument(
        "--track_orbitals",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optional feature, off by default. Record the occupied "
             "orbital energies and their symmetry labels (irrep=empty if "
             "--symmetry_aware_search is off) at every ADB cycle, and save "
             "them to '<output_dir>/<fname>.orbitals.csv'.",
    )

    args = parser.parse_args()

    basis = np.asarray(args.basis)
    molpath = args.mpath
    dec = args.decontractions
    variant = args.var
    lshells = args.linkshells
    conv_tol = args.convtol
    sapbasis = args.sapbasis
    nfunc_norm = args.nfunc_normalisation
    sapbasis = [sapbasis] if isinstance(sapbasis, str) else sapbasis
    init_guesses = args.init_guesses
    dft = args.dft
    unit = args.unit
    abd_init = args.abd
    q_tol = args.q_tol
    sym_occ_file = args.sym_occ_file
    pnt_grp_file = args.point_group_file
    run_mode = args.run_mode
    fn_output = args.fn_output
    sph_avg_fock = args.sph_avg_fock
    symm = args.symmetry
    odir = args.output_dir
    debug = args.debug
    symmetry_aware_search = args.symmetry_aware_search
    track_orbitals = args.track_orbitals
    bs = []
    bstemp = []

    if len(basis) == 1 and os.path.isfile(basis[0]):
        f = open(basis[0])
        for line in f:
            bstemp.extend(line.strip("\n").split(" "))
    else:
        bstemp = basis
    for b in bstemp:
        if b[0] == "#":
            continue
        if b.replace("-", "").replace("unc", "") not in gto.basis.ALIAS.keys():
            print(f"Basis set {b} not found in PySCF! Will still try from BSE.")
            bs.append(b)
        else:
            bs.append(b)

    if sym_occ_file is not None:
        if not os.path.isfile(sym_occ_file):
            RuntimeError(f'Path {sym_occ_file} is not a valid file.')
    
    if pnt_grp_file is not None:
        if not os.path.isfile(pnt_grp_file):
            RuntimeError(f'Path {pnt_grp_file} is not a valid file.')
    
    mols = get_molecules_in_dir(
        molpath, bs, get_decontractions = dec, unit = unit,
        symmetry = symm,
        symmetry_fname = pnt_grp_file )

    if 'all' in init_guesses:
        init_guesses = AVAIL_INIT_METHODS

    for mol in mols:
        mol[4] = add_initial_guesses(init_guesses, mol[4])
    
    match run_mode:
        case 'abs':
            run_abs(
                mols,
                variant = variant,
                lshells = lshells,
                conv_tol = conv_tol,
                sap_basis_sets = sapbasis,
                nfunc_normalisation = nfunc_norm,
                dft = dft, abd_init = abd_init,
                symmetry_occ_fname = sym_occ_file,
                q_tol = q_tol,
                ODIR = odir,
                debug = debug,
                symmetry_aware_search = symmetry_aware_search,
                track_orbitals = track_orbitals,
                )
        case 'full_crit':
            compute_fullbasis_criterion(
                mols,
                variant=variant,
                dft=dft,
                sap_basis_sets=sapbasis,
                output_file=fn_output,
            )
