#!/bin/python3

import sys
import os
import copy
import argparse

import adb
from pyscf import scf, gto
import numpy as np
import pandas as pd
import datetime
from time import time
import adbutils

AVAIL_INIT_METHODS = [
    'SCF',
    'atom',
    'sap',
    'huckel',
    'vsap'
]

def get_files_in_folder(folder: str):
    """Get all files in folder.

    Args:
        folder : str
            The folder to search for files in.

    Returns:
        List of files in the folder.
    """
    files = os.listdir(folder)
    return files


def get_molecules_in_dir(
    molpath: str,
    basis_sets: list,
    get_decontractions: bool = False,
    unit='Angstrom',
    symmetry_fname=None
):
    """Get molecule xyz files from molpath, can be directory or single file.
    """
    prefix = molpath


    if os.path.isdir(prefix):
        fs = get_files_in_folder(prefix)
        fs = [prefix + '/' + f for f in fs]
    else:
        fs = [molpath]
    molecules = []
    for fn in fs:
        molfname = fn.split("/")[-1]
        if molfname[0] == "#": # If mol fname starts with #, skip file
            continue
        print(f"reading file {fn}")

        if symmetry_fname is not None:
            irrep_occs, symm = adbutils.read_symmetry_occs_from_file(
                symmetry_fname, molfname=molfname)
        else:
            symm=True

        for bs in basis_sets:
            for unc in (
                ["", "unc-"] if get_decontractions and "unc-" not in bs else [""]
            ):
                fnparts = fn.split('/')[-1].split('.')
                if len(fnparts) > 2:
                    charge = [int(substring.replace('charge', '')) for substring in fnparts if 'charge' in substring]
                    charge = charge[0] if len(charge) != 0 else 0
                    spin = [int(substring.replace('spin','')) for substring in fnparts if 'spin' in substring]
                    spin = spin[0] if len(spin) != 0 else None
                else:
                    charge = 0
                    spin = None
                try:
                    mol = gto.M(
                        atom=fn,
                        basis=unc + bs,
                        ecp=unc + bs,
                        charge=charge,
                        spin=spin,
                        unit=unit,
                        symmetry=symm,
                        verbose=0,
                    )
                except:
                    mol = gto.M(
                        atom=fn,
                        basis=unc + bs,
                        charge=charge,
                        spin=spin,
                        unit=unit,
                        symmetry=symm,
                        verbose=0,
                    )
                # if symmetry_fname is not None:
                #     mol.irrep_name = list(irrep_occs.keys())
                mol = adb.create_shell_separated_mol(mol, verbose=mol.verbose)
                smask = adb.init_smask(mol)
                molecules.append(
                    [fn.split("/")[-1], mol, adb.create_shell_separated_mol(mol), smask, None, bs]
                )

    # Sort by number of electrons, then by the basis, then by number of basis fcts
    molecules.sort(key=lambda x: (x[1].tot_electrons(), x[1].basis, x[1].nao_nr()))
    print(
        f"read a total of {len(molecules)} molecular structures, with the following numbers of functions: {[int(m[1].nao_nr()) for m in molecules]}"
    )
    print(f"with filenames {[name[0] for name in molecules]}")
    return molecules


def add_initial_guesses(ig_list, mol_list):
    if mol_list is None:
        mol_list = ig_list
    else:
        mol_list.extend(ig_list)

    return mol_list


def run_abs(
    mol_list,
    variant='enocc',
    lshells=True,
    conv_tol=1e-4,
    sap_basis_sets='sapgraspsmall',
    nfunc_normalisation=True,
    dft=False,
    xc='b3lyp',
    grid_level=7,
    abd_init=True,
    use_psi4=True,
    symmetry_occ_fname=None,
    q_tol=1.0
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

    for molfilename, mol, shellsep_mol, shells, init_guess, basisname in mol_list:
        # Open the output file
        bsname = basisname#mol.basis
        molname = molfilename.split(".")[0]
        charge = mol.charge
        spin = mol.spin
        if len(bsname) > 25:
            bsname = "basis_NA"
        
        if init_guess is None:
            init_guess = ['atom']

        if not os.path.isdir('output'):
            os.mkdir('output')

        is_restricted = mol.spin == 0

        if use_psi4:
            _, docc, socc, wfn_full, irrep_labels, irrep_symb = adbutils.psi4_fullbasis(
                mol,
                basis=basis,
                init_guess=init_guess,
                dft=dft, xc=xc
            )
            AOCC = wfn_full.nalphapi().to_tuple()
            BOCC = wfn_full.nbetapi().to_tuple()
            symmetry_occs = list(zip(irrep_labels, AOCC, BOCC))
            # Create symmetry occupation dict
            #             IRREP: 2*alpha                      (alpha, beta)
            # Example:    'A1':  1                            (2, 2)
            irrep_nelec = {x[0]: 2*x[1] if is_restricted else (x[1], x[2]) for x in symmetry_occs}
        if symmetry_occ_fname is not None:
            irrep_nelec, _ = adbutils.read_symmetry_occs_from_file(
                symmetry_occ_fname, molfname=molfilename)
        
        # Set up Hartree-Fock, remove linear dependencies from basis
        if is_restricted:
            myhf = mol.RHF()
        else:
            myhf = mol.UHF()#.apply(scf.addons.remove_linear_dep_)
        myhf = myhf.apply(scf.addons.remove_linear_dep_)
        if dft:
            myhf = myhf.to_ks()
            myhf.xc = xc
            myhf.grids.level = grid_level
            myhf.grids.prune = None
        myhf.eig = adb.eigh


        start = time()
        # Set the symmetry occupations if present
        if irrep_nelec is not None:
            myhf.irrep_nelec = {}
            for key in irrep_nelec:
                if irrep_nelec[key] != 0:
                    if key not in myhf.mol.irrep_name:
                        raise RuntimeError(f'irrep {key} not found in subbasis')
                    myhf.irrep_nelec[key] = irrep_nelec[key]
        else:
            print('Symmetry occupations not set explicitly! This may cause convergence issues.', file=sys.stderr)

        myhf.kernel()
        end = time()
        fullbasis_hf_time = end - start
        F_scf = myhf.get_fock()

        # Save the SCF matrices
        mo_coeff_scf = copy.deepcopy(myhf.mo_coeff)
        mo_energy_scf = copy.deepcopy(myhf.mo_energy)
        mo_occ_scf = copy.deepcopy(myhf.mo_occ)

        for ig in init_guess:
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
                with open(f'output/{fname}.out', "w") as f:
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

                    dm0=None
                    start = time()
                    if ig == 'SCF':
                        F = F_scf
                    else:
                        if ig == 'vsap':
                            tempmf = mol.KS().set(xc='b3lyp')
                            dm0 = tempmf.get_init_guess(key='vsap')
                        else:
                            myhf.sap_basis = sapbs
                            dm0 = myhf.get_init_guess(key=ig)
                        F = myhf.get_fock(dm=dm0)
                    end = time()
                    S = myhf.get_ovlp()
                    f.write("time stats [s]\n")
                    f.write("{:<17s}{:<17s}{:<17s}\n".format("t_HF", "t_fbyf", "t_sbys"))
                    f.write(f"{(fullbasis_hf_time + end - start):15.9e}  ")

                    if variant == 'enocc':
                        start = time()
                        # maskhistory = adb.find_subspace(
                        #     F, S, mol, myhf,
                        #     conv_tol=conv_tol,
                        #     variant=variant,
                        #     return_mask_history=True,
                        #     nfunc_normalisation=nfunc_normalisation,
                        #     abd_initialization=abd_init
                        # )
                        # data_fbyf = adb.mask_analysis(
                        #     maskhistory, mol, myhf,
                        #     F, S
                        # )
                        end = time()

                    if variant == 'enocc':
                        f.write(f"{end-start:15.9e}  ")
                    else:
                        f.write("{:<15s}".format("-"))

                    start = time()
                    print(adbutils.get_uncontracted_basis(mol))
                    smaskhistory = adb.find_subspace(
                        F, S, mol, myhf,
                        conv_tol=conv_tol,
                        get_smask=True,
                        variant=variant,
                        link_shells=lshells,
                        nfunc_normalisation=nfunc_normalisation,
                        return_mask_history=True,
                        abd_initialization=abd_init,
                        abd_Q_tol=q_tol
                    )
                    data_sbys = adb.mask_analysis(
                        smaskhistory, shellsep_mol, myhf,
                        #F_scf, S, dft=dft, xc=xc, grid_level=grid_level,
                        F, S, dft=dft, xc=xc, grid_level=grid_level,
                        use_psi4=use_psi4, molfname=molfilename,
                        sym_occ_fname=symmetry_occ_fname,
                        C_full=mo_coeff_scf
                    )
                    end = time()

                    e_tot = myhf.e_tot
                    if use_psi4:
                        e_tot_psi4, docc_full, socc_full, wfn_full, irrep_labels = adbutils.psi4_fullbasis(
                            mol,
                            basis=basisname,
                            init_guess=myhf.init_guess,
                            dft=dft, xc=xc
                        )
                        e_tot = e_tot_psi4
                    f.write(f"{end-start:15.9e}\n\n")

                    # if variant == 'enocc':
                    #     df_fbyf = pd.DataFrame(data_fbyf, columns=datacols)
                    df_sbys = pd.DataFrame(data_sbys, columns=datacols)

                    f.write("{:<15s} {:<30s} {:<15s}\n".format("N_occ", "E_HF", "nfunc"))
                    f.write(
                        "{:<15d} {:<30.20f} {:<15d}\n\n".format(
                            np.sum(mol.nelec), e_tot, mol.nao_nr()
                        )
                    )

                    # if variant == 'enocc':
                        # f.write("function-by-function iteration\n")
                        # df_fbyf.to_csv(f, index=False)
                        # f.write("\n\n")

                    f.write("shell-by-shell iteration\n")
                    df_sbys.to_csv(f, index=False)
                    f.write("\n\n")


def run_occupations(
    mol_list,
    dft=False,
    xc='b3lyp',
    grid_level=7):

    print('molfilename;occs;irrep_symbol')
    for molfilename, mol, shellsep_mol, shells, init_guess, basisname in mol_list:
        print(molfilename, end=';')
        
        init_guess = 'atom'

        # Set up Hartree-Fock, remove linear dependencies from basis
        is_restricted = mol.spin == 0

        _, docc, socc, wfn_full, irrep_labels, irrep_symb = adbutils.psi4_fullbasis(
            mol,
            basis=basis,
            init_guess=init_guess,
            dft=dft, xc=xc
        )
        AOCC = wfn_full.nalphapi().to_tuple()
        BOCC = wfn_full.nbetapi().to_tuple()
        symmetry_occs = list(zip(irrep_labels, AOCC, BOCC))
        # Create symmetry occupation dict
        #             IRREP: 2*alpha                      (alpha, beta)
        # Example:    'A1':  1                            (2, 2)
        irrep_nelec = {x[0]: 2*x[1] if is_restricted else (x[1], x[2]) for x in symmetry_occs}

        print(f'{irrep_nelec};{irrep_symb.capitalize().rstrip()}')

    return None


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
        "-u", "--unit", type=str, required=False, default='Angstrom',
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
        default=True,
        help="Initialize with atomic block decomposition, optional. Default is True.",
    )
    parser.add_argument(
        "--use_psi4",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Psi4 for SCF calculations instead of PySCF, optional. Default is True.",
    )
    parser.add_argument(
        "--q_tol", type=float, required=False, default=1.0,
        help="charge tolerance, default 1.0"
    )
    parser.add_argument(
        "--sym_occ_file", type=str, required=False, default='occupations.dat',
        help="path to file with required symmetry occupations."
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
    use_psi4 = args.use_psi4
    q_tol = args.q_tol
    sym_occ_file = args.sym_occ_file
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
    mols = get_molecules_in_dir(
        molpath, bs, get_decontractions=dec, unit=unit,
    symmetry_fname='occupations.dat')
    # print(mols)
    if 'all' in init_guesses:
        init_guesses = AVAIL_INIT_METHODS
    for mol in mols:
        mol[4] = add_initial_guesses(init_guesses, mol[4])
    
    if not os.path.isfile(sym_occ_file):
        RuntimeError(f'Path {sym_occ_file} is not a valid file.')
    # run_occupations(
    #     mols,
    #     dft=dft, xc='b3lyp')
    run_abs(
        mols,
        variant=variant,
        lshells=lshells,
        conv_tol=conv_tol,
        sap_basis_sets=sapbasis,
        nfunc_normalisation=nfunc_norm,
        dft=dft, abd_init=abd_init,
        use_psi4=use_psi4,
        symmetry_occ_fname=sym_occ_file,
        q_tol=q_tol
        )
