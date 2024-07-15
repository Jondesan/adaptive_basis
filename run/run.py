#!/bin/python3

import sys
import os
import argparse

sys.path.append("../adbmodule/")
import adb
from pyscf import scf, gto
import numpy as np
import pandas as pd
import datetime
from time import time


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
    dirpath: str, basis_sets: list, get_decontractions: bool = False
):
    prefix = dirpath
    fs = get_files_in_folder(prefix)
    fs = [prefix + f for f in fs]
    molecules = []
    for fn in fs:
        if fn.split("/")[-1][0] == "#":
            continue
        print(f"reading file {fn}")
        for bs in basis_sets:
            for unc in (
                ["", "unc-"] if get_decontractions and "unc-" not in bs else [""]
            ):
                mol = gto.M(
                    atom=fn,
                    basis=unc + bs,
                    verbose=0,
                )
                smask = adb.init_smask(mol)
                molecules.append(
                    [fn.split("/")[-1], mol, adb.create_shell_separated_mol(mol), smask, None]
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


def run_abs(mol_list, variant=0, lshells=True, conv_tol=1e-4):
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

    for molfilename, mol, uncmol, shells, init_guess in mol_list:
        # Open the output file
        bsname = mol.basis
        molname = molfilename.split(".")[0]
        if len(bsname) > 25:
            bsname = "basis_NA"

        # Set up Hartree-Fock, remove linear dependencies from basis
        myhf = mol.HF().apply(scf.addons.remove_linear_dep_)
        
        if init_guess is None:
            init_guess = ['minao']

        for ig in init_guess:
            with open(f'output/{".".join([molname, bsname, ig])}.out', "w") as f:
                f.write("{:<15s} {:<15s} {:<15s} {:<15s}\n".format("molecule", "basis_set", "variant", "init_guess"))
                f.write(f"{molname:<15s} {bsname:<15s} {variant:<15d} {ig:<15s}\n")
                f.write(f"Calculations done on {datetime.datetime.now()}\n\n")

                start = time()
                myhf.kernel()
                if ig == 'SCF':
                    dm0 = None
                elif ig == 'sap':
                    # dm0 = adb.init_guess_by_sap(mol, sapbs_path='/home/joonahuh/uni/electronic_structure/bs/laikov_hf.nw')
                    myhf.sap_basis = '/home/joonahuh/uni/electronic_structure/bs/laikov_hfs.nw'
                    dm0 = myhf.init_guess_by_sap(mol)#, sap_basis='/home/joonahuh/uni/electronic_structure/bs/laikov_hfs.nw')
                else:
                    dm0 = myhf.get_init_guess(key=ig)
                end = time()

                F = myhf.get_fock()#dm=dm0)
                initF = myhf.get_fock(dm=dm0)
                S = myhf.get_ovlp()
                mo_energy, mo_coeff = adb.eigh(initF, S)
                nocc = np.count_nonzero(myhf.get_occ(mo_energy, mo_coeff))
                f.write("time stats [s]\n")
                f.write("{:<17s}{:<17s}{:<17s}\n".format("t_HF", "t_fbyf", "t_sbys"))
                f.write(f"{end-start:15.9e}  ")

                if variant == 0:
                    start = time()
                    _, data_fbyf = adb.find_subspace(
                        F,
                        S,
                        mol,
                        myhf,
                        conv_tol=conv_tol,
                        collect_data=True,
                        variant=variant,
                        dm0=dm0
                    )
                    end = time()

                if variant == 0:
                    f.write(f"{end-start:15.9e}  ")
                else:
                    f.write("{:<15s}".format("-"))

                start = time()
                smask, data_sbys = adb.find_subspace(
                    F,
                    S,
                    mol,
                    myhf,
                    conv_tol=conv_tol,
                    collect_data=True,
                    get_smask=True,
                    variant=variant,
                    link_shells=lshells,
                    dm0=dm0
                )
                end = time()

                f.write(f"{end-start:15.9e}\n\n")

                if variant == 0:
                    df_fbyf = pd.DataFrame(data_fbyf, columns=datacols)
                df_sbys = pd.DataFrame(data_sbys, columns=datacols)

                f.write("{:<15s} {:<15s} {:<15s}\n".format("N_occ", "E_HF", "nfunc"))
                f.write(
                    "{:<15d} {:<15f} {:<15d}\n\n".format(
                        nocc, myhf.e_tot, mol.nao_nr()
                    )
                )

                if variant == 0:
                    f.write("function-by-function iteration\n")
                    df_fbyf.to_csv(f, index=False)
                    f.write("\n\n")

                f.write("shell-by-shell iteration\n")
                df_sbys.to_csv(f, index=False)
                f.write("\n\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run adaptive basis Hartree-Fock calculations."
    )
    parser.add_argument(
        "--mpath", type=str, required=True, help="path to molecule directory"
    )
    parser.add_argument(
        "--bpath", type=str, required=True, help="path to basis input file"
    )
    parser.add_argument(
        "-c", "--convtol", type=float, required=False, default=1e-4,
        help="path to basis input file"
    )
    parser.add_argument(
        "--decontractions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="whether to run decontracted calculations too, optional.",
    )
    parser.add_argument(
        "--var",
        type=int,
        required=False,
        default=0,
        choices=[0, 1, 2],
        help="which minimisation criteria to use, optional. Default is 0",
    )
    parser.add_argument(
        "--linkshells",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Turn duplicate shell linking on/off during shell by shell calculations.",
    )

    args = parser.parse_args()

    basispath = args.bpath
    molpath = args.mpath
    dec = args.decontractions
    variant = args.var
    lshells = args.linkshells
    conv_tol = args.convtol
    bs = []
    bstemp = []
    f = open(basispath)
    for line in f:
        bstemp.extend(line.strip("\n").split(" "))
    for b in bstemp:
        if b[0] == "#":
            continue
        if b.replace("-", "") not in gto.basis.ALIAS.keys():
            print(f"Basis set {b} not found in PySCF!")
        else:
            bs.append(b)
    mols = get_molecules_in_dir(molpath, bs, get_decontractions=dec)
    for mol in mols:
        mol[4] = add_initial_guesses(
            ['atom', 'sap', 'SCF'],
            mol[4]
        )
    # print(mols)
    run_abs(mols, variant=variant, lshells=lshells, conv_tol=conv_tol)
