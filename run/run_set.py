#!/bin/python3

import sys
import os
import argparse

import adb
from pyscf import scf, gto, dft
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
    molpath: str,
    basis_sets: list,
    get_decontractions: bool = False,
    unit='Angstrom'
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
        if fn.split("/")[-1][0] == "#":
            continue
        print(f"reading file {fn}")
        for bs in basis_sets:
            fnparts = fn.split('/')[-1].split('.')
            if len(fnparts) > 2:
                charge = [int(substring.replace('charge', '')) for substring in fnparts if 'charge' in substring]
                charge = charge[0] if len(charge) != 0 else 0
                spin = [int(substring.replace('spin','')) for substring in fnparts if 'spin' in substring]
                spin = spin[0] if len(spin) != 0 else None
            else:
                charge = 0
                spin = None
            mol = gto.M(
                atom=fn,
                basis=bs,
                charge=charge,
                spin=spin,
                unit=unit,
                verbose=0,
            )
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Run adaptive basis Hartree-Fock calculations for given set."
    )
    parser.add_argument(
        "--mpath", type=str, required=True,
        help="path to molecule directory"
    )
    parser.add_argument(
        "--basis", type=str, required=False, default='def2-TZVP', nargs='+',
        help="path to molecule directory"
    )
    parser.add_argument(
        "--unit", type=str, required=False, default='Angstrom',
        choices=['Angstrom', 'Bohr'],
        help="units of the xyz files"
    )
    parser.add_argument(
        "--conv_tol", type=float, required=False, default=1e-1,
        help="convergence tolerance, default 1e-1"
    )
    
    args = parser.parse_args()
    
    mpath = args.mpath
    basis = args.basis
    units = args.unit
    conv_tol = args.conv_tol

    mols = get_molecules_in_dir(mpath, basis, unit=units)
    with open('output.out', 'w') as f:
        for molfilename, mol, uncmol, shells, init_guess, basisname in mols:
            xcfunc = 'PBE'
            grid_level = 3

            mf = dft.KS(mol)
            mf.grids.level = grid_level
            mf.xc = xcfunc
            mf.grids.prune = None
            mf.init_guess = 'atom'

            mf.kernel()
            fullbasis_energy = mf.e_tot

            dm0 = mf.get_init_guess(key=mf.init_guess)
            S = mf.get_ovlp()
            F = mf.get_fock(dm=dm0)

            smaskhistory = adb.find_subspace(
                F, S, mol, mf, conv_tol=conv_tol,
                collect_data=False, get_smask=True,
                return_mask_history=True,
            )

            data_sbys = adb.mask_analysis(
                smaskhistory, mol, mf,
                F, S, verbose=True,
                dft=True, xc=xcfunc, grid_level=grid_level,
            )

            f.write(f'{molfilename.split(".")[0]:20s}\t{fullbasis_energy:.20f}\t{data_sbys[-1][3]:.20f}\n')




