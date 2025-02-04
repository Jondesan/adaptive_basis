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
    molpath: list,
    basis_sets: list,
    get_decontractions: bool = False,
    unit='Angstrom'
):
    """Get molecule xyz files from molpath, can be directory or single file.
    """
    if len(molpath) == 1:
        prefix = molpath[0]
        if os.path.isdir(prefix):
            fs = get_files_in_folder(prefix)
            fs = [prefix + '/' + f for f in fs]
        else:
            fs = [prefix]
    else:
        fs = molpath
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
        "--mpath", type=str, required=True, nargs='+',
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
    parser.add_argument(
        "--q_tol", type=float, required=False, default=.5,
        help="charge tolerance, default .5"
    )
    parser.add_argument(
        "--normalisation", type=int, required=False, default=1,
        choices=[0,1],
        help="whether to use normalisation"
    )
    parser.add_argument(
        "--output", type=str, required=False, default='output.out',
        help="name of output file"
    )
    parser.add_argument(
        "--dft",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use DFT, optional. Default is True.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether output is robust or not, optional. Default is True.",
    )

    args = parser.parse_args()
    
    mpath = args.mpath
    basis = args.basis
    units = args.unit
    conv_tol = args.conv_tol
    q_tol = args.q_tol
    normalisation = args.normalisation
    output = args.output
    run_dft = args.dft
    verbose = args.verbose

    mols = get_molecules_in_dir(mpath, basis, unit=units)
    with open(output, 'w') as f:
        for molfilename, mol, uncmol, shells, init_guess, basisname in mols:
            xcfunc = 'PBE'
            grid_level = 3

            # mf = dft.RKS(mol) if run_dft else scf.RHF(mol)
            mf = dft.KS(mol) if run_dft else scf.HF(mol)
            if run_dft:
                mf.grids.level = grid_level
                mf.xc = xcfunc
                mf.grids.prune = None
            
            start = time()
            mf.kernel(init_guess='atom')
            end = time()
            fullbasis_time = end - start

            fullbasis_energy = mf.e_tot

            dm0 = mf.get_init_guess(key=mf.init_guess)
            S = mf.get_ovlp()
            F = mf.get_fock(dm=dm0)
            start = time()
            smaskhistory = adb.find_subspace(
                F, S, mol, mf, conv_tol=conv_tol,
                collect_data=False, get_smask=True,
                return_mask_history=True,
                nfunc_normalisation=normalisation,
                abd_Q_tol=q_tol,
                verbose=verbose,
            )
            end = time()
            subbasis_time = end - start

            data_sbys = adb.mask_analysis(
                smaskhistory, mol, mf,
                F, S, verbose=verbose,
                dft=run_dft, xc=xcfunc, grid_level=grid_level,
            )
            # print(f'{data_sbys=}')

            subbasis_mol = adb.create_shell_separated_mol(mol)
            mask = [sm[0] for sm in smaskhistory[-1][0]]
            newbas = mol._bas[mask]
            subbasis_mol._bas = newbas
            submf = dft.KS(subbasis_mol) if run_dft else scf.HF(subbasis_mol) 
            if run_dft:
                submf.grids.level = grid_level
                submf.xc = xcfunc
                submf.grids.prune = None
            # submf.init_guess = 'atom'

            mf = dft.KS(mol) if run_dft else scf.HF(mol)
            if run_dft:
                mf.grids.level = grid_level
                mf.xc = xcfunc
                mf.grids.prune = None
            # mf.init_guess = 'atom'

            # mask = adb.smask_to_mask(smaskhistory[-1][0])
            # start = time()
            # submf.kernel(init_guess='atom')
            # dm0 = submf.make_rdm1()
            # initdm = np.zeros_like(S)

            # idx = np.where(mask)[0]
            # for j,i in enumerate(idx):
            #     initdm[i][idx] = dm0[j]
            # mf.kernel(dm0=initdm, init_guess='atom')
            # end = time()

            # subinit_time = end - start

            func_mask = adb.smask_to_mask(smaskhistory[-1][0], mol.cart)
            f.write(f'{molfilename.split(".")[0]:20s}\t{fullbasis_energy:.20f}\t{data_sbys[-1][3]:.20f}\t')
            f.write(f'{subbasis_time:.4f}\t{fullbasis_time:.4f}\t')#{subinit_time:.4f}\t')
            f.write(f'{np.sum(func_mask)}\t{len(func_mask)}\n')
            f.flush()

            print(f'{molfilename.split(".")[0]:20s}\t{fullbasis_energy:.20f}\t{data_sbys[-1][3]:.20f}\t', end='')
            print(f'{subbasis_time:.4f}\t{fullbasis_time:.4f}\t', end='')#{subinit_time:.4f}\t')
            print(f'{np.sum(func_mask)}\t{len(func_mask)}\n')


