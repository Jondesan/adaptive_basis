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
                # Parse initial guess
                ig = [substring.replace('init_','') for substring in fnparts if 'init_' in substring]
                ig = ig[0] if len(ig) != 0 else 'atom'
            else:
                charge = 0
                spin = None
                ig = 'atom'
            try:
                mol = gto.M(
                    atom=fn,
                    basis=bs,
                    ecp=bs,
                    charge=charge,
                    spin=spin,
                    unit=unit,
                    verbose=0,
                )
                print(mol.unit)
            except:
                print('running except...')
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
                [fn.split("/")[-1], mol, adb.create_shell_separated_mol(mol), smask, ig, bs]
            )

    # Sort by number of electrons, then by the basis, then by number of basis fcts
    molecules.sort(key=lambda x: (x[1].tot_electrons(), x[1].basis, x[1].nao_nr()))
    print(
        f"read a total of {len(molecules)} molecular structures, with the following numbers of functions: {[int(m[1].nao_nr()) for m in molecules]}"
    )
    print(f"with filenames {[name[0] for name in molecules]}")
    return molecules


def run_wa_set(args):
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
    
    # print(f'\n{mols[264]=}')
    # print(f'{mols[265]=}')
    with open(output, 'w', buffering=1) as f:
        for molfilename, mol, uncmol, shells, init_guess, basisname in mols:
            xcfunc = 'PBE'
            grid_level = 7

            mf = dft.KS(mol) if run_dft else scf.HF(mol)
            if run_dft:
                mf.grids.level = grid_level
                mf.xc = xcfunc
                mf.grids.prune = None
            
            # Initialize init guess method
            mf.init_guess = init_guess

            # This produces the SAD density matrix
            dm0 = mf.get_init_guess(key=init_guess)
            # we need the corresponding Fock matrix
            F = mf.get_fock(dm=dm0)
            S = mf.get_ovlp()
            mf.mo_energy, mf.mo_coeff = mf.eig(F, S)

            
            start = time()
            smaskhistory = adb.find_subspace(
                F, S, mol, mf, conv_tol=conv_tol,
                collect_data=False, get_smask=True,
                return_mask_history=True,
                nfunc_normalisation=normalisation,
                abd_Q_tol=q_tol,
                dft=run_dft, xc=xcfunc, grid_level=grid_level,
                verbose=verbose,
            )
            end = time()
            subbasis_time = end - start

            subbasis_mol = adb.create_shell_separated_mol(mol)
            # adb.tk_debugger(smaskhistory[-1][0])
            # extracted_basis, ecp_bas = adb.extract_basis(smaskhistory[-1][0], adb.create_shell_separated_mol(subbasis_mol))
            # adb.basis_to_file_nwchem(
            #     extracted_basis, f'{molfilename}_output_basis_new.nw', ecp_basis=ecp_bas,
            #     commentstring='Test basis for the atomic block decomp initialized algorithm.')
            # print('Created the subbasis, output to file', f'{molfilename}_output_basis_new.nw')
            
            start = time()
            mf.kernel(dm0=dm0)
            end = time()
            fullbasis_time = end - start
            fullbasis_converged = mf.converged
            fullbasis_energy = mf.e_tot

            data_sbys = adb.mask_analysis(
                smaskhistory, mol, mf,
                F, S, verbose=verbose,
                dft=run_dft, xc=xcfunc, grid_level=grid_level,
            )
            # print(f'{data_sbys=}')

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
            submf.kernel(dm0=adb.mask_matrix(dm0, mask))
            subbasis_converged = submf.converged
            # dm0 = submf.make_rdm1()
            # initdm = np.zeros_like(S)

            # idx = np.where(mask)[0]
            # for j,i in enumerate(idx):
            #     initdm[i][idx] = dm0[j]
            # mf.kernel(dm0=initdm, init_guess='atom')
            # end = time()

            # subinit_time = end - start


            func_mask = adb.smask_to_mask(smaskhistory[-1][0], mol.cart)
            diff = data_sbys[-1][3] - fullbasis_energy
            f.write(f'{molfilename.split(".")[0]:20s}\t{fullbasis_energy:.20f}\t{data_sbys[-1][3]:.20f}\t')
            f.write(f'{diff:.8f}\t')#{subbasis_time:.4f}\t{fullbasis_time:.4f}\t')#{subinit_time:.4f}\t')
            f.write(f'{np.sum(func_mask)}\t{len(func_mask)}\t')
            f.write(f'{fullbasis_converged}\t{subbasis_converged}\t{init_guess}\n')

            print(f'{molfilename.split(".")[0]:20s}\t{fullbasis_energy:.20f}\t{data_sbys[-1][3]:.20f}\t', end='')
            print(f'{diff:.8f}\t', end='')#{subbasis_time:.4f}\t{fullbasis_time:.4f}\t', end='')#{subinit_time:.4f}\t')
            print(f'{np.sum(func_mask)}\t{len(func_mask)}\t')
            print(f'{fullbasis_converged}\t{subbasis_converged}\t{init_guess}\n', flush=True)


def run_multiplicities(args):
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

    print('Molecule_filename', '2S', 'converged', 'e_tot')
    for molfilename, mol, uncmol, shells, init_guess, basisname in mols:
        # for spin in [6,7,8]:
        for spin in [0,1,2,3,4,5]:
            mol.spin = spin
            if run_dft:
                mol.verbose = 4
                mf = mol.KS()
                mf.grids.level = 7
                mf.xc = 'PBE'
                mf.grids.prune = None
            else:
                mf = mol.HF()
            mf.init_guess = 'sap'
            try:
                mf.kernel()
                print(molfilename.split('/')[-1], spin, mf.converged, mf.e_tot, flush=True)
            except:
                print(molfilename.split('/')[-1], spin, 'inconsistent', '-', flush=True)

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
    
    run_wa_set(args)
    # run_multiplicities(args)


