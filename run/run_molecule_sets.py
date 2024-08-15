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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run adaptive basis Hartree-Fock calculation."
    )
    parser.add_argument(
        "-m", "--mpath", type=str, required=True, help="path to molecule .xyz file"
    )
    parser.add_argument(
        "-b", "--basis", type=str, required=True, help="basis set"
    )
    
    args = parser.parse_args()

    molpath = args.mpath
    basis = args.basis

    mol = gto.M(
        atom = molpath,
        basis = basis,
        verbose = 0,
        spin=None
    )

    mol.verbose = 4
    mf = scf.HF(mol)
    start = time()
    dm0 = mf.get_init_guess(key='atom')
    end = time()
    t_dm0 = end - start
    start = time()
    F = mf.get_fock(dm=dm0)
    end = time()
    t_F = end - start

    mf.init_guess = 'atom'
    mf.sap_basis = 'sapgrasplarge'
    S = mf.get_ovlp()

    start = time()
    smask = adb.find_subspace(
        F, S, mol, mf,
        conv_tol=1e-4, get_smask=True, dm0=dm0,
        verbose=False, variant='enocc',#return_mask_history=True
    )
    end = time()
    mask = adb.smask_to_mask(smask, mol.cart)
    t_find_subspace = end - start

    start = time()
    mf.kernel()
    end = time()
    t_fullHF = end - start
    Efull = mf.e_tot

    submol = adb.create_shell_separated_mol(mol)
    submol._bas = submol._bas[[sm[0] for sm in smask]]
    submf = scf.HF(submol)
    submf.init_guess = 'atom'

    start = time()
    submf.kernel(dm0=dm0[:, mask][mask, :])
    subdm = submf.make_rdm1()
    end = time()
    t_subHF = end - start
    esub = submf.e_tot

    mol.verbose = 4
    mf = scf.HF(mol)
    dm0 = np.zeros_like(dm0)
    idx = np.where(mask)[0]
    for j,i in enumerate(idx):
        dm0[i][idx] = subdm[j]

    # mf.max_cycle = 0
    start = time()
    mf.kernel(dm0=dm0)
    end = time()
    Esubinit = mf.e_tot
    t_correction = end -start

    print('\n\n')
    print(f't_dm0: {t_dm0:.3f}')
    print(f't_F: {t_F:.3f}')
    print(f't_find_subspace: {t_find_subspace:.3f}')
    print('Number of shells in subspace:  ', np.count_nonzero([m[0] for m in smask]))
    print('Number of shells in full space:', len(smask))
    print(f't_fullHF: {t_fullHF:.3f}')
    print(f't_subHF: {t_subHF:.3f}')
    print(f't_correction: {t_correction:.3f}')

    print(f'ABS: {(t_dm0 + t_dm0 + t_F + t_find_subspace + t_subHF + t_correction):.3f}')
    print(f'E_full: {Efull:.8f}\t\t E_sub: {esub:.8f}')