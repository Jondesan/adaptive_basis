#!/bin/python3

import sys
import os
import argparse

sys.path.append("../adbmodule/")
import adb
from pyscf import scf, gto
import numpy as np
import numpy
import pandas as pd
import datetime
from time import time

def write_info(msg: str, fpath: str):
    with open(fpath + '/info.txt', 'w') as f:
        msg = msg.replace(r'\n', '\n')
        for line in msg.split('\n'):
            f.write(f'{line}\n')

    return


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
    parser.add_argument(
        "-c", "--cutoff", type=float, required=False, default=None,
        help="mask cutoff percentage for ABS, must be between [0.0, 1.0]"
    )
    parser.add_argument(
        "-e", "--ecutoff", type=float, required=False, default=1e-1,
        help="energy cutoff for ABS"
    )
    parser.add_argument(
        "-u", "--unit", type=str, required=False, default='Angstrom',
        choices=['Angstrom', 'Bohr'],
        help="coordinate units of xyz file"
    )
    parser.add_argument(
        "--out", type=str, required=True,
        help="path to output directory"
    )
    parser.add_argument(
        "--info", type=str, required=False, default='No info provided.',
        help="message to write to info.txt"
    )
    parser.add_argument(
        "--id", type=str, required=True,
        help="identifying string to add to molecule directory to separate data from different runs"
    )
    parser.add_argument(
        "--soscf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="whether to run second order SCF, optional",
    )

    
    args = parser.parse_args()

    molpath = args.mpath
    basis = args.basis
    cutoff = args.cutoff
    ecutoff = args.ecutoff
    unit = args.unit
    outpath = args.out
    info = args.info
    id = args.id
    SOSCF = args.soscf

    if cutoff is not None and (cutoff < 0.0 or cutoff > 1.0):
        raise argparse.ArgumentTypeError('Value of cutoff has to be between 0 and 1')
    if not os.path.isdir(outpath):
        raise ValueError(f'Path {outpath} not a directory!')
    
    molname = molpath.split('/')[-1].split('.')[0]
    datadir = outpath + f'/{molname}.{id}'
    if os.path.isdir(datadir):
        raise RuntimeError(f'The directory {datadir} exists already!')
    os.mkdir(datadir)

    write_info(info, datadir)

    spin = charge = None

    if 'spin' in molpath:
        spin = molpath.split('/')[-1].split('.')
        spin = list(filter(lambda x: 'spin' in x, spin))[0]
        spin = int(spin[-1])
    if 'charge' in molpath:
        charge = molpath.split('/')[-1].split('.')
        charge = list(filter(lambda x: 'charge' in x, charge))[0]
        charge = int(charge[-1])
    mol = gto.M(
        atom = molpath,
        basis = basis,
        verbose = 0,
        spin=spin,
        charge=charge,
        unit=unit
    )

    mol.verbose = 4
    with open(datadir + '/fullHF.dat', 'w') as fullHFfile:
        mol.stdout = fullHFfile
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
            conv_tol=ecutoff, get_smask=True, dm0=dm0,
            verbose=False, variant='enocc',
            mask_cutoff=cutoff,
        )
        end = time()
        mask = adb.smask_to_mask(smask, mol.cart)
        t_find_subspace = end - start

        start = time()
        # mf.kernel(dm0=dm0)
        # mf.mol.verbose = 4
        if SOSCF:
            # guess Fock
            F = mf.get_fock(dm = dm0)
            # build orbitals
            guess_orbE, guess_orbs = mf.eig(F, S)
            guess_occs = mf.get_occ(guess_orbE, guess_orbs)
            # initialize mf orbs
            mf.mo_coeff = guess_orbs
            mf.mo_energy = guess_orbE
            mf = mf.newton()
            mf.kernel()
        else:
            mf.kernel()
        end = time()
    t_fullHF = end - start
    Efull = mf.e_tot

    submol = adb.create_shell_separated_mol(mol)
    submol.verbose = 4
    submol._bas = submol._bas[[sm[0] for sm in smask]]

    with open(datadir + '/subHF.dat', 'w') as subHFfile:
        submol.stdout = subHFfile
        
        submf = scf.HF(submol).newton()
        submf.init_guess = 'atom'

        start = time()
        submf.kernel(dm0=dm0[:, mask][mask, :])
        subdm = submf.make_rdm1()
        end = time()
    t_subHF = end - start
    esub = submf.e_tot

    with open(datadir + '/sub_initHF.dat', 'w') as subinitHFfile:
        mol.stdout = subinitHFfile
        mf = scf.HF(mol)
        dm0 = np.zeros_like(dm0)
        idx = np.where(mask)[0]
        for j,i in enumerate(idx):
            dm0[i][idx] = subdm[j]

        start = time()
        if SOSCF:
            # guess Fock
            F = mf.get_fock(dm = dm0)
            # build orbitals
            guess_orbE, guess_orbs = mf.eig(F, S)
            guess_occs = mf.get_occ(guess_orbE, guess_orbs)
            # initialize mf orbs
            mf.mo_coeff = guess_orbs
            mf.mo_energy = guess_orbE
            mf = mf.newton()
            mf.kernel()
        else:
            mf.kernel(dm0=dm0)
        end = time()
    Esubinit = mf.e_tot
    t_subinit = end - start

    with open(datadir + '/summary.dat', 'w') as f:
        f.write(f't_dm0: {t_dm0:.3f}\n')
        f.write(f't_F: {t_F:.3f}\n')
        f.write(f't_find_subspace: {t_find_subspace:.3f}\n')
        f.write(f'Number of shells in subspace:   {np.count_nonzero([m[0] for m in smask])}\n')
        f.write(f'Number of shells in full space: {len(smask)}\n')
        f.write(f't_fullHF: {t_fullHF + t_dm0:.3f}\n')
        f.write(f't_subHF: {t_subHF:.3f}\n')
        f.write(f't_subinitHF: {t_subinit:.3f}\n')
        f.write(f'ABS: {(t_dm0 + t_dm0 + t_F + t_find_subspace + t_subHF + t_subinit):.3f}\n')
        f.write(f'E_full: {Efull:.8f}\n')
        f.write(f'E_sub: {esub:.8f}\n')