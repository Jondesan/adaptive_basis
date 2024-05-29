#!/bin/python3
abs_path = '/home/joonahuh/uni/electronic_structure/'

import sys
sys.path.append(abs_path + 'pyscf-master/')
sys.path.append(abs_path)
from jondelys.analysis.comm import send_notification
from jondelys.sbi import *
import pyscf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from time import time
import os

def get_files_in_folder(folder:str):
    '''Get all files in folder.

    Args:
        folder : str
            The folder to search for files in.
    
    Returns:
        List of files in the folder.
    '''
    files = os.listdir(folder)
    return files

def get_molecules_in_dir(
    dirpath: str, basis_sets: list,
    get_decontractions: bool = False
    ):
    
    prefix = dirpath
    fs = get_files_in_folder(prefix)
    fs = [prefix + f for f in fs]
    molecules = []
    for fn in fs:
        print(f'reading file {fn}')
        if fn.split('/')[-1][0] == '#':
            continue
        for bs in basis_sets:
            for unc in ['', 'unc-'] if get_decontractions and 'unc-' not in bs else ['']:
                mol = pyscf.M(
                    atom = fn,
                    basis = unc + bs,
                    verbose = 0,
                )
                smask = init_smask(mol)
                molecules.append([fn.split('/')[-1], mol, create_uncontracted_molecule_copy(mol), smask])

    # Sort by number of electrons, then by the basis, then by number of basis fcts
    molecules.sort(key=lambda x: (x[1].tot_electrons(), x[1].basis, x[1].nao_nr()))
    print(f'read a total of {len(molecules)} molecular structures, with the following numbers of functions: {[m[1].nao_nr() for m in molecules]}')
    print(f'with filenames {[name[0] for name in molecules]}')
    return molecules

def run_sbi(mol_list, send_tg_notif=False):
    """Run subbasis iteration for molecules in mol_list
    """
    
    datacols = ['nfunc', 'cursum', 'diff', 'E_scf', 'E_orb', 'Qsqrd', 'smask']
    dataframe = []
    '''
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
    '''

    for molfilename, mol, uncmol, shells in mol_list:
        # Open the output file
        bsname = mol.basis
        molname = molfilename.split(".")[0]
        if len(bsname) > 25:
            bsname = 'basis_NA'

        f = open(abs_path + f'run/output/{".".join([molname, bsname])}.out', 'w')
        f.write('{:<15s} {:<15s}\n'.format('molecule', 'basis_set'))
        f.write(f'{molname:<15s} {bsname:<15s}\n')
        f.write(f'Calculations done on {datetime.datetime.now()}\n\n')

        # Set up Hartree-Fock, remove linear dependencies from basis
        myhf = mol.HF().apply(pyscf.scf.addons.remove_linear_dep_)
        start = time()
        myhf.kernel()
        end = time()

        # true_scf_energies.append(myhf.e_tot)
        S = myhf.get_ovlp()
        F = myhf.get_fock()
        f.write('time stats [s]\n')
        f.write('{:<17s}{:<17s}{:<17s}\n'.format('t_HF', 't_fbyf', 't_sbys'))
        f.write(f'{end-start:15.9e}  ')

        start = time()
        _, data_fbyf = find_subspace(F, S, mol, myhf, conv_tol=1e-4, collect_data=True)
        end = time()
        
        f.write(f'{end-start:15.9e}  ')

        start = time()
        smask, data_sbys = find_subspace(F, S, mol, myhf, conv_tol=1e-4, collect_data=True, get_smask=True)
        end = time()

        f.write(f'{end-start:15.9e}\n\n')

        df_fbyf = pd.DataFrame(data_fbyf, columns=datacols)
        df_sbys = pd.DataFrame(data_sbys, columns=datacols)

        f.write('{:<15s} {:<15s} {:<15s}\n'.format('N_occ', 'E_HF', 'nfunc'))
        f.write('{:<15d} {:<15f} {:<15d}\n\n'.format(np.count_nonzero(myhf.get_occ()), myhf.e_tot, mol.nao_nr()))

        f.write('function-by-function iteration\n')
        df_fbyf.to_csv(f, index=False)
        f.write('\n\n')

        f.write('shell-by-shell iteration\n')
        df_sbys.to_csv(f, index=False)
        f.write('\n\n')
        
        f.close()
        if send_tg_notif:
            # Telegram notification
            send_notification(f'Finished calculation for molecule {molname} with basis set {mol.basis}!')
    if send_tg_notif:
        send_notification(f'Finished all scheduled calculations!')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python3 run.py <molpath> <basispath> [<dec>]')
        print('\tmolpath:  \tthe path to molecule directory')
        print('\tbasispath:\tthe path to file listing basis sets to be used')
        print('\tdec:\tTrue/False, whether to run decontracted calculations too, optional')
        sys.exit()

    molpath = sys.argv[1]
    basispath = sys.argv[2]
    dec = False
    if len(sys.argv) == 4:
        dec = bool(sys.argv[3])

    bs = []
    bstemp = []
    f = open(basispath)
    for line in f:
        bstemp.extend(line.strip('\n').split(' '))
    for b in bstemp:
        if b.replace('-', '') not in pyscf.gto.basis.ALIAS.keys():
            print(f'Basis set {b} not found in PySCF!')
        else:
            bs.append(b)
    mols = get_molecules_in_dir(molpath, bs, get_decontractions=dec)
    run_sbi(mols)