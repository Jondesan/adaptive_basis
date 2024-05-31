#!/bin/python3

import sys, os, argparse
sys.path.append('../adbmodule/')
from adb import *
import pyscf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from time import time

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
        if fn.split('/')[-1][0] == '#':
            continue
        print(f'reading file {fn}')
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

def run_adb(mol_list, variant=0):
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

        f = open(f'output/{".".join([molname, bsname])}.out', 'w')
        f.write('{:<15s} {:<15s} {:<15s}\n'.format('molecule', 'basis_set', 'variant'))
        f.write(f'{molname:<15s} {bsname:<15s} {variant:<15d}\n')
        f.write(f'Calculations done on {datetime.datetime.now()}\n\n')

        # Set up Hartree-Fock, remove linear dependencies from basis
        myhf = mol.HF().apply(pyscf.scf.addons.remove_linear_dep_)
        start = time()
        myhf.kernel()
        end = time()

        S = myhf.get_ovlp()
        F = myhf.get_fock()
        f.write('time stats [s]\n')
        f.write('{:<17s}{:<17s}{:<17s}\n'.format('t_HF', 't_fbyf', 't_sbys'))
        f.write(f'{end-start:15.9e}  ')

        if variant == 0:
            start = time()
            _, data_fbyf = find_subspace(
                F, S, mol, myhf,
                conv_tol=1e-4, collect_data=True, variant=variant
                )
            end = time()
        
        if variant == 0:
            f.write(f'{end-start:15.9e}  ')
        else:
            f.write('{:<15s}'.format('-'))

        start = time()
        smask, data_sbys = find_subspace(
            F, S, mol, myhf,
            conv_tol=1e-4, collect_data=True, get_smask=True, variant=variant)
        end = time()

        f.write(f'{end-start:15.9e}\n\n')

        if variant == 0:
            df_fbyf = pd.DataFrame(data_fbyf, columns=datacols)
        df_sbys = pd.DataFrame(data_sbys, columns=datacols)

        f.write('{:<15s} {:<15s} {:<15s}\n'.format('N_occ', 'E_HF', 'nfunc'))
        f.write('{:<15d} {:<15f} {:<15d}\n\n'.format(np.count_nonzero(myhf.get_occ()), myhf.e_tot, mol.nao_nr()))

        if variant == 0:
            f.write('function-by-function iteration\n')
            df_fbyf.to_csv(f, index=False)
            f.write('\n\n')

        f.write('shell-by-shell iteration\n')
        df_sbys.to_csv(f, index=False)
        f.write('\n\n')
        
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run adaptive basis Hartree-Fock calculations.')
    parser.add_argument(
        '--mpath', type=str, required=True,
        help='path to molecule directory'
        )
    parser.add_argument(
        '--bpath', type=str, required=True,
        help='path to basis input file'
        )
    parser.add_argument(
        '--dec', type=bool, required=False, default=False,
        help='whether to run decontracted calculations too, optional. Default is False'
        )
    parser.add_argument(
        '--var', type=int, required=False, default=False, choices=[0,1,2],
        help='which minimisation criteria to use, optional. Default is 0'
        )


    args = parser.parse_args()

    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                     const=sum, default=max,
    #                     help='sum the integers (default: find the max)')

    # if len(sys.argv) < 3:
    #     print('Usage: python3 run.py <molpath> <basispath> [<dec> <variant>]')
    #     print('\tmolpath:  \t the path to molecule directory')
    #     print('\tbasispath:\t the path to file listing basis sets to be used')
    #     print('\tdec:      \t True/False, whether to run decontracted calculations too, optional')
    #     print('\tvariant:  \t 0,1,2, selects which variant to use for calculations. Optional, default is 0')
    #     sys.exit()

    # molpath = sys.argv[1]
    # basispath = sys.argv[2]
    # dec = False
    # if len(sys.argv) == 4:
    #     dec = bool(sys.argv[3])
    # if len(sys.argv) == 5:
    #     dec = bool(sys.argv[3])
    #     variant = int(sys.argv[4])
    basispath = args.bpath
    molpath = args.mpath
    dec = args.dec
    variant = args.var

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
    run_adb(mols, variant)