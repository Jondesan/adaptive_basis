#!/bin/python3

import sys
import os
import argparse

import adb
import adbutils as adbutils
from basis_set_exchange import convert_formatted_basis_file
from pyscf import scf, gto, dft
import numpy as np
import pandas as pd
import datetime
from time import time
import psi4
import io
from copy import deepcopy
import re


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


def get_subbasis(
        mol,
        conv_tol=1e-2,
        q_tol=1.0,
        init_guess='atom',
        normalisation=True,
        abd_init=True,
        run_dft=True
        ):
    
    xcfunc = 'PBE'
    grid_level = 7

    mf = dft.KS(mol) if run_dft else scf.HF(mol)
    if run_dft:
        mf.grids.level = grid_level
        mf.xc = xcfunc
        mf.grids.prune = None
    
    # Initialize init guess method
    mf.init_guess = init_guess

    # This produces the initial guess density matrix
    dm0 = mf.get_init_guess(key=init_guess)
    # we need the corresponding Fock matrix
    F = mf.get_fock(dm=dm0)
    S = mf.get_ovlp()
    # This gives the initial guess density matrix for the mf object
    mf.mo_energy, mf.mo_coeff = mf.eig(F, S)
    mf.mo_occs = mf.get_occ(mf.mo_energy)
    
    smask = adb.find_subspace(
        F, S, mol, mf, conv_tol=conv_tol,
        get_smask=True,
        return_mask_history=False,
        nfunc_normalisation=normalisation,
        abd_Q_tol=q_tol, abd_initialization=True,
        verbose=False,
    )

    return smask


def subbases_to_files(args):
    mpath = args.mpath
    basis = args.basis
    units = args.unit
    conv_tol = args.conv_tol
    q_tol = args.q_tol
    normalisation = args.normalisation
    output = args.output
    run_dft = args.dft
    abd_init = args.abd_init
    verbose = args.verbose

    mols = get_molecules_in_dir(mpath, basis, unit=units)
    
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

            # This produces the initial guess density matrix
            dm0 = mf.get_init_guess(key=init_guess)
            # we need the corresponding Fock matrix
            F = mf.get_fock(dm=dm0)
            S = mf.get_ovlp()
            # This gives the initial guess density matrix for the mf object
            mf.mo_energy, mf.mo_coeff = mf.eig(F, S)
            mf.mo_occs = mf.get_occ(mf.mo_energy)
            
            smaskhistory = adb.find_subspace(
                F, S, mol, mf, conv_tol=conv_tol,
                get_smask=True,
                return_mask_history=True,
                nfunc_normalisation=normalisation,
                abd_Q_tol=q_tol, abd_initialization=abd_init,
                verbose=verbose,
            )

            molname = molfilename.split('.')[0] # Extract molecule name from filename
            adbutils.subbasis_to_file(
                mol,
                smaskhistory[-1][0],
                basis_fname=f'{molname}_subbasis',
                basis_file_comment='Batch creation of subbasis files.'
            )


def extract_occ_values_from_string(occ):
    occ = ''.join(occ.split()[1:])
    occ = occ.translate({ord(c): None for c in '[]'}) # Remove '[' and ']'
    occ = np.fromstring(occ, dtype=int, sep=',')

    return occ


def extract_occupations_from_psi4_output(output_lines, is_unrestricted=False):
    docc = list(filter(lambda x: 'DOCC' in x, output_lines))
    if is_unrestricted: # Extract SOCC from output
        socc = list(filter(lambda x: 'SOCC' in x, output_lines))
    else:   # Create SOCC string of equal length with zeroes
        socc = deepcopy(docc)
        for i,so in enumerate(socc):
            socc[i] = so.replace('DOCC', 'SOCC')
            socc[i] = re.sub(r'(\d+)', '0', socc[i])
    return list(zip(docc, socc))


def extract_occupation_values(occ_tuple):
    """Takes as input a tuple of psi4 occupation strings from the output file
    and outputs a tuple of integer arrays with the corresponding occupation
    values.

    The output format is
    ('  DOCC   [     3,    0,    0,    0,    0,    2,    1,    1 ]',
     '  SOCC   [     3,    0,    0,    0,    0,    2,    1,    1 ]')
    """
    return (extract_occ_values_from_string(occ) for occ in occ_tuple)


def run_psi4(
        args,
        mol,
        basis,
        init_guess,
        dft=True):
    with open(mol.atom) as f:
        xyz = f.read()
    psi4mol = psi4.geometry(xyz)
    unit = mol.unit
    unit_identifier = {
        'angstrom': 0,
        'bohr': 1}[unit.lower()]
    psi4mol.set_units(psi4.core.GeometryUnits(unit_identifier))
    psi4mol.set_multiplicity(mol.spin + 1)
    is_unrestricted = mol.spin > 0
    method = 'PBE' if dft else 'SCF'
    psi4.set_options({'reference': 'uhf' if is_unrestricted else 'rhf'})
    
    f = io.BytesIO()
    converged = False
    with adbutils.stdout_redirector(f):
        try:
            e_tot, wfn = psi4.energy(
                method,
                basis=basis,
                return_wfn=True)
            converged = True
        except:
            pass
    psi4.core.clean()
    output_file = open('output.dat', 'w')
    output_file.write(f.getvalue().decode('utf-8'))
    
    # Filter lines with DOCC
    psi4output = f.getvalue().decode('utf-8').split('\n')
    occs = extract_occupations_from_psi4_output(psi4output, is_unrestricted)
    if converged:
        docc, socc = extract_occupation_values(occs[-1])
    else:
        # If SCF did not converge check which occupations were found and
        # determine which has lowest converged energy
        unique_occs = list(set(occs))

        print(f'Found the following occupations:\n{'\n\n'.join(
            map(lambda x: '\n'.join(x), unique_occs))}')
        print('Testing which provides lowest converged energy...')
        doccs = []
        for occ in unique_occs:
            docc, socc = extract_occupation_values(occ)
            psi4.set_options({'DOCC': list(docc)})
            psi4.set_options({'SOCC': list(socc)})
            f = io.BytesIO()
            with adbutils.stdout_redirector(f):
                e_tot_docc, wfn_docc = psi4.energy(
                    method,
                    basis=basis,
                    return_wfn=True)
            psi4.core.clean()
            doccs.append((docc, e_tot_docc, wfn_docc))

        doccs.sort(key=lambda x: x[1])
        print(doccs)
        docc, e_tot, wfn = doccs[0]
    
    subbasis_fname = mol.atom.split('/')[-1].split('.')[0] + '_subbasis'
    smask = get_subbasis(
        mol, args.conv_tol, q_tol=args.q_tol, init_guess=init_guess,
        normalisation=args.normalisation, abd_init=args.abd_init,
        run_dft=args.dft)
    adbutils.subbasis_to_file(mol, smask, basis_fname=subbasis_fname)
    convert_formatted_basis_file(subbasis_fname + '.nw', subbasis_fname + '.gbs')

    # SCF in the subbasis
    psi4.set_options({'DOCC': list(docc)})
    psi4.set_options({'SOCC': list(socc)})
    with adbutils.stdout_redirector(f):
        e_tot_sub, wfn_sub = psi4.energy(
            method,
            basis=subbasis_fname,
            return_wfn=True)
    
    print('Fullbasis energy:', e_tot, '\nSubbasis energy:', e_tot_sub)

def run_wa_set(args):
    mpath = args.mpath
    basis = args.basis
    units = args.unit
    conv_tol = args.conv_tol
    q_tol = args.q_tol
    normalisation = args.normalisation
    output = args.output
    run_dft = args.dft
    abd_init = args.abd_init
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

            # This produces the initial guess density matrix
            dm0 = mf.get_init_guess(key=init_guess)
            # we need the corresponding Fock matrix
            F = mf.get_fock(dm=dm0)
            S = mf.get_ovlp()
            # This gives the initial guess density matrix for the mf object
            mf.mo_energy, mf.mo_coeff = mf.eig(F, S)
            mf.mo_occs = mf.get_occ(mf.mo_energy)
            
            start = time()
            smaskhistory = adb.find_subspace(
                F, S, mol, mf, conv_tol=conv_tol,
                get_smask=True,
                return_mask_history=True,
                nfunc_normalisation=normalisation,
                abd_Q_tol=q_tol, abd_initialization=abd_init,
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
            mf.kernel()
            end = time()
            converged_F = mf.get_fock()
            fullbasis_time = end - start
            fullbasis_converged = mf.converged
            fullbasis_energy = mf.e_tot

            data_sbys = adb.mask_analysis(
                smaskhistory, mol, mf,
                converged_F, S, verbose=verbose,
                dft=run_dft, xc=xcfunc, grid_level=grid_level,
            )

            smask = smaskhistory[-1][0]
            # Create subbasis mol object
            extracted_basis, ecp_bas = adb.extract_basis(smask, adb.create_shell_separated_mol(mol))
            subbasis_mol = gto.M(
                atom = mol.atom, basis = extracted_basis,
                charge = mol.charge, spin = mol.spin,
                verbose = mol.verbose, unit=mol.unit,
                ecp = ecp_bas
                )
            # subbasis_mol.build()

            submf = dft.KS(subbasis_mol) if run_dft else scf.HF(subbasis_mol) 
            if run_dft:
                submf.grids.level = grid_level
                submf.xc = xcfunc
                submf.grids.prune = None

            mf = dft.KS(mol) if run_dft else scf.HF(mol)
            if run_dft:
                mf.grids.level = grid_level
                mf.xc = xcfunc
                mf.grids.prune = None

            # Run SCF algorithm for the subbasis
            smask = smaskhistory[-1][0]
            mask = adb.smask_to_mask(smask)
            submf.kernel(dm0=adb.mask_matrix(dm0, mask))
            subbasis_converged = submf.converged


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
        "--abd_init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to initialise the subbasis with atomic block decomposition, optional. Default is True.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether output is robust or not, optional. Default is True.",
    )
    parser.add_argument(
        "--calculate_subbasis_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to only calculate the subbasis, optional. Default is False.",
    )

    args = parser.parse_args()
    subbas_only = args.calculate_subbasis_only

    if subbas_only:
        subbases_to_files(args)
    else:
        # run_wa_set(args)
        # run_multiplicities(args)
        mols = get_molecules_in_dir(args.mpath, args.basis, unit=args.unit)
        run_psi4(args, mols[0][1], basis='def2-TZVP', init_guess=mols[0][4], dft=args.dft)


