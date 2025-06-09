"""
    Utility functions for the adaptive basis set library.


"""


import adb
from contextlib import contextmanager
import ctypes
import io
import os, sys
import tempfile
from basis_set_exchange import convert_formatted_basis_file
from pyscf.gto.basis.parse_nwchem import \
    convert_basis_to_nwchem,\
    convert_ecp_to_nwchem
from pyscf import scf, gto, dft, lib
import numpy as np
import pandas as pd
import datetime
from time import time
from copy import deepcopy
import re
from ast import literal_eval


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
    unit='Angstrom',
    symmetry_fname=None
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
        molfname = fn.split("/")[-1]
        if molfname[0] == "#": # If mol fname starts with #, skip file
            continue
        print(f"reading file {fn}")

        if symmetry_fname is not None:
            irrep_occs, symm = read_symmetry_occs_from_file(
                symmetry_fname, molfname=molfname)
        else:
            symm=True

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
                    symmetry=symm,
                    verbose=0,
                )
            except:
                print('running except...')
                mol = gto.M(
                    atom=fn,
                    basis=bs,
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
                [fn.split("/")[-1], mol, adb.create_shell_separated_mol(mol), smask, ig, bs]
            )

    # Sort by number of electrons, then by the basis, then by number of basis fcts
    molecules.sort(key=lambda x: (x[1].tot_electrons(), x[1].basis, x[1].nao_nr()))
    print(
        f"read a total of {len(molecules)} molecular structures, with the following numbers of functions: {[int(m[1].nao_nr()) for m in molecules]}"
    )
    print(f"with filenames {[name[0] for name in molecules]}")
    return molecules


def read_symmetry_occs_from_file(fname: str, molfname: str):
    """Read pyscf formatted symmetry occupations from file and return one
    with matching molecule filename.

    File is expected to follow this format:
    
    molfilename;occs
    h2.charge0.spin0.xyz;{'Ag': 2, 'B1g': 0, 'B2g': 0, 'B3g': 0, 'Au': 0, 'B1u': 0, 'B2u': 0, 'B3u': 0}
    ch4.charge0.spin0.xyz;{'A1': 6, 'A2': 0, 'B1': 2, 'B2': 2}
    h2o.charge0.spin0.xyz;{'A1': 6, 'A2': 0, 'B1': 2, 'B2': 2}

    """
    
    file = None
    with open(fname, 'r') as f:
        file = f.readlines()

    # Get lines from the file with matching molecule filename
    line = [x for x in file if molfname in x]
    # TODO: handle multiple lines with same name better
    line = None if len(line) < 1 else line[0]

    irrep_nelec = literal_eval(line.split(";")[1])
    irrep_symb = line.split(";")[2].rstrip()
    return irrep_nelec, irrep_symb


def get_uncontracted_basis(mol, fn=None):
    """Unravel the contracted basis of mol.

    Args:
        mol : pyscf.MoleBase object
            molecule object.
        fn : None or str
            the file name to which write the basis. If None, basis will
            not be written into a file, only returned as a str.

    Returns:
        The basis as a pySCF formatted string, which can be used with
        pyscf.gto.basis.parse.
    """
    line = 'BASIS "ao basis" PRINT\n'
    basis = ""

    if fn is not None:
        f = open("tempbasis/" + fn + ".dat", "w")
        f.write(line)

    asymb = list(set([mol.atom_pure_symbol(i) for i in range(len(mol._atom))]))
    for asy in asymb:
        line = "#BASIS SET:\n"
        basis += line
        if fn is not None:
            f.write(line)

        for shell in mol._basis[asy]:
            coeffs = np.array(shell[1:])
            contractions = coeffs.shape[1]
            for i in range(1, contractions):
                line = (
                    asy + "\t" + lib.param.ANGULAR[shell[0]].capitalize() + "\n"
                )
                basis += line
                if fn is not None:
                    f.write(line)
                for b in coeffs:
                    line = f"{b[0]:15.7f}\t{b[i]:15.7f}\n"
                    basis += line
                    if fn is not None:
                        f.write(line)
    line = "END\n"
    if fn is not None:
        f.write(line)
        f.close()
    return basis


def get_basis_dict(basis: str):
    """Convert a basis string into a dictionary to pass
    to pyscf.gto.basis.parse
    """

    dc = dict()
    for elem in basis.split("#")[1:]:
        dc[elem[11]] = gto.basis.parse(str(elem[11:]))
    return dc


def basis_to_file_nwchem(
    basis,
    fn,
    ecp_basis=None,
    commentstring="",
    bsname="ao basis",
    cart=False,
    print_noprint="print",
    additional_labels="",
):
    """Converts the basis to NWChem format and writes it into a file.

    Args:
        basis : dict
            PySCF formatted basis structure
        fn : str
            File name for basis file
        bsname : str
            Basis name for basis file data
        cart : bool
            Whether basis in cartesian or spherical geometry
        print_noprint : str
            NWChem print option
        additional_labels : str
            Additional NWChem options
    """
    sph_cart = "cartesian" if cart else "spherical"
    with open(fn + '.nw', "w") as f:
        if len(commentstring) != 0:
            for commentline in commentstring.split('#'):
                f.write(f"#{commentline}\n")
            f.write("\n")
        f.write(f'BASIS "{bsname}" {sph_cart} {print_noprint} ')
        f.write(f"{additional_labels}\n")

        for asymb, atom_basis in basis.items():
            bs_atom_nwchem = convert_basis_to_nwchem(asymb, atom_basis)
            f.write(f"{bs_atom_nwchem}\n")
        f.write("END")

        if ecp_basis is not None:
            f.write('\n\n\nECP\n')
            for asymb, atom_ecp in ecp_basis.items():
                ecp_atom_nwchem = convert_ecp_to_nwchem(asymb, atom_ecp)
                f.write(ecp_atom_nwchem)
                f.write('\n')
            f.write("END")

    return


def subbasis_to_file(
        mol,
        smask,
        basis_fname="subbasis",
        basis_file_comment="Lorem ipsum",
        verbose=False
):
    """Outputs the subbasis defined by shell mask smask to file.
    
    """
    subbasis_mol = adb.create_shell_separated_mol(mol)
    extracted_basis, ecp_bas = adb.extract_basis(
        smask,
        adb.create_shell_separated_mol(subbasis_mol))
    
    adb.basis_to_file_nwchem(
        extracted_basis, f'{basis_fname}', ecp_basis=ecp_bas,
        commentstring=basis_file_comment)
    if verbose:
        print('Created the subbasis, output to file', f'{basis_fname}.nw')


libc = ctypes.CDLL(None)
c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')

@contextmanager
def stdout_redirector(stream):
    # The original fd stdout points to. Usually 1 on POSIX systems.
    original_stdout_fd = sys.stdout.fileno()

    def _redirect_stdout(to_fd):
        """Redirect stdout to the given file descriptor."""
        # Flush the C-level buffer stdout
        libc.fflush(c_stdout)
        # Flush and close sys.stdout - also closes the file descriptor (fd)
        sys.stdout.close()
        # Make original_stdout_fd point to the same file as to_fd
        os.dup2(to_fd, original_stdout_fd)
        # Create a new sys.stdout that points to the redirected fd
        sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))

    # Save a copy of the original stdout fd in saved_stdout_fd
    saved_stdout_fd = os.dup(original_stdout_fd)
    try:
        # Create a temporary fileand redirect stdout to it
        tfile = tempfile.TemporaryFile(mode='w+b')
        _redirect_stdout(tfile.fileno())
        # Yield to caller, then redirect stdout back to the saved fd
        yield
        _redirect_stdout(saved_stdout_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        stream.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_stdout_fd)


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
            subbasis_to_file(
                mol,
                smaskhistory[-1][0],
                basis_fname=f'{molname}_subbasis',
                basis_file_comment='Batch creation of subbasis files.'
            )


def subbasis_to_gaussian_file(
        mol,
        smask,
        fname_addition='',
        verbose=False):
    subbasis_fname = mol.atom.split('/')[-1].split('.')[0] + '_subbasis' + fname_addition
    subbasis_to_file(mol, smask, basis_fname=subbasis_fname)
    convert_formatted_basis_file(subbasis_fname + '.nw', subbasis_fname + '.gbs')

    if verbose:
        print(f'Created NWChem formatted basis file {subbasis_fname}.nw and', end='')
        print(f'and made conversion to Gaussia94 formatted file {subbasis_fname}.gbs')

    return subbasis_fname


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
    import psi4
    with open(mol.atom) as f:
        xyz = f.read()
    psi4mol = psi4.geometry(xyz)
    unit = mol.unit
    unit_identifier = {
        'angstrom': 0,
        'bohr': 1}[unit.lower()]
    psi4mol.set_units(psi4.core.GeometryUnits(unit_identifier))
    psi4mol.set_multiplicity(mol.spin + 1)
    psi4mol.set_molecular_charge(mol.charge)

    is_unrestricted = mol.spin > 0
    method = 'B3LYP' if dft else 'SCF'
    psi4.set_options({'reference': 'uhf' if is_unrestricted else 'rhf'})
    
    f = io.BytesIO()
    converged = False
    with stdout_redirector(f):
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
        
        occupations_string = '\n\n'.join(map(lambda x: '\n'.join(x), unique_occs))
        print('Found the following occupations:')
        print(occupations_string)
        print('Testing which provides lowest converged energy...')
        doccs = []
        for occ in unique_occs:
            docc, socc = extract_occupation_values(occ)
            psi4.set_options({'DOCC': list(docc)})
            psi4.set_options({'SOCC': list(socc)})
            f = io.BytesIO()
            with stdout_redirector(f):
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
    subbasis_to_file(mol, smask, basis_fname=subbasis_fname)
    convert_formatted_basis_file(subbasis_fname + '.nw', subbasis_fname + '.gbs')

    # SCF in the subbasis
    psi4.set_options({'DOCC': list(docc)})
    psi4.set_options({'SOCC': list(socc)})
    with stdout_redirector(f):
        e_tot_sub, wfn_sub = psi4.energy(
            method,
            basis=subbasis_fname,
            return_wfn=True)
    
    return e_tot, e_tot_sub


def psi4_fullbasis(
        mol,
        basis,
        init_guess,
        dft=True,
        xc='B3LYP',
        verbose=False):
    import psi4
    
    init_guess = {
        'minao': 'atom', 
        'atom': 'atom',
        'huckel': 'huckel',
        'hcore': 'core',
        'sap': 'sapgau',
        'vsap': 'sap'
    }[init_guess]

    psi4mol = pyscf_mol_to_psi4(mol)
    irrep_symbol = psi4mol.find_point_group().symbol()
    irrep_labels = psi4mol.irrep_labels()

    # Output fullbasis to a file for psi4 in case it is not available
    # in Psi4 by default
    bas_str = get_uncontracted_basis(mol)
    pyscf_bas = get_basis_dict(bas_str)
    pyscf_ecpbas = mol._ecp if mol._ecp != {} else None
    molname = mol.atom.split('/')[-1].split('.')[0]
    basis_fname = f'{molname}_{basis}_full'
    basis_to_file_nwchem(
        pyscf_bas, basis_fname,
        ecp_basis=pyscf_ecpbas,
        cart=mol.cart)
    convert_formatted_basis_file(basis_fname + '.nw', basis_fname + '.gbs')

    is_unrestricted = mol.spin > 0
    method = xc.upper() if dft else 'SCF'
    psi4.set_options({'reference': 'uhf' if is_unrestricted else 'rhf'})

    f = io.BytesIO()
    converged = False
    with stdout_redirector(f):
        try:
            e_tot, wfn = psi4.energy(
                method,
                basis=basis_fname,
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

        if verbose:
            occupations_string = '\n\n'.join(map(lambda x: '\n'.join(x), unique_occs))
            print('Found the following occupations:')
            print(occupations_string)
            print('Testing which provides lowest converged energy...')
        doccs = []
        for occ in unique_occs:
            docc, socc = extract_occupation_values(occ)
            psi4.set_options({'DOCC': list(docc)})
            psi4.set_options({'SOCC': list(socc)})
            f = io.BytesIO()
            with stdout_redirector(f):
                e_tot_trial, wfn_trial = psi4.energy(
                    method,
                    basis=basis,
                    return_wfn=True)
            psi4.core.clean()
            doccs.append((docc, e_tot_trial, wfn_trial, socc))

        doccs.sort(key=lambda x: x[1])
        docc, e_tot, wfn, socc = doccs[0]

    
    return e_tot, docc, socc, wfn, irrep_labels, irrep_symbol


def pyscf_mol_to_psi4(mol):
    import psi4

    with open(mol.atom) as f:
        xyz = f.read()
    psi4mol = psi4.geometry(xyz)
    unit = mol.unit
    unit_identifier = {
        'angstrom': 0,
        'bohr': 1}[unit.lower()]
    psi4mol.set_units(psi4.core.GeometryUnits(unit_identifier))
    psi4mol.set_multiplicity(mol.spin + 1)
    psi4mol.set_molecular_charge(mol.charge)

    return psi4mol


def psi4_manual_basis(
        mol,
        basis,
        init_guess,
        dft=True,
        xc='B3LYP'):
    import psi4
    
    init_guess = {
        'minao': 'atom', 
        'atom': 'atom',
        'huckel': 'huckel',
        'hcore': 'core',
        'sap': 'sapgau',
        'vsap': 'sap'
    }[init_guess]

    psi4mol = pyscf_mol_to_psi4(mol)

    is_unrestricted = mol.spin > 0
    method = xc.upper() if dft else 'SCF'
    psi4.set_options({'reference': 'uhf' if is_unrestricted else 'rhf'})

    f = io.BytesIO()
    converged = False
    with stdout_redirector(f):
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
    
    if converged:
        return e_tot, wfn
    print('Problem with the subbasis SCF. Psi4 did not converge.', file=sys.stderr)
    return 0.0, None