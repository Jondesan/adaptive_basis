import numpy
from adb.molutil import create_shell_separated_mol
from adb.maskutil import init_smask
from copy import deepcopy
import os
from pyscf.gto import Mole

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
    unit = 'Angstrom',
    symmetry: bool | str = False,
    symmetry_fname = None
):
    """Get molecule xyz files from molpath, can be directory or single file.
    """
    from pyscf.gto.mole import format_atom
    from pyscf.gto.basis import load_ecp

    prefix = molpath


    if os.path.isdir(prefix):
        fs = get_files_in_folder(prefix)
        fs = [prefix + '/' + f for f in fs]
    else:
        fs = [molpath]
    molecules = []
    for fn in fs:
        molfname = fn.split("/")[-1]
        if molfname[0] == "#": # If mol fname starts with #, skip file
            continue
        print(f"reading file {fn}")

        if symmetry_fname is not None:
            symm = point_group_from_file(symmetry_fname, molfname)
        else:
            symm = symmetry

        for bs in basis_sets:
            for unc in (
                ["", "unc-"] if get_decontractions and "unc-" not in bs else [""]
            ):
                fnparts = fn.split('/')[-1].split('.')
                if len(fnparts) > 2:
                    charge = [int(substring.replace('charge', '')) for substring in fnparts if 'charge' in substring]
                    charge = charge[0] if len(charge) != 0 else 0
                    spin = [int(substring.replace('spin','')) for substring in fnparts if 'spin' in substring]
                    spin = spin[0] if len(spin) != 0 else None
                else:
                    charge = 0
                    spin = None
                asymbs = list(set([atom[0] for atom in format_atom(fn)]))

                # If ECPs are present, set the ECP basis dictionary, else None
                ecp_bs = {}
                for asymb in asymbs:
                    ecp = load_ecp(unc + bs, asymb)
                    if not ecp:
                        continue
                    ecp_bs[asymb] = ecp
                # If only None in ecp dict, set object to None so
                # pyscf interprets it correctly
                ecp_bs = None if not ecp_bs else ecp_bs
                print(f"{ecp_bs=}")

                mol = Mole(
                    atom=fn,
                    basis=unc + bs,
                    ecp=ecp_bs,
                    charge=charge,
                    spin=spin,
                    unit=unit,
                    symmetry=symm,
                    verbose=0,
                ).build()
                
                mol = create_shell_separated_mol(mol, verbose=mol.verbose)
                smask = init_smask(mol)
                print(f'Created molecule {molfname}, with charge {charge}, spin {spin} and symmetry set at {symm}')
                molecules.append(
                    [fn.split("/")[-1], mol, create_shell_separated_mol(mol), smask, None, bs]
                )

    # Sort by number of electrons, then by the basis, then by number of basis fcts
    molecules.sort(key=lambda x: (x[1].tot_electrons(), x[1].basis, x[1].nao_nr()))
    print(
        f"read a total of {len(molecules)} molecular structures, with the following numbers of functions: {[int(m[1].nao_nr()) for m in molecules]}"
    )
    print(f"with filenames {[name[0] for name in molecules]}")
    return molecules


def point_group_from_file(path, mol_filename):
    """ Looks through file at 'path' for the point group label of a molecule
    with filename 'mol_filename'. If not found, return True.
    """
    pnt_grp = True
    print(f'Reading file with point group information at path {path}')
    print(f'Searching for point group match for molecule with filename {mol_filename}')
    with open(path, 'r') as file:
        for line in file:
            name, point_grp_label = line.split()
            if mol_filename == name:
                pnt_grp = point_grp_label
                print(f'Found point group {pnt_grp} for molecule with filename {mol_filename}')
    
    return pnt_grp


def read_symmetry_occs_from_file(fname: str, molfname: str):
    """Read pyscf formatted symmetry occupations from file and return one
    with matching molecule filename.

    File is expected to follow this format:
    
    molfilename;occs
    h2.charge0.spin0.xyz;{'Ag': 2, 'B1g': 0, 'B2g': 0, 'B3g': 0, 'Au': 0, 'B1u': 0, 'B2u': 0, 'B3u': 0}
    ch4.charge0.spin0.xyz;{'A1': 6, 'A2': 0, 'B1': 2, 'B2': 2}
    h2o.charge0.spin0.xyz;{'A1': 6, 'A2': 0, 'B1': 2, 'B2': 2}

    """
    from ast import literal_eval
    
    if fname is None or not os.path.isfile(fname):
        return None, None
    
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


def write_orbital_history(
        orbital_history:    list,
        fn:                 str,
        molname:            str = "",
        basisname:          str = "",
        ) -> None:
    """Write a find_subspace(track_orbitals=True) orbital_history to a CSV
    file (one row per occupied orbital per ADB cycle): nfunc,energy,irrep.
    `fn` gets '.csv' appended. `irrep` is left blank for symmetry-blind
    entries (irrep_nelec/orbsym not given to find_subspace).
    """
    with open(fn + ".csv", "w") as f:
        if molname or basisname:
            f.write(f"# molecule={molname} basis={basisname}\n")
        f.write("nfunc,energy,irrep\n")
        for entry in orbital_history:
            nfunc = entry["nfunc"]
            for energy, irrep in entry["orbitals"]:
                f.write(f"{nfunc},{energy:.12f},{irrep if irrep is not None else ''}\n")


def print_data_header() -> None:
    print(
            f'\n{"N_func":>10s}  {"New funcs":>12s}  {"Criteria val":>15s}' +\
            f'  {"Difference":>15s}  {"E_subbasSCF":>15s}  {"Q^2":>18s}'
        )


def print_data(
    mask:               numpy.ndarray,
    criteria_value:     float,
    diff:               float,
    ao_or_shell_label:  str,
    E_scf:              float | str = "-",
    Qsqrd:              float | str = "-",
    print_header:       bool        = False ) -> None:
    """Data printout function
    
    """

    if print_header:
        print_data_header()

    if E_scf is None:
        E_scf = "-"
    if Qsqrd is None:
        Qsqrd = "-"

    print(f"{sum(mask):10d}", end="")
    print(f" {ao_or_shell_label:>13s}", end="")
    print(f" {criteria_value:16.9f}", end="")
    print(f'  {diff:{">15s" if isinstance(diff, str) else "15.9f"}}', end="")
    print(f'  {E_scf:{">15s" if isinstance(E_scf, str) else "15.9f"}}', end="")
    print(f'  {Qsqrd:{">15s" if isinstance(Qsqrd, str) else "18.12f"}}')


def print_labels_of_functions_in_mask(
        mask,
        mol,
        print_actual_minimal_basis = False):
    atom_dict = function_labels_from_mask(mask, mol)
    # Sort dictionary by the internal atom index
    atom_dict = dict(sorted(
        atom_dict.items(),
        key=lambda item: int(item[0].split()[0])))

    if print_actual_minimal_basis:
        minimal_mol = deepcopy(mol)
        minimal_mol.basis = 'sto3g'
        minimal_mol.build()
        minimal_mask = numpy.ones(minimal_mol.nao_nr(), dtype=bool)

        print("\nFunctions in the 'actual' minimal basis:")
        minimal_atom_dict = function_labels_from_mask(minimal_mask, minimal_mol)
        # Sort dictionary by the internal atom index
        minimal_atom_dict = dict(sorted(
            minimal_atom_dict.items(),
            key=lambda item: int(item[0].split()[0])))
        for key, elem in minimal_atom_dict.items():
            print(f'{key}: {elem}')

    print('\nFunctions in the pseudominimal basis:')
    for key, elem in atom_dict.items():
        print(f'{key}: {elem}')
    print()


def orbital_key(orb):
    import re

    shell_order = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5}
    # Extract number and shell letter from the start of the string
    match = re.match(r'^(\d+)([spdfghi])', orb.lower())
    if match:
        n, shell = match.groups()
        return (shell_order.get(shell, 99), int(n))
    else:
        # fallback for malformed or unexpected orbitals
        return (float('inf'), int('inf'))


def function_labels_from_mask(mask, mol):
    """ Return a dictionary with all 
    
    """
    from pyscf.gto.mole import cart_labels, sph_labels
    import re

    # Get function labels
    labels = []
    all_labels = numpy.array(cart_labels(mol)) if mol.cart \
            else numpy.array(sph_labels(mol))
    for label in all_labels[mask]:
        # Split label strings of the form
        # 'Atom_idx Atom_symb sph/cart_label', e.g. '0 H 1s' or '1 O 2px'
        atom_num = label.split()[0]
        asymb    = label.split()[1]
        pattern  = re.compile(r'^([0-9]+[spdfgh])')
        match    = pattern.match(label.split()[-1])
        if match is None:
            raise ValueError('No regex patter match found!')
        labels.append(
            ' '.join([atom_num, asymb, match.group(1)]))
    labels = list(set(labels))
    labels = [label.split() for label in labels]
    labels = sorted(labels, key=lambda x: ' '.join(x))
    atom_dict = {}
    for label in labels:
        key = ' '.join(label[:2])
        if key not in atom_dict.keys():
            atom_dict[key] = [label[2]]
        else:
            atom_dict[key].append(label[2])
    for key in atom_dict:
        atom_dict[key].sort(key=orbital_key)

    return atom_dict