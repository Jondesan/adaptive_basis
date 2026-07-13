import numpy


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
        minimal_mask = np.ones(minimal_mol.nao_nr(), dtype=bool)

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