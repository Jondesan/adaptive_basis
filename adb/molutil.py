from pyscf.gto import Mole
from .basisutil import get_uncontracted_basis, extract_basis
import numpy
from pyscf.gto.mole import format_atom
from pyscf.gto.basis import load_ecp

def create_shell_separated_mol(
        mol:        Mole,
        verbose:    int           = 0) -> Mole:
    """Creates a copy of mol with shells separated."""
    shell_sep_basis = get_uncontracted_basis(mol)
    cmol = Mole(
        atom=mol.atom, basis=shell_sep_basis,
        charge=mol.charge, spin=mol.spin,
        unit=mol.unit, symmetry=mol.symmetry,
        ecp=mol.ecp,
        verbose=verbose).build()
    return cmol


def get_shells(mol: Mole) -> numpy.ndarray:
    """Get the shell structure of mol object.

    Args:
        mol : pyscf.gto.Mole
            The molecule object.

    Returns:
        A 1D ndarray with the number of functions per shell as elements.
        Shells are ordered in the pyscf internal format.
    """
    shells = numpy.array([], dtype=int)  # Number of functions per shell

    for ib in range(mol.nbas):  # nbas = number of shells (basis fcts)
        angl = mol.bas_angular(ib)  # angular momentum l of given basis function
        nc = mol.bas_nctr(ib)  # number of CGTOs for given shell

        shells = numpy.append(
            shells, nc * (angl + 1) * (angl + 2) // 2 if mol.cart else nc * (2 * angl + 1)
        )

    if sum(shells) != mol.nao_nr():
        raise Exception(
            "Number of functions in the mask does not correspond with number of functions in the molecule!"
        )

    return shells


def basis_functions_per_atom(mol: Mole) -> numpy.ndarray:
    basis_struct = mol._bas
    atoms = mol._atom
    nat = len(atoms)
    func_per_atom = numpy.zeros(nat, dtype=int)
    for i in range(nat):
        angl = basis_struct[basis_struct[:,0]==i][:,1]
        numc = basis_struct[basis_struct[:,0]==i][:,3] # Number of CGTOs
        func_per_atom[i] = numpy.sum((2*angl+1) * numc) if not mol.cart \
                           else numpy.sum((angl + 1)*(angl + 2) // 2 * numc)
    
    return func_per_atom


def create_mol_from_file(fn: str, basis_set: str, charge=0, 
                         spin=0, unit="Angstrom", symmetry=False,
                         **kwargs) -> Mole:
    asymbs = list(set([atom[0] for atom in format_atom(fn)]))

    # If ECPs are present, set the ECP basis dictionary, else None
    ecp_bs = {}
    for asymb in asymbs:
        ecp = load_ecp(basis_set, asymb)
        if not ecp:
            continue
        ecp_bs[asymb] = ecp
    # If only None in ecp dict, set object to None so
    # pyscf interprets it correctly
    ecp_bs = None if not ecp_bs else ecp_bs

    mol = Mole(
        atom=fn,
        basis=basis_set,
        ecp=ecp_bs,
        charge=charge,
        spin=spin,
        unit=unit,
        symmetry=symmetry,
        verbose=0,
        **kwargs
    )
    mol = mol.build()

    return mol


def create_mol_from_template(template: Mole, **kwargs) -> Mole:
    from copy import deepcopy

    mol = deepcopy(template)
    for key,val in kwargs.items():
        if not hasattr(mol, key):
            raise RuntimeError(f"Molecule object does not have attribute {key}")
        setattr(mol, key, val)
    mol = mol.build()

    return mol


def create_subbasis_mol(
        mol:        Mole,
        smask:      numpy.ndarray    ) -> Mole:

    extracted_basis, ecp_bas = extract_basis(smask, create_shell_separated_mol(mol))
    subbasis_mol = Mole(
        atom = mol.atom, basis = extracted_basis,
        charge = mol.charge, spin = mol.spin,
        verbose = mol.verbose, unit = mol.unit,
        ecp = ecp_bas, symmetry = mol.symmetry
    )
    subbasis_mol.build()

    return subbasis_mol


def get_array_of_angular_momenta_and_atom_id(mol):
    angls_aid = numpy.zeros((mol.nao_nr(), 2), dtype=int)
    input_idx = 0
    for i, bas in enumerate(mol._bas):
        nfuncs = funcs_on_shell(bas[1], mol.cart)
        angls_aid[input_idx:input_idx + nfuncs, 0] = bas[1]
        angls_aid[input_idx:input_idx + nfuncs, 1] = bas[0]
        input_idx += nfuncs
    return angls_aid


def funcs_on_shell(angl, cart=False):
    return (angl + 1) * (angl + 2) / 2 if cart else 2 * angl + 1