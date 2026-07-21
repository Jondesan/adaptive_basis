from copy import deepcopy

import numpy as np
from pyscf.gto import Mole
from pyscf.gto.basis import load_ecp
from pyscf.gto.mole import format_atom

from .basisutil import extract_basis, get_uncontracted_basis


def create_shell_separated_mol(mol: Mole, verbose: int = 0) -> Mole:
    """Build a copy of `mol` with every contraction split into its own shell.

    Uses `adb.basisutil.get_uncontracted_basis` to unravel `mol`'s basis,
    so the returned mol's ``_bas``/shell mask are in 1:1 correspondence
    with individual contractions -- the representation `adb`'s shell
    masks (`init_smask` etc.) operate on.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object to shell-separate.
    verbose : int, default 0
        Verbosity passed to the new `Mole`.

    Returns
    -------
    pyscf.gto.Mole
        The shell-separated copy, built and ready to use.
    """
    shell_sep_basis = get_uncontracted_basis(mol)
    return Mole(
        atom=mol.atom, basis=shell_sep_basis,
        charge=mol.charge, spin=mol.spin,
        unit=mol.unit, symmetry=mol.symmetry,
        ecp=mol.ecp,
        verbose=verbose).build()


def get_shells(mol: Mole) -> np.ndarray:
    """Number of AO functions per shell, in pyscf's internal shell order.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object.

    Returns
    -------
    ndarray, dtype=int
        Number of functions per shell.

    Raises
    ------
    Exception
        If the shell function counts don't sum to `mol.nao_nr()`.
    """
    shells = np.array([], dtype=int)

    for ib in range(mol.nbas):
        angl = mol.bas_angular(ib)
        nc = mol.bas_nctr(ib)
        shells = np.append(
            shells, nc * (angl + 1) * (angl + 2) // 2 if mol.cart else nc * (2 * angl + 1)
        )

    if sum(shells) != mol.nao_nr():
        raise Exception(
            "Number of functions in the mask does not correspond with number of functions in the molecule!"
        )

    return shells


def basis_functions_per_atom(mol: Mole) -> np.ndarray:
    """Number of AO functions on each atom of `mol`.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object.

    Returns
    -------
    ndarray, dtype=int, shape (natm,)
        Number of functions per atom, in atom order.
    """
    basis_struct = mol._bas
    atoms = mol._atom
    func_per_atom = np.zeros(len(atoms), dtype=int)
    for i in range(len(atoms)):
        angl = basis_struct[basis_struct[:, 0] == i][:, 1]
        numc = basis_struct[basis_struct[:, 0] == i][:, 3]
        func_per_atom[i] = np.sum((2 * angl + 1) * numc) if not mol.cart \
            else np.sum((angl + 1) * (angl + 2) // 2 * numc)

    return func_per_atom


def create_mol_from_file(
        fn:         str,
        basis_set:  str,
        charge:     int             = 0,
        spin:       int             = 0,
        unit:       str             = "Angstrom",
        symmetry:   bool | str      = False,
        **kwargs,
        ) -> Mole:
    """Build a `Mole` from a geometry file, auto-attaching ECPs where available.

    Parameters
    ----------
    fn : str
        Path to a geometry file (any format `pyscf.gto.mole.format_atom`
        accepts, e.g. XYZ).
    basis_set : str
        Basis set name, looked up per element (and, via
        `pyscf.gto.basis.load_ecp`, for a matching ECP).
    charge : int, default 0
        Molecular charge.
    spin : int, default 0
        Number of unpaired electrons (pyscf's ``2S`` convention).
    unit : str, default "Angstrom"
        Coordinate unit.
    symmetry : bool or str, default False
        Passed straight to `Mole(symmetry=...)`.
    **kwargs
        Additional keyword arguments forwarded to `Mole`.

    Returns
    -------
    pyscf.gto.Mole
        The built molecule.
    """
    asymbs = list(set(atom[0] for atom in format_atom(fn)))

    ecp_bs = {}
    for asymb in asymbs:
        ecp = load_ecp(basis_set, asymb)
        if ecp:
            ecp_bs[asymb] = ecp
    ecp_bs = ecp_bs or None

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
    return mol.build()


def create_mol_from_template(template: Mole, **kwargs) -> Mole:
    """Copy `template` and rebuild it with the given attributes overridden.

    Parameters
    ----------
    template : pyscf.gto.Mole
        Molecule to copy (deep-copied; `template` itself is untouched).
    **kwargs
        Attribute names and values to set on the copy before rebuilding,
        e.g. ``basis='sto3g'``.

    Returns
    -------
    pyscf.gto.Mole
        The rebuilt copy.

    Raises
    ------
    RuntimeError
        If `template` has no such attribute.
    """
    mol = deepcopy(template)
    for key, val in kwargs.items():
        if not hasattr(mol, key):
            raise RuntimeError(f"Molecule object does not have attribute {key}")
        setattr(mol, key, val)
    return mol.build()


def create_subbasis_mol(mol: Mole, smask: np.ndarray) -> Mole:
    """Build the `Mole` described by a shell mask.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Full-basis molecule the subbasis is drawn from.
    smask : ndarray
        Shell mask (see `adb.maskutil.init_smask`) indexing into `mol`'s
        shell-separated basis.

    Returns
    -------
    pyscf.gto.Mole
        The built subbasis molecule.
    """
    extracted_basis, ecp_bas = extract_basis(smask, create_shell_separated_mol(mol))
    subbasis_mol = Mole(
        atom=mol.atom, basis=extracted_basis,
        charge=mol.charge, spin=mol.spin,
        verbose=mol.verbose, unit=mol.unit,
        ecp=ecp_bas, symmetry=mol.symmetry
    )
    subbasis_mol.build()

    return subbasis_mol


def get_array_of_angular_momenta_and_atom_id(mol: Mole) -> np.ndarray:
    """Per-AO angular momentum and owning atom id.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object.

    Returns
    -------
    ndarray, dtype=int, shape (nao, 2)
        Column 0 is each AO's angular momentum `l`; column 1 is its atom
        id. Row `n` describes the `n`-th AO in `mol`'s internal order.
    """
    angls_aid = np.zeros((mol.nao_nr(), 2), dtype=int)
    input_idx = 0
    for bas in mol._bas:
        nfuncs = funcs_on_shell(bas[1], mol.cart) * bas[3]
        angls_aid[input_idx:input_idx + nfuncs, 0] = bas[1]
        angls_aid[input_idx:input_idx + nfuncs, 1] = bas[0]
        input_idx += nfuncs
    return angls_aid


def funcs_on_shell(angl: int, cart: bool = False) -> int:
    """Number of AO functions in one shell of angular momentum `angl`.

    Parameters
    ----------
    angl : int
        Angular momentum quantum number (0=S, 1=P, ...).
    cart : bool, default False
        Whether to count Cartesian (rather than spherical) functions.

    Returns
    -------
    int
        ``(angl + 1) * (angl + 2) // 2`` Cartesian functions, or
        ``2 * angl + 1`` spherical functions.
    """
    return (angl + 1) * (angl + 2) // 2 if cart else 2 * angl + 1
