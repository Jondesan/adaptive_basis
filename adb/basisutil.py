from copy import deepcopy

import numpy as np
from pyscf.gto import MoleBase
from pyscf.gto.basis.parse_nwchem import (
    convert_basis_to_nwchem,
    convert_ecp_to_nwchem,
    to_general_contraction,
)

from . import CONSTANTS


def get_uncontracted_basis(mol: MoleBase, fn: str | None = None) -> str:
    """Unravel `mol`'s contracted basis into one shell per contraction.

    Parameters
    ----------
    mol : pyscf.gto.MoleBase
        Molecule object.
    fn : str, optional
        If given, also write the basis to ``tempbasis/<fn>.dat`` in NWChem
        format.

    Returns
    -------
    str
        The basis as an NWChem-formatted string, parseable by
        `pyscf.gto.basis.parse`.
    """
    line = 'BASIS "ao basis" PRINT\n'
    basis = ""

    if fn is not None:
        f = open("tempbasis/" + fn + ".dat", "w")
        f.write(line)

    asymb = list(set(mol.atom_pure_symbol(i) for i in range(len(mol._atom))))
    for asy in asymb:
        line = "#BASIS SET:\n"
        basis += line
        if fn is not None:
            f.write(line)

        for shell in mol._basis[asy]:
            coeffs = np.array(shell[1:])
            contractions = coeffs.shape[1]
            for i in range(1, contractions):
                line = asy + "\t" + CONSTANTS.ANGULAR[shell[0]].capitalize() + "\n"
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


def get_basis_dict(basis: str) -> dict:
    """Parse an NWChem-formatted basis string into a pyscf basis dict.

    Parameters
    ----------
    basis : str
        NWChem-formatted basis string, e.g. as returned by
        `get_uncontracted_basis`.

    Returns
    -------
    dict
        Maps each atom symbol to its `pyscf.gto.basis.parse`-formatted
        basis.
    """
    from pyscf.gto.basis import parse

    dc = {}
    for elem in basis.split("#")[1:]:
        dc[elem[11]] = parse(str(elem[11:]))
    return dc


def basis_to_file_nwchem(
        basis:              dict,
        fn:                 str,
        ecp_basis:          dict | None = None,
        commentstring:      str         = "",
        bsname:             str         = "ao basis",
        cart:               bool        = False,
        print_noprint:      str         = "print",
        additional_labels:  str         = "",
        ) -> None:
    """Write a pyscf-format basis (and optional ECP) to an NWChem-format file.

    Parameters
    ----------
    basis : dict
        pyscf-formatted basis, one entry per atom symbol.
    fn : str
        Output file name (``.nw`` is appended).
    ecp_basis : dict, optional
        pyscf-formatted ECP basis, written as a trailing ``ECP`` block if
        given.
    commentstring : str, default ""
        Comment lines to prepend, ``'#'``-separated.
    bsname : str, default "ao basis"
        Basis set name recorded in the file header.
    cart : bool, default False
        Whether the basis is Cartesian (rather than spherical).
    print_noprint : str, default "print"
        NWChem ``print``/``noprint`` option.
    additional_labels : str, default ""
        Extra text appended to the ``BASIS`` header line.
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


def extract_basis(smask: np.ndarray, shellsep_mol: MoleBase) -> tuple[dict, dict | None]:
    """Extract the basis described by a shell mask, as a pyscf-format dict.

    Parameters
    ----------
    smask : ndarray
        Shell mask (see `adb.maskutil.init_smask`). The basis is extracted
        according to which shells are selected.
    shellsep_mol : pyscf.gto.MoleBase
        Shell-separated molecule (see
        `adb.molutil.create_shell_separated_mol`) whose basis `smask`
        indexes into.

    Returns
    -------
    basis : dict
        The masked basis, one entry per atom symbol, in pyscf format. An
        atom whose every selected shell filters out to nothing (see the
        implementation note below) maps to `None`, consistent with an atom
        that was never populated.
    ecp_basis : dict or None
        The ECP basis dict, if `shellsep_mol` has one, else `None`.

    Raises
    ------
    ValueError
        If `smask`'s length doesn't match `shellsep_mol._bas`.
    """
    if len(smask) != len(shellsep_mol._bas):
        raise ValueError(
            "Shell mask does not match with _bas attribute!"
            + " Make sure the shellsep_mol objects shells have been separated"
            + " using the create_shell_separated_mol method."
        )

    asymb = list(shellsep_mol._basis.keys())
    basis = dict.fromkeys(asymb)

    duplicate_removed_smask = []
    found_atoms = []
    current_id = -1
    # Collect unique atom smasks (if same atom is present in the shellsep_mol
    # more than once, ignore its mask after the first one)
    for elem in deepcopy(smask[np.asarray(smask[:, 0], dtype=bool)]):
        if elem[3][1] not in found_atoms:
            found_atoms.append(elem[3][1])
            current_id = elem[3][0]
        elif current_id != elem[3][0]:
            continue
        duplicate_removed_smask.append(elem)

    duplicate_removed_smask = np.array(duplicate_removed_smask)
    # Initialize distinct atoms' dictionary formatted basis structures
    # with angular momentum angl
    for angl, shl in duplicate_removed_smask[:, [2, 3]]:
        if basis[shl[1]] is None:
            basis[shl[1]] = []
        if angl not in [x[0] for x in basis[shl[1]]]:
            basis[shl[1]].append([angl])

    # Append exponents and contraction coefficients
    for key in asymb:
        ogbas = to_general_contraction(shellsep_mol._basis[key])
        # Look up by each entry's own angular momentum rather than its
        # position in ogbas -- to_general_contraction only emits one entry
        # per angular momentum actually present, so position and angular
        # momentum coincide only when the basis has no l-gaps.
        ogbas_by_l = {entry[0]: entry for entry in ogbas}
        # Important when initialization does not put functions on all
        # atoms in the molecule, would result in error
        if basis[key] is None:
            continue
        kept_shells = []
        for shell in basis[key]:
            i = shell[0]
            key_smask = [drs for drs in duplicate_removed_smask if drs[3][1] == key]
            idxs = [idx[3][4] - idx[2] for idx in key_smask if idx[2] == i]
            coeff_table = np.asarray(ogbas_by_l[i][1:], dtype=float)[:, [0] + idxs]
            # Remove rows and columns with all 0 contraction coeffs
            filtered_shell = coeff_table[
                ~((coeff_table[:, 0] != 0) &
                  (coeff_table[:, 1:] == 0).all(axis=1))]
            filtered_shell = filtered_shell[~np.all(filtered_shell == 0, axis=1)]
            if filtered_shell.tolist():
                shell.extend(filtered_shell.tolist())
                kept_shells.append(shell)
        # Keep the "no shells for this atom" convention consistent with the
        # None check above (an atom whose every selected shell filtered out
        # is indistinguishable from one that was never populated).
        basis[key] = kept_shells if kept_shells else None
    ecp = shellsep_mol._ecp if shellsep_mol._ecp != {} else None
    return basis, ecp
