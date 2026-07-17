from pyscf.gto import MoleBase
import numpy
from copy import deepcopy
from . import CONSTANTS
from pyscf.gto.basis.parse_nwchem import convert_basis_to_nwchem, convert_ecp_to_nwchem, to_general_contraction

def get_uncontracted_basis(
        mol:    MoleBase,
        fn:     str | None    = None) -> str:
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
    line  = 'BASIS "ao basis" PRINT\n'
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
            coeffs = numpy.array(shell[1:])
            contractions = coeffs.shape[1]
            for i in range(1, contractions):
                line = (
                    asy + "\t" + CONSTANTS.ANGULAR[shell[0]].capitalize() + "\n"
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


def get_basis_dict(basis: str) -> dict:
    """Convert a basis string into a dictionary to pass
    to pyscf.gto.basis.parse
    """
    from pyscf.gto.basis import parse

    dc = dict()
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
    additional_labels:  str         = "" ) -> None:
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


def extract_basis(
        smask:          numpy.ndarray,
        shellsep_mol:   MoleBase
    ) -> tuple[dict, dict | None]:
    """Extract a basis from given shell mask as python dictionary in
    pySCF format.

    Args:
        smask : ndarray
            Shell mask. Basis will be extracted according to this.

        shellsep_mol : pyscf.MoleBase object
            molecule object from whose basis the new basis will be
            extracted.

    Returns:
        basis : dict
            the masked basis of the molecule as a dictionary according
            pySCF format.
        ecp_basis : none | dict
            the ECP basis dictionary if present in the full basis of
            shellsep_mol. Otherwise returns None.
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
    for elem in deepcopy(smask[numpy.asarray(smask[:, 0], dtype = bool)]):
        if elem[3][1] not in found_atoms:
            found_atoms.append(elem[3][1])
            current_id = elem[3][0]
        elif current_id != elem[3][0]:
            continue
        duplicate_removed_smask.append(elem)

    duplicate_removed_smask = numpy.array(duplicate_removed_smask)
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
            coeff_table = numpy.asarray(ogbas_by_l[i][1:], dtype=float)[:, [0] + idxs]
            # Remove rows and columns with all 0 contraction coeffs
            filtered_shell = coeff_table[
                ~((coeff_table[:, 0] != 0) &
                (coeff_table[:, 1:] == 0).all(axis = 1))]
            filtered_shell = filtered_shell[~numpy.all(filtered_shell == 0, axis = 1)]
            if filtered_shell.tolist():
                shell.extend(filtered_shell.tolist())
                kept_shells.append(shell)
        # Keep the "no shells for this atom" convention consistent with the
        # None check above (an atom whose every selected shell filtered out
        # is indistinguishable from one that was never populated).
        basis[key] = kept_shells if kept_shells else None
    ecp = shellsep_mol._ecp if shellsep_mol._ecp != {} else None
    return basis, ecp