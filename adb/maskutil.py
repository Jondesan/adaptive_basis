from pyscf import gto
from pyscf.gto.ecp import core_configuration
from pyscf.data.elements import _std_symbol
import numpy
from copy import deepcopy
from itertools import count
from . import CONSTANTS


def init_smask(
        mol:    gto.MoleBase,
        cart:   bool          = False) -> numpy.ndarray:
    """Initialize the shell mask array. smask will be a list of lists,
    with length equal to the number of uncontracted shells, and each
    element is a two element list, first is bool that specifies the mask
    for the current shell, the other how many primitives in this shell.
    """

    smask = []

    count = numpy.zeros((mol.natm, 9), dtype=int)
    for ib in range(mol.nbas):
        ia = mol.bas_atom(ib)       # atom that given basis function sits on
        angl = mol.bas_angular(ib)  # angular momentum angl of given basis function
        nc = mol.bas_nctr(ib)       # number of CGTOs for given shell
        symb = mol.atom_symbol(ia)  # label of given atom
        nelec_ecp = mol.atom_nelec_core(ia)  # Number of ecp electrons
        if nelec_ecp == 0 or angl > 3:
            shl_start = count[ia, angl] + angl + 1
        else:
            coreshl = core_configuration(nelec_ecp, atom_symbol = _std_symbol(symb))
            shl_start = coreshl[angl] + count[ia, angl] + angl + 1
        count[ia, angl] += nc
        for n in range(shl_start, shl_start + nc):
            if nelec_ecp == 0 or angl > 3:
                n_remove_ecp = n
            else:
                n_remove_ecp = n - coreshl[angl]
            smask.append(
                [
                    False,
                    (angl + 1) * (angl + 2) // 2 if cart else 2 * angl + 1,
                    angl,
                    (ia, symb, n, CONSTANTS.ANGULAR[angl].capitalize(), n_remove_ecp),
                ]
            )

    return numpy.array(smask, dtype=object)


def smask_to_mask(
        smask:  numpy.ndarray,
        cart:   bool        = False) -> numpy.ndarray:
    """Convert current shell mask into function mask."""
    funcs_per_shell = [
        ((s[2] + 1) * (s[2] + 2) // 2 if cart else 2 * s[2] + 1) for s in smask
    ]
    mask = [False] * sum(funcs_per_shell)
    for i, sm in enumerate(smask):
        if sm[0]:
            rb = sum(funcs_per_shell[:i])
            re = rb + sm[1]
            mask[rb:re] = [True] * (re - rb)
    return numpy.array(mask, dtype=bool)


def mask_to_smask(
        mask:   numpy.ndarray,
        smask:  numpy.ndarray,
        cart:   bool        = False) -> numpy.ndarray:
    """Flip shells of smask to True that have 1 or more functions set to
    True in mask.
    """
    mapping = maskidx_to_smaskidx(mask, smask, cart)
    for i in numpy.argwhere(mask):
        smask[mapping[i[0]]][0] = True

    return smask


def linked_shell_idx(smask: numpy.ndarray) -> numpy.ndarray:
    """ Return smask indices that correspond to duplicate shells, i.e.
    if molecule has more than one of same atom type, the shells of that
    atom will be duplicated.

    Args:
        smask : ndarray
            Shell mask array

    Returns:
        shl_indices : ndarray
            Duplicate shell indices
    """
    atoms_found = []
    shells = ["".join([str(s) for s in sm[3][1:]]) for sm in smask]

    shl_indices = []
    for i, sm in enumerate(smask):
        if "".join([str(s) for s in sm[3][1:]]) not in atoms_found:
            atoms_found.append("".join([str(s) for s in sm[3][1:]]))
            indices = [
                ind
                for ind, ele in zip(count(), shells)
                if ele == "".join([str(s) for s in sm[3][1:]])
            ]
            shl_indices.append(indices)
    return shl_indices


def get_all_shell_labels(mol: gto.MoleBase) -> list[str]:
    count = numpy.zeros((mol.natm, 9), dtype=int)
    labels = []
    for ib in range(mol.nbas):  # nbas = number of shells (basis fcts)
        ia = mol.bas_atom(ib)   # atom that given basis function sits on
        l = mol.bas_angular(ib) # angular momentum l of basis function
        strl = CONSTANTS.ANGULAR[l] # angular momentum label
        nc = mol.bas_nctr(ib)   # number of CGTOs for given shell
        symb = mol.atom_symbol(ia)  # label of given atom
        nelec_ecp = mol.atom_nelec_core(ia) # Number of ecp electrons

        if nelec_ecp == 0 or l > 3:
            shl_start = count[ia,l]+l+1
        else:
            coreshl = core_configuration(nelec_ecp, atom_symbol=_std_symbol(symb))
            shl_start = coreshl[l]+count[ia,l]+l+1
        count[ia,l] += nc
        for n in range(shl_start, shl_start+nc):
            labels.append((ia, symb, '%d%s' % (n, strl)))

    return labels


def link_shells(mol, mask):
    """Toggle functions corresponding to same atoms on and toggle all
    functions within a shell.

    Args:
        mol : pyscf.got.MoleBase
            The molecule object
        mask : numpy.ndarray
            The function mask
    """

    smask = mask_to_smask(
        mask,
        init_smask(mol, mol.cart),
        mol.cart)

    return smask_to_mask(smask, mol.cart)


def get_atom_shell_label(
        mol: gto.MoleBase,
        shl_idx: int,
        link_shells: bool = False
    ) -> str:
    labels = get_all_shell_labels(mol)

    if link_shells:
        return '%s %s' % labels[shl_idx][1:]
    return '%d %s %s' % labels[shl_idx]


def print_shells(mol: gto.MoleBase, smask: numpy.ndarray) -> None:
    labels = get_all_shell_labels(mol)
    for i,sm in enumerate(smask):
        if not sm[0]:
            continue
        print('Atom %d, symb: %s, shell: %s' % labels[i])


def maskidx_to_smaskidx(
        mask:   numpy.ndarray,
        smask:  numpy.ndarray,
        cart:   bool        = False) -> list:
    """Create mapping between mask and smask"""
    mapping = [0] * len(mask)
    counter = 0
    for i, sm in enumerate(smask):
        angl = sm[2]
        for j in range((angl + 1) * (angl + 2) // 2 if cart else 2 * angl + 1):
            mapping[counter] = i
            counter += 1
    return mapping


def set_linked_shells(
        smask:  numpy.ndarray,
        val:    bool        ) -> numpy.ndarray:
    """Set smask to 'val' at linked shell positions.
    """
    copysmask = deepcopy(smask)
    selected_shells = numpy.argwhere([sm[0] for sm in copysmask])
    all_shells = numpy.array([''.join(map(str, sm[3][1:])) for sm in copysmask])
    same_as_selected = numpy.argwhere(all_shells == all_shells[selected_shells])[:,1]
    temp = copysmask[same_as_selected][:,0]
    temp = val
    copysmask[same_as_selected,0] = temp
    return copysmask


def mask_matrix(
        mat:                numpy.ndarray,
        mask:               numpy.ndarray,
        is_restricted:      bool        = True ) -> numpy.ndarray:
    """Return masked matrix

    Args:
        mat : ndarray
            Matrix, e.g. Fock matrix. If using restricted Hartree-Fock,
            should be 2D. If using unrestricted, 3D with alpha and beta
            matrices as the array elements along axis 0 if such matrix.
            (overlap matrix will only have one matrix in UHF)
        mask : array
            The basis mask.
        is_restricted : bool
            Whether using restricted or unrestricted HF. Optional,
            default is True.

    Returns:
        masked_mat : ndarray
            The masked matrix
    """
    is_restricted = (len(mat.shape) == 2)
    return mat[mask, :][:, mask] if is_restricted else mat[:, mask, :][:, :, mask]