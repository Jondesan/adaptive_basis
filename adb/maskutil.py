from copy import deepcopy
from itertools import count

import numpy as np
from pyscf import gto
from pyscf.gto.ecp import core_configuration
from pyscf.data.elements import _std_symbol

from . import CONSTANTS


def init_smask(mol: gto.MoleBase, cart: bool = False) -> np.ndarray:
    """Build the initial (all-`False`) shell mask for `mol`.

    A shell mask ("smask") is an object array with one row per uncontracted
    shell (i.e. per row of a shell-separated mol's ``_bas``), each row
    ``[selected, nfuncs, angl, (atom_id, atom_symbol, shell_index,
    angl_label, ecp_adjusted_shell_index)]``:

    - ``selected`` : bool -- whether this shell is toggled on. Always
      `False` here.
    - ``nfuncs`` : int -- number of AO functions this shell contributes
      (spherical or Cartesian, per `cart`).
    - ``angl`` : int -- angular momentum quantum number.
    - the 5-tuple identifies the shell for labeling/linking purposes; ECP
      core shells are excluded from the shell numbering (`ecp_adjusted_shell_index`).

    Parameters
    ----------
    mol : pyscf.gto.MoleBase
        Molecule object, typically shell-separated (see
        `adb.molutil.create_shell_separated_mol`).
    cart : bool, default False
        Whether `mol` uses Cartesian (rather than spherical) AOs.

    Returns
    -------
    ndarray, dtype=object, shape (nbas, 4)
        The initial shell mask, every shell deselected.
    """
    smask = []

    shell_count = np.zeros((mol.natm, 9), dtype=int)
    for ib in range(mol.nbas):
        ia = mol.bas_atom(ib)
        angl = mol.bas_angular(ib)
        nc = mol.bas_nctr(ib)
        symb = mol.atom_symbol(ia)
        nelec_ecp = mol.atom_nelec_core(ia)
        if nelec_ecp == 0 or angl > 3:
            shl_start = shell_count[ia, angl] + angl + 1
        else:
            coreshl = core_configuration(nelec_ecp, atom_symbol=_std_symbol(symb))
            shl_start = coreshl[angl] + shell_count[ia, angl] + angl + 1
        shell_count[ia, angl] += nc
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

    return np.array(smask, dtype=object)


def smask_to_mask(smask: np.ndarray, cart: bool = False) -> np.ndarray:
    """Expand a shell mask into a per-function (AO) mask.

    Parameters
    ----------
    smask : ndarray
        Shell mask, as returned by `init_smask`.
    cart : bool, default False
        Whether `smask` describes Cartesian (rather than spherical) AOs.

    Returns
    -------
    ndarray, dtype=bool, shape (nao,)
        `True` for every AO belonging to a selected shell.
    """
    funcs_per_shell = [
        ((s[2] + 1) * (s[2] + 2) // 2 if cart else 2 * s[2] + 1) for s in smask
    ]
    mask = [False] * sum(funcs_per_shell)
    for i, sm in enumerate(smask):
        if sm[0]:
            rb = sum(funcs_per_shell[:i])
            re = rb + sm[1]
            mask[rb:re] = [True] * (re - rb)
    return np.array(mask, dtype=bool)


def mask_to_smask(mask: np.ndarray, smask: np.ndarray, cart: bool = False) -> np.ndarray:
    """Select every shell in `smask` that has one or more functions selected in `mask`.

    Parameters
    ----------
    mask : ndarray
        Per-function (AO) mask.
    smask : ndarray
        Shell mask to update in place (and return).
    cart : bool, default False
        Whether `mask`/`smask` describe Cartesian (rather than spherical)
        AOs.

    Returns
    -------
    ndarray
        `smask`, with every shell containing a selected AO now selected.
    """
    mapping = maskidx_to_smaskidx(mask, smask, cart)
    for i in np.argwhere(mask):
        smask[mapping[i[0]]][0] = True

    return smask


def linked_shell_idx(smask: np.ndarray) -> list[list[int]]:
    """Group shell-mask indices by symmetry-equivalent (duplicate) shells.

    If a molecule has more than one atom of the same element, each of
    those atoms' shells are duplicates of one another; `expand_mask` uses
    this grouping to toggle all of them together (`link_shells=True`).

    Parameters
    ----------
    smask : ndarray
        Shell mask array.

    Returns
    -------
    list of list of int
        One list of `smask` indices per distinct shell "identity" (angular
        momentum + shell label, shared across symmetry-equivalent atoms).
    """
    atoms_found = []
    shells = ["".join([str(s) for s in sm[3][1:]]) for sm in smask]

    shl_indices = []
    for sm in smask:
        shell_id = "".join([str(s) for s in sm[3][1:]])
        if shell_id not in atoms_found:
            atoms_found.append(shell_id)
            indices = [ind for ind, ele in zip(count(), shells) if ele == shell_id]
            shl_indices.append(indices)
    return shl_indices


def get_all_shell_labels(mol: gto.MoleBase) -> list[tuple[int, str, str]]:
    """List every shell of `mol` as ``(atom_id, atom_symbol, shell_label)``.

    ``shell_label`` follows the usual quantum-chemistry convention, e.g.
    ``'2S'``, ``'3P'`` (shell index, then angular-momentum letter); ECP
    core shells are excluded from the shell numbering.

    Parameters
    ----------
    mol : pyscf.gto.MoleBase
        Molecule object.

    Returns
    -------
    list of (int, str, str)
        One ``(atom_id, atom_symbol, shell_label)`` tuple per shell, in
        `mol`'s internal shell order.
    """
    shell_count = np.zeros((mol.natm, 9), dtype=int)
    labels = []
    for ib in range(mol.nbas):
        ia = mol.bas_atom(ib)
        l = mol.bas_angular(ib)
        strl = CONSTANTS.ANGULAR[l]
        nc = mol.bas_nctr(ib)
        symb = mol.atom_symbol(ia)
        nelec_ecp = mol.atom_nelec_core(ia)

        if nelec_ecp == 0 or l > 3:
            shl_start = shell_count[ia, l] + l + 1
        else:
            coreshl = core_configuration(nelec_ecp, atom_symbol=_std_symbol(symb))
            shl_start = coreshl[l] + shell_count[ia, l] + l + 1
        shell_count[ia, l] += nc
        for n in range(shl_start, shl_start + nc):
            labels.append((ia, symb, '%d%s' % (n, strl)))

    return labels


def link_shells(mol: gto.MoleBase, mask: np.ndarray) -> np.ndarray:
    """Round a per-function mask up to whole shells, atom-linked.

    Any AO in `mask` pulls its whole shell in (via `mask_to_smask`), for
    every symmetry-equivalent atom of the same element (via `init_smask`'s
    duplicate-shell bookkeeping) -- i.e. toggling one function on one atom
    toggles the matching function on every chemically-equivalent atom too.

    Parameters
    ----------
    mol : pyscf.gto.MoleBase
        Molecule object.
    mask : ndarray
        Per-function (AO) mask.

    Returns
    -------
    ndarray, dtype=bool
        The shell-and-atom-rounded mask.
    """
    smask = mask_to_smask(mask, init_smask(mol, mol.cart), mol.cart)
    return smask_to_mask(smask, mol.cart)


def get_atom_shell_label(mol: gto.MoleBase, shl_idx: int, link_shells: bool = False) -> str:
    """Human-readable label for one shell.

    Parameters
    ----------
    mol : pyscf.gto.MoleBase
        Molecule object.
    shl_idx : int
        Index into `get_all_shell_labels(mol)`.
    link_shells : bool, default False
        If `True`, omit the atom index (e.g. ``'H 2S'`` instead of
        ``'1 H 2S'``) -- used when the shell was toggled for every
        symmetry-equivalent atom at once.

    Returns
    -------
    str
        ``'<atom_symbol> <shell_label>'`` if `link_shells`, else
        ``'<atom_id> <atom_symbol> <shell_label>'``.
    """
    labels = get_all_shell_labels(mol)

    if link_shells:
        return '%s %s' % labels[shl_idx][1:]
    return '%d %s %s' % labels[shl_idx]


def print_shells(mol: gto.MoleBase, smask: np.ndarray) -> None:
    """Print one line per selected shell in `smask`.

    Parameters
    ----------
    mol : pyscf.gto.MoleBase
        Molecule object.
    smask : ndarray
        Shell mask.
    """
    labels = get_all_shell_labels(mol)
    for i, sm in enumerate(smask):
        if not sm[0]:
            continue
        print('Atom %d, symb: %s, shell: %s' % labels[i])


def maskidx_to_smaskidx(mask: np.ndarray, smask: np.ndarray, cart: bool = False) -> list[int]:
    """Map every AO index in `mask` to its owning shell index in `smask`.

    Parameters
    ----------
    mask : ndarray
        Per-function (AO) mask (only its length is used).
    smask : ndarray
        Shell mask.
    cart : bool, default False
        Whether `mask`/`smask` describe Cartesian (rather than spherical)
        AOs.

    Returns
    -------
    list of int
        `mapping[i]` is the `smask` row index owning AO `i`.
    """
    mapping = [0] * len(mask)
    counter = 0
    for i, sm in enumerate(smask):
        angl = sm[2]
        for _ in range((angl + 1) * (angl + 2) // 2 if cart else 2 * angl + 1):
            mapping[counter] = i
            counter += 1
    return mapping


def set_linked_shells(smask: np.ndarray) -> np.ndarray:
    """Select every shell that is symmetry-linked to an already-selected shell.

    For each already-selected shell, finds every other shell sharing the
    same identity (angular momentum + shell label -- see
    `linked_shell_idx`) across symmetry-equivalent atoms, and selects those
    too.

    Parameters
    ----------
    smask : ndarray
        Shell mask.

    Returns
    -------
    ndarray
        A copy of `smask` with all symmetry-linked shells selected.
    """
    csmask = deepcopy(smask)
    selected_shells = np.argwhere([sm[0] for sm in csmask])
    all_shells = np.array([''.join(map(str, sm[3][1:])) for sm in csmask])
    same_as_selected = np.argwhere(all_shells == all_shells[selected_shells])[:, 1]
    csmask[same_as_selected, 0] = True
    return csmask


def mask_matrix(mat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Restrict a matrix to the AOs selected by `mask`, along every AO axis.

    Parameters
    ----------
    mat : ndarray, shape (nao, nao) or (2, nao, nao)
        Matrix to mask, e.g. a Fock or overlap matrix. Restrictedness is
        inferred from `mat.ndim`: 2D is masked on both axes; a leading
        spin axis of length 2 (unrestricted) is masked on the trailing two
        axes only.
    mask : ndarray, dtype=bool, shape (nao,)
        The AO mask.

    Returns
    -------
    ndarray
        The masked matrix, shape ``(n_sel, n_sel)`` or ``(2, n_sel, n_sel)``.
    """
    is_restricted = (mat.ndim == 2)
    return mat[mask, :][:, mask] if is_restricted else mat[:, mask, :][:, :, mask]
