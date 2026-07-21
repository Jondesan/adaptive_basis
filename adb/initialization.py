"""Initial minimal-basis mask construction strategies for adb.search.find_subspace."""

import numpy as np
from pyscf import gto
from pyscf.gto.mole import aoslice_by_atom, intor_cross

from .calculations import eig, get_q_sqrd, spherical_average
from .CONSTANTS import ELEMENTS
from .ioutil import (
    print_atomic_block_atom_header,
    print_atomic_block_energies_debug,
    print_atomic_block_state_energies,
    print_restricted_atom_orbital_summary,
    print_unrestricted_atom_orbital_summary,
)
from .maskutil import (
    init_smask,
    link_shells,
    mask_matrix,
    mask_to_smask,
    smask_to_mask
)
from .molutil import (
    basis_functions_per_atom,
    create_mol_from_template,
    funcs_on_shell,
    get_array_of_angular_momenta_and_atom_id,
)


def atomic_block_minimal_basis(
        mol:                        gto.Mole,
        F:                          np.ndarray,
        S:                          np.ndarray,
        Q_tol:                      float   = 1.0,
        by_shell:                   bool    = True,
        verbose:                    bool    = False,
        spherically_average_fock:   bool    = True,
        ) -> np.ndarray:
    """Construct a minimal-basis mask via atomic block decomposition (ABD).

    For each atom, diagonalizes its own diagonal block of the Fock matrix
    to get atom-local pseudo-orbitals, then greedily selects the AOs
    contributing most to those orbitals' density until the selected
    subspace accounts for at least ``Qlim - Q_tol`` of the atom's charge
    (`Qlim` being the number of atom-local occupied states).

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object.
    F : ndarray, shape (nao, nao) or (2, nao, nao)
        Fock matrix (guess or converged). A stacked ``(2, nao, nao)`` array
        is treated as unrestricted.
    S : ndarray, shape (nao, nao)
        Overlap matrix.
    Q_tol : float, default 1.0
        Charge tolerance: how much of an atom's charge the minimal basis
        is allowed to not account for. Must be smaller than the atom's
        number of occupied states.
    by_shell : bool, default True
        Whether to select whole shells rather than individual functions.
        Must be `True` -- `by_shell=False` is not implemented (see Raises).
    verbose : bool, default False
        Whether to print per-atom diagnostic output.
    spherically_average_fock : bool, default True
        Whether to spherically average each atom's Fock block (see
        `adb.calculations.spherical_average`) before diagonalizing it, to
        avoid the atom-in-molecule environment's angular anisotropy biasing
        the atomic-orbital selection.

    Returns
    -------
    ndarray, dtype=bool, shape (nao,)
        The minimal-basis AO mask.

    Raises
    ------
    ValueError
        If `Q_tol` is not smaller than some atom's number of occupied
        states.
    NotImplementedError
        If `by_shell` is `False`.
    """
    func_per_atom = basis_functions_per_atom(mol)
    assert np.sum(func_per_atom) == mol.nao

    minimal_basis_mask = np.zeros(mol.nao, dtype=bool)
    smask = init_smask(mol, mol.cart)

    atoms = list(map(lambda x: x[0], mol._atom))

    restricted = (F.ndim == 2)

    # Loop through atomic blocks in the Fock matrix
    nfuncs_min_tot = 0
    for i, funcs_and_atom in enumerate(zip(func_per_atom, atoms)):
        nfuncs, atom = funcs_and_atom
        smask_atom = list(filter(lambda x: x[3][0] == i, smask))
        mask = np.zeros(mol.nao, dtype=bool)
        mask_atom = np.zeros(func_per_atom[i], dtype=bool)
        func_offset = np.sum(func_per_atom[:i])
        mask[func_offset:func_offset + nfuncs] = True
        S_atom = mask_matrix(S, mask)
        F_atom = mask_matrix(F, mask)
        # Number of functions in minimal basis of current atom,
        # not counting ECP electrons
        nfunc_per_minimal_atom = int(np.ceil(
            (ELEMENTS.index(atom) - mol.atom_nelec_core(i)) / 2))
        if verbose:
            print_atomic_block_atom_header(atom, nfunc_per_minimal_atom, nfuncs)
        nfuncs_min_tot += nfunc_per_minimal_atom

        F_ave = F_atom.copy()
        if spherically_average_fock:
            F_ave = spherical_average(F_ave, [shell[1] for shell in smask_atom])

        e_atom, c_atom = eig(F_ave, S_atom.copy())

        def number_of_states(energies, thresh=1e-3):
            """Number of atom-local states to occupy, expanding past
            `nfunc_per_minimal_atom` to absorb any near-degenerate states
            within `thresh` of the highest minimal-basis state."""
            if verbose:
                print_atomic_block_energies_debug(energies)
            nfuncs_include = nfunc_per_minimal_atom
            while nfuncs_include < nfuncs \
              and energies[nfuncs_include] - energies[nfunc_per_minimal_atom - 1] < thresh:
                nfuncs_include += 1
            return nfuncs_include

        if restricted:
            nocca, noccb = number_of_states(e_atom), number_of_states(e_atom)
            if verbose:
                print_restricted_atom_orbital_summary(nocca, noccb, e_atom)
            occs = np.zeros(c_atom.shape[1])
            occs[:nocca] = 2
            P_atom = np.abs(c_atom @ np.diag(occs) @ c_atom.conj().T)
        else:
            nocca, noccb = number_of_states(e_atom[0]), number_of_states(e_atom[1])
            if verbose:
                print_unrestricted_atom_orbital_summary(nocca, noccb, e_atom)
            occs = np.zeros((2, c_atom.shape[2]))
            occs[0, :nocca] = 1
            occs[1, :noccb] = 1
            P_atom = np.abs(
                c_atom[0] @ np.diag(occs[0]) @ c_atom[0].conj().T +
                c_atom[1] @ np.diag(occs[1]) @ c_atom[1].conj().T
            )
        Qlim = nocca + noccb
        if verbose:
            print_atomic_block_state_energies(Qlim, e_atom, nocca, noccb, restricted)

        atom_indices = set()
        Q = 0
        eps = Q_tol
        if eps >= Qlim:
            raise ValueError(f'Tolerance for Q must be smaller than the number of states, {Qlim=}!')
        P_atom = np.round(P_atom, 12)
        while np.abs(Q - Qlim) > eps:
            # Find largest element of density matrix
            P_atom_idx = np.unravel_index(np.argmax(P_atom, axis=None), P_atom.shape)
            Pat_i, Pat_j = P_atom_idx

            if by_shell:
                # Set functions of same shell to True
                mask_atom[Pat_i] = True
                mask_atom[Pat_j] = True
                smask_atom = mask_to_smask(mask_atom, smask_atom, mol.cart)
                mask_atom = smask_to_mask(smask_atom, mol.cart)
                _, c_mask = eig(
                    mask_matrix(F_ave.copy(), mask_atom),
                    mask_matrix(S_atom.copy(), mask_atom)
                )

                # Zero the rows/columns just claimed so they aren't picked again
                P_atom[mask_atom, :] = 0
                P_atom[:, mask_atom] = 0

                atom_indices.update(np.where(mask_atom)[0].tolist())
                Q = get_q_sqrd(
                    c_atom.copy(), c_mask,
                    S_atom[:, mask_atom].copy(),
                    (nocca, noccb),
                )
            else:
                raise NotImplementedError('The function-by-function branch has not been implemented yet.')

        atom_indices = list(atom_indices)
        minimal_basis_mask[func_offset + np.asarray(atom_indices)] = True

    assert np.sum(minimal_basis_mask) >= nfuncs_min_tot
    return minimal_basis_mask


def find_projected_minimal_basis_mask(mol: gto.Mole) -> np.ndarray:
    """Construct a minimal-basis mask by projecting onto an STO-3G reference.

    For each STO-3G shell, picks the single AO of `mol`'s own (larger)
    basis with the largest overlap onto it, then rounds the selection up
    to whole shells for every symmetry-equivalent atom (`link_shells`).

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object (in its own, larger basis).

    Returns
    -------
    ndarray, dtype=bool, shape (nao,)
        The projected minimal-basis AO mask.

    Raises
    ------
    RuntimeError
        If the final (shell-and-atom-rounded) mask doesn't select exactly
        as many functions as the STO-3G reference has.
    """
    mol_sto3g = create_mol_from_template(mol, basis='sto3g')
    mask = np.zeros(mol.nao_nr(), dtype=bool)
    s21 = intor_cross('int1e_ovlp', mol, mol_sto3g)

    # Find the AO-id offsets of the atoms
    atom_offsets = aoslice_by_atom(mol)[:, 2]

    # Generate arrays with the angular momentum l and atom-id
    # for all functions. Element n corresponds the l and atom-id
    # of the n:th basis function
    sto3g_angls = get_array_of_angular_momenta_and_atom_id(mol_sto3g)
    sto3g_aid = sto3g_angls[:, 1]
    sto3g_angls = sto3g_angls[:, 0]

    for (ovlp_col, angl, atom_id) in zip(s21.T, sto3g_angls, sto3g_aid):
        # Count functions of angular momentum angl in large basis
        # for the current atom, consider contractions
        nfunc_angl = sum(
            x[3] for x in mol._bas if x[0] == atom_id and x[1] == angl
        )
        # Remember to multiply by the number of allowed
        # magnetic quantum numbers
        nfunc_angl *= funcs_on_shell(angl, mol.cart)

        # Count the offset of the current shell
        shell_offset = atom_offsets[atom_id]
        # Make sure to multiply by the number of allowed magnetic quant. nums
        shell_offset += sum(
            x[3] * funcs_on_shell(x[1], mol.cart)
            for x in mol._bas if x[0] == atom_id and x[1] < angl
        )

        # This guarantees no function will be chosen twice by removing already
        # selected functions from the pool of available ones
        ovlp_col[mask] = 0.0

        idx = np.argmax(np.abs(ovlp_col[shell_offset:shell_offset + nfunc_angl]))
        mask[shell_offset + idx] = 1

    mask = link_shells(mol, mask)
    if np.sum(mask) != mol_sto3g.nao_nr():
        raise RuntimeError(
            f"Number of functions in the projected minimal basis [{np.sum(mask)}] "
            f"does not match the number of functions in the actual minimal basis "
            f"[{mol_sto3g.nao_nr()}]!"
        )

    return mask
