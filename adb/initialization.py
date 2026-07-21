"""Initial minimal-basis mask construction strategies for adb.search.find_subspace"""

import copy
import numpy as np
from pyscf import gto
from pyscf.gto.mole import intor_cross, aoslice_by_atom
from .molutil import (
    basis_functions_per_atom, create_mol_from_template,
    get_array_of_angular_momenta_and_atom_id, funcs_on_shell,
    )
from .maskutil import init_smask, mask_to_smask, smask_to_mask, mask_matrix, link_shells
from .calculations import eig, get_q_sqrd, spherical_average
from .CONSTANTS import ELEMENTS
from .ioutil import (
    print_atomic_block_atom_header, print_atomic_block_energies_debug,
    print_restricted_atom_orbital_summary, print_unrestricted_atom_orbital_summary,
    print_atomic_block_state_energies,
    )


def atomic_block_minimal_basis(
    mol:                        gto.Mole,
    F:                          np.ndarray,
    S:                          np.ndarray,
    Q_tol:                      float           = 1.0,
    by_shell:                   bool            = True,
    verbose:                    bool            = False,
    spherically_average_fock:   bool            = True,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Create minimal basis from atomic block decomposition.
    """
    func_per_atom = basis_functions_per_atom(mol)
    assert np.sum(func_per_atom) == mol.nao

    minimal_basis_mask = np.zeros(mol.nao, dtype=bool)
    # if by_shell:
    smask = init_smask(mol, mol.cart)

    atoms = list(map(lambda x: x[0], mol._atom))

    restricted = (len(F.shape) == 2)

    # Loop through atomic blocks in the Fock matrix
    nfuncs_min_tot = 0
    for i,funcs_and_atom in enumerate(zip(func_per_atom, atoms)):
        nfuncs, atom = funcs_and_atom
        smask_atom = list(filter(lambda x: x[3][0] == i, smask))
        mask = np.zeros(mol.nao, dtype=bool)
        mask_atom = np.zeros(func_per_atom[i], dtype=bool)
        func_offset = np.sum(func_per_atom[:i])
        mask[func_offset:func_offset+nfuncs] = True
        S_atom = mask_matrix(S, mask)
        F_atom = mask_matrix(F, mask)
        # Number of functions in minimal basis of current atom,
        # not counting ECP electrons
        nfunc_per_minimal_atom = int(np.ceil(
            (ELEMENTS.index(atom)-mol.atom_nelec_core(i)) / 2))
        if verbose:
            print_atomic_block_atom_header(atom, nfunc_per_minimal_atom, nfuncs)
        # Add to molecule minimal number of functions
        nfuncs_min_tot += nfunc_per_minimal_atom

        F_ave = F_atom.copy()
        if spherically_average_fock:
            F_ave = spherical_average(F_ave, [shell[1] for shell in smask_atom])

        e_atom, c_atom = eig(F_ave, S_atom.copy())

        def number_of_states(energies, thresh=1e-3):
            if verbose:
                print_atomic_block_energies_debug(energies)
            nfuncs_include = nfunc_per_minimal_atom
            # Handle degeneracies
            while nfuncs_include < nfuncs \
              and energies[nfuncs_include]-energies[nfunc_per_minimal_atom-1] < thresh:
                nfuncs_include += 1
            return nfuncs_include


        if restricted:
            nocca, noccb = number_of_states(e_atom), number_of_states(e_atom)
            if verbose:
                print_restricted_atom_orbital_summary(nocca, noccb, e_atom)
            occs = np.zeros(c_atom.shape[1])
            occs[:nocca] = 2
            P_atom = np.abs(
                c_atom @ np.diag(occs) @ c_atom.conj().T
            )
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
        Qlim = nocca+noccb
        if verbose:
            print_atomic_block_state_energies(Qlim, e_atom, nocca, noccb, restricted)

        atom_indices = set()
        Q = 0
        eps = Q_tol
        if eps >= Qlim:
            raise ValueError(f'Tolerance for Q must be smaller than the number of states, {Qlim=}!')
        # while len(atom_indices) < nfunc_per_minimal_atom:
        P_atom = np.round(P_atom, 12)
        while np.abs(Q - Qlim) > eps:
            # Find largest element of density matrix
            P_atom_idx = np.unravel_index(np.argmax(P_atom, axis=None),P_atom.shape)
            
            Pat_i, Pat_j = P_atom_idx

            # Set functions of same shell to True
            if by_shell:
                mask_atom[Pat_i] = True
                mask_atom[Pat_j] = True
                smask_atom = mask_to_smask(mask_atom, smask_atom, mol.cart)
                mask_atom = smask_to_mask(smask_atom, mol.cart)
                _, c_mask = eig(
                    mask_matrix(F_ave.copy(), mask_atom),
                    mask_matrix(S_atom.copy(), mask_atom)
                    )

                # set elements i,j and j,i of P_atom to zero
                P_atom[mask_atom, :] = 0
                P_atom[:, mask_atom] = 0

                # add indices where mask is True to atom_indices
                atom_indices.update(np.where(mask_atom)[0].tolist())
                Q = get_q_sqrd(
                    c_atom.copy(), c_mask,
                    S_atom[:, mask_atom].copy(),
                    (nocca, noccb),
                    )
            else:
                raise NotImplementedError('The function-by-function branch has not been implemented yet.')
                atom_indices.extend(list(set((Pat_i, Pat_j))))
                P_atom[Pat_i, Pat_j] = 0
                P_atom[np.flip((Pat_i, Pat_j))] = 0
                # fixme: update c_mask and Q

        # Mask
        atom_indices = list(atom_indices)
        minimal_basis_mask[func_offset + np.asarray(atom_indices)] = True

    assert np.sum(minimal_basis_mask) >= nfuncs_min_tot
    return minimal_basis_mask


def find_projected_minimal_basis_mask(
        mol,
    ):
    mol_sto3g = create_mol_from_template(mol, basis='sto3g')
    mask = np.zeros(mol.nao_nr(), dtype=bool)
    s21 = intor_cross('int1e_ovlp', mol, mol_sto3g)

    # Find the AO-id offsets of the atoms
    atom_offsets = aoslice_by_atom(mol)[:,2]

    # Generate arrays with the angular momentum l and atom-id
    # for all functions. Element n corresponds the l and atom-id
    # of the n:th basis function
    sto3g_angls = get_array_of_angular_momenta_and_atom_id(mol_sto3g)
    sto3g_aid = sto3g_angls[:, 1]
    sto3g_angls = sto3g_angls[:, 0]

    prev_angl = None
    prev_aid = None
    shell_offset = 0

    for (ovlp_col, angl, atom_id) in zip(s21.T, sto3g_angls, sto3g_aid):
        # Count functions of angular momentum angl in large basis
        # for the current atom, consider contractions
        nfunc_angl = sum(
            x[3] for x in mol._bas if x[0] == atom_id and x[1] == angl
        )
        # Remember to multiply by the number of allowed
        # magnetic quantum numbers
        nfunc_angl *= funcs_on_shell(angl, mol.cart)

        # Initialise the shell mask if first round, new shell or new atom
        if prev_angl != angl or prev_angl is None or prev_aid != atom_id or prev_aid is None:
            prev_aid = atom_id
            prev_angl = angl

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
        raise RuntimeError(f"Number of functions in the projected minimal basis [{np.sum(mask)}] does not match the number of functions in the actual minimal basis [{mol_sto3g.nao_nr()}]!")

    return mask
