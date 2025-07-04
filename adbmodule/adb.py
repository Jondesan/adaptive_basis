"""Adaptive basis set method"""

# import pyscf
import numpy as np
from scipy import linalg
from itertools import count
from pyscf.gto.basis.parse_nwchem import convert_basis_to_nwchem,\
    to_general_contraction, convert_ecp_to_nwchem
from pyscf.gto.ecp import core_configuration
from pyscf.data.elements import _std_symbol, ELEMENTS
from pyscf.gto.basis.parse_nwchem import load
from pyscf.gto.mole import *
from pyscf.scf import *
from pyscf.scf.addons import canonical_orth_
from pyscf import lib, gto, scf
from warnings import warn
from operator import itemgetter
import adbutils
import copy
import sys
import re

VARIANTS = [
    'enocc', # Energy sum of occupied orbitals
    #'ecore', # Energy sum of orbitals and core H
    'elden', # Electron density
]
NFUNCS = {
    'S': 1,
    'P': 3,
    'D': 5,
    'F': 7,
    'G': 9,
    'H': 11,
    'I': 13,
    'J': 15,
}

def tk_debugger(*vars):
    print('######## TEEKKARIN DEBUGGER #############################')
    for var in vars:
        print(var, end=' ')
    print()
    print('######## TEEKKARIN DEBUGGER END #########################')

def eigh(
        h: np.ndarray,
        s: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
    """Wrapper for eigh, calculates orthogonalisation for RHF and UHF.
    """
    if len(np.asarray(h).shape) == 3:
        ea, ca = canonical_orth(h[0], s)
        eb, cb = canonical_orth(h[1], s)
        return np.asarray([ea, eb]), np.asarray([ca, cb])
    else:
        return canonical_orth(h, s)


def canonical_orth(
        h: np.ndarray | tuple[np.ndarray, np.ndarray],
        s: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
    """Modified canonical orthogonalisation.

    Args:
        h : ndarray
            Fock matrix
        s : ndarray
            Overlap matrix
        get_idx : bool
            Whether to return the indices that sort the eigenvalues.
            Default is False

    Returns:
        Sorted eigenvalues (ascending) and coefficients, if get_idx is
        True the indices that sort the eigenvalues are also returned
    """
    x = canonical_orth_(s, 1e-8)
    xhx = x.conj().T @ h @ x
    e, c = linalg.eigh(xhx)
    c = x @ c
    idx = np.argsort(e)
    return e[idx], c[:,idx]


def extract_basis(
        smask:          np.ndarray,
        shellsep_mol:   gto.MoleBase
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
    for elem in copy.deepcopy(smask[np.asarray(smask[:, 0], dtype=bool)]):
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
    for key in asymb:#basis.keys():
        ogbas = to_general_contraction(shellsep_mol._basis[key])
        # Important when initialization does not put functions on all
        # atoms in the molecule, would result in error
        if basis[key] is None:
            continue
        for shell in basis[key]:
            i = shell[0]
            key_smask = [drs for drs in duplicate_removed_smask if drs[3][1] == key]
            idxs = [idx[3][4] - idx[2] for idx in key_smask if idx[2] == i]
            coeff_table = np.asarray(ogbas[i][1:], dtype=float)[:, [0] + idxs]
            # Remove rows and columns with all 0 contraction coeffs
            filtered_shell = coeff_table[
                ~((coeff_table[:, 0] != 0) &
                (coeff_table[:, 1:] == 0).all(axis=1))]
            filtered_shell = filtered_shell[~np.all(filtered_shell == 0, axis=1)]
            if not filtered_shell.tolist():
                basis[key].pop(i)
            else:
                shell.extend(filtered_shell.tolist())
    ecp = shellsep_mol._ecp if shellsep_mol._ecp != {} else None
    return basis, ecp


def basis_to_file_nwchem(
    basis:              dict,
    fn:                 str,
    ecp_basis:          dict | None = None,
    commentstring:      str = "",
    bsname:             str = "ao basis",
    cart:               bool = False,
    print_noprint:      str = "print",
    additional_labels:  str = "" ) -> None:
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


def get_uncontracted_basis(
        mol:    gto.MoleBase,
        fn:     str | None = None) -> str:
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
    line = 'BASIS "ao basis" PRINT\n'
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
            coeffs = np.array(shell[1:])
            contractions = coeffs.shape[1]
            for i in range(1, contractions):
                line = (
                    asy + "\t" + lib.param.ANGULAR[shell[0]].capitalize() + "\n"
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

    dc = dict()
    for elem in basis.split("#")[1:]:
        dc[elem[11]] = gto.basis.parse(str(elem[11:]))
    return dc


def get_shells(mol: gto.MoleBase) -> np.ndarray:
    """Get the shell structure of mol object.

    Args:
        mol : pyscf.gto.MoleBase
            The molecule object.

    Returns:
        A 1D ndarray with the number of functions per shell as elements.
        Shells are ordered in the pyscf internal format.
    """
    shells = np.array([], dtype=int)  # Number of functions per shell

    for ib in range(mol.nbas):  # nbas = number of shells (basis fcts)
        angl = mol.bas_angular(ib)  # angular momentum l of given basis function
        nc = mol.bas_nctr(ib)  # number of CGTOs for given shell

        shells = np.append(
            shells, nc * (angl + 1) * (angl + 2) // 2 if mol.cart else nc * (2 * angl + 1)
        )

    if sum(shells) != mol.nao_nr():
        raise Exception(
            "Number of functions in the mask does not correspond with number of functions of the molecule!"
        )

    return shells


def maskidx_to_smaskidx(
        mask:   np.ndarray,
        smask:  np.ndarray,
        cart:   bool = False) -> list:
    """Create mapping between mask and smask"""
    mapping = [0] * len(mask)
    counter = 0
    for i, sm in enumerate(smask):
        angl = sm[2]
        for j in range((angl + 1) * (angl + 2) // 2 if cart else 2 * angl + 1):
            mapping[counter] = i
            counter += 1
    return mapping


def init_smask(
        mol:    gto.MoleBase,
        cart:   bool = False) -> np.ndarray:
    """Initialize the shell mask array. smask will be a list of lists,
    with length equal to the number of uncontracted shells, and each
    element is a two element list, first is bool that specifies the mask
    for the current shell, the other how many primitives in this shell.
    """

    smask = []

    count = np.zeros((mol.natm, 9), dtype=int)
    for ib in range(mol.nbas):
        ia = mol.bas_atom(ib)       # atom that given basis function sits on
        angl = mol.bas_angular(ib)  # angular momentum angl of given basis function
        nc = mol.bas_nctr(ib)       # number of CGTOs for given shell
        symb = mol.atom_symbol(ia)  # label of given atom
        nelec_ecp = mol.atom_nelec_core(ia)  # Number of ecp electrons
        if nelec_ecp == 0 or angl > 3:
            shl_start = count[ia, angl] + angl + 1
        else:
            coreshl = core_configuration(nelec_ecp, atom_symbol=_std_symbol(symb))
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
                    (ia, symb, n, lib.param.ANGULAR[angl].capitalize(), n_remove_ecp),
                ]
            )

    return np.array(smask, dtype=object)


def smask_to_mask(
        smask:  np.ndarray,
        cart:   bool = False) -> np.ndarray:
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
    return np.array(mask, dtype=bool)


def mask_to_smask(
        mask:   np.ndarray,
        smask:  np.ndarray,
        cart:   bool = False) -> np.ndarray:
    """Flip shells of smask to True that have 1 or more functions set to
    True in mask.
    """
    mapping = maskidx_to_smaskidx(mask, smask, cart)
    for i in np.argwhere(mask):
        smask[mapping[i[0]]][0] = True

    return smask


def get_iteration_criteria_value(
    variant:    str,
    epsilon_i:  np.ndarray | None   = None,
    nocc:       tuple | None        = None,
    sub_hcore:  np.ndarray | None   = None,
    Csub:       np.ndarray | None   = None,
    Cfull:      np.ndarray | None   = None,
    ovlp:       np.ndarray | None   = None
    ) -> float:
    """Calculates the value of the chosen variants criteria.

    Args:
        variant : string
            Which variant to calculate.
        The needed variables for calculating the different criteria.
        enocc:
        epislon_i : ndarray
            energy eigenvalues
        nocc : tuple
            number of occupations
        ecore: 
        evals : ndarray
            energy eigenvalues
        nocc : tuple
            number of occupations
        sub_hcore : 2D array
            subbasis core hamiltonian hcore,
        Csub : ndarray
            subbasis coefficient matrix Csub
        elden:
        Cfull : ndarray
            full basis coeff matrix
        Csub : ndarray
            subbasis coeff matrix
        ovlp : 2D array
            overlap matrix
        nocc : tuple
            number of occupations
    
    Returns:
        criteria : float
            Value of the criteria
    """
    criteria = 0.0
    if variant not in VARIANTS:
        raise RuntimeError(
            'The variant you are trying to use was not recognised!')
    match variant:
        case 'enocc':
            RHF = (len(np.asarray(epsilon_i).shape) == 1)
            if RHF:
                criteria = np.sum(epsilon_i[:nocc[0]])
            else:
                criteria  = np.sum(epsilon_i[0,:nocc[0]])
                criteria += np.sum(epsilon_i[1,:nocc[1]]) 
            return criteria
        case 'ecore':
            mocc = Csub[:,:nocc]
            P = mocc @ mocc.conj().T
            return np.sum(
                P[:nocc,:nocc]
                @ (sub_hcore + np.diag(epsilon_i))[:nocc,:nocc] )
        case 'elden':
            return get_q_sqrd(Cfull, Csub, ovlp, nocc)


def linked_shell_idx(smask: np.ndarray) -> np.ndarray:
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
    return np.asarray(shl_indices)

def expand_mask(
    F:                      np.ndarray,
    S:                      np.ndarray,
    nocc:                   tuple,
    mask:                   np.ndarray,
    smask:                  np.ndarray | None   = None,
    variant:                str                 = 'enocc',
    hcore:                  np.ndarray | None   = None,
    Cfull:                  np.ndarray | None   = None,
    link_shells:            bool                = True,
    nfunc_normalisation:    bool                = True,
    ) -> tuple[np.ndarray, float, float, np.ndarray | None]:
    r"""Expands the current mask by either one function or one shell
    based on smask.

    Args:
        F : ndarray
            Full Fock matrix
        S : ndarray
            Full overlap matrix
        nocc : tuple
            Number of occupied alpha and beta orbitals
        mask : ndarray
            The current mask. A logical 1d array
        smask : None or ndarray
            If None functions are tested individually. Else shell by
            shell testing is used where shells are determined by the
            smask array, where the elements represent the number of
            functions per current shell. The shells are ordered in the
            PySCF internal format
        variant : str
            Which variant to use. Specifies what will be the
            minimisation criteria for adding a function/shell.
            enocc: $\sum_{i}^{nocc}\epsilon_i$,
               where $epsilon_i$ are the occupied diagonal Fock matrx
               elements
            ecore: $\frac{1}{2}\sum_{i}^{occ}(\epsilon_i+h_{ii})$,
               where $h_{ii}=C_i^\dagger H_{core}C_i$
            elden: $\Delta Q$,
               which is $1-\frac{1}{nocc}
                * \sum_{i,j}^{nocc}<i^{subbasis}|j^{fullbasis}>$
        link_shells : bool
            Whether to link shells of atoms of same type in the mask
            Optional, default is True
        nfunc_normalisation : bool
            Whether to normalise the criteria with the number of added
            functions.
            Optional, deault is True
        dft : bool
            Hartree-Fock or DFT.
            Optional, default is False
        xc : str
            XC functional string accepted by PySCF.
            Optional, default is 'b3lyp'.
        grid_level : int
            predefined integration grid levels, 0-9
            (0 very sparse, 9 very dense). Optional, default is 3.

    Returns:
        The new mask (boolean ndarray), the current difference in
        eigenvalue sums and the current sum (energy sum of occupied
        orbitals), shell mask if smask is provided.
    """
    RHF = (len(F.shape) == 2)
    maskedF = mask_matrix(F, mask, RHF)
    maskedS = mask_matrix(S, mask)
    evals, coeffs = eigh(maskedF, maskedS)
    last_sum = 0.0
    if Cfull is None and variant == 'elden':
        _, Cfull = eigh(F, S)
    last_sum = get_iteration_criteria_value(
        variant, epsilon_i=evals, nocc=nocc,
        sub_hcore=mask_matrix(hcore, mask), Csub=coeffs,
        Cfull=Cfull, ovlp=S[:, mask])

    test_sums = []    
    if smask is None:
        for i, m in enumerate(mask):
            if m:
                continue

            test_mask = copy.deepcopy(mask)
            test_mask[i] = True
            maskedF = mask_matrix(F, test_mask, RHF)
            maskedS = mask_matrix(S, test_mask)
            evals, coeffs = eigh(maskedF, maskedS)

            test_sums.append(
                (i,
                get_iteration_criteria_value(
                    'enocc', epsilon_i=evals, nocc=nocc,
                    sub_hcore=mask_matrix(hcore, mask), Csub=coeffs,
                    Cfull=Cfull, ovlp=S[:, test_mask]),
                1))
    else:
        # Gather indices of duplicate shells if link_shells enabled
        # (if system has more than 1 atom of same type,
        #  shells will be duplicated.)
        if link_shells:
            shl_indices = linked_shell_idx(smask)
        else:
            shl_indices = [[i] for i in range(len(smask))]

        for i, sidx in enumerate(shl_indices):
            if smask[sidx][0, 0]:
                continue
            test_smask = copy.deepcopy(smask)

            submask = test_smask[sidx]
            submask[:, 0] = True
            test_smask[sidx] = submask
            test_mask = smask_to_mask(test_smask)

            maskedF = mask_matrix(F, test_mask, RHF)
            maskedS = mask_matrix(S, test_mask)
            evals, coeffs = eigh(maskedF, maskedS)
            
            func_keys = [shell[3] for shell in submask[:,3]]
            nfuncs = np.sum(itemgetter(*func_keys)(NFUNCS))
            test_sums.append(
                (i,
                get_iteration_criteria_value(
                    variant, epsilon_i=evals, nocc=nocc,
                    sub_hcore=mask_matrix(hcore, mask), Csub=coeffs,
                    Cfull=Cfull, ovlp=S[:, test_mask]),
                nfuncs))

    if nfunc_normalisation:
        test_differences = [(test_sum[1] - last_sum) / test_sum[2] for test_sum in test_sums]
    else:
        test_differences = [(test_sum[1] - last_sum) for test_sum in test_sums]
    if variant == 'elden':
        array_index = np.argmax(test_differences)
    else:
        array_index = np.argmin(test_differences)
    current_idx_to_flip = test_sums[array_index][0]

    if smask is None:
        mask[current_idx_to_flip] = True
    else:
        submask = smask[shl_indices[current_idx_to_flip]]
        submask[:, 0] = True
        smask[shl_indices[current_idx_to_flip]] = submask
        mask = smask_to_mask(smask)
    return mask, test_differences[array_index], test_sums[array_index][1], smask


def get_all_shell_labels(mol: gto.MoleBase) -> list[str]:
    count = np.zeros((mol.natm, 9), dtype=int)
    labels = []
    for ib in range(mol.nbas):  # nbas = number of shells (basis fcts)
        ia = mol.bas_atom(ib)   # atom that given basis function sits on
        l = mol.bas_angular(ib) # angular momentum l of basis function
        strl = lib.param.ANGULAR[l] # angular momentum label
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


def get_atom_shell_label(
        mol: gto.MoleBase,
        shl_idx: int,
        link_shells: bool = False
    ) -> str:
    labels = get_all_shell_labels(mol)

    if link_shells:
        return '%s %s' % labels[shl_idx][1:]
    return '%d %s %s' % labels[shl_idx]


def print_shells(mol: gto.MoleBase, smask: np.ndarray) -> None:
    labels = get_all_shell_labels(mol)
    for i,sm in enumerate(smask):
        if not sm[0]:
            continue
        print('Atom %d, symb: %s, shell: %s' % labels[i])


def get_sub_scf_attributes(
    mol:            gto.MoleBase,
    fock:           np.ndarray,
    overlap:        np.ndarray,
    dft:            bool            = False,
    xc:             str             = 'b3lyp',
    grid_level:     int             = 7,
    ) -> tuple[float, float, np.ndarray]:
    """Calculates converged attributes for the system.

    Args:
        mol : pyscf.gto.MoleBase
            The molecule object
        dft : bool
            Hartree-Fock or DFT.
            Optional, default is False
        xc : str
            XC functional string accepted by PySCF.
            Optional, default is 'b3lyp'.
        grid_level : int
            predefined integration grid levels, 0-9 (0 very sparse, 9 very dense).
            Optional, default is 3.

    Returns:
        The SCF energy, sum of occupied orbital energies of the
        subbasis, the MO coefficient matrix of the subbasis.
    """
    RHF = (len(fock.shape) == 2)
    # if RHF:
    #     mf = mol.RHF()
    # else:
    #     mf = mol.UHF()
    mf = mol.HF()
    mf = mf.apply(scf.addons.remove_linear_dep_)
    if dft:
        mf = mf.to_ks(xc=xc)
        mf.grids.level = grid_level
        mf.grids.prune = None

    # Diagonalize fock matrix and form guess density matrix
    if fock.shape[1] > 1:
        e, c = eigh(fock, overlap)
        occ = mf.get_occ(e, c)
        dm = mf.make_rdm1(c, occ)
        mf.init_guess = dm
    mf.kernel(dump_chk=False)

    scf_energy = mf.e_tot
    # sum over occupied orbital energies
    if RHF:
        nocc_sb = len(mf.mo_occ > 0)
        scf_orbital_energy = sum(np.sort(mf.mo_energy)[:nocc_sb])
    else:
        nocc_sb = [len(mf.mo_occ[0] > 0), len(mf.mo_occ[1] > 0)]
        scf_orbital_energy = .5 * sum(
            np.sort(mf.mo_energy[0])[:nocc_sb[0]] +
            np.sort(mf.mo_energy[1])[:nocc_sb[1]])
        
    return scf_energy, scf_orbital_energy, mf.mo_coeff


def create_subbasis_mol(
        mol:        gto.MoleBase,
        smask:      np.ndarray) -> gto.MoleBase:
    extracted_basis, ecp_bas = extract_basis(smask, create_shell_separated_mol(mol))
    subbasis_mol = gto.Mole(
        atom = mol.atom, basis = extracted_basis,
        charge = mol.charge, spin = mol.spin,
        verbose = mol.verbose, unit = mol.unit,
        ecp = ecp_bas, symmetry = mol.symmetry
    )
    subbasis_mol.build()
    
    return subbasis_mol


def create_shell_separated_mol(
        mol:        gto.MoleBase,
        verbose:    int = 0) -> gto.MoleBase:
    """Creates a copy of mol with shells separated."""
    shell_sep_basis = get_uncontracted_basis(mol)
    cmol = gto.M(
        atom=mol.atom, basis=shell_sep_basis,
        charge=mol.charge, spin=mol.spin,
        unit=mol.unit, symmetry=mol.symmetry,
        ecp=mol.ecp,
        verbose=0)
    return cmol


def print_data_header() -> None:
    print(
            f'\n{"N_func":>10s}  {"New funcs":>12s}  {"Criteria val":>15s}' +\
             '  {"Difference":>15s}  {"E_subbasSCF":>15s}  {"Q^2":>18s}'
        )


def print_data(
    mask:               np.ndarray,
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


def get_q_sqrd(
        Cfull:      np.ndarray,
        Csub:       np.ndarray,
        ovlp:       np.ndarray,
        nocc:       np.ndarray  ) -> float:
    """Calculates the square of the projection Q"""
    RHF = (len(Cfull.shape) == 2)
    if RHF:
        Q = Cfull[:, :nocc[0]].T @ ovlp @ Csub[:, :nocc[0]]
        return 2.0 * np.sum(np.sum(Q**2))
    else:
        Q = [
            Cfull[0, :, :nocc[0]].T @ ovlp @ Csub[0, :, :nocc[0]],
            Cfull[1, :, :nocc[1]].T @ ovlp @ Csub[1, :, :nocc[1]]
        ]
        return (np.sum(np.sum(Q[0]**2)) + np.sum(np.sum(Q[1]**2)))


def set_linked_shells(
        smask:  np.ndarray,
        val:    bool        ) -> np.ndarray:
    """Set smask to 'val' at linked shell positions.
    """
    copysmask = copy.deepcopy(smask)
    selected_shells = np.argwhere([sm[0] for sm in copysmask])
    all_shells = np.array([''.join(map(str, sm[3][1:])) for sm in copysmask])
    same_as_selected = np.argwhere(
        all_shells == all_shells[selected_shells]
    )[:,1]
    temp = copysmask[same_as_selected][:,0]
    temp = val
    copysmask[same_as_selected,0] = temp
    return copysmask


def mask_matrix(
        mat:                np.ndarray,
        mask:               np.ndarray,
        is_restricted:      bool        = True ) -> np.ndarray:
    """Return masked matrix

    Args:
        mat : ndarray
            Matrix, e.g. Fock matrix. If using restricted Hartree-Fock,
            should be 2D. If using unrestricted, 3D with alpha and beta
            matrices as the array elements along axis 0 if such matrix.
            (overlap matrix will only have one matrix in UHF)
        mask : array
            The basis mask.
        RHF : bool
            Whether using restricted or unrestricted HF. Optional,
            default is True.

    Returns:
        masked_mat : ndarray
            The masked matrix
    """
    is_restricted = (len(mat.shape) == 2)
    return mat[mask, :][:, mask] if is_restricted else mat[:, mask, :][:, :, mask]


def basis_functions_per_atom(mol: gto.MoleBase) -> np.ndarray:
    basis_struct = mol._bas
    atoms = mol._atom
    nat = len(atoms)
    func_per_atom = np.zeros(nat, dtype=int)
    for i in range(nat):
        angl = basis_struct[basis_struct[:,0]==i][:,1]
        func_per_atom[i] = np.sum(2*angl+1) if not mol.cart else (angl + 1)*(angl + 2) // 2
    
    return func_per_atom


def spherical_average(mat: np.ndarray, ml: np.ndarray) -> np.ndarray:
    """Calculate the spherical average of a matrix.

    Args:
        mat : ndarray
            The (Fock) matrix which will be spherically averaged.
        ml : ndarray | arraylike
            An array with the numbers of functions on the shells.
    """

    mat_copy = mat.copy()
    restricted = (len(mat_copy.shape) == 2)
    if restricted:
        return sph_avg(mat_copy, ml)
    mat_out = np.ndarray(mat_copy.shape)
    mat_out[0] = sph_avg(mat_copy[0], ml)
    mat_out[1] = sph_avg(mat_copy[1], ml)
    return mat_out


def sph_avg(mat: np.ndarray, ml: np.ndarray) -> np.ndarray:
    mat_copy = mat.copy()
    offset = 0
    for nfunc in ml:
        if nfunc == 1:
            offset += nfunc
            continue
        shell_mat = mat_copy[offset:offset+nfunc, offset:offset+nfunc]
        # Extract diagonal of the shell block
        diag = np.diag(shell_mat)
        avg = np.mean(diag)
        shell_mat = np.diag([avg]*diag.shape[0])

        for i in range(nfunc):
            mat_copy[offset+i,offset:offset+nfunc] = shell_mat[i,:]

        offset += nfunc
    return mat_copy


def atomic_block_minimal_basis(
    mol:                        gto.MoleBase,
    F:                          np.ndarray,
    S:                          np.ndarray,
    Q_tol:                      float           = 1.0,
    by_shell:                   bool            = True,
    get_mask_history:           bool            = True,
    link_shells:                bool            = False,
    verbose:                    bool            = True,
    spherically_average_fock:   bool            = True,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Create minimal basis from atomic block decomposition.
    """
    func_per_atom = basis_functions_per_atom(mol)
    assert np.sum(func_per_atom) == mol.nao

    minimal_basis_mask = np.zeros(mol.nao, dtype=bool)
    # if by_shell:
    smask = init_smask(mol, mol.cart)
    if get_mask_history:
        mask_history = []
        full_mask = np.zeros(mol.nao, dtype=bool)

    atoms = list(map(lambda x: x[0], mol._atom))

    restricted = (len(F.shape) == 2)
    
    # Loop through atomic blocks in the Fock matrix
    nfuncs_min_tot = 0
    for i,funcs_and_atom in enumerate(zip(func_per_atom, atoms)):
        nfuncs, atom = funcs_and_atom
        print(f'{atom=}')
        smask_atom = list(filter(lambda x: x[3][0] == i, smask))
        mask = np.zeros(mol.nao, dtype=bool)
        mask_atom = np.zeros(func_per_atom[i], dtype=bool)
        func_offset = np.sum(func_per_atom[:i])
        mask[func_offset:func_offset+nfuncs] = True
        S_atom = mask_matrix(S, mask)
        F_atom = mask_matrix(F, mask)
        # Number of functions in minimal basis of current atom
        nfunc_per_minimal_atom = int(np.ceil(
            (ELEMENTS.index(atom)-mol.atom_nelec_core(i)) / 2))
        print(f'{nfunc_per_minimal_atom=}')
        # Add to molecule minimal number of functions
        nfuncs_min_tot += nfunc_per_minimal_atom
        
        # TODO: fix the shell array (currently for whole mol, not just atomic block)
        F_ave = F_atom.copy()
        if spherically_average_fock:
            F_ave = spherical_average(F_ave, [shell[1] for shell in smask_atom])

        e_atom, c_atom = eigh(F_ave, S_atom.copy())

        def number_of_states(energies, thresh=1e-3):
            nfuncs_include = nfunc_per_minimal_atom
            # Handle degeneracies
            while nfuncs_include < nfuncs \
              and energies[nfuncs_include]-energies[nfunc_per_minimal_atom-1] < thresh:
                print(f'{nfuncs_include=} {energies[nfuncs_include]=}')
                nfuncs_include += 1
            return nfuncs_include


        if restricted:
            nocca, noccb = number_of_states(e_atom), number_of_states(e_atom)
            print(f'Energy of highest orbital {e_atom[nocca-1]*27.2114} eV')
            occs = np.zeros(c_atom.shape[1])
            occs[:nocca] = 2
            P_atom = np.abs(
                c_atom @ np.diag(occs) @ c_atom.conj().T
            )
        else:
            nocca, noccb = number_of_states(e_atom[0]), number_of_states(e_atom[1])
            print(f'Energy of highest alpha orbital {e_atom[0, nocca-1]*27.2114} eV')
            print(f'Energy of highest beta  orbital {e_atom[1, noccb-1]*27.2114} eV')
            occs = np.zeros((2, c_atom.shape[2]))
            occs[0, :nocca] = 1
            occs[1, :noccb] = 1
            P_atom = np.abs(
                c_atom[0] @ np.diag(occs[0]) @ c_atom[0].conj().T +
                c_atom[1] @ np.diag(occs[1]) @ c_atom[1].conj().T
                )
        Qlim = nocca+noccb
        print(f'{Qlim=}')

        if verbose:
            with np.printoptions(precision=2, suppress=True):
                if restricted:
                    print(f'Bound state energies [eV]: {e_atom[e_atom<0]*27.2114}')
                    print(f'Occupied state energies [eV]: {e_atom[:nocca]*27.2114}')
                else:
                    print(f'Bound alpha state energies [eV]: {e_atom[0, e_atom[0,:]<0]*27.2114}')
                    print(f'Bound beta  state energies [eV]: {e_atom[1, e_atom[1,:]<0]*27.2114}')
                    print(f'Occupied alpha state energies [eV]: {e_atom[0, :nocca]*27.2114}')
                    print(f'Occupied beta  state energies [eV]: {e_atom[1, :noccb]*27.2114}')

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
            # Check whether only 1 index tuple was found (no two equal 
            # elements in P_atom), otherwise set P_atom_idx to the first
            # found index tuple
            if not isinstance(P_atom_idx[0], np.int64):
                P_atom_idx = P_atom_idx[0]
            Pat_i, Pat_j = P_atom_idx
            
            # Set functions of same shell to True
            if by_shell:
                mask_atom[Pat_i] = True
                mask_atom[Pat_j] = True
                smask_atom = mask_to_smask(mask_atom, smask_atom, mol.cart)
                mask_atom = smask_to_mask(smask_atom, mol.cart)
                _, c_mask = eigh(
                    mask_matrix(F_ave.copy(), mask_atom),
                    mask_matrix(S_atom.copy(), mask_atom)
                    )
                
                if get_mask_history:
                    # set Pat_i and Pat_j in the full mask to True in
                    # the current atom block
                    full_mask[func_offset + Pat_i] = True
                    full_mask[func_offset + Pat_j] = True
                    full_smask = mask_to_smask(full_mask, smask.copy(), mol.cart)
                    mask_history.append(full_smask)

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
                atom_indices.extend(list(set((Pat_i, Pat_j))))
                P_atom[Pat_i, Pat_j] = 0
                P_atom[np.flip((Pat_i, Pat_j))] = 0
                # fixme: update c_mask and Q
                raise RuntimeError('not implemented')

        # Mask
        atom_indices = list(atom_indices)
        minimal_basis_mask[func_offset + np.asarray(atom_indices)] = True

    assert np.sum(minimal_basis_mask) >= nfuncs_min_tot
    if get_mask_history:
        return minimal_basis_mask, mask_history
    return minimal_basis_mask

def find_subspace(
    F:                      np.ndarray,
    S:                      np.ndarray,
    mol:                    gto.MoleBase,
    scf_obj:                scf.hf.SCF | scf.hf.RHF | scf.uhf.UHF | scf.rohf.ROHF | scf.ghf.GHF,
    conv_tol:               float           = 1e-2,
    verbose:                bool            = True,
    get_smask:              bool            = False,
    variant:                str             = 'enocc',
    link_shells:            bool            = True,
    nfunc_normalisation:    bool            = True,
    return_mask_history:    bool            = False,
    abd_initialization:     bool            = True,
    spherical_average:      bool            = False,
    abd_Q_tol:              float           = .5,
    ) -> np.ndarray:
    r"""Looks for a Fock matrix subspace that approximately solves the
    Roothaan equation FC=SCE below a convergence of conv_tol.

    Args:
        F : ndarray
            The full Fock matrix that will be sampled.
        S : ndarray
            The overlap matrix.
        mol : MoleBase
            The MoleBase molecule object
        scf_obj : SCF
            The SCF object corresponding to mol
        conv_tol : float
            Convergence criteria used to determine when to stop the
            subspace iteration.
        verbose : bool
            Determines whether some output will be printed during
            calculation.
        get_smask : bool
            Whether to return the shell mask and run iteration shell by
            shell instead of function by function. May provide faster
            convergence but can also provide more functions overall.
        variant : str
            Which variant to use. Specifies what will be the
            minimisation criteria for adding a function/shell.
            enocc: $\sum_{i}^{nocc}\epsilon_i$,
               where $epsilon_i$ are the occupied diagonal Fock matrx
               elements
            ecore: $\frac{1}{2}\sum_{i}^{occ}(\epsilon_i+h_{ii})$,
               where $h_{ii}=C_i^\dagger H_{core}C_i$
            elden: $\Delta Q$,
               which is $1-\frac{1}{nocc}\sum_{i,j}^{nocc}<i^{subbasis}|j^{fullbasis}>$
        link_shells : bool
            Whether to link shells of atoms of same type in the mask.
            Default is True.
        nfunc_normalisation : bool
            Whether to normalise the criteria with the number of added
            functions.
            Optional, deault is True
        dft : bool
            Hartree-Fock or DFT.
            Optional, default is False
        xc : str
            XC functional string accepted by PySCF.
            Optional, default is 'b3lyp'.
        grid_level : int
            predefined integration grid levels, 0-9 (0 very sparse, 9 very dense).
            Optional, default is 3.
        return_mask_history : bool
            Whether to return the mask/smask at every iteration or only
            the final converged one, default is False.
        mask_cutoff : float
            The ratio of toggled functions to all functions after which
            subspace is considered converged. If None, conv_tol will be used,
            if supplied conv_tol will be ignored.
        abd_initialization : bool
            Toggles atomic block decomposition minimal basis initialization
            on. Optional, default is True.
        spherical_average : bool
            Whether ABD spherically averages the Fock matrix. Optional,
            default is False.
        abd_Q_tol : float
            The atomic block decomposition charge tolerance, i.e. how much
            of the charge of the molecule the minimal basis is allowed to
            not account for. Optional, default 0.5.

    Returns:
        1D boolean ndarray. A mask with selected function indices set to
        True. If collect_data is True, an ndarray is also returned with
        data as described in Args section. Shell mask is returned
        instead of function mask if get_smask is True.
    """
    if verbose:
        print('Running find_subspace for mol ', mol.atom)
    scf_obj_copy = scf_obj.copy()
    fullbasis_mol = create_shell_separated_mol(mol)

    # mask or smask initialization
    is_restricted = len(F.shape) == 2
    if is_restricted:
        Fii = np.diag(F)
    else:
        Fii = .5 * np.sum(np.diagonal(F, axis1=1, axis2=2), axis=0)
    if abd_initialization:
        mask, minimal_basis_history = atomic_block_minimal_basis(
            mol,
            F,
            S,
            Q_tol=abd_Q_tol,
            link_shells=link_shells,
            spherically_average_fock=spherical_average,
            verbose=verbose)
        mask_init_idx = np.where(mask)[0]
    else:
        mask_init_idx = [np.argmin(Fii)]
        mask = [False] * fullbasis_mol.nao_nr()
        mask[mask_init_idx[0]] = True
    smask = None
    nocc = mol.nelec

    if get_smask:
        smask = init_smask(fullbasis_mol, fullbasis_mol.cart)
        smask = mask_to_smask(mask, smask, fullbasis_mol.cart)
        if link_shells:
            # If link_shells true, set same shells of same atoms to True
            smask = set_linked_shells(smask, True)
            # if verbose:
            #     print('\nLinked shells: ON\n')

        mask = smask_to_mask(smask, fullbasis_mol.cart)

    sub_hcore = Cfull = Csub = None
    if variant == 'ecore':
        sub_hcore = scf_obj_copy.hf.get_hcore(mol)[mask_init_idx, mask_init_idx]
    elif variant == 'elden':
        _, Cfull = eigh(F, S)
        _, Csub = eigh(mask_matrix(F, mask, is_restricted=is_restricted), mask_matrix(S, mask))
    previous_sum = get_iteration_criteria_value(
        variant, epsilon_i=Fii, nocc=nocc, sub_hcore=sub_hcore,
        Csub=Csub, Cfull=Cfull, ovlp=S[:, mask])
    if return_mask_history:
        if abd_initialization:
            mask_history = []
            for mb_mask in minimal_basis_history:
                mask_history.append(
                    (mb_mask,
                    0.0,
                    0.0,
                    'Atomic Block Decomposition')
                )
        else:
            mask_history = [(
                copy.deepcopy(smask) if get_smask else copy.deepcopy(mask),
                previous_sum,
                0.0,
                'Max element of Fock matrix')]

    subbasis_mol = create_shell_separated_mol(fullbasis_mol)
    basis_initialized = False
    while True and not np.all(mask):
        mask, difference, current_criteria_val, smask = expand_mask(
            F, S, nocc, mask,
            hcore=scf_obj_copy.get_hcore(),
            Cfull=scf_obj_copy.mo_coeff,
            smask=smask, variant=variant, link_shells=link_shells,
            nfunc_normalisation=nfunc_normalisation,
        )

        if return_mask_history:
            if basis_initialized:
                mask_history.append( (
                    copy.deepcopy(smask) if get_smask else copy.deepcopy(mask),
                    current_criteria_val,
                    difference) )
            else:
                mask_history.append( (
                    copy.deepcopy(smask) if get_smask else copy.deepcopy(mask),
                    0.0,
                    0.0,
                    'Max element of Fock matrix') )

        subbasis_mol = create_shell_separated_mol(fullbasis_mol)

        previous_sum = current_criteria_val

        if not basis_initialized:
            basis_initialized = np.sum(mask) >= np.max(nocc)
            continue
        
        if  abs(difference) < conv_tol  or \
            sum(mask) == len(mask):
            break

    if get_smask:
        mask = smask
        
    if return_mask_history:
        mask = mask_history

    return mask


def mask_analysis(
    mask_history:   np.ndarray,
    mol:            gto.MoleBase,
    scf_obj:        scf.hf.SCF | scf.hf.RHF | scf.uhf.UHF | scf.rohf.ROHF | scf.ghf.GHF,
    fock:           np.ndarray,
    ovlp:           np.ndarray,
    verbose:        bool                = True,
    sym_occ_fname:  str                 = 'occupations.dat',
    molfname:       str | None          = None,
    basis:          str                 = 'def2-tzvp',
    link_shells:    bool                = True,
    dft:            bool                = False,
    xc:             str                 = 'b3lyp',
    grid_level:     int                 = 7,
    use_psi4:       bool                = False,
    C_full:         np.ndarray | None   = None
    ) -> list:
    """Run mask analysis.

    Args:
        mask_history : array
            The mask/smask history for which to run the analysis on. Elements
            will be tuples, with i:th tuple being
            (i:th mask/smask, i:th criteria value, i:th difference)
        mol : pyscf.gto.MoleBase object
            The molecule object
        scf_obj : pyscf.scf.(U/R/RO/D/-)HF object
            The self-consistent field object
        fock : numpy.ndarray
            The converged Fock matrix
        ovlp : numpy.ndarray
            The overlap matrix
        verbose : bool
            Whether to print data during analysis. Default is True.
        basis : str
            Name of the full basis set. Will be used in psi4 full
            basis calculations.
        link_shells : bool
            Whether the shells of same atoms have been linked during ABS
            calculation. Not strictly required even in the case of linked shell
            ABS calculation, but will result in slightly inccorrect data prints.
            Default is True.
        dft : bool
            Hartree-Fock or DFT.
            Optional, default is False
        xc : str
            XC functional string accepted by PySCF.
            Optional, default is 'b3lyp'.
        grid_level : int
            predefined integration grid levels, 0-9 (0 very sparse, 9 very dense).
            Optional, default is 3.
        use_psi4 : bool
            If True, psi4 will be used for SCF computation instead of PySCF.
            Optional, default is False.

    Return:
        dataframe : array
            A python array with number of functions,
            current_sum, difference, total SCF energy, SCF energy of
            occupied orbitals and the projection onto the converged
            full basis wave function will on every iteration.
    """
    scf_obj_copy = scf_obj.copy()
    fullbasis_mol = create_shell_separated_mol(mol)
    is_restricted = len(fock.shape) == 2
    nocc = fullbasis_mol.nelec

    #init_method = mask_history[0][3]
    initialized = False
    dataframe = []
    last_mask = [False] * fullbasis_mol.nao_nr()
    is_smask = isinstance(mask_history[0][0], np.ndarray)
    if is_smask:
        last_smask = init_smask(mol, mol.cart)
    scf_energy = None
    scf_orbital_energy = None
    
    # Filter mask_history of initialization and ABS
    mask_history_init = list(filter(lambda x: len(x)>=4, mask_history))
    mask_history = list(filter(lambda x: len(x)<=3, mask_history))

    if verbose:
        init_method = mask_history_init[0][3]
        print('\n' + 20*'#' + ' INITIALIZATION: ' + f'{init_method.upper():<30s} ' + 33*'#')
        i = 1
        for mask_i, current_val, difference, *init in mask_history_init:
            if is_smask:
                smask = mask_i
                changes = [i for i in range(len(smask)) if smask[i][0] != last_smask[i][0]]
                label = get_atom_shell_label(mol, changes[0], link_shells=link_shells*initialized)
                last_smask = smask
                last_mask = smask_to_mask(last_smask, mol.cart)
            else:
                mask = mask_i
                changes = [i for i in range(len(mask)) if mask[i] != last_mask[i]]
                aolabels = fullbasis_mol.ao_labels()
                aolabels = [aolabels[i] for i in changes]
                label = ' '.join(aolabels)
                last_mask = mask
            print(f'{label},  ', end='')
            if i % 10 == 0: print()
            i += 1
        
        print('\nNumber of toggled functions:', np.sum(last_mask))
        print(20*'#' + ' INITIALIZATION END ' + 61*'#')

        if link_shells:
            print('\nLink shells: ON')
            print('Additional functions may be added due to shell linking!')
        
        print_data_header()

    # If using Psi4, determine occupations on the fly
    if use_psi4:
        _, docc, socc, wfn_full, irrep_labels, irrep_symb = adbutils.psi4_fullbasis(
            mol,
            basis=basis,
            init_guess=scf_obj_copy.init_guess,
            dft=dft, xc=xc
        )
        AOCC = wfn_full.nalphapi().to_tuple()
        BOCC = wfn_full.nbetapi().to_tuple()
        symmetry_occs = list(zip(irrep_labels, AOCC, BOCC))
        # Create symmetry occupation dict
        #             IRREP: 2*alpha                      (alpha, beta)
        # Example:    'A1':  1                            (2, 2)
        irrep_nelec = {x[0]: 2*x[1] if is_restricted else (x[1], x[2]) for x in symmetry_occs}
        if use_psi4:
            # Get coefficient matrices
            Ca = wfn_full.Ca_subset('AO', 'ALL').to_array(copy=True)
            if not is_restricted:
                Cb = wfn_full.Cb_subset('AO', 'ALL').to_array(copy=True)
                C_full = np.asarray([Ca, Cb])
            else:
                C_full = Ca
    elif molfname is not None:
        irrep_nelec, irrep_symb = adbutils.read_symmetry_occs_from_file(sym_occ_fname, molfname=molfname)
        if irrep_nelec is None:
            irrep_symb = True
    else:
        irrep_symb = True        

    for mask_i, current_val, difference, *init in mask_history:
        if is_smask:
            smask = mask_i
            extracted_basis, ecp_bas = extract_basis(smask, create_shell_separated_mol(fullbasis_mol))
            
            subbasis_mol = Mole(
                atom = fullbasis_mol.atom, basis = extracted_basis,
                charge = fullbasis_mol.charge, spin = fullbasis_mol.spin,
                verbose = fullbasis_mol.verbose, unit = fullbasis_mol.unit,
                ecp = ecp_bas, symmetry = irrep_symb
                )
            subbasis_mol.build()
            submf = scf.HF(subbasis_mol)
            
            mask = smask_to_mask(smask, fullbasis_mol.cart)
            maskedF = mask_matrix(fock, mask, is_restricted)
            maskedS = mask_matrix(ovlp, mask)
            maskedHcore = mask_matrix(scf_obj_copy.get_hcore(), mask)
            if not np.allclose(maskedS, submf.get_ovlp()):
                raise RuntimeError('The masked overlap and the full overlap of masked molecule do not match!')
            if not np.allclose(maskedHcore, submf.get_hcore()):
                raise RuntimeError('The masked core Hamiltonian and the full core Hamiltonian of masked molecule do not match!')

            if dft:
                submf = submf.to_ks(xc=xc)
                submf.grids.level = grid_level
                submf.grids.prune = None

            # SCF initial guess
            subbasis_energies, submf.mo_coeff = eigh(maskedF, maskedS)
            submf.mo_occs = submf.get_occ(subbasis_energies)
            # Set the symmetry occupations if present
            if irrep_nelec is not None:
                submf.irrep_nelec = {}
                for key in irrep_nelec:
                    if irrep_nelec[key] != 0:
                        if key not in submf.mol.irrep_name:
                            raise RuntimeError(f'irrep {key} not found in subbasis')
                        submf.irrep_nelec[key] = irrep_nelec[key]
            else:
                print('Symmetry occupations not set explicitly! This may cause convergence issues.', file=sys.stderr)
            submf.kernel(dump_chk=False)
            scf_energy = submf.e_tot
 
            if is_restricted:
                nocc_sb = len(submf.mo_occ > 0)
                scf_orbital_energy = sum(submf.mo_energy[:nocc_sb])
            else:
                nocc_sb = [len(submf.mo_occ[0] > 0), len(submf.mo_occ[1] > 0)]
                scf_orbital_energy = .5 * sum(
                    submf.mo_energy[0][:nocc_sb[0]] +
                    submf.mo_energy[1][:nocc_sb[1]])

            if is_restricted:
                Q_sqrd = get_q_sqrd(
                    C_full, submf.mo_coeff,
                    ovlp[:,mask], nocc
                )
            else:
                Q_sqrd = get_q_sqrd(
                    np.asarray(C_full), np.asarray(submf.mo_coeff),
                    ovlp[:,mask], nocc
                )
            
            if not submf.converged:
                print('The SCF did not converge in the subbasis. Results may be unreliable.', file=sys.stderr)
        else:
            mask = mask_i
            e, subbasis_coeffs = eigh(mask_matrix(fock, mask, is_restricted=is_restricted), mask_matrix(ovlp, mask))
            Q_sqrd = get_q_sqrd(
                C_full, subbasis_coeffs,
                ovlp[:,mask], nocc
            )
        
        if verbose:
            if is_smask:
                changes = [i for i in range(len(smask)) if smask[i][0] != last_smask[i][0]]
                label = get_atom_shell_label(mol, changes[0], link_shells=link_shells*initialized)
                if link_shells:
                    label = ' '.join(label.split(' ')[1:])
                    label = '*'+label
            else:
                changes = [i for i in range(len(mask)) if mask[i] != last_mask[i]]
                aolabels = fullbasis_mol.ao_labels()
                aolabels = [aolabels[i] for i in changes]
                label = ' '.join(aolabels)
            print_data(
                mask, current_val, difference, label, scf_energy, Q_sqrd,
                print_header=False
            )
        dataframe.append([
                sum(mask),
                current_val,
                difference,
                scf_energy,
                scf_orbital_energy,
                Q_sqrd,
                copy.deepcopy(smask if is_smask else mask),
            ])
        last_mask = copy.deepcopy(mask)
        if is_smask:
            last_smask = copy.deepcopy(smask)
    return dataframe
