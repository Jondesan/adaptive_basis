"""Adaptive basis method"""

import pyscf
import numpy as np
from itertools import count
from pyscf.gto.basis.parse_nwchem import convert_basis_to_nwchem, to_general_contraction
from pyscf.gto.ecp import core_configuration
from pyscf.data.elements import _std_symbol, ELEMENTS
from pyscf.gto.basis.parse_nwchem import load
from pyscf.gto.mole import *
from pyscf.scf import *
from pyscf.scf.addons import canonical_orth_
from pyscf import lib
from pyscf import gto
from pyscf import scf
import copy
import re


def eigh(h, s, get_idx=False):
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
    e, c = np.linalg.eigh(xhx)
    c = x @ c
    idx = np.argsort(e)
    # e = np.sort(e)

    if get_idx:
        return e[idx], c[idx], idx
    return e[idx], c[idx]


def extract_basis(smask, shellsep_mol):
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
    """

    if len(smask) != len(shellsep_mol._bas):
        raise ValueError(
            "Shell mask does not match with _bas attribute!"
            + "Make sure the shellsep_mol objects shells have been separated"
            + "using the create_shell_separated_mol method."
        )

    atom_id = shellsep_mol._atm[:, 0]
    asymb = [ELEMENTS[i] for i in list(atom_id)]

    basis = dict.fromkeys(asymb)

    duplicate_removed_smask = []
    found_atoms = []
    current_id = -1
    # Collect unique atom smasks (if same atom is present in the shellsep_mol
    # more than once, ignore its mask after the first one)
    for elem in copy.deepcopy(smask[smask[:, 0] == True]):
        if elem[3][1] not in found_atoms:
            found_atoms.append(elem[3][1])
            current_id = elem[3][0]
        elif current_id != elem[3][0]:
            continue
        duplicate_removed_smask.append(elem)

    duplicate_removed_smask = np.array(duplicate_removed_smask)

    # Initialize distinct atoms' dictionary formatted basis structures
    # with angular momentum l
    for l, shl in duplicate_removed_smask[:, [2, 3]]:
        if basis[shl[1]] is None:
            basis[shl[1]] = []
        elif l not in [x[0] for x in basis[shl[1]]]:
            basis[shl[1]].append([l])

    # Append exponents and contraction coefficients
    for key in basis.keys():
        ogbas = to_general_contraction(
            shellsep_mol._basis[key]
        )
        for shell in basis[key]:
            i = shell[0]
            key_smask = [drs for drs in duplicate_removed_smask if drs[3][1] == key]
            idxs = [idx[3][2] - idx[2] for idx in key_smask if idx[2] == i]
            shell.extend(np.array(ogbas[i][1:])[:, [0] + idxs].tolist())

    return basis


def basis_to_file_nwchem(
    basis,
    fn,
    commentstring="",
    bsname="ao basis",
    cart=False,
    print_noprint="print",
    additional_labels="",
):
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
    with open(fn, "w") as f:
        if len(commentstring) != 0:
            f.write(f"{commentstring}\n\n")
        f.write(f'BASIS "{bsname}" {sph_cart} {print_noprint} ')
        f.write(f"{additional_labels}\n")

        for asymb in basis.keys():
            bs_atom = convert_basis_to_nwchem(asymb, basis[asymb])
            f.write(f"{bs_atom}\n")

        f.write("END")

    return


def get_uncontr_basis(mol, fn=None):
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


def get_basis_dict(basis: str):
    """Convert a basis string into a dictionary to pass
    to pyscf.gto.basis.parse
    """

    dc = dict()
    for elem in basis.split("#")[1:]:
        dc[elem[11]] = gto.basis.parse(str(elem[11:]))
    return dc


def get_shells(mol):
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
        ia = mol.bas_atom(ib)  # atom that given basis function sits on
        l = mol.bas_angular(ib)  # angular momentum l of given basis function
        nc = mol.bas_nctr(ib)  # number of CGTOs for given shell
        symb = mol.atom_symbol(ia)  # label of given atom

        shells = np.append(
            shells, nc * (l + 1) * (l + 2) // 2 if mol.cart else nc * (2 * l + 1)
        )

    if sum(shells) != mol.nao_nr():
        raise Exception(
            "Number of functions in the mask does not correspond with number of functions of the molecule!"
        )

    return shells


def maskidx_to_smaskidx(mask, smask, cart=False):
    """Create mapping between mask and smask"""
    mapping = [0] * len(mask)  # mol.nao_nr()
    counter = 0
    for i, sm in enumerate(smask):
        l = sm[2]
        for j in range((l + 1) * (l + 2) // 2 if cart else 2 * l + 1):
            mapping[counter] = i
            counter += 1
    return mapping


def init_smask(mol, cart=False):
    """Initialize the shell mask array. smask will be a list of lists,
    with length equal to the number of uncontracted shells, and each
    element is a two element list, first is bool that specifies the mask
    for the current shell, the other how many primitives in this shell.
    """

    smask = []

    count = np.zeros((mol.natm, 9), dtype=int)
    for ib in range(mol.nbas):
        # atom that given basis function sits on
        ia = mol.bas_atom(ib)
        # angular momentum angl of given basis function
        angl = mol.bas_angular(ib)
        # number of CGTOs for given shell
        nc = mol.bas_nctr(ib)
        symb = mol.atom_symbol(ia)  # label of given atom
        nelec_ecp = mol.atom_nelec_core(ia)  # Number of ecp electrons
        if nelec_ecp == 0 or angl > 3:
            shl_start = count[ia, angl] + angl + 1
        else:
            coreshl = core_configuration(nelec_ecp, atom_symbol=_std_symbol(symb))
            shl_start = coreshl[angl] + count[ia, angl] + angl + 1
        count[ia, angl] += nc
        for n in range(shl_start, shl_start + nc):
            smask.append(
                [
                    False,
                    (angl + 1) * (angl + 2) // 2 if cart else 2 * angl + 1,
                    angl,
                    (ia, symb, n, lib.param.ANGULAR[angl].capitalize()),
                ]
            )

    return np.array(smask, dtype=object)


def smask_to_mask(smask, cart=False):
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


def mask_to_smask(mask, smask, cart=False):
    """Flip shells of smask to True that have 1 or more functions set to
    True in mask.
    """
    mapping = maskidx_to_smaskidx(mask, smask, cart)
    for i in np.argwhere(mask):
        smask[mapping[i[0]]][0] = True

    return smask


def get_iteration_criteria_value(variant, params):
    """Calculates the value of the chosen variants criteria.

    Args:
        variant : int
            Which variant to calculate.
        params : list
            The needed variables for calculating the different criteria.
            0: params[0] = energy eigenvalues,
               params[1] = number of occupations
            1: params[0] = energy eigenvalues,
               params[1] = number of occupations,
               params[2] = subbasis core hamiltonian hcore,
               params[3] = subbasis coefficient matrix Csub
            2: params[0] = coeff matrix fullbasis Cfull,
               params[1] = coeff matrix subbasis Csub,
               params[2] = fullbasis mol object,
               params[3] = subbasis mol object,
               params[4] = number of occupations

    """
    criteria = 0.0
    match variant:
        case 0:
            epsilon_i, nocc = params
            criteria = np.sum(epsilon_i[:nocc])
        case 1:
            epsilon_i, nocc, hcore, Csub = params
            # criteria = 0.5 * np.sum(
            #     epsilon_i[:nocc] + np.diag(Csub.T @ hcore @ Csub)[:nocc]
            # )
            # print(Csub.shape, Csub.T.shape, (hcore + np.diag(epsilon_i)).shape)
            # criteria = np.sum(Csub[:nocc,:nocc] @ Csub.T[:nocc,:nocc] @ (hcore + np.diag(epsilon_i))[:nocc,:nocc])
            occupations = [0]*Csub.shape[0]
            occupations[:nocc] = [2]*nocc
            occupations = np.array(occupations)
            mocc = Csub[:,:nocc]
            # print(mocc.conj().T.shape, mocc.shape)
            P = mocc @ mocc.conj().T
            criteria = np.sum(P[:nocc,:nocc] @ (hcore + np.diag(epsilon_i))[:nocc,:nocc])
            # print(criteria)
        case 2:
            Cfull, Csub, mol_full, mol_sub, nocc = params
            criteria = get_q_sqrd(Cfull, Csub, mol_full, mol_sub, nocc)
    return criteria


def expand_mask(
    F,
    S,
    nocc,
    mask,
    smask=None,
    variant=0,
    fullbasis_mol=None,
    subbasis_mol=None,
    Cfull=None,
    link_shells=True,
):
    """Expands the current mask by either one function or one shell
    based on smask.

    Args:
        F : ndarray
            Full Fock matrix
        S : ndarray
            Full overlap matrix
        nocc : int
            Number of occupied orbitals
        mask : ndarray
            The current mask. A logical 1d array
        smask : None or ndarray
            If None functions are tested individually. Else shell by
            shell testing is used where shells are determined by the
            smask array, where the elements represent the number of
            functions per current shell. The shells are ordered in the
            PySCF internal format
        variant : int
            Which variant to use. Specifies what will be the
            minimisation criteria for adding a function/shell.
            0: $\sum_{i}^{nocc}\epsilon_i$,
               where $epsilon_i$ are the occupied diagonal Fock matrx
               elements
            1: $\frac{1}{2}\sum_{i}^{occ}(\epsilon_i+h_{ii})$,
               where $h_{ii}=C_i^\dagger H_{core}C_i$
            2: $\Delta Q$,
               which is $1-\frac{1}{nocc}\sum_{i,j}^{nocc}<i^{subbasis}|j^{fullbasis}>$
        link_shells : bool
            Whether to link shells of atoms of same type in the mask.
            Default is True.

    Returns:
        The new mask (boolean ndarray), the current difference in
        eigenvalue sums and the current sum (energy sum of occupied
        orbitals), shell mask if smask is provided.
    """
    evals, coeffs = eigh(F[mask, :][:, mask], S[mask, :][:, mask])
    last_sum = 0.0
    hcore = scf.hf.get_hcore(fullbasis_mol)

    if smask is None or variant == 0:
        last_sum = np.sum(evals[:nocc])
    else:
        subbasis_mol = create_shell_separated_mol(fullbasis_mol)
        newmask = [sm[0] for sm in smask]
        newbas = fullbasis_mol._bas[newmask]
        subbasis_mol._bas = newbas
        match variant:
            case 1:
                criteria_input = [evals, nocc, hcore[mask, :][:, mask], coeffs]
            case 2:
                _, Cfull = eigh(F, S)
                criteria_input = [Cfull, coeffs, fullbasis_mol, subbasis_mol, nocc]
        last_sum = get_iteration_criteria_value(variant, criteria_input)

    test_sums = []
    if smask is None:
        for i, m in enumerate(mask):
            if m:
                continue

            test_mask = copy.deepcopy(mask)
            test_mask[i] = True
            evals, coeffs = eigh(
                F[test_mask, :][:, test_mask], S[test_mask, :][:, test_mask]
            )

            criteria_input = [evals, nocc]

            test_sums.append((i, get_iteration_criteria_value(0, criteria_input)))
    else:
        # Gather indices of duplicate shells if link_shells enabled
        # (if system has more than 1 atom of same type,
        #  shells will be duplicated.)
        shl_indices = []
        if link_shells:
            atoms_found = []
            shells = ["".join([str(s) for s in sm[3][1:]]) for sm in smask]

            for i, sm in enumerate(smask):
                if "".join([str(s) for s in sm[3][1:]]) not in atoms_found:
                    atoms_found.append("".join([str(s) for s in sm[3][1:]]))
                    indices = [
                        ind
                        for ind, ele in zip(count(), shells)
                        if ele == "".join([str(s) for s in sm[3][1:]])
                    ]
                    shl_indices.append(indices)
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

            evals, coeffs = eigh(
                F[test_mask, :][:, test_mask], S[test_mask, :][:, test_mask]
            )
            if variant != 0:
                subbasis_mol = create_shell_separated_mol(fullbasis_mol)
                newmask = test_smask[:, 0].astype(bool)
                newbas = fullbasis_mol._bas[newmask]
                subbasis_mol._bas = newbas
            match variant:
                case 0:
                    criteria_input = [evals, nocc]
                case 1:
                    criteria_input = [
                        evals,
                        nocc,
                        hcore[test_mask, :][:, test_mask],
                        coeffs,
                    ]
                case 2:
                    criteria_input = [Cfull, coeffs, fullbasis_mol, subbasis_mol, nocc]

            test_sums.append((i, get_iteration_criteria_value(variant, criteria_input)))

    test_differences = [test_sum[1] - last_sum for test_sum in test_sums]
    if variant == 2:
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

        ####################################################
        # FIGURES FOR DEBUGGING, all seems to work!
        ####################################################
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        # x = np.arange(len(test_differences))
        # ax.plot(x, test_differences, '.')
        # for i, td in enumerate(test_differences):
        #     text = smask[test_sums[i][0]][3]
        #     text = ''.join(map(str, text[1:]))
        #     ax.annotate(
        #         text,
        #         (x[i], td)
        #     )
        # ax.grid(alpha=.5)
        # plt.show()
        ####################################################

    return mask, test_differences[array_index], test_sums[array_index][1], smask


def get_sub_scf_attributes(mol, fock, overlap):
    """Calculates converged attributes for the system using the subbasis
    determined by mask.

    Args:
        mol : pyscf.gto.MoleBase
            The molecule object

    Returns:
        The SCF energy, sum of occupied orbital energies of the
        subbasis, the MO coefficient matrix of the subbasis.
    """
    mf = mol.HF().apply(scf.addons.remove_linear_dep_)

    # Diagonalize fock matrix and form guess density matrix
    if fock.shape[0] > 1:
        e, c = eigh(fock, overlap)
        occ = mf.get_occ(e, c)
        dm = scf.hf.make_rdm1(c, occ)
        mf.init_guess = dm
    mf.kernel()

    # electronic_energy = mf.energy_elec()
    # hcore = pyscf.scf.hf.get_hcore(mol)
    # idx = np.count_nonzero(mf.get_occ())#[mf.get_occ()>0]
    # print(idx)
    # print(electronic_energy[0])# + np.sum(np.diag(hcore)[:idx]))
    scf_energy = mf.e_tot
    nocc_sb = len(mf.mo_occ > 0)
    scf_orbital_energy = sum(np.sort(mf.mo_energy)[:nocc_sb])
    return scf_energy, scf_orbital_energy, mf.mo_coeff


def create_shell_separated_mol(mol, verbose=0):
    """Creates a copy of mol with shells separated."""
    shell_sep_basis = get_uncontr_basis(mol)
    cmol = pyscf.gto.M(atom=mol.atom, basis=shell_sep_basis, verbose=0)
    return cmol


def print_data(
    mask,
    criteria_value,
    diff,
    ao_or_shell_label,
    E_scf="-",
    Qsqrd="-",
    print_header=False,
):
    if print_header:
        print(
            f'\n{"N_func":>10s}  {"Added ao/shell(s)":>18s}  {"Criteria val":>15s}  {"Difference":>15s}  {"E_subbasSCF":>15s}  {"Q^2":>15s}'
        )

    if E_scf is None:
        E_scf = "-"
    if Qsqrd is None:
        Qsqrd = "-"

    print(f"{sum(mask):10}  {ao_or_shell_label:>18}  {criteria_value:15.9f}", end="")
    print(f'  {diff:{">15s" if isinstance(diff, str) else "15.9f"}}', end="")
    print(f'  {E_scf:{">15s" if isinstance(E_scf, str) else "15.9f"}}', end="")
    print(f'  {Qsqrd:{">15s" if isinstance(Qsqrd, str) else "15.9f"}}')


def get_q_sqrd(Cfull, Csub, mol_full, mol_sub, nocc):
    """Calculates the square of the projection Q"""
    if mol_full.cart and mol_sub.cart:
        ovlp = gto.intor_cross("int1e_ovlp", mol_full, mol_sub)
    elif not mol_full.cart and not mol_sub.cart:
        ovlp = gto.intor_cross("int1e_ovlp_sph", mol_full, mol_sub)
    else:
        raise RuntimeError("One of the mol objects is cartesian ans other spherical!")
    Q = Cfull[:, :nocc].T @ ovlp @ Csub[:, :nocc]

    return np.sum(np.sum(Q**2))


def find_subspace(
    F,
    S,
    mol,
    scf,
    conv_tol=1e-2,
    verbose=True,
    collect_data=False,
    get_smask=False,
    variant=0,
    link_shells=True,
    dm0=None
):
    """Looks for a Fock matrix subspace that approximately solves the
    Roothaan equation FC=SCE below a convergence of conv_tol.

    Args:
        F : ndarray
            The full Fock matrix that will be sampled.
        S : ndarray
            The overlap matrix.
        mol : MoleBase
            The MoleBase molecule object
        scf : SCF
            The converged SCF object corresponding to mol
        conv_tol : float
            Convergence criteria used to determine when to stop the
            subspace iteration.
        verbose : bool
            Determines whether some output will be printed during
            calculation.
        collect_data : bool
            If true, np.ndarray will be created and number of functions,
             current_sum, difference, total SCF energy, SCF energy of
             occupied orbitals and the projection onto the converged
             full basis wave function will be appended on every
             iteration.
        get_smask : bool
            Whether to return the shell mask and run iteration shell by
            shell instead of function by function. May provide faster
            convergence but can also provide more functions overall.
        variant : int
            Which variant to use. Specifies what will be the
            minimisation criteria for adding a function/shell.
            0: $\sum_{i}^{nocc}\epsilon_i$,
               where $epsilon_i$ are the occupied diagonal Fock matrx
               elements
            1: $\frac{1}{2}\sum_{i}^{occ}(\epsilon_i+h_{ii})$,
               where $h_{ii}=C_i^\dagger H_{core}C_i$
            2: $\Delta Q$,
               which is $1-\frac{1}{nocc}\sum_{i,j}^{nocc}<i^{subbasis}|j^{fullbasis}>$
        link_shells : bool
            Whether to link shells of atoms of same type in the mask.
            Default is True.
        dm0 : ndarray
            Initial guess density matrix. Can be used to test different
            intial guesses and their convergence with ABS method.

    Returns:
        1D boolean ndarray. A mask with selected function indices set to
        True. If collect_data is True, an ndarray is also returned with
        data as described in Args section. Shell mask is returned
        instead of function mask if get_smask is True.
    """
    fullbasis_mol = create_shell_separated_mol(mol)
    convF = copy.deepcopy(F)
    F = scf.get_fock(dm=dm0)
    # mask or smask initialization
    Fii = np.diag(F)
    mask = [False] * fullbasis_mol.nao_nr()
    min_idx = np.argmin(Fii)
    smask = None
    mask[min_idx] = True

    if not scf.converged:
        raise RuntimeWarning('SCF has not converged!\n')

    if get_smask:
        smask = init_smask(fullbasis_mol, fullbasis_mol.cart)
        smask = mask_to_smask(mask, smask, fullbasis_mol.cart)
        if link_shells: # If link_shells, set same shells of same atoms
        # to True
            selected_shells = np.argwhere([sm[0] for sm in smask])
            all_shells = np.array([''.join(map(str, sm[3][1:])) for sm in smask])
            same_as_selected = np.argwhere(
                all_shells == all_shells[selected_shells]
            )[:,1]
            temp = smask[same_as_selected][:,0]
            temp = True
            smask[same_as_selected,0] = temp

        mask = smask_to_mask(smask, fullbasis_mol.cart)

    previous_sum = np.sum(Fii[mask])
    if dm0 is not None:
        fock = scf.get_fock(dm=dm0)  # = h1e + vhf, no DIIS
        s1e = scf.get_ovlp()
        mo_energy, mo_coeff = scf.eig(fock, s1e)
        occ = scf.get_occ(mo_energy, mo_coeff)
        subbasis_coeffs = mo_coeff
    else:
        occ = scf.get_occ()
        subbasis_coeffs = scf.mo_coeff
    nocc = np.count_nonzero(occ)
    scf_energy = None
    scf_orbital_energy = None
    subbasis_mol = create_shell_separated_mol(fullbasis_mol)

    if get_smask:
        newmask = [sm[0] for sm in smask]
        newbas = fullbasis_mol._bas[newmask]
        subbasis_mol._bas = newbas

        fullbasis_coeffs = scf.mo_coeff
        # fullbasis_coeffs = copy.deepcopy(subbasis_coeffs)#scf.mo_coeff
        scf_energy, scf_orbital_energy, subbasis_coeffs = get_sub_scf_attributes(
            subbasis_mol, convF[mask, :][:, mask], S[mask, :][:, mask]
        )
        Q_sqrd = get_q_sqrd(
            fullbasis_coeffs, subbasis_coeffs, fullbasis_mol, subbasis_mol, nocc
        )
    else:
        Q_sqrd = None

    if collect_data:
        dataframe = [
            [
                sum(mask),
                previous_sum,
                0.0,
                scf_energy,
                scf_orbital_energy,
                Q_sqrd,
                copy.deepcopy(smask),
            ]
        ]

    if verbose:
        ao_labels = np.array(fullbasis_mol.ao_labels())[mask]
        if get_smask:
            num, symb = ao_labels[0].split()[:2]
            label = "" if link_shells else f"{num} "
            shllbl = re.findall(r'(\d+[a-zA-Z])', ao_labels[0].split()[2])[0]
            label += f"{symb} {shllbl}"
        else:
            label = ao_labels[0].strip()
        print_data(
            mask, previous_sum, "-", label, scf_energy, Q_sqrd, print_header=True
        )

    last_mask = copy.deepcopy(mask)
    while True:
        mask, difference, current_criteria_val, smask = expand_mask(
            F,
            S,
            nocc,
            mask,
            fullbasis_mol=fullbasis_mol,
            subbasis_mol=subbasis_mol,
            Cfull=scf.mo_coeff,
            smask=smask,
            variant=variant,
            link_shells=link_shells,
        )

        subbasis_mol = create_shell_separated_mol(fullbasis_mol)
        if get_smask:
            subbasis_mol._bas = fullbasis_mol._bas[[sm[0] for sm in smask]]
            scf_energy, scf_orbital_energy, subbasis_coeffs = get_sub_scf_attributes(
                subbasis_mol, convF[mask, :][:, mask], S[mask, :][:, mask]
            )

        if collect_data:
            if get_smask:
                Q_sqrd = get_q_sqrd(
                    fullbasis_coeffs, subbasis_coeffs, fullbasis_mol, subbasis_mol, nocc
                )

            dataframe.append(
                [
                    sum(mask),
                    current_criteria_val,
                    difference,
                    scf_energy,
                    scf_orbital_energy,
                    Q_sqrd,
                    copy.deepcopy(smask),
                ]
            )
        if verbose:
            # Get most recent function/shell label
            changes = [i for i in range(len(mask)) if mask[i] != last_mask[i]]
            ao_labels = np.array(fullbasis_mol.ao_labels())[changes]
            if get_smask:
                num, symb = ao_labels[0].split()[:2]
                label = ""
                label += "" if link_shells else f"{num} "
                shllbl = re.findall(r'(\d+[a-zA-Z])', ao_labels[0].split()[2])[0]
                label += f"{symb} {shllbl}"
            else:
                label = ao_labels[0].strip()
            print_data(
                mask, current_criteria_val, difference, label, scf_energy, Q_sqrd
            )

        if abs(difference) < conv_tol or sum(mask) == len(mask):
            break
        previous_sum = current_criteria_val
        last_mask = copy.deepcopy(mask)

    if get_smask:
        mask = smask

    if collect_data:
        return mask, dataframe
    return mask


def fakemol_for_charges_sap(coords, expnt=1e16, contr_coeff=1):
    '''Construct a fake Mole object that holds the charges on the given
    coordinates (coords).  The shape of the charge can be a normal
    distribution with the Gaussian exponent (expnt).
    '''
    nbas = coords.shape[0]
    expnt = np.asarray(expnt).ravel()
    contr_coeff = numpy.asarray(contr_coeff).ravel()

    fakeatm = np.zeros((nbas,ATM_SLOTS), dtype=np.int32)
    fakebas = np.zeros((nbas,BAS_SLOTS), dtype=np.int32)
    fakeenv = [0] * PTR_ENV_START
    ptr = PTR_ENV_START
    fakeatm[:,PTR_COORD] = numpy.arange(ptr, ptr+nbas*3, 3)
    fakeenv.append(coords.ravel())
    ptr += nbas*3
    fakebas[:,ATOM_OF] = numpy.arange(nbas)
    fakebas[:,NPRIM_OF] = contr_coeff.size
    fakebas[:,NCTR_OF] = 1
    if expnt.size == 1:
        expnt = expnt[0]
        # approximate point charge with gaussian distribution exp(-1e16*r^2)
        fakebas[:,PTR_EXP] = ptr
        fakebas[:,PTR_COEFF] = ptr+1
        fakeenv.append([expnt, contr_coeff / (2*np.sqrt(np.pi)*gaussian_int(2,expnt))])
        ptr += 2
    else:
        assert expnt.size == contr_coeff.size#nbas
        # approximate point charge with gaussian distribution exp(-expnt*r^2)
        fakebas[:,PTR_EXP] = ptr + np.arange(nbas) * 2
        fakebas[:,PTR_COEFF] = ptr + np.arange(nbas) * 2 + contr_coeff.size
        coeff = contr_coeff / (2 * np.sqrt(np.pi) * gaussian_int(2, expnt))
        fakeenv.append(np.vstack((expnt, coeff)).ravel())

    fakemol = Mole()
    fakemol._atm = fakeatm
    fakemol._bas = fakebas
    fakemol._env = np.hstack(fakeenv)
    fakemol._built = True
    return fakemol

def vsap_pot(mol, sapbs_path='bs/sap_grasp_small.nw'):
    '''Calculates the superposition of atomic potentials guess potential matrix V

    Args:
        mol : pyscf.gto.Mole
            PySCF Mole object for which the potential is calculated

    Returns:
        V : np.array
            The SAP potential matrix
    '''
    atom_coords = np.asarray([coord[1] for coord in mol._atom])
    atoms = np.asarray([coord[0] for coord in mol._atom])

    sapbas = {
        atom: np.asarray(load(sapbs_path, atom)[0][1:]) for atom in set(atoms)
    }

    # charge sumcheck
    Z_eff = sum([np.sum(sapbas[a][:,1]) for a in atoms])
    if np.abs(Z_eff + mol.nelectron) > 1e-3:
        raise RuntimeError(
            '\n'.join([f'SAP basis coefficients must be equal or close to total electronic charge: {Z_eff} !≃ {mol.nelectron}',
            f'Check fails with value {np.abs(Z_eff + mol.nelectron)})']))

    V = np.zeros((mol.nao_nr(), mol.nao_nr()))
    cmol = copy.deepcopy(mol)
    nbas = cmol.nbas
    for i, atom in enumerate(atoms):
        expnt = sapbas[atom][:,0]
        coeff = sapbas[atom][:,1]
        nucleon_fakemol = fakemol_for_charges_sap(np.asarray([atom_coords[i]]), expnt, coeff)

        cmol += nucleon_fakemol

    shls_slice = (0, nbas, 0, nbas, nbas, cmol.nbas)
    int3c2e = cmol.intor('int3c2e', comp=1, shls_slice=shls_slice)
    V = -np.einsum('pqk->pq', int3c2e)

    return V

def vsap_coeffs(mol, mf=None, sapbs_path='bs/sap_grasp_small.nw'):
    '''Get SAP guess orbital MO coefficients

    Args:
        mol : pyscf.gto.Mole
            PySCF Mole object
        mf : pyscf.scf.RHF
            PySCF scf object corresponding to mol

    Returns:
        e : numpy.array
            Orbital energies
        c : numpy.array
            Orbital coefficients
    '''
    V = vsap_pot(mol, sapbs_path=sapbs_path)

    if mf is None:
        mf = scf.RHF(mol)

    hcore = mf.get_hcore()
    S = mf.get_ovlp()
    e, c = mf.eig(hcore + V, S)

    return e, c

def init_guess_by_vsap(mol, sapbs_path='bs/sap_grasp_small.nw'):
    '''Forms the SAP initial guess density matrix dm0 for mol

    Args:
        mol : pyscf.gto.Mole
            PySCF Mole object

    Returns:
        dm0 : numpy.array
            Initial guess density matrix
    '''

    mf = scf.RHF(mol)
    e, coeff = vsap_coeffs(mol, mf, sapbs_path=sapbs_path)

    occ = mf.get_occ(e, coeff)

    dm0 = scf.hf.make_rdm1(coeff, occ)

    return dm0

def calculate_projection(C1, C2, S, nocc):
    Q = C1[:,:nocc].T @ S @ C2[:,:nocc]
    return np.sum(np.sum(Q**2))/nocc


def calculate_sap_projection(mol, mf=None):

    if mf is None:
        mf = scf.RHF(mol)
        mf.kernel()
    elif not mf.converged:
        raise RuntimeWarning('SCF has not converged!\n'
        + 'Make sure you have run the self-consistent field procedure.')

    mf.init_guess = 'sap'
    _, C_sap = vsap_coeffs(mol)
    C_scf = mf.mo_coeff
    S = mf.get_ovlp()
    nocc = np.count_nonzero(mf.mo_occ)

    return calculate_projection(C_sap, C_scf, S, nocc)

def sad_guess(mol):
    from scf import atom_hf
    atm_scf = atom_hf.get_atm_nrhf(mol)
    aoslice = mol.aoslice_by_atom()
    atm_dms = []
    mo_coeff = []
    mo_occ = []
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        if symb not in atm_scf:
            symb = mol.atom_pure_symbol(ia)

        if symb in atm_scf:
            e_hf, e, c, occ = atm_scf[symb]
        else:  # symb's basis is not specified in the input
            nao_atm = aoslice[ia,3] - aoslice[ia,2]
            c = numpy.zeros((nao_atm, nao_atm))
            occ = numpy.zeros(nao_atm)

        atm_dms.append(numpy.dot(c*occ, c.conj().T))
        mo_coeff.append(c)
        mo_occ.append(occ)

    dm = scipy.linalg.block_diag(*atm_dms)
    mo_coeff = scipy.linalg.block_diag(*mo_coeff)
    mo_occ = numpy.hstack(occ)

    if mol.cart:
        cart2sph = mol.cart2sph_coeff(normalized='sp')
        dm = reduce(lib.dot, (cart2sph, dm, cart2sph.T))
        mo_coeff = lib.dot(cart2sph, mo_coeff)

    return mo_coeff