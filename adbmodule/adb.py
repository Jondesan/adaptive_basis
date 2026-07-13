"""Adaptive basis set method"""



# import pyscf
import numpy as np
from scipy import linalg
# from itertools import count
from pyscf.gto.basis.parse_nwchem import convert_basis_to_nwchem,\
    to_general_contraction, convert_ecp_to_nwchem
from pyscf.gto.ecp import core_configuration
from pyscf.data.elements import _std_symbol, ELEMENTS
from pyscf.gto.basis.parse_nwchem import load
from pyscf.gto.mole import *
from pyscf.scf import *
from pyscf.scf.addons import canonical_orth_, project_dm_nr2nr
from pyscf import lib, gto, scf, symm
import pyscf
from warnings import warn
from operator import itemgetter
import adbutils
import atomic_block_util
import copy
import sys
import re
from calculations import eig, symmetrized_eig, get_iteration_criteria_value, get_q_sqrd
from mask import init_smask, mask_to_smask, smask_to_mask, set_linked_shells, linked_shell_idx, get_atom_shell_label
from CONSTANTS import NFUNCS, EXPAND_MASK_EPS


def dual_basis_energy_correction(
    scf_obj: scf.hf.SCF | scf.hf.RHF | scf.uhf.UHF | scf.rohf.ROHF | scf.ghf.GHF,
    P_full_projected: np.ndarray
    ) -> tuple[float, float]:

    F_full = scf_obj.get_fock(dm = P_full_projected)
    E_new, C_new = scf_obj.eig(F_full, scf_obj.get_ovlp())
    P_new = scf_obj.make_rdm1(
        mo_coeff = C_new,
        mo_occ = scf_obj.get_occ(mo_energy = E_new, mo_coeff = C_new))
    dP = P_new - P_full_projected
    # If unrestricted calculation
    if len(dP.shape) > 2:
        dE = (np.trace(dP[0] @ F_full[0]) + np.trace(dP[1] @ F_full[1])) / 2
    else:
        dE = np.trace(dP @ F_full)
    return np.real(dE), scf_obj.e_tot


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
    for elem in copy.deepcopy(smask[np.asarray(smask[:, 0], dtype = bool)]):
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
                (coeff_table[:, 1:] == 0).all(axis = 1))]
            filtered_shell = filtered_shell[~np.all(filtered_shell == 0, axis = 1)]
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


def get_uncontracted_basis(
        mol:    gto.MoleBase,
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
            "Number of functions in the mask does not correspond with number of functions in the molecule!"
        )

    return shells


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
    restricted = (len(fock.shape) == 2)
    mf = mol.HF()
    mf = mf.apply(scf.addons.remove_linear_dep_)
    if dft:
        mf = mf.to_ks(xc=xc)
        mf.grids.level = grid_level
        mf.grids.prune = None

    # Diagonalize fock matrix and form guess density matrix
    if fock.shape[1] > 1:
        e, c = eig(fock, overlap)
        occ = mf.get_occ(e, c)
        dm = mf.make_rdm1(c, occ)
        mf.init_guess = dm
    mf.kernel(dump_chk=False)

    scf_energy = mf.e_tot
    # sum over occupied orbital energies
    if restricted:
        nocc_sb = np.sum(mf.mo_occ > 0)
        scf_orbital_energy = sum(np.sort(mf.mo_energy)[:nocc_sb])
    else:
        nocc_sb = [np.sum(mf.mo_occ[0] > 0), np.sum(mf.mo_occ[1] > 0)]
        scf_orbital_energy = .5 * sum(
            np.sort(mf.mo_energy[0])[:nocc_sb[0]] +
            np.sort(mf.mo_energy[1])[:nocc_sb[1]])
        
    return scf_energy, scf_orbital_energy, mf.mo_coeff


def create_subbasis_mol(
        mol:        gto.MoleBase,
        smask:      np.ndarray    ) -> gto.MoleBase:

    extracted_basis, ecp_bas = extract_basis(smask, create_shell_separated_mol(mol))
    subbasis_mol = gto.Mole(
        atom = mol.atom, basis = extracted_basis,
        charge = mol.charge, spin = mol.spin,
        verbose = mol.verbose, unit = mol.unit,
        ecp = ecp_bas, symmetry = mol.symmetry
    )
    subbasis_mol.build()

    return subbasis_mol


def diagonalize_masked(
        maskedF:    np.ndarray,
        maskedS:    np.ndarray,
        mol:        gto.MoleBase | None = None,
        smask:      np.ndarray | None   = None,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Diagonalize an already-masked (Fock, overlap) pair.

    Plain adb.eig (symmetry-blind) when `mol` is None. When `mol` is given
    (the shell-separated mol whose shells `smask` indexes into -- see
    expand_mask's docstring), builds a fresh subbasis Mole via
    create_subbasis_mol(mol, smask) and diagonalizes block-by-irrep with
    symmetrized_eig instead, returning a name-tagged `orbsym` array.

    Shared by expand_mask's symmetry-aware branch, find_subspace's
    symmetry-aware pre-loop baseline, and find_subspace's optional
    track_orbitals bookkeeping -- pulled out to a top-level function so
    those three call sites diagonalize a masked subbasis identically
    instead of maintaining three copies of the same branch.

    Returns:
        evals, coeffs, orbsym (None when mol is None).
    """
    if mol is None:
        evals, coeffs = eig(maskedF, maskedS)
        return evals, coeffs, None
    sub_mol = create_subbasis_mol(mol, smask)
    evals, coeffs, orbsym_id = symmetrized_eig(
        maskedF, maskedS, sub_mol.symm_orb, sub_mol.irrep_id)
    id_to_name = dict(zip(sub_mol.irrep_id, sub_mol.irrep_name))
    orbsym = np.asarray([id_to_name[i] for i in orbsym_id])
    return evals, coeffs, orbsym


def get_occupied_orbitals(
        epsilon_i:      np.ndarray,
        nocc:           tuple,
        irrep_nelec:    dict | None         = None,
        orbsym:         np.ndarray | None   = None,
        restricted:     bool                = True,
        ) -> list:
    """Extract the occupied orbital set implied by the same selection rule
    get_iteration_criteria_value's 'enocc' branch (and _enocc_by_irrep)
    use to compute their criterion sum -- but returning the individual
    selected (energy, irrep) pairs instead of just their sum.

    Symmetry-blind (irrep_nelec/orbsym not given): the lowest nocc[0]
    (restricted) or nocc[0]/nocc[1] (unrestricted) eigenvalues overall.

    Symmetry-aware (irrep_nelec/orbsym given): the lowest target-count
    eigenvalues *within each irrep*, per pyscf's mf.irrep_nelec convention
    (int per irrep for restricted, (n_alpha, n_beta) tuple for
    unrestricted) -- mirroring _enocc_by_irrep's selection exactly, minus
    its shortfall-penalty bookkeeping (irrelevant here: this is only ever
    called on an already-accepted mask/step, where by construction every
    targeted irrep has enough capacity).

    Args:
        epsilon_i : ndarray
            Orbital energies, as returned by adb.eig/symmetrized_eig.
            Shape (nmo,) for restricted, (2, nmo) for unrestricted.
        nocc : tuple
            (n_alpha, n_beta) occupied counts, used only when irrep_nelec
            is None.
        irrep_nelec : dict | None
            Target occupation per irrep name (pyscf mf.irrep_nelec
            format). When given, `orbsym` must be given too.
        orbsym : ndarray | None
            Irrep name (string) for each entry along epsilon_i's last
            axis, as returned by diagonalize_masked/symmetrized_eig (after
            translating irrep ids to names).
        restricted : bool
            Whether epsilon_i is restricted (2D h) or unrestricted-shaped.

    Returns:
        List of (energy, irrep_label) tuples, one per occupied orbital.
        irrep_label is None throughout when irrep_nelec/orbsym are not
        given (symmetry-blind case).
    """
    occupied = []
    if irrep_nelec is not None:
        if orbsym is None:
            raise ValueError(
                "'orbsym' must be provided together with 'irrep_nelec'.")
        for irname, target in irrep_nelec.items():
            spin_targets = [(None, target // 2)] if restricted else \
                [(0, target[0]), (1, target[1])]
            for spin, n_need in spin_targets:
                if n_need == 0:
                    continue
                e_ir = epsilon_i[orbsym == irname] if spin is None \
                    else epsilon_i[spin][orbsym == irname]
                idx = np.argsort(np.real(e_ir))[:n_need]
                for e in np.real(e_ir)[idx]:
                    occupied.append((float(e), irname))
    elif restricted:
        for e in epsilon_i[:nocc[0]]:
            occupied.append((float(np.real(e)), None))
    else:
        for e in epsilon_i[0][:nocc[0]]:
            occupied.append((float(np.real(e)), None))
        for e in epsilon_i[1][:nocc[1]]:
            occupied.append((float(np.real(e)), None))
    return occupied


def get_occupied_orbitals_from_scf(mf) -> list:
    """Extract the occupied orbital energies and (if mf.mol.symmetry is
    enabled) their symmetry labels from a converged pyscf SCF object.

    Companion to get_occupied_orbitals: that one works from a raw
    (epsilon_i, orbsym) pair produced during the ADB search itself, before
    any SCF exists (a fixed guess Fock matrix, not self-consistent). This
    one instead reads mo_energy/mo_occ/mo_coeff straight off a *converged*
    mf object -- used by mask_analysis's track_orbitals to record the
    genuinely self-consistent occupied-orbital spectrum for each subbasis,
    as opposed to find_subspace/expand_mask's guess-Fock-matrix spectrum.

    Returns a list of (energy, irrep_label) tuples, one per occupied MO.
    irrep_label is None throughout when mf.mol.symmetry is off/C1.
    """
    mol = mf.mol
    has_symmetry = bool(mol.symmetry) and mol.groupname != 'C1'
    restricted = (np.asarray(mf.mo_occ, dtype=object).ndim == 1)

    def _labels(mo_coeff):
        if not has_symmetry:
            return [None] * mo_coeff.shape[1]
        return list(symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo_coeff))

    occupied = []
    if restricted:
        for e, occ, lbl in zip(mf.mo_energy, mf.mo_occ, _labels(mf.mo_coeff)):
            if occ > 0:
                occupied.append((float(e), lbl))
    else:
        for spin in (0, 1):
            for e, occ, lbl in zip(
                    mf.mo_energy[spin], mf.mo_occ[spin], _labels(mf.mo_coeff[spin])):
                if occ > 0:
                    occupied.append((float(e), lbl))
    return occupied


def write_orbital_history(
        orbital_history:    list,
        fn:                 str,
        molname:            str = "",
        basisname:          str = "",
        ) -> None:
    """Write a find_subspace(track_orbitals=True) orbital_history to a CSV
    file (one row per occupied orbital per ADB cycle): nfunc,energy,irrep.
    `fn` gets '.csv' appended. `irrep` is left blank for symmetry-blind
    entries (irrep_nelec/orbsym not given to find_subspace).
    """
    with open(fn + ".csv", "w") as f:
        if molname or basisname:
            f.write(f"# molecule={molname} basis={basisname}\n")
        f.write("nfunc,energy,irrep\n")
        for entry in orbital_history:
            nfunc = entry["nfunc"]
            for energy, irrep in entry["orbitals"]:
                f.write(f"{nfunc},{energy:.12f},{irrep if irrep is not None else ''}\n")


def create_shell_separated_mol(
        mol:        gto.MoleBase,
        verbose:    int           = 0) -> gto.MoleBase:
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
            f'  {"Difference":>15s}  {"E_subbasSCF":>15s}  {"Q^2":>18s}'
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
        is_restricted : bool
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
        numc = basis_struct[basis_struct[:,0]==i][:,3] # Number of CGTOs
        func_per_atom[i] = np.sum((2*angl+1) * numc) if not mol.cart \
                           else np.sum((angl + 1)*(angl + 2) // 2 * numc)
    
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
    get_mask_history:           bool            = False,
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
    if get_mask_history:
        mask_history = []
        full_mask = np.zeros(mol.nao, dtype=bool)

    atoms = list(map(lambda x: x[0], mol._atom))

    restricted = (len(F.shape) == 2)
    
    # Loop through atomic blocks in the Fock matrix
    nfuncs_min_tot = 0
    for i,funcs_and_atom in enumerate(zip(func_per_atom, atoms)):
        nfuncs, atom = funcs_and_atom
        if verbose:
            print(f'{atom=}')
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
            print(f'{nfunc_per_minimal_atom=}')
            print(f'{nfuncs=}')
        # Add to molecule minimal number of functions
        nfuncs_min_tot += nfunc_per_minimal_atom
        
        F_ave = F_atom.copy()
        if spherically_average_fock:
            F_ave = spherical_average(F_ave, [shell[1] for shell in smask_atom])

        e_atom, c_atom = eig(F_ave, S_atom.copy())

        def number_of_states(energies, thresh=1e-3):
            if verbose:
                print(f'{energies=}')
            nfuncs_include = nfunc_per_minimal_atom
            # Handle degeneracies
            while nfuncs_include < nfuncs \
              and energies[nfuncs_include]-energies[nfunc_per_minimal_atom-1] < thresh:
                nfuncs_include += 1
            return nfuncs_include


        if restricted:
            nocca, noccb = number_of_states(e_atom), number_of_states(e_atom)
            if verbose:
                print(f'{nocca=}, {noccb=}')
                print(f'Energy of highest orbital {e_atom[nocca-1]*27.2114} eV')
            occs = np.zeros(c_atom.shape[1])
            occs[:nocca] = 2
            P_atom = np.abs(
                c_atom @ np.diag(occs) @ c_atom.conj().T
            )
        else:
            nocca, noccb = number_of_states(e_atom[0]), number_of_states(e_atom[1])
            if verbose:
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
        if verbose:
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
                _, c_mask = eig(
                    mask_matrix(F_ave.copy(), mask_atom),
                    mask_matrix(S_atom.copy(), mask_atom)
                    )
                
                if get_mask_history:
                    # set Pat_i and Pat_j in the full mask to True in
                    # the current atom block
                    full_mask[func_offset + Pat_i] = True
                    full_mask[func_offset + Pat_j] = True
                    full_smask = mask_to_smask(full_mask, smask.copy(), mol.cart)
                    mask_history.append(copy.deepcopy(full_smask))

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


def pair_by_nearest_neighbor(
        mol: gto.MoleBase,
        inclusive_uneven: bool = False
    ) -> list:
    paired_coord_indices = []

    for idx, atom in enumerate(mol._atom):
        # Get flattened list of already paired indices
        already_paired = list(sum(paired_coord_indices, ()))
        if idx in already_paired:
            continue

        # If only single index left: add to indices, break loop
        if len(already_paired) >= mol.natm - 1:
            if inclusive_uneven:
                raise NotImplementedError("'inclusive_uneven has not been implemented.'")
            else:
                paired_coord_indices.append((idx, ))
            break

        # Add the index of the atom being currently looked at
        # so we do not consider the atom itself as nearest neighbor
        already_paired.append(idx)

        # Consider only atoms not already paired up
        # Set subtraction removes the already paired indices from all indices
        atoms_to_consider = [(mol._atom[i], i) for i \
                             in list(set(range(mol.natm)) - set(already_paired))]

        position = np.array(atom[1])
        # Calculate the distance of all atoms to be considered
        # to the current atom
        distances = [
            (np.sqrt(np.sum((position - np.array(atom_to_consider[0][1]))**2)),
             atom_to_consider[1]) for atom_to_consider in atoms_to_consider
            ]
        neighbor_idx = sorted(distances, key=lambda x: x[0])[0][1]

        paired_coord_indices.append((idx, neighbor_idx))
    return paired_coord_indices


def pseudominimal_basis_nearest_neighbor(
    mol:                        gto.MoleBase,
    F:                          np.ndarray,
    S:                          np.ndarray,
    Q_tol:                      float           = 1.0,
    by_shell:                   bool            = True,
    verbose:                    bool            = False,
    spherically_average_fock:   bool            = True,
    inclusive_uneven:           bool            = False,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Create minimal basis from atomic block decomposition
       by pairing the atoms in the molecule by their nearest
       neighbor.

    Args:
        mol :

        F :
        
        S :

        Q_tol :

        by_shell :

        verbose : 

        spherically_average_fock : 

        inclusive_uneven : bool
            Boolean flag, controls whether an unpaired atom will be included as
            a third atom in a pair closest to it when the mol object has an
            uneven number of atoms.
    """
    func_per_atom = basis_functions_per_atom(mol)
    assert np.sum(func_per_atom) == mol.nao

    pseudominimal_basis_mask = np.zeros(mol.nao, dtype=bool)
    smask = init_smask(mol, mol.cart)

    atoms = list(map(lambda x: x[0], mol._atom))

    restricted = (len(F.shape) == 2)
    
    # Loop through atomic blocks in the Fock matrix
    nfuncs_min_tot = 0
    block_idx_pairs = pair_by_nearest_neighbor(
        mol,
        inclusive_uneven = inclusive_uneven)

    for i, block_idxs in enumerate(block_idx_pairs):
        mask = np.zeros(mol.nao, dtype=bool)
        nfunc_in_minimal_block = 0

        for block in block_idxs:
            nfuncs = func_per_atom[block]
            atom = atoms[block]

            func_offset = np.sum(func_per_atom[:block])
            mask[func_offset:func_offset+nfuncs] = True
            
            # Number of functions in minimal basis of current atom,
            # not counting ECP electrons
            nfunc_in_minimal_block += int(np.ceil(
                (ELEMENTS.index(atom)-mol.atom_nelec_core(block)) / 2))

        # smask_atom = list(filter(lambda x: x[3][0] == block, smask))
        smask_block = list(filter(lambda x: x[3][0] in block_idxs, smask))
        mask_block = np.zeros(
            np.sum(func_per_atom[list(block_idxs)]),
            dtype=bool)
        S_block = mask_matrix(S, mask)
        F_block = mask_matrix(F, mask)
        
        # Add to molecule minimal number of functions
        nfuncs_min_tot += nfunc_in_minimal_block
        
        F_ave = F_block.copy()
        if spherically_average_fock:
            F_ave = spherical_average(F_ave, [shell[1] for shell in smask_block])

        e_block, c_block = eig(F_ave, S_block.copy())

        def number_of_states(energies, thresh=1e-3):
            if verbose:
                print(f'{energies=}')
            nfuncs_include = nfunc_in_minimal_block
            # Handle degeneracies
            while nfuncs_include < nfuncs \
              and energies[nfuncs_include] - energies[nfunc_in_minimal_block - 1] < thresh:
                nfuncs_include += 1
            return nfuncs_include


        if restricted:
            nocca, noccb = number_of_states(e_block), number_of_states(e_block)
            if verbose:
                print(f'{nocca=}, {noccb=}')
                print(f'Energy of highest orbital {e_block[nocca-1]*27.2114} eV')
            occs = np.zeros(c_block.shape[1])
            occs[:nocca] = 2
            P_block = np.abs(
                c_block @ np.diag(occs) @ c_block.conj().T
            )
        else:
            nocca, noccb = number_of_states(e_block[0]), number_of_states(e_block[1])
            if verbose:
                print(f'Energy of highest alpha orbital {e_block[0, nocca-1]*27.2114} eV')
                print(f'Energy of highest beta  orbital {e_block[1, noccb-1]*27.2114} eV')
            occs = np.zeros((2, c_block.shape[2]))
            occs[0, :nocca] = 1
            occs[1, :noccb] = 1
            P_block = np.abs(
                c_block[0] @ np.diag(occs[0]) @ c_block[0].conj().T +
                c_block[1] @ np.diag(occs[1]) @ c_block[1].conj().T
                )
        Qlim = nocca+noccb
        if verbose:
            print(f'{Qlim=}')

        if verbose:
            with np.printoptions(precision=2, suppress=True):
                if restricted:
                    print(f'Bound state energies [eV]:    {e_block[e_block<0]*27.2114}')
                    print(f'Occupied state energies [eV]: {e_block[:nocca]*27.2114}')
                else:
                    print(f'Bound alpha state energies [eV]: {e_block[0, e_block[0,:]<0]*27.2114}')
                    print(f'Bound beta  state energies [eV]: {e_block[1, e_block[1,:]<0]*27.2114}')
                    print(f'Occupied alpha state energies [eV]: {e_block[0, :nocca]*27.2114}')
                    print(f'Occupied beta  state energies [eV]: {e_block[1, :noccb]*27.2114}')

        block_indices = set()
        Q = 0
        eps = Q_tol
        if eps >= Qlim:
            raise ValueError(f'Tolerance for Q must be smaller than the number of states, {Qlim=}!')
        # while len(atom_indices) < nfunc_per_minimal_atom:
        P_block = np.round(P_block, 12)
        while np.abs(Q - Qlim) > eps:
            # Find largest element of density matrix
            P_block_idx = np.unravel_index(np.argmax(P_block, axis=None),P_block.shape)
            # Check whether only 1 index tuple was found (no two equal 
            # elements in P_block), otherwise set P_block_idx to the first
            # found index tuple
            if not isinstance(P_block_idx[0], np.int64):
                P_block_idx = P_block_idx[0]
            Pbl_i, Pbl_j = P_block_idx
            
            # Set functions of same shell to True
            if by_shell:
                mask_block[Pbl_i] = True
                mask_block[Pbl_j] = True
                smask_block = mask_to_smask(mask_block, smask_block, mol.cart)
                mask_block  = smask_to_mask(smask_block, mol.cart)
                _, c_mask = eig(
                    mask_matrix(F_ave.copy(),  mask_block),
                    mask_matrix(S_block.copy(), mask_block)
                    )

                # set elements i,j and j,i of P_atom to zero
                P_block[mask_block, :] = 0
                P_block[:, mask_block] = 0

                # add indices where mask is True to atom_indices
                block_indices.update(np.where(mask_block)[0].tolist())
                Q = get_q_sqrd(
                    c_block.copy(), c_mask,
                    S_block[:, mask_block].copy(),
                    (nocca, noccb),
                    )
            else:
                block_indices.extend(list(set((Pbl_i, Pbl_j))))
                P_block[Pbl_i, Pbl_j] = 0
                P_block[np.flip((Pbl_i, Pbl_j))] = 0
                # fixme: update c_mask and Q
                raise RuntimeError('not implemented')

        # # Mask
        block_indices = np.array(list(block_indices))
        # Set mask within block
        # NOTE: This only works if everything up to this point has been
        #       done in order, so that within a block the atoms are
        #       ordered from smallest internal index to the largest
        for i, block in enumerate(block_idxs):
            nfuncs = func_per_atom[block]
            atom = atoms[block]
            # Count the offset of the atomic block
            block_offset = np.sum(func_per_atom[:block])
            # Index range of the atomic block
            rb = block_offset # np.sum(block_idxs[:i], dtype=int)
            re = rb + nfuncs
            block_indices_of_atom = block_indices[(block_indices >= 0) & (block_indices < re - rb)]
            assert len(block_indices_of_atom) <= nfuncs
            pseudominimal_basis_mask[block_offset + block_indices_of_atom] = True

    assert np.sum(pseudominimal_basis_mask) >= nfuncs_min_tot
    return pseudominimal_basis_mask


def find_subspace(
    F:                          np.ndarray,
    S:                          np.ndarray,
    mol:                        gto.MoleBase,
    scf_obj:                    scf.hf.SCF | scf.hf.RHF | scf.uhf.UHF | scf.rohf.ROHF | scf.ghf.GHF,
    conv_tol:                   float           = 1e-2,
    verbose:                    bool            = True,
    get_smask:                  bool            = False,
    variant:                    str             = 'enocc',
    link_shells:                bool            = True,
    nfunc_normalisation:        bool            = True,
    return_mask_history:        bool            = False,
    abd_initialization:         bool            = False,
    initialize_by_projection:   bool            = True,
    spherical_average:          bool            = False,
    abd_Q_tol:                  float           = .5,
    symmetry_aware:             bool            = False,
    irrep_nelec:                dict | None     = None,
    track_orbitals:             bool            = False,
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
        symmetry_aware : bool
            Optional feature, off by default. When True, the search
            diagonalizes trial Fock matrices block-by-irrep and targets
            `irrep_nelec`'s per-irrep occupation instead of the lowest N
            eigenvalues overall (see expand_mask/symmetrized_eig). Requires
            `mol.symmetry` truthy, `mol.groupname != 'C1'`, `irrep_nelec`
            given, `get_smask=True` and `link_shells=True` -- a partial,
            unlinked mask is not guaranteed to stay symmetry-closed, which
            block-by-irrep diagonalization depends on. Default False:
            behaviour is byte-identical to before this option existed.
        irrep_nelec : dict | None
            Target occupation per irrep name (pyscf mf.irrep_nelec format,
            e.g. from scf_obj.get_irrep_nelec() on the converged reference
            full-basis SCF). Required when `symmetry_aware=True`.
        track_orbitals : bool
            Optional feature, off by default. When True, records the
            occupied orbital energies and their symmetry labels (via
            get_occupied_orbitals) at every accepted ADB cycle -- the
            initial mask and every subsequent growth step, mirroring
            return_mask_history's per-step bookkeeping. Symmetry labels
            are only meaningful (non-None) when `symmetry_aware=True`; with
            it False they're still recorded, just with irrep=None
            throughout. When True, the return value becomes a 2-tuple
            `(result, orbital_history)` instead of just `result` --
            existing call sites that assign the return value to a single
            variable are unaffected as long as they leave this at its
            default. `orbital_history` is a list of
            `{'nfunc': int, 'orbitals': [(energy, irrep_label), ...]}`
            dicts, one per recorded cycle, saveable via
            write_orbital_history. Default False: behaviour/return shape
            identical to before this option existed.

    Returns:
        1D boolean ndarray. A mask with selected function indices set to
        True. If collect_data is True, an ndarray is also returned with
        data as described in Args section. Shell mask is returned
        instead of function mask if get_smask is True. If
        `track_orbitals=True`, a `(result, orbital_history)` tuple is
        returned instead of just `result` -- see the `track_orbitals` Args
        entry above.
    """
    if symmetry_aware:
        if not (mol.symmetry and mol.groupname != 'C1'):
            raise RuntimeError(
                "find_subspace(symmetry_aware=True) requires mol.symmetry "
                "to be enabled with a non-C1 point group.")
        if irrep_nelec is None:
            raise RuntimeError(
                "find_subspace(symmetry_aware=True) requires irrep_nelec "
                "(the target per-irrep occupation) to be provided.")
        if not get_smask:
            raise RuntimeError(
                "find_subspace(symmetry_aware=True) requires get_smask=True: "
                "a per-function mask cannot be guaranteed symmetry-closed.")
        if not link_shells:
            raise RuntimeError(
                "find_subspace(symmetry_aware=True) requires link_shells=True: "
                "a mask that toggles individual symmetry-equivalent atoms' "
                "shells independently is not guaranteed to stay "
                "symmetry-closed, which block-by-irrep diagonalization "
                "depends on.")

    if verbose:
        print('Running find_subspace for mol ', mol.atom)
    scf_obj_copy = scf_obj.copy()
    fullbasis_mol = create_shell_separated_mol(mol)

    if abd_initialization and initialize_by_projection:
        warn("Both 'abd_initialization' and 'initialize_by_projection' cannot be True simultaneously.\nInitialization by projection takes precedent.")

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
            spherically_average_fock=spherical_average,
            verbose=verbose)
        mask_init_idx = np.where(mask)[0]
    elif initialize_by_projection:
        print("--- Initializing the dual basis by minimal basis projection ---")
        mask = atomic_block_util.find_projected_minimal_basis_mask(mol)
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
        if link_shells and not initialize_by_projection:
            # If link_shells true, set same shells of same atoms to True
            smask = set_linked_shells(smask, True)
            # if verbose:
            #     print('\nLinked shells: ON\n')

        mask = smask_to_mask(smask, fullbasis_mol.cart)

    sub_hcore = Cfull = Csub = None
    if variant == 'ecore':
        sub_hcore = scf_obj_copy.hf.get_hcore(mol)[mask_init_idx, mask_init_idx]
    elif variant == 'elden':
        _, Cfull = eig(F, S)

    e_sub, Csub, orbsym = diagonalize_masked(
        mask_matrix(F, mask, is_restricted=is_restricted), mask_matrix(S, mask),
        fullbasis_mol if symmetry_aware else None, smask)

    previous_sum = get_iteration_criteria_value(
        variant, epsilon_i=e_sub, nocc=nocc,
        Csub=Csub, Cfull=Cfull, ovlp=S[:, mask],
        irrep_nelec=irrep_nelec if symmetry_aware else None, orbsym=orbsym)

    orbital_history = None
    if track_orbitals:
        orbital_history = [{
            'nfunc': int(np.sum(mask)),
            'orbitals': get_occupied_orbitals(
                e_sub, nocc, irrep_nelec if symmetry_aware else None, orbsym,
                restricted=is_restricted),
        }]

    basis_initialized = False
    if return_mask_history:
        mask_history = []
        if abd_initialization:
            for mb_mask in minimal_basis_history:
                mask_history.append(
                    (mb_mask,
                    0.0,
                    0.0,
                    'Atomic Block Decomposition')
                )
            basis_initialized = True
        elif initialize_by_projection:
            mask_history.append((
                copy.deepcopy(smask) if get_smask else copy.deepcopy(mask),
                previous_sum,
                0.0,
                'Minimal basis projection'))
            basis_initialized = True
        else:
            mask_history.append((
                copy.deepcopy(smask) if get_smask else copy.deepcopy(mask),
                previous_sum,
                0.0,
                'Max element of Fock matrix'))

    while True and not np.all(mask):
        mask, difference, current_criteria_val, n_added, smask = expand_mask(
            F, S, nocc, mask,
            hcore=scf_obj_copy.get_hcore(),
            Cfull=scf_obj_copy.mo_coeff,
            smask=smask, variant=variant, link_shells=link_shells,
            nfunc_normalisation=nfunc_normalisation,
            mol=fullbasis_mol if symmetry_aware else None,
            irrep_nelec=irrep_nelec if symmetry_aware else None,
        )

        if n_added == 0:
            # expand_mask found no remaining candidate that genuinely
            # improves the criterion (see its docstring/EXPAND_MASK_EPS) --
            # the mask above is unchanged, and calling again would return
            # the identical no-op forever. Stop instead of looping.
            break

        if track_orbitals:
            # Record every accepted cycle unconditionally (unlike
            # return_mask_history below, which skips both the
            # not-yet-basis_initialized bootstrap steps and the final,
            # conv_tol-satisfying step) -- track_orbitals is meant to give
            # a complete per-cycle record.
            e_step, _, orbsym_step = diagonalize_masked(
                mask_matrix(F, mask, is_restricted=is_restricted),
                mask_matrix(S, mask),
                fullbasis_mol if symmetry_aware else None, smask)
            orbital_history.append({
                'nfunc': int(np.sum(mask)),
                'orbitals': get_occupied_orbitals(
                    e_step, nocc, irrep_nelec if symmetry_aware else None,
                    orbsym_step, restricted=is_restricted),
            })

        if not basis_initialized:
            basis_initialized = np.sum(mask) >= np.max(nocc)
            continue

        if  abs(n_added * difference) < conv_tol  or \
            sum(mask) == len(mask):
            break
        
        if return_mask_history:
            if basis_initialized:
                mask_history.append(
                    (copy.deepcopy(smask) if get_smask else copy.deepcopy(mask),
                    current_criteria_val,
                    difference))
            else:
                mask_history.append( (
                    copy.deepcopy(smask) if get_smask else copy.deepcopy(mask),
                    0.0,
                    0.0,
                    'Max element of Fock matrix') )

        previous_sum = current_criteria_val


    if get_smask:
        mask = smask

    if return_mask_history:
        mask = mask_history

    if track_orbitals:
        return mask, orbital_history

    return mask


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
    mol:                    gto.MoleBase | None = None,
    irrep_nelec:            dict | None         = None,
    ) -> tuple[np.ndarray, float, float, int, np.ndarray | None]:
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
        mol : MoleBase | None
            Optional. When given together with `irrep_nelec`, every trial
            (and the current) masked Fock/overlap matrix is diagonalized
            block-by-irrep (symmetrized_eig) instead of with the plain,
            symmetry-blind adb.eig, and the 'enocc' criterion targets
            `irrep_nelec`'s per-irrep occupation instead of the lowest N
            eigenvalues overall. Requires `smask` (shell mode) and
            `mol.symmetry` truthy. Must be the *shell-separated* mol whose
            shells `smask`/`mask` index into (i.e. what find_subspace calls
            `fullbasis_mol`, not necessarily its own `mol` argument) -- it
            is passed straight to create_subbasis_mol to build each trial's
            symmetry-adapted basis. Default None: behaviour is identical to
            before this option existed.
        irrep_nelec : dict | None
            Optional. Target occupation per irrep name, pyscf
            mf.irrep_nelec format. Must be given together with `mol`.

    Returns:
        The new mask (boolean ndarray), the current difference in
        eigenvalue sums and the current sum (energy sum of occupied
        orbitals), the number of functions added (0 if no candidate was a
        genuine improvement -- see EXPAND_MASK_EPS -- in which case mask/
        smask are returned unchanged), shell mask if smask is provided.
    """
    restricted = (len(F.shape) == 2)
    symmetry_aware = mol is not None and irrep_nelec is not None
    if symmetry_aware and smask is None:
        raise RuntimeError(
            "expand_mask's symmetry-aware mode (mol/irrep_nelec given) "
            "requires shell mode (smask must not be None); a per-function "
            "mask cannot be guaranteed symmetry-closed.")

    def _eig(maskedF, maskedS, test_smask):
        """Plain adb.eig, or symmetrized_eig + name-tagged orbsym when
        symmetry_aware."""
        evals, coeffs, orbsym = diagonalize_masked(
            maskedF, maskedS, mol if symmetry_aware else None, test_smask)
        return (evals, coeffs), orbsym

    maskedF = mask_matrix(F, mask, restricted)
    maskedS = mask_matrix(S, mask)
    (evals, coeffs), orbsym = _eig(maskedF, maskedS, smask)
    last_sum = 0.0
    if Cfull is None and variant == 'elden':
        _, Cfull = eig(F, S)
    last_sum = get_iteration_criteria_value(
        variant, epsilon_i=evals, nocc=nocc, 
        Csub=coeffs, Cfull=Cfull, ovlp=S[:, mask],
        irrep_nelec=irrep_nelec, orbsym=orbsym)

    test_sums = []
    if smask is None:
        for i, m in enumerate(mask):
            if m:
                continue

            test_mask = copy.deepcopy(mask)
            test_mask[i] = True
            maskedF = mask_matrix(F, test_mask, restricted)
            maskedS = mask_matrix(S, test_mask)
            evals, coeffs = eig(maskedF, maskedS)

            test_sums.append(
                (i,
                get_iteration_criteria_value(
                    'enocc', epsilon_i=evals, nocc=nocc,
                    Csub=coeffs, Cfull=Cfull,
                    ovlp=S[:, test_mask]),
                1))
    else:
        # Gather indices of duplicate shells if link_shells enabled
        # ( if system has more than 1 atom of same type, shells will be
        #   duplicated. )
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

            maskedF = mask_matrix(F, test_mask, restricted)
            maskedS = mask_matrix(S, test_mask)
            (evals, coeffs), test_orbsym = _eig(maskedF, maskedS, test_smask)

            func_keys = [shell[3] for shell in submask[:,3]]
            nfuncs = np.sum(itemgetter(*func_keys)(NFUNCS))
            test_sums.append(
                (i,
                get_iteration_criteria_value(
                    variant, epsilon_i=evals, nocc=nocc, Csub=coeffs,
                    Cfull=Cfull, ovlp=S[:, test_mask],
                    irrep_nelec=irrep_nelec, orbsym=test_orbsym),
                nfuncs))

    if nfunc_normalisation:
        test_differences = [(test_sum[1] - last_sum) / test_sum[2] for test_sum in test_sums]
    else:
        test_differences = [(test_sum[1] - last_sum) for test_sum in test_sums]
    if variant == 'elden':
        array_index = np.argmax(test_differences)
        no_improvement = test_differences[array_index] <= EXPAND_MASK_EPS
    else:
        array_index = np.argmin(test_differences)
        no_improvement = test_differences[array_index] >= -EXPAND_MASK_EPS

    if no_improvement:
        # Every remaining candidate is tied with (or worse than) the
        # current mask -- none of them is a genuine improvement. This
        # happens once a target (e.g. a symmetry-aware irrep_nelec) is
        # already fully satisfied and no untried candidate can affect it:
        # np.argmin/argmax would otherwise just tie-break by array order
        # and commit an arbitrary, meaningless addition. Leave the mask
        # unchanged and signal "nothing to add" via n_added=0 instead --
        # find_subspace stops on seeing this rather than looping forever.
        return mask, 0.0, last_sum, 0, smask

    current_idx_to_flip = test_sums[array_index][0]

    if smask is None:
        mask[current_idx_to_flip] = True
    else:
        submask = smask[shl_indices[current_idx_to_flip]]
        submask[:, 0] = True
        smask[shl_indices[current_idx_to_flip]] = submask
        mask = smask_to_mask(smask)

    nfuncs_in_trial = test_sums[array_index][2]
    return mask, test_differences[array_index], test_sums[array_index][1], nfuncs_in_trial, smask


def mask_analysis(
    mask_history:           np.ndarray,
    mol:                    gto.MoleBase,
    scf_obj:                scf.hf.SCF | scf.hf.RHF | scf.uhf.UHF | scf.rohf.ROHF | scf.ghf.GHF,
    fock:                   np.ndarray,
    ovlp:                   np.ndarray,
    verbose:                bool                = True,
    sym_occ_fname:          str                 = 'occupations.dat',
    molfname:               str | None          = None,
    link_shells:            bool                = True,
    dft:                    bool                = False,
    xc:                     str                 = 'b3lyp',
    grid_level:             int                 = 7,
    C_full:                 np.ndarray | None   = None,
    calculate_correction:   bool                = False,
    irrep_nelec:            dict | None         = None,
    debug:                  bool                = False,
    track_orbitals:         bool                = False,
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
        C_full : np.ndarray | None
            Full basis coefficient matrix, used for projection calculations.
        calculate_correction : bool
            Whether to calculate the dual basis correction of Liang, Steele,
            Head-Gordon et al. for every subbasis.
        track_orbitals : bool
            Optional feature, off by default. When True, records the
            *self-consistent* occupied orbital energies and their symmetry
            labels (via get_occupied_orbitals_from_scf) for every subbasis
            in mask_history that gets its own SCF run (i.e. every smask
            entry; the plain-mask/non-SCF branch has no self-consistent
            solution to record). This is the converged counterpart to
            find_subspace(track_orbitals=True), which records the
            unconverged guess-Fock-matrix spectrum seen *during* the ADB
            search rather than the actual SCF solution of each subbasis.
            When True, the return value becomes a 2-tuple
            `(dataframe, orbital_history)` instead of just `dataframe` --
            existing call sites assigning the return value to a single
            variable are unaffected as long as this stays at its default.
            Default False: behaviour/return shape identical to before this
            option existed.
    Return:
        dataframe : array
            A python array with number of functions,
            current_sum, difference, total SCF energy, SCF energy of
            occupied orbitals and the projection onto the converged
            full basis wave function will on every iteration. If
            `track_orbitals=True`, a `(dataframe, orbital_history)` tuple
            is returned instead -- see the `track_orbitals` Args entry.
    """
    scf_obj_copy = scf_obj.copy()
    if mol.symmetry and mol.groupname != 'C1':
        original_irre_nelec = scf_obj_copy.get_irrep_nelec()
    fullbasis_mol = create_shell_separated_mol(mol)
    is_restricted = len(fock.shape) == 2
    nocc = fullbasis_mol.nelec

    #init_method = mask_history[0][3]
    initialized = False
    dataframe = []
    orbital_history = [] if track_orbitals else None
    last_mask = [False] * fullbasis_mol.nao_nr()
    is_smask = isinstance(mask_history[0][0], np.ndarray)
    if is_smask:
        last_smask = init_smask(mol, mol.cart)
    scf_energy = None
    scf_orbital_energy = None
    
    # Filter mask_history of initialization and ABS
    mask_history_init = list(filter(lambda x: len(x)>=4, mask_history))
    mask_history = list(filter(lambda x: len(x)<=3, mask_history))
    mask_history.insert(0, mask_history_init[-1][:3])

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
        if is_smask:
            minimal_mask = smask_to_mask(mask_history_init[-1][0], mol.cart)
        else:
            minimal_mask = mask_history_init[-1][0]
        print(np.sum(minimal_mask))
        print(atomic_block_util.function_labels_from_mask(minimal_mask, mol))
        
        print('\nNumber of toggled functions:', np.sum(last_mask))
        print(20*'#' + ' INITIALIZATION END ' + 61*'#')

        if link_shells:
            print('\nLink shells: ON')
            print('Additional functions may be added due to shell linking!')
        
        print_data_header()

    for mask_i, current_val, difference, *init in mask_history:
        dE = 0.0
        E_HF_largebasis = 0.0
        if is_smask:
            smask = mask_i
            extracted_basis, ecp_bas = extract_basis(smask, create_shell_separated_mol(fullbasis_mol))
            
            subbasis_mol = Mole(
                atom = fullbasis_mol.atom, basis = extracted_basis,
                charge = fullbasis_mol.charge, spin = fullbasis_mol.spin,
                verbose = fullbasis_mol.verbose, unit = fullbasis_mol.unit,
                ecp = ecp_bas, symmetry = fullbasis_mol.symmetry
                )
            subbasis_mol.build()
            is_restricted = subbasis_mol.spin == 0
            if is_restricted:
                submf = subbasis_mol.RHF()#.newton()
            else:
                submf = subbasis_mol.UHF()#.newton()
            
            mask = smask_to_mask(smask, fullbasis_mol.cart)
            maskedF = mask_matrix(fock, mask, is_restricted)
            maskedS = mask_matrix(ovlp, mask)
            maskedHcore = mask_matrix(scf_obj_copy.get_hcore(), mask)
            if not np.allclose(maskedS, submf.get_ovlp()):
                raise RuntimeError('The masked overlap and the full overlap of masked molecule do not match!')
            if not np.allclose(maskedHcore, submf.get_hcore()):
                raise RuntimeError('The masked core Hamiltonian and the full core Hamiltonian of masked molecule do not match!')

            if dft:
                if is_restricted:
                    submf = subbasis_mol.RKS()
                else:
                    submf = subbasis_mol.UKS()
                submf.xc = xc
                submf.grids.level = grid_level
                submf.grids.prune = None
            submf = submf.apply(scf.addons.remove_linear_dep_)

            # SCF initial guess by projecting the density from full basis to current basis
            dm0_init = scf.addons.project_dm_nr2nr(scf_obj.mol,
                                                   scf_obj.make_rdm1(scf_obj.mo_coeff,
                                                                     scf_obj.mo_occ),
                                                   subbasis_mol)

            # Set the symmetry adapted occupations if present
            if not mol.symmetry or mol.groupname == 'C1':
                submf.irrep_nelec = None
            else:
                if irrep_nelec is not None:
                    submf.irrep_nelec = irrep_nelec
                else:
                    submf.irrep_nelec = scf_obj_copy.get_irrep_nelec()
                # Remove any irreps that are not present in mol
                irrep_list = copy.deepcopy(submf.irrep_nelec)
                for irname in submf.irrep_nelec:
                    if irname not in subbasis_mol.irrep_name:
                        if is_restricted:
                            assert submf.irrep_nelec[irname] == 0
                        else:
                            assert submf.irrep_nelec[irname] == (0, 0)
                        del irrep_list[irname]
                submf.irrep_nelec = irrep_list
            
            if debug:
                submf.verbose = 4

            # Using level_shift, do a few first order SCF cycles
            submf.level_shift = 1.0
            submf.max_cycle = 3
            submf.kernel(dm0=dm0_init)

            # Use the level shift calulcation density as initial guess
            # Restore default parameters and switch to second order CIAH
            submf = submf.newton()
            submf.level_shift = 0.0
            submf.max_cycle = 50
            submf.kernel()

            # Double check that occupations of the irreps have not changed
            if mol.symmetry and mol.groupname != 'C1':
                if not all([elem == submf.get_irrep_nelec()[key] for key, elem in submf.get_irrep_nelec().items()]):
                    raise RuntimeError(f'The irrep occupations have changed from the ones dictated by the full basis solution.\nOriginal: {original_irre_nelec}\nThis cycle: {submf.get_irrep_nelec()}')
            
            subbasis_converged = submf.converged
            scf_energy = submf.e_tot

            if track_orbitals:
                orbital_history.append({
                    'nfunc': int(np.sum(mask)),
                    'orbitals': get_occupied_orbitals_from_scf(submf),
                })

            if is_restricted:
                # nocc_sb = np.sum(submf.mo_occ > 0)
                nocc_sb = len(submf.mo_occ > 0)
                scf_orbital_energy = sum(submf.mo_energy[:nocc_sb])
            else:
                # nocc_sb = [np.sum(submf.mo_occ[0] > 0), np.sum(submf.mo_occ[1] > 0)]
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
            
            # Calculate dual basis correction as per Liang, Steele,
            # Head-Gordon et al.
            if calculate_correction:
                P_sub = submf.make_rdm1()
                P_full_projected = project_dm_nr2nr(
                    subbasis_mol,
                    P_sub,
                    fullbasis_mol)
                dE, E_HF_largebasis = dual_basis_energy_correction(
                    fullbasis_mol.HF(),
                    P_full_projected,
                )


            if not submf.converged:
                print('The SCF did not converge in the subbasis. Results may be unreliable.', file=sys.stderr)
        else:
            mask = mask_i
            e, subbasis_coeffs = eig(mask_matrix(fock, mask, is_restricted=is_restricted), mask_matrix(ovlp, mask))
            Q_sqrd = get_q_sqrd(
                C_full, subbasis_coeffs,
                ovlp[:,mask], nocc
            )
        
        if verbose:
            if is_smask:
                changes = [i for i in range(len(smask)) if smask[i][0] != last_smask[i][0]]
                if len(changes) == 0:
                    label = "Init"
                else:
                    label = get_atom_shell_label(mol, changes[0], link_shells=link_shells*initialized)
                    if link_shells:
                        label = ' '.join(label.split(' ')[1:])
                        label = '*' + label
            else:
                changes = [i for i in range(len(mask)) if mask[i] != last_mask[i]]
                if len(changes) == 0:
                    label = "Init"
                else:
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
                dE,
                E_HF_largebasis,
                subbasis_converged,
            ])
        last_mask = copy.deepcopy(mask)
        if is_smask:
            last_smask = copy.deepcopy(smask)

    if track_orbitals:
        return dataframe, orbital_history

    return dataframe



"""

█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
███████████████████████████████████████████████████████▓▒▒▒▒▒▒▒▒▒▓▓▓█████████████████████████████████████████████████████
███████████████████████████████████████████████████▓▒▒░░░░░░░░░░░░░░░▒▒▒▓████████████████████████████████████████████████
██████████████████████████████████████████▒▒░░░░░▒░░░░░░░░░░░░░░░░░░░░░░░░▒▒▓████████████████████████████████████████████
██████████████████████████████████████▓▒░░░░░░░▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒█████████████████████████████████████████
███████████████████████████████████▓▒░░░░░░░░░▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒██████████████████████████████████████
█████████████████████████████████▓▒░░░░░░░░░░░▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▓███████████████████████████████████
███████████████████████████████▒░░░░░░░░░░░░░░▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▓█████████████████████████████████
█████████████████████████████▒░░░░░░░░░░░░░░░░▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒███████████████████████████████
███████████████████████████▓▒░░░░░░░░░░░░░░░░░▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒█████████████████████████████
██████████████████████████▒░░░░░░░░░░░░░░░░▒▒░▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒░░░░░░░░░░▓███████████████████████████
█████████████████████████▒░░░░░░░░░░░░░░░▒▒░░░▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒░░░░░░░░░░▒▓█████████████████████████
███████████████████████▓▒░░░░░░░░░░░░░░▒▒░░░░░▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒░░░░░░░░░░░░▒████████████████████████
██████████████████████▓▒░░░░░░░░▒░░░░▒░░░░░░░░▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒░░░░░░░░░░░░▒███████████████████████
█████████████████████▓▒░░░░▒░░░░▒░░▒▒░░░░░░░░░▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒░░░░░░░░░░░░░▒██████████████████████
█████████████████████▒░░░░▒▒░░░▒▒░▒░░░░░░░░░░░▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒░░░░░░░░░░░░░▒█████████████████████
████████████████████▒░░░░▒▒▒░░░▒▒▒░░░░░░░░░░░░▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒░░░░░░░░░░░░░░▒████████████████████
███████████████████▓░░▒░▒▒▒▒░░░▒▒░░░░░░░░░░░░▒▒▒░░░░░░░░░░░░▒░░░░░░░░░░░░░░░░░░░░░░░░▒░░░░░░░░░░░░░░░▒███████████████████
███████████████████▒░░▒▒▒▒▒▒░░░▒░░░░░░░░░░░░▒░▒▒░░░░░░░░░░░░▒░░░░░░░░░░░░░░░░░▒▒░░░░░▒▒░░░░░░░░░░░░░░░▓██████████████████
██████████████████▒░░▒▒▒▒▒▒▒░░░▒░░░░░░░░░░▒▒░░░▒░░░░░░░░░░░░▒▒░░░░░░░░░░░░░░░░▒▒░░░░░░▒░░░░░░░░░░░░░░░░▓█████████████████
█████████████████▓░░░▒▒▒▒▒▒▒░░░▒░░░░░░░░░▒▒░░░░▒░░░░░░░░░░░░░▒▒░░░░░░░░░░░░░░░░▒░░░░░░▒▒░░░░░░░░░░░░░░░▒█████████████████
█████████████████▓░░░▒▒▒▒▒▒▒░░░▒░░░░░░░░▒░░░░░░▒░░░░░░░░░░░░░░▒░░░░░░░░░░░░░░░░▒░░░░░░░▒░░░░░░░░░░░▒░░░▒▓████████████████
█████████████████▒░░░▒▒▒▒▒▒▒░░░▒░░░░░░▒▒░░░░░░░▒▒░░░▒▒░░░░░░░░▒▒░░░░░░░░░░░░░░░▒░░░░░░░▒░░░░░░░░░░░▒░░░░▒████████████████
█████████████████▒░░▒▒▒▒▒▒▒▒░░░▒░░▒░▒▒░░░░░░░░░▒▒░░░▒▒░░░░░░░░░▒▒░░░░░░░░░░░░░░▒░░░░░░░▒░░░░░░░░░░░▒░░░░▒████████████████
████████████████▓▒░░▒▒▒▒▒▒▒▒░░▒▒▒▒░░░░░░░░░░░░░▒▒░░░▒▒▒░░░░░░░░░▒░░░░░░░░░░░░░░▒░░░░░░░▒▒░░░░░░░░░░▒▒░░░░▓███████████████
████████████████▒▒░▒▒▒▒▒▒▒▒▒░░▒░░░░▒▒░░░░░░░░░░░▒░░░▒▒▒░░░░░░░░░▒▒░░░░░░░░░░░░░▒▒░░░░░░░▒░░░░░░░░░░▒▒░░░░▒███████████████
████████████████▒▒░▒▒▒▒▒▒▒▒▒░░▒░░░░▒░░░░░░░░░░░░▒░░░░▒▒░░░░░░░░░▒▒░░░░░░░░░░░░░▒▒░░░░░░░▒░░░░░░░░░░░▒░░░░▒███████████████
████████████████▒▒░▒▒▒▒▒▒▒▒▒░░▒░░░▒░░░░░░░░░░░░░▒░░░░▒▒░░░░░░░░░░▒▒░░░░░░░░░░░░▒▒░░░░░░░▒▒░░░░░░░░░░▒░░░░▒███████████████
███████████████▓▒▒▒▒▒▒▒▒▒▒▒▒░░▒░░▒▒░░░░░░░░░░░░░░▒░░░▒▒▒░░░░░░░░░░▒▒░░░░░░░░░░░▒▒░░░░░░░▒▒░░░░░░░░░▒▒░░░░▒▓██████████████
███████████████▓▒▒▒▒▒▒▒▒▒▒▒▒░░▒░░▒░░░░░░░░░░░░░░░▒░░░▒▒▒░░░░░░░░░░▒▒░░░░░░░░░░░▒▒░░░░░░░▒▒░░░░░░░░▒▒▒▒▒▒▒▒▒██████████████
███████████████▓▒▒▒▒▒▓▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░▒░░░▒▒▒░░░░░░░░░▒▒▒░░░░░░░░░░▒▒░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██████████████
███████████████▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░▒░▒▒▒▒▒░░░░░░░░░▒▒▒░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██████████████
███████████████▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░▒░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██████████████
███████████████▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██████████████
███████████████▓▒▒▒▒▒▓▓▒▒▓▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██████████████
███████████████▓▒▒▒▒▒▓▒▒▓▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██████████████
███████████████▓▒▒▒▒▒▓▒▒▒▓▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██████████████
███████████████▓▒▒▒▒▒▓▒▒▒▒▒▒▒▒▒░░░░░▒▒▒▒▒░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██████████████
███████████████▓▒▒▒▒▒▓▒▒▓▒▓▒▒▒▒░░░░░░░░░░░░▒▒▒▒▒▒▒░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██████████████
████████████████▒▒▒▒▒▓▒▒▓▒▒▒▒▒▒░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▒▒▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██████████████
████████████████▒▒▒▒▒▓▒▒▓▓▒▒▒▒▒▒▒▓████████████▓▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░▒▒▓▒▒████████████████▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓██████████████
████████████████▓▒▒▒▒█▒▒▒▒▓▒▒▒▒░░▒▓██▒▒▒██▒▒▒▒▒▒██▓█▒▒▒▒▒▒░░░░░░░░░▒▓▒░▒█▓▒█▒▒▓█▓▒▓▓▒▒▒██▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓██████████████
████████████████▓▒▒▒▒▒█▒▒▒▒▒▒▒▒░░░░▒█▓░░▓█▒▒▓█▒░▒░░░▒▒░▒▒░░░░░░░░░▒▓░░░░░░░▓▒▒▒▓▒▒▒▒░░█▒░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒███████████████
█████████████████▒▒▒▒▒▓▓▒▒▒▒▒▒▒▒░░░░░▓▒░░▒▓▒▒▒▒▒▓░░░░░░░░░░░░░░░░░▒▒░░░░░░░░▒▒▒▒▒▓▒░▒█▒░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒███████████████
█████████████████▓▒▒▒▒▓▓▓▓▒▒▒▒▒▒▒░░░░░░▒▒░░▒▓▓▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▓▒░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒███████████████
█████████████████▓▒▒▒▒▓▓▓▓█▒▒▒▒▒▒▒░░░░░░░▒▓▓▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▓▓▓▓▒▒░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓███████████████
██████████████████▒▒▒▒▓▓▓▓▓▓▒▒▒▒▒▒▒░░░░░░░░▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒████████████████
██████████████████▒▒▒▒▓▓▓▓▓▓▒▒▒▓▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒████████████████
██████████████████▓▒▒▒▒▓▓▓▓▓▓▒▒▒▓▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓████████████████
███████████████████▒▒▒▒▓▓▓▓▓▓▒▒▒▓▓▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒█████████████████
███████████████████▓▒▒▒▓▓▓▓▓▓▒▒▒▓█▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓█████████████████
████████████████████▒▒▒▒▓▓▓▓▓▓▒▒▓██▒▒▒▒░░░░░░░░░░░░░░░░░░░░▒▒▒░▒▒░░░░░░░░░░░░░░░░░░░░░░▒█▒▒▒▒▒▒▒▒▒▓▒▒▒▒██████████████████
████████████████████▓▒▒▒▓▓▓▓▓▓▒▒▓███▒▒▒▒░░░░░░░░░░░░░░░░░░░░▒▒▒▓░░░░░░░░░░░░░░░░░░░░░░░▓█▒▒▒▒▒▒▒▒▒▓▒▒▒▓██████████████████
█████████████████████▓▒▒▓▓▓▓▓█▓▒▒████▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▓██▒▒▒▒▒▒▒▒▒█▒▒▒███████████████████
█████████████████████▓▒▒▓▓▓████▒▒▓████▒▒▒░░░░░░░░░░░░░▒░░░░░░░░░░░░░▒▒░░░░░░░░░░░░░░▒████▒▒▒▒▒▒▒▒▒█▒▒████████████████████
██████████████████████▓▒▓▓█████▓▒▓█████▒▒▒▒░░░░░░░░░░░░░▒▒▓▒▒▒▒▒▓▓▒▒░░░░░░░░░░░░░░░▒█████▒▒▒▒▒▒▒▒▒▓▒▓████████████████████
███████████████████████▓▒▓██████▒▒██████▓▒▒▒▒░░░░░░░░░░░░▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░▓█████▓▒▒▒▒▒▒▒▒▓▓▓█████████████████████
████████████████████████▒▓██████▒▒████████▓▒▒▒▒░░░░░░░░░░░░░▒▒▒▒░░░░░░░░░░░░░░░░▒███████▓▒▒▒▒▒▒▒▒▓▓██████████████████████
████████████████████████▓▓██████▓▒▓█████████▓▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒█████████▓▒▒▒▒▒▒▒▒▓███████████████████████
█████████████████████████▓███████▒▒████████████▓▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░▒▓███████████▓▒▒▒▒▒▒▒▒████████████████████████
█████████████████████████████████▓▒███████████▒▓██▓▒▒▒░░░░░░░░░░░░░░░░░░▒▓▒░▒███████████▒▒▒▒▒▒▒▒▒████████████████████████
██████████████████████████████████▒▓██████████▒▒▒▒▓██▓▒▒░░░░░░░░░░░░░▒▒▓▒░░░▒███████████▒▒▒▒▒▒▒░▒████████████████████████
███████████████████████████████████▒██████████▒▒▒▒▒▒▒▒██▒▒░░░░░░░░░▒▓▒▒░░░░░▒███████████▒▒▒▒▒▒░░▓████████████████████████
███████████████████████████████████▓▓████████▓▓▓▒▒▒▒▒▒▒▒▒▓█▒░░░▒▓▓▒▒▒▒░░░░▒▓▓███████████▒▒▒▒▒░░▒█████████████████████████
████████████████████████████████████▓█████████▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░▒▓▓▓▓███████████▒▒▒▒░░░▒█████████████████████████
█████████████████████████████████████▓█████████▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░▒▓▓▓▓█▓▓█████████▓▒▒▒░░░░▒█████████████████████████
█████████████████████████████████████████████████▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒░░░░░▓▓▓▓█▓▓▒▓█████████▓▒▒░░░░░▓█████████████████████████
███████████████████████████████████████████████████▓▓▓██▒▒▒▒▒▒░░░░▒█▓▒▓▓█▓▓▓▒▓█████████▒▒░░░░░░██████████████████████████
███████████████████████████████████████████████████▓▒▒▒░░▒▒▒░░░░░░░░░░▒█▓▓▓▓▒▓▓████████▒░░░░░░▒██████████████████████████
███████████████████████████████████████████████████▓▒▒▒░░░░░░░░░░░░░░░▓█▓▓▓▓▒▓▓████████▒░░░░░░▒██████████████████████████
█████████████████████████████████████████▓██████████▓▒▒░░░░░░░░░░░░░░▒██▓▓▓▓▓▓▓▓██████▓░░░░░░░▓██████████████████████████
███████████████████████████████████████▓▓▓▓▓████████▓▒▒▒░░░░░░░░░░░░░▒▓█▓▓▓▓▓▓▓▓▓▓████▒░░░░░░▒███████████████████████████
█████████████████████████████████████▓▓▓▓▓▓▓▓▓████▓▓▓▒▒▒░░░░░░░░░░░░░▓▓▓▓█▓▓▓▓▓▓▓▓████▒░░░░░░▓███████████████████████████
█████████████████████████████████▓▓▓▓▓▓▓▓▓▓▓▓▓▓█▓▓▓▓▒▒▒▒▒░░░░░░░░░░░░░░▒▓▓▓█▓▓▓▓▓▓▓██▓░░░░░░░▓███████████████████████████
██████████████████████████████▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░▒▓▓▓█▓▓▓▓▓▓██▒░░░░░░▒▓███████████████████████████
████████████████████████▓▓▓█▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░▒▓▒▒▒▒░░░░░░░░░░░░░░▒▓▓▓▓▓▓▓▓▓███▒░░░░░░▓▓██▓▓▓██████████████████████
██████████████████▓▒░░▒▓▓▓▓▓█▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█▓▓▓▒░░░▒▒▒▒░░░░░░░░░░░░░▒▓▓▓▓▒▓▓▓▓▓▓█▒░░░░░░▒▓██▓▓▓█▒░▒▓██████████████████
██████████████▓▒░░░░░░░▒█▓▓▓██▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█▓▓▓░░░░▒▒▒▒░░░░░░░░░░░░▓▓▓▓▓▒▓▓▓▓▓█▓░░░░░░░▓▓█▓▓▓█▒░░░░░░░▒██████████████
██████████▒▒░░░░░░░░░░░▒▓▓▓▓▓█▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█▓▓▓▓▒░░░▒▒▒░░░░░░░░░░░▓▓▓▓▓▒▒▓▓▓▓▓▓▒░░░░░░▒▓▓█▓▓▓▓░░░░░░░░░░░░▒▒▓████████
▓▓▓▒▒▒░░░░░░░░░░░░░░░░░░▒█▓▓▓██▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█▓▓▓▓▓▒░▒▓▒░░░░░░░░▒▓▓▓▓▓▓▓▒▓▓▓▓▓▓▓▒░░░░░▒▓▓██▓▓▓▓░░░░░░░░░░░░░░░░░░░▒▒▓▓
░░░░░░░░░░░░░░░░░░░░░░░░░▓▓▓▓▓█▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█▓▓▓▓▓▓▒▒▒░░░░░░░▓▓▓▓▓▓▓▓▒▓▓▓▓▓▓▓▒░░░░▒▒▒▓▓██▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░
░░░░░░░░░░░░░░▒█▓▓▓▒▒▒▒▒▒▒▓▓▓▓█▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█▓▓▓▓▓▒▒░░░░░░▓▓▓▓▓▓█▒▓▓▓▓▓▓▓▓▓▒░░░▒▒▒▓▓▓█▓▓▓▓▓▒▒▒▓▓▓▓▒▒▒▓░░░░░░░░░░░░
░░░░░░░░░░░░░▒▒░░░░░░░░░░▓▓▓▓▓█▓▓████████████▓▓▓▓▓▓▓█▓▓▓▓▓▒▒▒▒▒▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒░░▒▒▒▒▓██▓▓█▓▓▓▓░░░░░░░░░░░▒░░░░░░░░░░░
░░░░░░░░░░░░░▒░░░░░░░░░░▒█▓▓▓██▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓███▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓██▓▓▓▓▓▒▒▒░▒▒▒▒▒▓▓▓▓▒▓▓▓▓▓░░░░░░░░░░░▒▒░░░░░░░░░░
░░░░░░░░░░░░▒▒░░░░░░░░░░▓▓▓▓▓██▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓██▓▓▓█▓▓▓▓▓▓▓▓▓▓▓█▓▓██▒▒▒▒▓▓▓▓▒▒▒▒▒▒▒▓▓▓▓▓▓█▓▓▓▓▒░░░░░░░░░▒▓▒░░░░░░░░░░

"""