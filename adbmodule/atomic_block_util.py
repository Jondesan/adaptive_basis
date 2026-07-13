import numpy as np
import pyscf
from copy import deepcopy
import adb
import os
import argparse
import re
import adbutils
from mask import link_shells

num_of_occupying_electrons = 1

def eig(h, s):
            '''Solver for generalized eigenvalue problem

            .. math:: HC = SCE
            '''
            from scipy.linalg import eigh
            e, c = eigh(h, s)
            idx = np.argmax(abs(c.real), axis=0)
            c[:,c[idx,np.arange(len(e))].real<0] *= -1
            return e, c, idx


class BlockDecomposedMol:
    def __init__(
        self,
        mol,
        initial_guess='atom',
    ) -> None:
        self.mol = mol
        self.atom_labels = [atom[0] for atom in self.mol._atom]
        self.nblocks = len(self.atom_labels)
        self.is_restricted = self.mol.spin == 0
        self.mf = mol.HF()
        self.initial_guess = initial_guess
        self.bfpa = adb.basis_functions_per_atom(mol)
        self.scf_initialized = False
        bfpa_cumsum = np.cumsum(self.bfpa)
        # Store the atomic block index ranges
        self.atomic_function_ranges = np.array([[0, bfpa_cumsum[0]-1]])
        self.atomic_function_ranges = np.append(
            self.atomic_function_ranges, 
            np.array(list(
                zip(bfpa_cumsum[:-1], bfpa_cumsum[1:]-1)
            )),
            axis=0
        )

        self.ao_labels = np.array(pyscf.gto.mole.cart_labels(self.mol)) if self.mol.cart \
            else np.array(pyscf.gto.mole.sph_labels(self.mol))
        self.dm0 = None
        self.fock = None


    def list_all_ao_labels(self) -> None:
        print("Printing all available atomic orbitals:")
        for i, label in enumerate(self.ao_labels):
            print(f'{i:4}: {label}')
        print()


    def get_orbitals_in_block(self, which_block) -> np.ndarray:
        """Return list of indices of the available orbitals
        in the given atomic block.

        Indices range from 0 to N_k-1 where N_k is the number of
        functions in the atomic block.
        
        Args:
        
        
        Returns:
        """
        ra, rb = self.atomic_function_ranges[which_block]
        return np.arange(0, rb - ra + 1)


    def initialize_scf_object(self) -> None:
        print('###############################')
        print(' Initializing the SCF object')
        print('  and calculating the initial')
        print('  guess orbitals in full basis')
        print('###############################\n\n')

        self.mf.sap_basis = 'sapgraspsmall'
        self.dm0 = self.mf.get_init_guess(key = self.initial_guess)
        self.fock = self.mf.get_fock(dm = self.dm0)
        self.scf_initialized = True


    def get_asymb_of_atomic_block(self, which_block) -> str:
        """Return the atomic symbol of the atom on the given block.
        
        Args:

        
        Returns:
        """
        return self.mol._atom[which_block][0]


    def calculate_orbitals_of_atomic_block(
        self,
        which_block=0,
    ) -> tuple:
        
        if which_block < 0 or \
           which_block > len(self.atomic_function_ranges) - 1:
            raise ValueError("The atomic block being accessed does not exist")

        if self.fock is None:
            self.initialize_scf_object()
        
        ra, rb = self.atomic_function_ranges[which_block]
        Fatom = deepcopy(self.fock[ra:rb + 1, ra:rb + 1])
        Satom = deepcopy(self.mf.get_ovlp()[ra:rb + 1, ra:rb + 1])

        energies, coeffs, idx = eig(Fatom, Satom)
        
        return energies, coeffs, idx


    def output_orbital_cub(
        self,
        coeff,
        which_block,
        which_orbital,
        cubefilename,
    ) -> None:

        ra, _ = self.atomic_function_ranges[which_block]
        # ioo = index of orbital in full system
        ioo = ra + which_orbital

        padded_coeff = self.pad_Cmatrix(coeff, which_block, which_orbital)
        pyscf.tools.cubegen.orbital(
            self.mol,
            cubefilename,
            padded_coeff[:,ioo]
        )


    def output_density_cub(
        self,
        coeff,
        which_block,
        which_orbital,
        cubefilename,
        num_of_occupying_electrons=1
    ) -> None:
        
        occupations = np.zeros(mol.nao_nr())
        ra, _ = self.atomic_function_ranges[which_block]
        # ioo = index of orbital in full system
        ioo = ra + which_orbital
        occupations[ioo] = num_of_occupying_electrons
        padded_coeff = self.pad_Cmatrix(coeff, which_block, which_orbital)
        dm = pyscf.scf.hf.make_rdm1(padded_coeff, occupations)
        pyscf.tools.cubegen.density(
            mol,
            cubefilename,
            dm
        )


    def index_of_orbital_in_full_sys(
            self,
            which_block,
            which_orbital
    ) -> int:
        """Finds the index of the orbital with respect to the full system.
        
        Args:
        
        
        Returns:
        """
        ra, _ = self.atomic_function_ranges[which_block]
        # ioo = index of orbital in full system
        ioo = ra + which_orbital    

        return ioo


    def pad_Cmatrix(
        self,
        coeff,
        which_block,
        which_orbital
    ) -> np.ndarray:
        
        ra, rb = self.atomic_function_ranges[which_block]
        # ioo = index of orbital in full system
        ioo = ra + which_orbital

        # Check if coeff is from a restricted calculation
        is_restricted = len(coeff.shape) == 2
        if is_restricted:
            padded_coeff = np.zeros((mol.nao_nr(), mol.nao_nr()))
            padded_coeff[ra:rb+1, ioo] = deepcopy(coeff[:, which_orbital])
        else:
            padded_coeff = np.zeros((2, mol.nao_nr(), mol.nao_nr()))
            padded_coeff[:, ra:rb+1, ioo] = deepcopy(coeff[:, :, which_orbital])


        return padded_coeff


    def calc_isoval(
        self,
        coeff                   : np.ndarray,
        which_block             : int,
        which_orbital           : int = 0,
        fraction                : float = .9,
    ) -> tuple:

        print()
        print( 'Calculating density and orbital isovalue')
        print(f'containing {fraction} of the electron density')

        ra, _ = self.atomic_function_ranges[which_block]
        # ioo = index of orbital in full system
        ioo = ra + which_orbital
        
        print(f'Calculating for atomic orbital {self.ao_labels[ioo]}\n')
        
        padded_coeff = self.pad_Cmatrix(coeff, which_block, which_orbital)

        return calculate_isovalue_for_fraction_of_charge(
            padded_coeff,
            self.mol,
            which_mo_to_calculate=ioo,
            fraction=fraction,
        )


    def generate_atomic_coefficients(self, which_block) -> np.ndarray:
        """Do an atomic SCF calculation to obtain converged orbitals
        for the atom in the given atomic block.
        
        Args:
        
        
        Returns:
        """
        asymb = self.get_asymb_of_atomic_block(which_block)
        acoords = ' '.join([str(x) for x in self.mol._atom[which_block][1]])
        basis = self.mol.basis

        atom_def = f'{asymb} {acoords}'
        e_and_c = []
        for spin in range(7):
            try:
                atomic_mol = pyscf.M(
                    atom=atom_def,
                    basis=basis,
                    spin=spin)
                mf = atomic_mol.HF()
                mf.kernel()
                e_and_c.append([mf.e_tot, mf.mo_coeff])
            except:
                continue
        coeff = e_and_c[np.argmin([etot[0] for etot in e_and_c])][1] 
        
        return coeff


    def generate_orbital_cub_from_atomic_calc(
            self,
            coeff,
            which_block,
            which_orbital,
            cubefilename,
    ) -> tuple:
        """
        """

        print()
        print('Calculating isovalues and orbital')
        print('output for the atomic SCF')
        isovalue_data = self.calc_isoval(coeff, which_block, which_orbital)
        self.output_orbital_cub(coeff, which_block, which_orbital, cubefilename)
        
        return isovalue_data


    def __str__(self) -> str:
        string_rep = ""
        string_rep += "Block decomposed molecule with atoms\n"
        string_rep += f"{str(self.atom_labels)}\n\n"
        string_rep += "Atomic function index ranges:\n"
        for i, index_range in enumerate(self.atomic_function_ranges):
            string_rep += f"{self.atom_labels[i]} -> {index_range}"
            string_rep += f" ({index_range[1] - index_range[0] + 1}"
            string_rep += " functions in atomic basis)\n"

        string_rep += "\n"
        string_rep += "The SCF calculation object has "
        string_rep += f"{'' if self.scf_initialized else 'not '}been initialized!\n"

        return string_rep


def orbital_key(orb):
    shell_order = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5}
    # Extract number and shell letter from the start of the string
    match = re.match(r'^(\d+)([spdfghi])', orb.lower())
    if match:
        n, shell = match.groups()
        return (shell_order.get(shell, 99), int(n))
    else:
        # fallback for malformed or unexpected orbitals
        return (float('inf'), int('inf'))


def print_labels_of_functions_in_mask(
        mask,
        mol,
        print_actual_minimal_basis = False):
    atom_dict = function_labels_from_mask(mask, mol)
    # Sort dictionary by the internal atom index
    atom_dict = dict(sorted(
        atom_dict.items(),
        key=lambda item: int(item[0].split()[0])))

    if print_actual_minimal_basis:
        minimal_mol = deepcopy(mol)
        minimal_mol.basis = 'sto3g'
        minimal_mol.build()
        minimal_mask = np.ones(minimal_mol.nao_nr(), dtype=bool)

        print("\nFunctions in the 'actual' minimal basis:")
        minimal_atom_dict = function_labels_from_mask(minimal_mask, minimal_mol)
        # Sort dictionary by the internal atom index
        minimal_atom_dict = dict(sorted(
            minimal_atom_dict.items(),
            key=lambda item: int(item[0].split()[0])))
        for key, elem in minimal_atom_dict.items():
            print(f'{key}: {elem}')

    print('\nFunctions in the pseudominimal basis:')
    for key, elem in atom_dict.items():
        print(f'{key}: {elem}')
    print()


def find_pseudominimal_basis_mask(
        mol,
        F,
        S,
        q_tol                    : float = .9,
        init_guess               : str   = 'atom',
        sap_basis                : str   = 'sapgraspsmall',
        sph_avg_fock             : bool  = False,
        run_dft                  : bool  = False,
        xcfunc                   : str   = 'PBE',
        pair_by_nearest_neighbor : bool  = False
    ):

    grid_level = 7
    mf = pyscf.dft.KS(mol) if run_dft else pyscf.scf.HF(mol)
    if run_dft:
        mf.grids.level = grid_level
        mf.xc = xcfunc
        mf.grids.prune = None
    
    # Initialize init guess method
    mf.init_guess = init_guess
    if init_guess == 'sap':
        mf.sap_basis = sap_basis
    dm0 = mf.get_init_guess(key=init_guess)
    # we need the corresponding Fock matrix
    F = mf.get_fock(dm=dm0)
    S = mf.get_ovlp()
    # Find minimal basis using atomic block decomposition
    if pair_by_nearest_neighbor:
        return adb.pseudominimal_basis_nearest_neighbor(
            mol, F, S, Q_tol=q_tol, by_shell=True,
            verbose=False,
            spherically_average_fock=sph_avg_fock,
        )
    else:
        return adb.atomic_block_minimal_basis(
            mol, F, S, Q_tol=q_tol, by_shell=True,
            get_mask_history=False, verbose=False,
            spherically_average_fock=sph_avg_fock,
        )


def funcs_on_shell(angl, cart=False):
    return (angl + 1) * (angl + 2) / 2 if cart else 2 * angl + 1


def get_array_of_angular_momenta(mol):
    angls = np.zeros(mol.nao_nr(), dtype=int)
    input_idx = 0
    for i, bas in enumerate(mol._bas):
        nfuncs = funcs_on_shell(bas[1], mol.cart)
        angls[input_idx:input_idx + nfuncs] = bas[1]
        input_idx += nfuncs
    return angls


def get_array_of_angular_momenta_and_atom_id(mol):
    angls_aid = np.zeros((mol.nao_nr(), 2), dtype=int)
    input_idx = 0
    for i, bas in enumerate(mol._bas):
        nfuncs = funcs_on_shell(bas[1], mol.cart)
        angls_aid[input_idx:input_idx + nfuncs, 0] = bas[1]
        angls_aid[input_idx:input_idx + nfuncs, 1] = bas[0]
        input_idx += nfuncs
    return angls_aid


def find_projected_minimal_basis_mask(
        mol,
    ):
    try:
        mol_sto3g = pyscf.M(
            atom = mol.atom,
            basis = 'sto3g',
            ecp = mol.ecp,
            spin = mol.spin,
            charge = mol.charge,
            cart = mol.cart,
            unit = mol.unit,
            symmetry = mol.symmetry,
        )
    except:
        mol_sto3g = pyscf.M(
            atom = mol.atom,
            basis = 'sto3g',
            spin = mol.spin,
            charge = mol.charge,
            cart = mol.cart,
            unit = mol.unit,
            symmetry = mol.symmetry,
        )
    mask = np.zeros(mol.nao_nr(), dtype=bool)    
    s21 = pyscf.gto.mole.intor_cross(
        'int1e_ovlp', mol, mol_sto3g)

    # Find the AO-id offsets of the atoms
    atom_offsets = pyscf.gto.mole.aoslice_by_atom(mol)[:,2]

    # Generate arrays with the angular momentum l and atom-id
    # for all functions. Element n corresponds the l and atom-id
    # of the n:th basis function
    sto3g_angls = get_array_of_angular_momenta_and_atom_id(mol_sto3g)
    sto3g_aid = sto3g_angls[:, 1]
    sto3g_angls = sto3g_angls[:, 0]

    prev_angl = None
    prev_aid = None
    shell_offset = 0

    ao_labels = mol.ao_labels()
    for (ovlp_col, angl, atom_id) in zip(s21.T, sto3g_angls, sto3g_aid):
        # Count functions of angular momentum angl in large basis
        # for the current atom
        nfunc_angl = len(list(filter(
            lambda x: x[0] == atom_id and x[1] == angl, mol._bas)))
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
        angls_atom = [x[1] for x in list(filter(lambda x: x[0] == atom_id and x[1] < angl, mol._bas))]
        shell_offset += sum([funcs_on_shell(angll) for angll in angls_atom])
        
        # This guarantees no function will be chosen twice by removing already
        # selected functions from the pool of available ones
        ovlp_col[mask] = 0.0

        idx = np.argmax(np.abs(ovlp_col[shell_offset:shell_offset + nfunc_angl]))
        mask[shell_offset + idx] = 1

    mask = link_shells(mol, mask)
    if np.sum(mask) != mol_sto3g.nao_nr():
        raise RuntimeError(f"Number of functions in the projected minimal basis [{np.sum(mask)}] does not match the number of functions in the actual minimal basis [{mol_sto3g.nao_nr()}]!")

    return mask


def function_labels_from_mask(mask, mol):
    """ Return a dictionary with all 
    
    """
    # Get function labels
    labels = []
    all_labels = np.array(pyscf.gto.mole.cart_labels(mol)) if mol.cart \
            else np.array(pyscf.gto.mole.sph_labels(mol))
    for label in all_labels[mask]:
        # Split label strings of the form
        # 'Atom_idx Atom_symb sph/cart_label', i.e. '0 H 1s' or '1 O 2px'
        atom_num = label.split()[0]
        asymb    = label.split()[1]
        pattern  = re.compile(r'^([0-9]+[spdfgh])')
        match    = pattern.match(label.split()[-1])
        if match is None:
            raise ValueError('No regex patter match found!')
        labels.append(
            ' '.join([atom_num, asymb, match.group(1)]))
    labels = list(set(labels))
    labels = [label.split() for label in labels]
    labels = sorted(labels, key=lambda x: ' '.join(x))
    atom_dict = {}
    for label in labels:
        key = ' '.join(label[:2])
        if key not in atom_dict.keys():
            atom_dict[key] = [label[2]]
        else:
            atom_dict[key].append(label[2])
    for key in atom_dict:
        atom_dict[key].sort(key=orbital_key)

    return atom_dict


def exract_basis_data_from_molecule(
    mol:    pyscf.gto.MoleBase
    ) -> list:
    """Extracts the basis exponents and contraction coefficients.

    """

    basis = deepcopy(mol._basis) # Extract basis data in PySCF format
    atoms = [at[0] for at in mol._atom] # Exract atom symbols

    functions = []
    for atom in atoms:
        atom_basis = basis[atom]
        for angular_basis in atom_basis:
            angl = angular_basis[0]
            exps = np.asarray(angular_basis[1:]).T[0]
            for contr_coeffs in np.asarray(angular_basis[1:])[:,1:].T:
                if mol.cart:
                    for i in range((angl + 1) * (angl + 2) / 2):
                        functions.append([angl, exps, list(contr_coeffs)])
                else:
                    for i in range(2 * angl + 1):
                        functions.append([angl, list(exps), list(contr_coeffs)])
            
    return functions

def number_of_states(
        energies,
        nfunc_per_minimal_atom,
        nfuncs,
        thresh = 1e-3):
    nfuncs_include = nfunc_per_minimal_atom
    # Handle degeneracies
    while nfuncs_include < nfuncs \
        and energies[nfuncs_include] - energies[nfunc_per_minimal_atom - 1] < thresh:
        nfuncs_include += 1
    return nfuncs_include


def init_atomic_mask(nao, funcs_per_atom, offset_idx, nfunc_tot):
    mask = np.zeros(nao, dtype = bool)
    func_offset = np.sum(funcs_per_atom[:offset_idx])
    mask[func_offset:func_offset+nfunc_tot] = True
    return mask

def nfunc_in_atom_minimal_basis(asymb, nelec_ECP):
    Z = pyscf.data.elements.ELEMENTS.NUC[asymb]
    return int(np.ceil((Z - nelec_ECP) / 2))

def atomic_block_orbital_output(
    mol:                        pyscf.gto.MoleBase,
    F:                          np.ndarray,
    S:                          np.ndarray,
    output:                     str,
    spherically_average_fock:   bool            = False,
    which_mo_to_calculate:      int             = 0,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Create atomic block cube files.
    """
    func_per_atom = adb.basis_functions_per_atom(mol)
    assert np.sum(func_per_atom) == mol.nao

    # if by_shell:
    smask = adb.init_smask(mol, mol.cart)

    atoms = list(map(lambda x: x[0], mol._atom))

    restricted = (len(F.shape) == 2)
    # Loop through atomic blocks in the Fock matrix
    nfuncs_min_tot = 0
    for i,funcs_and_atom in enumerate(zip(func_per_atom, atoms)):
        # if i != 0:
        #     break   # Skip the rest of the atoms, this is temporary, have
        #             # fun trying to remember why
        
        nfuncs, atom = funcs_and_atom
        smask_atom = list(filter(lambda x: x[3][0] == i, smask))
        # Create the function mask for the atom
        mask = init_atomic_mask(mol.nao, func_per_atom, i, nfuncs)

        S_atom = adb.mask_matrix(S, mask)
        F_atom = adb.mask_matrix(F, mask)

        # Number of functions in minimal basis of current atom,
        # not counting ECP electrons, add to molecule minimal
        # number of functions
        nfuncs_min_tot += nfunc_in_atom_minimal_basis(
            atom,
            mol.atom_nelec_core(i))

        F_ave = F_atom.copy()
        if spherically_average_fock:
            F_ave = adb.spherical_average(F_ave, [shell[1] for shell in smask_atom])

        

        e_atom, c_atom, sorted_idx = eig(F_ave, S_atom.copy())
        # e_atom, c_atom = pyscf.scf.hf.eig(F_ave, S_atom.copy())
        coeff = np.zeros(F.shape)
        c_atom_idx = 0
        for j, flag in enumerate(mask):
            if flag:
                coeff[j, mask] = c_atom[c_atom_idx]
                c_atom_idx += 1
        occupations = np.zeros(mol.nao_nr())
        func_offset = np.sum(func_per_atom[:i])
        occupations[func_offset + which_mo_to_calculate] = num_of_occupying_electrons
        sorted_idx = 0
        print_orbital_label(mol, func_offset,
                            which_mo_to_calculate = which_mo_to_calculate,
                            reindexing = sorted_idx)
        dm = pyscf.scf.hf.make_rdm1(coeff, occupations)
        # pyscf.tools.cubegen.density(
        #     mol,
        #     'density_atomic_block.cub',
        #     dm
        # )

        calculate_isovalue_for_fraction_of_charge(
            coeff,
            mol,
            which_mo_to_calculate=func_offset + which_mo_to_calculate,
            fraction = .9)
        print('\n')
        # pyscf.tools.cubegen.orbital(
        #     mol,
        #     output,
        #     coeff[:,which_mo_to_calculate]
        # )

    return None


def print_orbital_label(
        mol,
        func_offset,
        which_mo_to_calculate = 0,
        reindexing = 0):
    """Print a given orbital label

    Args:
    mol : pyscf.gto.MoleBase
        The PySCF molecule object
    func_offset : int
        The index offset with respect to the whole molecule.
    which_mo_to_calculate : int
        Index of the MO which is considered.
    reindexing : int | array | np.array
        Either a scalar or an array of ints. Reorders the labels. If scalar,
        no reordering is done.
    """
    if np.isscalar(reindexing):
        reindexing = np.asarray(range(0, mol.nao - 1 - func_offset))
    if mol.cart:
        print(np.asarray(pyscf.gto.mole.cart_labels(mol))[func_offset + reindexing][which_mo_to_calculate])
    else:
        print(np.asarray(pyscf.gto.mole.sph_labels(mol))[func_offset + reindexing][which_mo_to_calculate])


def calculate_isovalue_for_fraction_of_charge(
        coeff,
        mol,
        which_mo_to_calculate : int = 0,
        fraction : float            = .9,
    ):
    """Calculate an isovalue for the electron density of a given orbital
    which contains 'fraction' of the charge.

    Args:
    coeff : np.ndarray
        The orbital coefficients of the orbitals considered. Must be
        padded so dimensions correspond to those of mol.
    mol : pyscf.gto.MoleBase
        The PySCF molecule object.
    which_mo_to_calculate : int
        Index of the MO which is considered.
    fraction : float

    
    Returns:

    """
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError("Given 'fraction' is not a within suitable range: "+
                         f"fraction={fraction}, should be within [0.0, 1.0]!")
    mf = mol.HF()
    nao_in_mol = mol.nao_nr()

    is_restricted = len(coeff.shape) == 2
    if is_restricted:
        occupations = np.zeros(nao_in_mol)
        occupations[which_mo_to_calculate] = num_of_occupying_electrons
        dm = pyscf.scf.hf.make_rdm1(coeff, occupations)
    else:
        occupations = np.zeros((2,nao_in_mol))
        occupations[0, which_mo_to_calculate] = num_of_occupying_electrons
        dm = pyscf.scf.uhf.make_rdm1(coeff, occupations)

    dft_grid = pyscf.dft.gen_grid.Grids(mol)
    dft_grid.level = 9
    dft_grid.build()
    weights = dft_grid.weights
    coords = dft_grid.coords

    aos = pyscf.dft.numint.eval_ao(mol, coords)
    print(f'{aos.shape=}')
    # rho = pyscf.dft.numint.eval_rho2(
    #     mol, aos, coeff, occupations
    # )
    rho = pyscf.dft.numint.eval_rho(
        mol, aos, dm
    )
    aos = aos[:, which_mo_to_calculate]
    

    # Sort from high -> low
    sort_idx = np.argsort(rho)[::-1]
    sorted_rho = rho[sort_idx]
    # Calculate charge from integrated density and weights
    charges = sorted_rho * weights[sort_idx]
    cumulative_charge = np.cumsum(charges)
    Qtot = cumulative_charge[-1]
    target_charge = fraction * Qtot
    # Binary search for the cutoff
    idx = np.searchsorted(cumulative_charge, target_charge)
    achieved_Q = cumulative_charge[idx]
    iso_target = sorted_rho[idx]
    
    achieved_fraction = achieved_Q / Qtot
    print(f'{iso_target=}\n{achieved_fraction=}\n{achieved_Q=}')
    print(f'The isovalue for the orbital: {np.sqrt(iso_target)}')
    print(f'{np.sum(rho*weights)=}')

    return iso_target, achieved_fraction, achieved_Q


if __name__ == "__main__":
    
    default_mol = '/home/joonahuh/uni/electronic_structure/benchmarks/pom_geom/xyz/h2o.charge0.spin0.xyz'
    parser = argparse.ArgumentParser(
        description="Atomic block decomposition utility module"
    )
    parser.add_argument(
        "--mpath", "-m", type=str, required=True, nargs='+',
        help="path to molecule .xyz file(s)", default=default_mol
    )
    parser.add_argument(
        "--basis", "-b", type=str, required=False, default='def2-TZVP', nargs=1,
        help="basis set name"
    )

    args = parser.parse_args()
    molfiles = args.mpath
    basis = args.basis

    # Make sure a directory to store output exists
    # datadir = 'data'
    # if not os.path.isdir(datadir):
    #     print(f"No directory named 'data' exists at {os.getcwd()}, creating one...")
    #     os.mkdir(datadir)

    mols = adbutils.get_molecules_in_dir(molfiles, basis)
    init_guess = 'atom'
    run_dft = False
    for molfilename, mol, uncmol, shells, init_guess, basisname in mols:
        print()
        print(molfilename)
        xcfunc = 'PBE'
        grid_level = 7

        mf = pyscf.dft.KS(mol) if run_dft else pyscf.scf.HF(mol)
        if run_dft:
            mf.grids.level = grid_level
            mf.xc = xcfunc
            mf.grids.prune = None
        
        # Initialize init guess method
        mf.init_guess = init_guess

        # This produces the initial guess density matrix
        dm0 = mf.get_init_guess(key=init_guess)
        # we need the corresponding Fock matrix
        F = mf.get_fock(dm=dm0)
        S = mf.get_ovlp()
        # This gives the initial guess density matrix for the mf object
        mf.mo_energy, mf.mo_coeff = mf.eig(F, S)
        mf.mo_occs = mf.get_occ(mf.mo_energy)
        mask = find_pseudominimal_basis_mask(
            mol,
            F, S,
            init_guess=init_guess,
            q_tol=0.9
        )
        print_labels_of_functions_in_mask(
            mask, mol, print_actual_minimal_basis = True)

        # Nearest neighbor pairing
        nn_mask = find_pseudominimal_basis_mask(
            mol,
            F, S,
            init_guess = init_guess,
            pair_by_nearest_neighbor = True,
            q_tol=0.9
        )
        pair_indices = adb.pair_by_nearest_neighbor(mol)
        print('Pair by nearest neighbor:')
        print(f'Paired atoms: {pair_indices}', end='')
        print_labels_of_functions_in_mask(nn_mask, mol)




    # for molfile in molfiles:
    #     mol = pyscf.M(atom = molfile, basis = basis)
    #     bdmol = BlockDecomposedMol(mol)
    #     bdmol.initialize_scf_object()
    #     print(bdmol)

    #     bdmol.list_all_ao_labels()

        # for atomic_block_to_compute in range(bdmol.nblocks):
        #     _, coeff, orbital_order = bdmol.calculate_orbitals_of_atomic_block(
        #         which_block = atomic_block_to_compute)
        #     atomic_scf_coefficients = bdmol.generate_atomic_coefficients(
        #         atomic_block_to_compute)
        #     if len(atomic_scf_coefficients.shape) > 2:
        #         atomic_scf_coefficients = atomic_scf_coefficients[0]
            
        #     # Use ground state atomic occupation as maximum number of
        #     # orbitals that are generated
        #     asymb = bdmol.get_asymb_of_atomic_block(atomic_block_to_compute)
        #     gs_atomic_occupation = pyscf.data.elements.NUC[asymb]
        #     occupied_orbitals_in_block = bdmol.get_orbitals_in_block(
        #         atomic_block_to_compute)[:gs_atomic_occupation]
            
        #     # Offset the orbital order indices by the number of functions
        #     # in preceeding atomic blocks
        #     orbital_order += bdmol.atomic_function_ranges[atomic_block_to_compute,0]
        #     # Reorder the orbital labels in the correct order and
        #     # update this order in the object
        #     ra, rb = bdmol.atomic_function_ranges[atomic_block_to_compute]
        #     bdmol.ao_labels[ra:rb+1] = bdmol.ao_labels[orbital_order]
        #     # reordered_ao_labels = bdmol.ao_labels[orbital_order]


        #     for orbital in occupied_orbitals_in_block:
        #         idx_in_full_system = bdmol.index_of_orbital_in_full_sys(
        #             atomic_block_to_compute, orbital)
        #         # orbital_label = reordered_ao_labels[idx_in_full_system]
        #         # orbital_label = reordered_ao_labels[orbital]
        #         orbital_label = bdmol.ao_labels[idx_in_full_system]
        #         orbital_label = '_'.join(orbital_label.split())
                
        #         output_path = datadir + '/' + orbital_label
        #         if not os.path.isdir(output_path):
        #             os.mkdir(output_path)
                

        #         isoval, achieved_fraction, achieved_Q = bdmol.calc_isoval(
        #             coeff,
        #             atomic_block_to_compute, orbital)
                
        #         bdmol.output_orbital_cub(
        #             coeff,
        #             atomic_block_to_compute, orbital,
        #             output_path + '/' + 'orbital_atomic_block.cub')
                
        #         # Atomic SCF isovalue data
        #         asid = bdmol.generate_orbital_cub_from_atomic_calc(
        #             atomic_scf_coefficients,
        #             atomic_block_to_compute, orbital,
        #             output_path + '/' + 'orbital_atomic_scf.cub')
                
        #         with open(output_path + '/' + 'isovalue_output.dat', 'w') as file:
        #             file.write(f'Density and orbital isovalue calculation data\n\n')

        #             file.write(f'!--- Atomic block calculation ---!\n')
        #             file.write(f'Density isovalue:  {isoval:.8f}\n')
        #             file.write(f'Orbital isovalue:  {np.sqrt(isoval):.8f}\n')
        #             file.write(f'Achieved charge:   {achieved_Q:.8f}\n')
        #             file.write(f'Achieved fraction: {achieved_fraction:.8f}\n')

        #             file.write('\n')
        #             file.write(f'!---- Atomic SCF calculation ----!\n')
        #             file.write(f'Density isovalue:  {asid[0]:.8f}\n')
        #             file.write(f'Orbital isovalue:  {np.sqrt(asid[0]):.8f}\n')
        #             file.write(f'Achieved charge:   {asid[2]:.8f}\n')
        #             file.write(f'Achieved fraction: {asid[1]:.8f}\n')
        #         # print(isovalue_data)
            
