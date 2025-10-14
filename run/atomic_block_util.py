import numpy as np
import pyscf
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import compress
import adb
import os

num_of_occupying_electrons = 1

def gto(r, zeta, angl):
    """Calculate the values of Gaussian type orbital at r for given parameters.

    Args:
        r : float or arraylike
            Value(s) of r at which the function is evaluated
        zeta : float
            Exponent
        angl : int | float
            Angular momentum
    """
    zeta = np.asarray(zeta)
    return r**angl * np.exp(-zeta * r**2)

def cgto(r, params):
    """Calculate the values of a contracted Gaussian type orbital at r
    for given parameters.

    Args:
        params : arraylike
            The parameters of the form
            [angl, [exp1, exp2, ...], [contr1, contr2, ...]]
    """
    if isinstance(r, float):
        return np.sum(np.asarray(params[2]) * gto(r, params[1], params[0]))
    else:
        vals = np.zeros(len(r))
        for i in range(vals.shape[0]):
            vals[i] = np.sum(np.asarray(params[2]) * gto(r[i], params[1], params[0]))
    return vals

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
                    for i in range((angl+1)*(angl+2)/2):
                        functions.append([angl, exps, list(contr_coeffs)])
                else:
                    for i in range(2*angl+1):
                        functions.append([angl, list(exps), list(contr_coeffs)])
            
    return functions

def number_of_states(energies, nfunc_per_minimal_atom, nfuncs, thresh=1e-3):
    nfuncs_include = nfunc_per_minimal_atom
    # Handle degeneracies
    while nfuncs_include < nfuncs \
        and energies[nfuncs_include]-energies[nfunc_per_minimal_atom-1] < thresh:
        nfuncs_include += 1
    return nfuncs_include


def init_atomic_mask(nao, funcs_per_atom, offset_idx, nfunc_tot):
    mask = np.zeros(nao, dtype=bool)
    func_offset = np.sum(funcs_per_atom[:offset_idx])
    mask[func_offset:func_offset+nfunc_tot] = True
    return mask

def nfunc_in_atom_minimal_basis(asymb, nelec_ECP):
    Z = pyscf.data.elements.ELEMENTS.index(asymb)
    return int(np.ceil( (Z-nelec_ECP) / 2 ))

def atomic_block_orbital_output(
    mol:                        pyscf.gto.MoleBase,
    F:                          np.ndarray,
    S:                          np.ndarray,
    output:                     str,
    get_mask_history:           bool            = True,
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

        e_atom, c_atom = pyscf.scf.hf.eig(F_ave, S_atom.copy())
        coeff = np.zeros(F.shape)
        c_atom_idx = 0
        for j, flag in enumerate(mask):
            if flag:
                coeff[j, mask] = c_atom[c_atom_idx]
                c_atom_idx += 1
        
        occupations = np.zeros(mol.nao_nr())
        func_offset = np.sum(func_per_atom[:i])
        occupations[func_offset + which_mo_to_calculate] = num_of_occupying_electrons
        dm = pyscf.scf.hf.make_rdm1(coeff, occupations)
        # pyscf.tools.cubegen.density(
        #     mol,
        #     'density_atomic_block.cub',
        #     dm
        # )

        calculate_isovalue_for_fraction_of_charge(
            coeff,
            mol,
            mol.nao_nr(),
            which_mo_to_calculate=func_offset + which_mo_to_calculate,
            fraction=.9)
        print('\n')
        # pyscf.tools.cubegen.orbital(
        #     mol,
        #     output,
        #     coeff[:,which_mo_to_calculate]
        # )

    return None


def calculate_isovalue_for_fraction_of_charge(
        coeff,
        mol,
        nao_in_mol,
        which_mo_to_calculate : int = 0,
        fraction : float            = .9,
        MAX_ITER : int              = 50
):
    mf = mol.HF()

    occupations = np.zeros(nao_in_mol)
    occupations[which_mo_to_calculate] = num_of_occupying_electrons

    dft_grid = pyscf.dft.gen_grid.Grids(mol)
    dft_grid.level = 9
    dft_grid.build()
    weights = dft_grid.weights
    coords = dft_grid.coords

    # return None
    aos = pyscf.dft.numint.eval_ao(mol, coords)
    # aos_3d_grid = aos[:,which_mo_to_calculate].reshape((Ngrid, Ngrid, Ngrid))
    rho = pyscf.dft.numint.eval_rho2(
        mol, aos, coeff, occupations
    )
    # rho_3d_grid = rho.reshape((Ngrid, Ngrid, Ngrid))
    aos = aos[:, which_mo_to_calculate]

    def Q_of_iso(iso):
        mask = rho >= iso
        return np.sum(rho[mask] * weights if np.isscalar(weights) \
               else rho[mask] * weights[mask])

    Qtot = np.sum(rho * weights)
    Qtarget = fraction * Qtot
    iso_lo = rho.min()
    iso_hi = rho.max()
    # Bisection
    q_lo = Q_of_iso(iso_lo)  # should be Qtot

    # ensure monotonicity holds
    if q_lo < Qtarget:
        raise RuntimeError("Q(t_lo) < Qtarget: unexpected")

    tol_charge = 1e-6
    tol_iso = 1e-9
    for _ in range(MAX_ITER):
        iso_mid = 0.5 * (iso_lo + iso_hi)
        q_mid = Q_of_iso(iso_mid)

        if  abs(q_mid - Qtarget) <= tol_charge or \
               (iso_hi - iso_lo) <= tol_iso:
            break

        # If q_mid > Qtarget, need larger t to reduce enclosed charge
        if q_mid > Qtarget:
            iso_lo = iso_mid
        else:
            iso_hi = iso_mid
    

    achieved_fraction = q_mid / Qtot
    print(f'{iso_mid=}\n{achieved_fraction=}\n{q_mid=}')
    print(f'The isovalue for the orbital: {np.sqrt(iso_mid)}')
    print(f'{np.sum(aos*weights)=}')
    print(f'{np.sum(rho*weights)=}')

    return iso_mid, achieved_fraction, q_mid


if __name__ == "__main__":
    mol = pyscf.M(atom='/home/joonahuh/uni/electronic_structure/benchmarks/pom_geom/xyz/h2o.charge0.spin0.xyz', basis='aug-pc-1')
    mf = mol.HF()

    mf.sap_basis = 'sapgraspsmall'
    dm0 = mf.get_init_guess(key='atom')
    F = mf.get_fock(dm=dm0)

    which_mo_to_calculate = 3

    atomic_block_orbital_output(
        mol,
        F,
        mf.get_ovlp(),
        'orbital_atomic_block.cub',
        which_mo_to_calculate=which_mo_to_calculate)
    
    # Generate orbital from atomic calculation
    mol2 = pyscf.M(atom=f'{mol._atom[0][0]} {' '.join([str(x) for x in mol._atom[0][1]])}', basis='aug-pc-1')
    mf = mol2.HF()
    mf.kernel()
    coeff = mf.mo_coeff
    # Pad the array with zeros to fit the length of the molecule array
    coeff = np.pad(coeff, (0, mol.nao_nr() - len(coeff)), constant_values=(0.0,0.0))
    print('\nIsovalue calculation for the atomic SCF:\n')
    calculate_isovalue_for_fraction_of_charge(
        mf.mo_coeff,
        mol2,
        mol2.nao_nr(),
        which_mo_to_calculate=which_mo_to_calculate,
        fraction=.9,
    )

    occupations = np.zeros(F.shape[0])
    occupations[which_mo_to_calculate] = 2
    dm = pyscf.scf.hf.make_rdm1(coeff, occupations)
    # pyscf.tools.cubegen.density(
    #     mol,
    #     'density_scf.cub',
    #     dm
    # )

    # pyscf.tools.cubegen.orbital(
    #         mol,
    #         'orbital_scf.cub',
    #         coeff[:,which_mo_to_calculate]
    #     )


    # functions = exract_basis_data_from_molecule(mol)
    # for i, func in enumerate(functions):
    #     print(i, func)

    # fig, ax = plt.subplots(1, 1, figsize=(10,4), tight_layout=True)

    # x = np.linspace(-10, 10, 1000)

    # coeffs = mf.mo_coeff[24] # Coefficients to form the
    #                         # linear combination for the MO
    # idx = coeffs > 1e-10

    # y = np.sum(
    #     [contr*cgto(x, func) \
    #         for func, contr in \
    #         zip(list(compress(functions, idx)), list(compress(coeffs, idx)))],
    #         axis=0)
    
    # ax.plot(x, y)
    # plt.show()