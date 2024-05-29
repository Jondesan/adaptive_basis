"""Adaptive basis method


"""
import sys
sys.path.append('/home/joonahuh/uni/electronic_structure/pyscf-master/')
import pyscf
import numpy as np
from scipy.linalg import fractional_matrix_power, eig
from time import time
import copy
import os


def eigh(h, s, get_idx=False):
    '''Modified canonical orthogonalisation.

    Args:
        h : ndarray
            Fock matrix
        s : ndarray
            Overlap matrix
        get_idx : bool
            Whether to return the indices that sort the eigenvalues. Default is False
    
    Returns:
        Sorted eigenvalues (ascending) and coefficients, if get_idx is True the indices that sort the eigenvalues are also returned
    '''
    x = pyscf.scf.addons.canonical_orth_(s, 1e-8)
    xhx = x.conj().T @ h @ x
    e, c = np.linalg.eigh(xhx)
    c = x @ c
    idx = np.argsort(e)
    # e = np.sort(e)

    if get_idx:
        return e[idx], c[idx], idx
    return e[idx], c[idx]

def get_uncontr_basis(mol, fn=None):
    '''Unravel the contracted basis of mol.

    Args:
        mol : pyscf.MoleBase object
            molecule object.
        fn : None or str
            the file name to which write the basis. If None, basis will not be written into a file, only returned as a str.

    Returns:
        The basis as a pySCF formatted string, which can be used with pyscf.gto.basis.parse.
    '''
    line = 'BASIS "ao basis" PRINT\n'
    basis = ''
    
    if fn is not None:
        f = open('tempbasis/' + fn + '.dat', 'w')
        f.write(line)

    asymb = list(set([mol.atom_pure_symbol(i) for i in range(len(mol._atom))]))
    for asy in asymb:
        line = '#BASIS SET:\n'
        basis += line
        if fn is not None: f.write(line)

        for shell in mol._basis[asy]:
            coeffs = np.array(shell[1:])
            contractions = coeffs.shape[1]
            for i in range(1, contractions):
                line = asy + '\t' + pyscf.lib.param.ANGULAR[shell[0]].capitalize() + '\n'
                basis += line
                if fn is not None: f.write(line)
                for b in coeffs:
                    line = f'{b[0]:15.7f}\t{b[i]:15.7f}\n'
                    basis += line
                    if fn is not None: f.write(line)
    line = 'END\n'
    if fn is not None:
        f.write(line)
        f.close()
    return basis

def get_basis_dict(basis: str):
    '''Convert a basis string into a dictionary to pass to pyscf.gto.basis.parse
    '''

    dc = dict()
    for elem in basis.split('#')[1:]:
        dc[elem[11]] = pyscf.gto.basis.parse(str(elem[11:]))
    return dc

def get_shells(mol):
    '''Get the shell structure of mol object.

    Args:
        mol : pyscf.gto.MoleBase
            The molecule object.

    Returns:
        A 1D ndarray with the number of functions per shell as elements. Shells are ordered in the pyscf internal format.
    '''
    shells = np.array([], dtype=int)  # Number of functions per shell

    for ib in range(mol.nbas):  # nbas = number of shells (basis fcts)
        ia = mol.bas_atom(ib)   # atom that given basis function sits on
        l = mol.bas_angular(ib) # angular momentum l of given basis function
        nc = mol.bas_nctr(ib)   # number of CGTOs for given shell
        symb = mol.atom_symbol(ia)  # label of given atom
        
        shells = np.append( shells, nc * ( l+1 ) * ( l+2 ) // 2 if mol.cart else nc * ( 2*l+1 ) )

    if sum(shells) != mol.nao_nr():
        raise Exception('Number of functions in the mask does not correspond with number of functions of the molecule!')

    return shells

def maskidx_to_smaskidx(mask, smask, cart=False):
    '''Create mapping between mask and smask
    '''
    mapping = [0]*len(mask)#mol.nao_nr()
    counter = 0
    for i,sm in enumerate(smask):
        l = sm[2]
        for j in range( (l + 1)*(l + 2) // 2 if cart else 2*l+1 ):
            # print(counter, j, i, l, sm)
            mapping[counter] = i
            counter += 1
    return mapping

def init_smask(mol, cart=False):
    '''Initialize the shell mask array. smask will be a list of lists, with length equal to the number of uncontracted shells, and each element is a two element list, first is bool that specifies the mask for the current shell, the other how many primitives in this shell.
    '''
    # ibs = mol._bas # internal basis struct
    smask = []
    
    count = np.zeros((mol.natm, 9), dtype=int)
    for ib in range(mol.nbas):
        ia = mol.bas_atom(ib)   # atom that given basis function sits on
        l = mol.bas_angular(ib) # angular momentum l of given basis function
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
            smask.append([False, (l + 1)*(l + 2) // 2 if cart else 2*l+1, l])

    return smask

def smask_to_mask(smask, cart=False):
    '''Convert current shell mask into function mask.
    '''
    funcs_per_shell = [( ( s[2] + 1 ) * ( s[2] + 2 ) // 2 if cart else 2*s[2] + 1 ) for s in smask]
    mask = [False] * sum(funcs_per_shell)
    for i,sm in enumerate(smask):
        if sm[0]:
            rb = sum(funcs_per_shell[:i])
            re = rb + sm[1]
            mask[rb:re] = [True]*(re-rb)
    return mask

def mask_to_smask(mask, smask, cart=False):
    '''Flip shells of smask to True that have 1 or more functions set to True in mask.
    '''
    mapping = maskidx_to_smaskidx(mask, smask, cart)
    for i in np.argwhere(mask):
        smask[mapping[i[0]]][0] = True
    
    return smask

def expand_mask(F, S, nocc, mask, smask=None, variant=0, fock1e=None):
    '''Expands the current mask by either one function or one shell based on smask.

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
            If None functions are tested individually. Else shell by shell testing is used where shells are determined by the smask array, where the elements represent the number of functions per current shell. The shells are ordered in the PySCF internal format
        variant : int
            Which energy model to use. 2 corresponds to the second variant, where the sum .5(e_i + h_ii) is used instead where h_ii are the expectation values of the one-electron operator. Has no effect if fock1e is not provided. Otherwise the sum is taken only over the occupied orbital energies.
        fock1e : None or ndarray
            1-electron operator matrix. If variant is set to 2, fock1e is expected to be given.
    
    Returns:
        The new mask (boolean ndarray), the current difference in eigenvalue sums and the current sum (energy sum of occupied orbitals), shell mask if smask is provided.
    '''
    eigenvalues, _ = eigh(F[mask,:][:,mask], S[mask,:][:,mask])
    last_sum = np.sum(eigenvalues[:nocc])
    
    current_difference = 0
    current_sum = 0

    test_sums = []
    if (smask is None):
        for i,m in enumerate(mask):
            if m == True:
                continue
            
            test_mask = copy.deepcopy(mask)
            test_mask[i] = True

            evals, _ = eigh(F[test_mask,:][:,test_mask], S[test_mask,:][:,test_mask])
            test_sums.append((i, np.sum(evals[:nocc])))
                
    else:
        for i, sm in enumerate(smask):
            if sm[0]:
                continue
            test_smask = copy.deepcopy(smask)
            test_smask[i][0] = True
            test_mask = smask_to_mask(test_smask)

            evals, _ = eigh(F[test_mask,:][:,test_mask], S[test_mask,:][:,test_mask])
            test_sums.append((i, np.sum(evals[:nocc])))

    test_differences = [test_sum[1] - last_sum for test_sum in test_sums]
    array_index = np.argmin(test_differences)
    current_idx_to_flip = test_sums[array_index][0]
    if smask is None:
        mask[current_idx_to_flip] = True
    else:
        smask[current_idx_to_flip][0] = True
        mask = smask_to_mask(smask)
    return mask, test_differences[array_index], test_sums[array_index][1], smask

def get_sub_scf_attributes(mol, fock, overlap):
    '''Calculates converged attributes for the system using the subbasis determined by mask.

    Args:
        mol : pyscf.gto.MoleBase
            The molecule object

    Returns:
        The SCF energy, sum of occupied orbital energies of the subbasis, the MO coefficient matrix of the subbasis.
    '''
    mf = mol.HF().apply(pyscf.scf.addons.remove_linear_dep_)

    # Diagonalize fock matrix and form guess density matrix
    if fock.shape[0] > 1:
        e, c = eigh(fock, overlap)
        occ = mf.get_occ(e, c)
        dm = pyscf.scf.hf.make_rdm1(c, occ)
        mf.init_guess = dm#'atom'
    mf.kernel()

    scf_energy = mf.e_tot
    nocc_sb = len(mf.mo_occ>0)
    scf_orbital_energy = sum(np.sort(mf.mo_energy)[:nocc_sb])
    return scf_energy, scf_orbital_energy, mf.mo_coeff

def create_uncontracted_molecule_copy(mol, verbose=0):
    unctr_basis = get_uncontr_basis(mol)
    cmol = pyscf.gto.M(
                atom = mol.atom,
                basis = unctr_basis,
                verbose = 0
            )
    return cmol

def print_data(mask, cursum, diff, E_scf="-", E_scfocc="-", Qsqrd="-", print_header=False):
    if print_header:
        print(f'{"N_func":>10s}\t{"Fock sum":>15s}\t{"Difference":>15s}\t{"E_subbasSCF":>15s}\t{"E_subb-occ":>15s}\t{"Q^2":>15s}')
    
    if E_scf is None: E_scf = '-'
    if E_scfocc is None: E_scfocc = '-'
    if Qsqrd is None: Qsqrd = '-'

    print(f'{sum(mask):10}\t{cursum:15.9f}\t{diff:{">15s" if type(diff) is str else "15.9f"}}\t', end='')
    print(f'{E_scf:{">15s" if type(E_scf) is str else "15.9f"}}\t', end='')
    print(f'{E_scfocc:{">15s" if type(E_scfocc) is str else "15.9f"}}\t', end='')
    print(f'{Qsqrd:{">15s" if type(Qsqrd) is str else "15.9f"}}')

def find_subspace(F, S, mol, scf, conv_tol=1e-2, verbose=True, collect_data=False, get_smask=False):
    '''Looks for a Fock matrix subspace that approximately solves the Roothaan equation FC=SCE below a convergence of conv_tol.

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
            Convergence criteria used to determine when to stop the subspace iteration.
        verbose : bool
            Determines whether some output will be printed during calculation.
        collect_data : bool
            If true, np.ndarray will be created and number of functions, current_sum, difference, total SCF energy, SCF energy of occupied orbitals and the projection onto the converged full basis wave function will be appended on every iteration.
        get_smask : bool
            Whether to return the shell mask and run iteration shell by shell instead of function by function. May provide faster convergence but can also provide more functions overall.

    Returns:
        1D boolean ndarray. A mask with selected function indices set to True. If collect_data is True, an ndarray is also returned with data as described in Args section. Shell mask is returned instead of function mask if get_smask is True.
    '''
    unctr_mol = create_uncontracted_molecule_copy(mol)
    # print('unctr_mol has', len(unctr_mol._bas), 'shells')
    # print('decontracted has', len(mol.decontract_basis()[0]._bas), 'shells')
    # mask or smask initialization
    Fii = np.diag(F)
    mask = [False] * unctr_mol.nao_nr()
    min_idx = np.argmin(Fii)
    smask = None
    mask[min_idx] = True

    if get_smask:
        smask = init_smask(unctr_mol, unctr_mol.cart)
        smask = mask_to_smask(mask, smask, unctr_mol.cart)
        mask = smask_to_mask(smask, unctr_mol.cart)
    
    previous_sum = np.sum(Fii[mask])
    occ = scf.get_occ()
    nocc = np.count_nonzero(occ)
    scf_energy = None
    scf_orbital_energy = None
    subbasis_coeffs = scf.mo_coeff


    if collect_data:
        if get_smask:
            cmol = create_uncontracted_molecule_copy(unctr_mol)
            newmask = [sm[0] for sm in smask]
            newbas = unctr_mol._bas[newmask]
            cmol._bas = newbas
            scf_energy, scf_orbital_energy, subbasis_coeffs = get_sub_scf_attributes(cmol, F[mask,:][:,mask], S[mask,:][:,mask])
            ovlp = pyscf.gto.intor_cross('int1e_ovlp_sph', unctr_mol, cmol)
            fullbasis_coeffs = scf.mo_coeff
            Q = fullbasis_coeffs[:,:nocc].T @ ovlp @ subbasis_coeffs[:,:nocc]
            Q_sqrd = np.sum(np.sum(Q**2))
        else:
            Q_sqrd = None
        dataframe = [[sum(mask), previous_sum, 0.0, scf_energy, scf_orbital_energy, Q_sqrd, smask]]
    
    if verbose:
        print_data(mask, previous_sum, "-", scf_energy, scf_orbital_energy, Q_sqrd, print_header=True)

    while(True):
        start = time()
        mask, difference, current_sum, smask = expand_mask(
            F, S, nocc, mask,
            smask=smask
            )
        end = time()
        
        cmol = create_uncontracted_molecule_copy(unctr_mol)
        if get_smask:
            cmol._bas = unctr_mol._bas[[sm[0] for sm in smask]]
            start2 = time()
            scf_energy, scf_orbital_energy, subbasis_coeffs = get_sub_scf_attributes(cmol, F[mask,:][:,mask], S[mask,:][:,mask])
            end2 = time()

        if verbose:
            print_data(mask, current_sum, difference, scf_energy, scf_orbital_energy, Q_sqrd)
        if collect_data:
            if get_smask:
                ovlp = pyscf.gto.intor_cross('int1e_ovlp_sph', unctr_mol, cmol)
                Q = fullbasis_coeffs[:,:nocc].T @ ovlp @ subbasis_coeffs[:,:nocc]
                Q_sqrd = np.sum(np.sum(Q**2))
            dataframe.append(
                [sum(mask), current_sum, difference, scf_energy,
                scf_orbital_energy, Q_sqrd, smask]
            )

        if ( abs(difference) < conv_tol or sum(mask) == len(mask)):
            break
        previous_sum = current_sum
    
    if get_smask:
        mask = smask
        
    if collect_data:
        return mask, dataframe
    return mask
