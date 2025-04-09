import adb
import pyscf
import numpy as np

if __name__ == '__main__':
    mol = pyscf.M(
        atom='be2f4.xyz',
        basis='def2-tzvp',
        # basis='321G',
        verbose=0)

    # mf = pyscf.scf.HF(mol)
    # F = mf.get_fock()
    # S = mf.get_ovlp()
    adb.atomic_block_minimal_basis(mol)
