import pyscf
import numpy as np
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create visualisations of ABS runs."
    )
    parser.add_argument(
        "-m", "--mols", type=str, required=True, nargs='+',
        help="path(s) to molecule files"
    )

    args = parser.parse_args()
    mols = args.mols

    basis_sets = ['aug-pc-1', 'aug-pc-2', 'aug-pc-3', 'aug-pc-4']

    for molpath in mols:
        for bs in basis_sets:
            for unc in ['', 'unc-']:
                mol = pyscf.M(atom=molpath, basis=unc + bs, verbose=0)

                mf = pyscf.scf.HF(mol)
                mf.init_guess = 'atom'
                mf.kernel()

                print(molpath.split('/')[-1], unc+bs, mf.e_tot)