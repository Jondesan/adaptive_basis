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
    with open('hf_energies.out', 'a') as outfile:
        for molpath in mols:
            for bs in basis_sets:
                for unc in ['', 'unc-']:
                    fnparts = molpath.split('/')[-1].split('.')
                    if len(fnparts) > 2:
                        charge = [int(substring.replace('charge', '')) for substring in fnparts if 'charge' in substring]
                        charge = charge[0] if len(charge) != 0 else 0
                        spin = [int(substring.replace('spin','')) for substring in fnparts if 'spin' in substring]
                        spin = spin[0] if len(spin) != 0 else None
                    else:
                        charge = 0
                        spin = None
                    mol = pyscf.M(
                        atom=molpath, basis=unc + bs,
                        charge=charge, spin=spin,
                        verbose=0)

                    mf = pyscf.scf.HF(mol)
                    mf.init_guess = 'atom'
                    mf.kernel()

                    outfile.write(f'{molpath.split("/")[-1]}\t{unc+bs}\t{mf.e_tot}\n')