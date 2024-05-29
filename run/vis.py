#!/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
import matplotlib
font = {'weight' : 'bold',
        'size'   : 14}
matplotlib.rc('font', **font)

class MolData:
    name = ''
    basis_set = ''
    thf = 0.0
    tfbf = 0.0
    tsbs = 0.0
    nocc = 0
    ehf = 0.0
    nfunc = 0
    fbfdat = pd.DataFrame()
    sbsdat = pd.DataFrame()

    def __str__(self):
        out = f'\n{self.name} with basis {self.basis_set}\n\n'
        out += f'{self.thf} {self.tfbf} {self.tsbs}\n\n'
        out += f'Nocc: {self.nocc},   E_HF: {self.ehf},   nfunc: {self.nfunc}\n\n'
        out += f'{self.fbfdat.to_string()}\n\n'
        out += f'{self.sbsdat.loc[:,"nfunc":"Qsqrd"].to_string()}\n'
        return out

def read_data(path: str):
    '''Read data file from path
    '''
    dat = MolData()

    fbf_begin = 0
    sbs_begin = 0
    fbf_end = 0
    sbs_end = 0

    f = open(path)
    for i,line in enumerate(f):
        if i == 1:
            dat.name, dat.basis_set = line.strip().split()
        if i == 6:
            dat.thf, dat.tfbf, dat.tsbs = list(map(lambda x: float(x), line.strip().split()))
        if i == 9:
            nocc, ehf, nfunc = line.strip().split()
            dat.nocc = int(nocc)
            dat.ehf = float(ehf)
            dat.nfunc = int(nfunc)

        if 'function-by-function' in line:
            fbf_begin = i+1
            continue
        if fbf_begin != 0 and line in ['\n', '\r\n'] and sbs_begin == 0:
            fbf_end = i-1
            continue
        if 'shell-by-shell' in line:
            sbs_begin = i+1
            continue
        if sbs_begin != 0 and sbs_end == 0 and line in ['\n', '\r\n']:
            sbs_end = i
            continue

    dat.fbfdat = pd.read_csv(path, skiprows=lambda x: x not in range(fbf_begin, fbf_end))
    dat.sbsdat = pd.read_csv(path, skiprows=lambda x: x not in range(sbs_begin, sbs_end))

    return dat

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 vis.py <datpath>')
        print('\tdatpath:  \tthe path to data file')
        sys.exit()
    datpath = sys.argv[1]

    dat = read_data(datpath)

    # Convergence panels
    fig, ax = plt.subplots(1, 1, figsize=(10,8), tight_layout=True)
    ax.semilogy(
        dat.fbfdat['nfunc'][1:], -dat.fbfdat['diff'][1:],
        '.-', label=r'fct by fct: $\Delta E_{orb}=\sum E^{occ orb}_{n+1}-\sum E^{occ orb}_{n}$'
        )
    ax.semilogy(
        dat.sbsdat['nfunc'][1:], -dat.sbsdat['diff'][1:],
        '.-', label=r'shl by shl: $\Delta E_{orb}=\sum E^{occ orb}_{n+1}-\sum E^{occ orb}_{n}$'
        )
    ax.semilogy(
        dat.sbsdat['nfunc'], dat.sbsdat['E_scf'] - dat.ehf,
        '.-', label=r'$\Delta E^{scf}=E^{scf}_{subbasis}-E^{scf}_{full basis}$'
    )
    ax.grid(alpha=.5)
    ax.legend()
    ax.set_title(
        f'${{{dat.name}}}$\nBasis: {dat.basis_set}, nfunc: {dat.nfunc}',
        fontsize=24, fontweight='bold'
        )
    ax.set_ylabel('$\Delta E_{(n-1,n)}$ [hartree]', fontsize=16)
    ax.set_xlabel('Subbasis size N', fontsize=16, fontweight='bold')

    # Projection panels
    fig2, ax2 = plt.subplots(1, 1, figsize=(10,8), tight_layout=True)    
    
    ax2.semilogy(
        dat.sbsdat['nfunc'], 1 - dat.sbsdat['Qsqrd']/dat.nocc,
        '.-', label=r'$\Delta Q_\sigma$'
    )
    ax2.grid(alpha=.5)
    ax2.legend()
    ax2.set_title(
        f'${{{dat.name}}}$\nBasis: {dat.basis_set}, nfunc: {dat.nfunc}',
        fontsize=24, fontweight='bold'
        )
    ax2.set_ylabel(
            r'$\Delta Q_\sigma = 1 - \frac{1}{N_{occ}}\sum_{i,j}^{N_{occ}}|<i^{subbasis}|j^{full basis}>|^2$',
            fontsize=16)
    ax2.set_xlabel('Subbasis size N', fontsize=16, fontweight='bold')

    plt.show()