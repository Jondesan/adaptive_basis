#!/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
import matplotlib
font = {'weight' : 'bold',
        'size'   : 14}
matplotlib.rc('font', **font)

ylabels = [
    r'$\Delta E_{(n-1,n)}$ [hartree]',
    r'$\Delta \frac{1}{2}\sum_i^{occ}(\epsilon_i + h_{ii})$',
    r'$\Delta Q$'
]
legend_labels = [
    r'shl by shl: $\Delta E_{orb}=\sum E^{occ orb}_{n+1}-\sum E^{occ orb}_{n}$',
    r'shl by shl: $\Delta (\frac{1}{2}\sum_i^{occ}(\epsilon_i + h_{ii}))_{n-1, n}$',
    r'shl by shl: $\Delta Q_{n-1, n}=Q_{n}-Q_{n-1}$'
]

class MolData:
    name = ''
    basis_set = ''
    variant = None
    thf = 0.0
    tfbf = 0.0
    tsbs = 0.0
    nocc = 0
    ehf = 0.0
    nfunc = 0
    fbfdat = pd.DataFrame()
    sbsdat = pd.DataFrame()

    def __str__(self):
        out = f'\n{self.name} with basis {self.basis_set}, variant {self.variant}\n\n'
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
            dat.name, dat.basis_set, dat.variant = line.strip().split()
            dat.variant = int(dat.variant)
        if i == 6:
            dat.thf, dat.tfbf, dat.tsbs = list(map(lambda x: (float(x) if x != "-" else x), line.strip().split()))
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
        
    if fbf_begin != 0:
        dat.fbfdat = pd.read_csv(path, skiprows=lambda x: x not in range(fbf_begin, fbf_end))
    if sbs_begin != 0:
       dat.sbsdat = pd.read_csv(path, skiprows=lambda x: x not in range(sbs_begin, sbs_end))

    return dat

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 vis.py <datpath>')
        print('\tdatpath:  \tthe path to data file')
        sys.exit()
    datpath = sys.argv[1]
    datadir = '/'.join(datpath.split('/')[:-1])
    handle_decontraction = False

    dat = read_data(datpath)
    uncdatapath = datadir + '/' + '.'.join([dat.name,'-'.join(['unc', dat.basis_set]),'out'])
    handle_decontraction = os.path.isfile(uncdatapath)

    dat = [dat]
    if handle_decontraction:
        dat.append(read_data(uncdatapath))

    # Convergence panels
    figs = [
        plt.figure(i, figsize=(10,8), tight_layout=True) 
        for i in range(3 if handle_decontraction else 2)
        ]
    axs = [figs[i].add_subplot() for i in range(len(figs))]
    
    for i in range(len(dat)):
        if dat[i].variant == 0:
            axs[i].semilogy(
                dat[i].fbfdat['nfunc'][1:], -dat[i].fbfdat['diff'][1:],
                '.-', label=r'fct by fct: $\Delta E_{orb}=\sum E^{occ orb}_{n+1}-\sum E^{occ orb}_{n}$'
                )
        x, y = dat[i].sbsdat['nfunc'][1:], -dat[i].sbsdat['diff'][1:]
        if dat[i].variant == 2:
            y *= -1
        axs[i].semilogy(
            x, y,
            '.-', label=legend_labels[dat[i].variant]
            )
        axs[i].semilogy(
            dat[i].sbsdat['nfunc'], dat[i].sbsdat['E_scf'] - dat[i].ehf,
            '.-', label=r'$\Delta E^{scf}=E^{scf}_{subbasis}-E^{scf}_{full basis}$'
        )
        axs[i].grid(alpha=.5)
        axs[i].legend()
        axs[i].set_title(
            f'${{{dat[i].name}}}$\nBasis: {dat[i].basis_set}, nfunc: {dat[i].nfunc}',
            fontsize=24, fontweight='bold'
            )
        axs[i].set_ylabel(ylabels[dat[i].variant], fontsize=16)
        axs[i].set_xlabel('Subbasis size N', fontsize=16, fontweight='bold')

        # Projection panels
        axs[-1].semilogy(
            dat[i].sbsdat['nfunc'], 1 - dat[i].sbsdat['Qsqrd']/dat[i].nocc,
            '.-', label=r'$\Delta Q_\sigma$, ' + f'{dat[i].basis_set}, nfunc: {dat[i].nfunc}'
        )
    
    axs[-1].grid(alpha=.5)
    axs[-1].legend()
    axs[-1].set_title(
        f'${{{dat[0].name}}}$\nBasis: {dat[0].basis_set}, nfunc: {dat[0].nfunc}',
        fontsize=24, fontweight='bold'
        )
    axs[-1].set_ylabel(
            r'$\Delta Q_\sigma = 1 - \frac{1}{N_{occ}}\sum_{i,j}^{N_{occ}}|<i^{subbasis}|j^{full basis}>|^2$',
            fontsize=16)
    axs[-1].set_xlabel('Subbasis size N', fontsize=16, fontweight='bold')

    plt.show()