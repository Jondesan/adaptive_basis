#!/bin/python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import matplotlib
import argparse
import operator
from cmcrameri import cm

font = {"weight": "bold", "size": 14}
matplotlib.rc("font", **font)

ls = [
    'solid',
    'dashed',
    'dashdot'
]

lw = 2.0

ylabels = {
    'enocc': r"$E_{\rm{subscf}} - E_{\rm{fullscf}}$ [hartree]",
    'ecore': r"$\Delta \frac{1}{2}\sum_i^{occ}(\epsilon_i + h_{ii})$",
    'elden': r"$\Delta Q$",
}
legend_labels = {
    'enocc': r"shl by shl: $\Delta E_{orb}$",#=\sum E^{occ orb}_{n+1}-\sum E^{occ orb}_{n}$",
    'ecore': r"shl by shl: $\Delta (\frac{1}{2}\sum_i^{occ}(\epsilon_i + h_{ii}))_{n-1, n}$",
    'elden': r"shl by shl: $\Delta Q_{n-1, n}$",#=Q_{n}-Q_{n-1}$",
}


class MolData:
    name = ""
    basis_set = ""
    variant = None
    init_guess = ""
    sapbasisname = None
    thf = 0.0
    tfbf = 0.0
    tsbs = 0.0
    nocc = 0
    ehf = 0.0
    nfunc = 0
    fbfdat = pd.DataFrame()
    sbsdat = pd.DataFrame()

    def __str__(self):
        out = f"\n{self.name} with basis {self.basis_set}, variant {self.variant}"
        out += f", initial guess: {self.init_guess}"
        if self.sapbasisname is not None:
            out += f"  sap basis: {self.sapbasisname}"
        out += "\n\n"
        out += f"{self.thf} {self.tfbf} {self.tsbs}\n\n"
        out += f"Nocc: {self.nocc},   E_HF: {self.ehf},   nfunc: {self.nfunc}\n\n"
        out += f"{self.fbfdat.to_string()}\n\n"
        out += f'{self.sbsdat.loc[:,"nfunc":"Qsqrd"].to_string()}\n'
        return out


def read_data(path: str):
    """Read data file from path"""
    dat = MolData()

    fbf_begin = 0
    sbs_begin = 0
    fbf_end = 0
    sbs_end = 0

    f = open(path)
    for i, line in enumerate(f):
        if i == 1:
            baseinfo = line.strip().split()
            dat.name, dat.basis_set, dat.variant, dat.init_guess = baseinfo[:4]
            if len(baseinfo) == 5:
                dat.sapbasisname = baseinfo[-1]
            dat.variant = str(dat.variant)
        if i == 6:
            dat.thf, dat.tfbf, dat.tsbs = list(
                map(lambda x: (float(x) if x != "-" else x), line.strip().split())
            )
        if i == 9:
            nocc, ehf, nfunc = line.strip().split()
            dat.nocc = int(nocc)
            dat.ehf = float(ehf)
            dat.nfunc = int(nfunc)

        if "function-by-function" in line:
            fbf_begin = i + 1
            continue
        if fbf_begin != 0 and line in ["\n", "\r\n"] and sbs_begin == 0:
            fbf_end = i - 1
            continue
        if "shell-by-shell" in line:
            sbs_begin = i + 1
            continue
        if sbs_begin != 0 and sbs_end == 0 and line in ["\n", "\r\n"]:
            sbs_end = i
            continue

    if fbf_begin != 0:
        dat.fbfdat = pd.read_csv(
            path, skiprows=lambda x: x not in range(fbf_begin, fbf_end)
        )
    if sbs_begin != 0:
        dat.sbsdat = pd.read_csv(
            path, skiprows=lambda x: x not in range(sbs_begin, sbs_end)
        )

    return dat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create visualisations of ABS runs."
    )
    parser.add_argument(
        "-d", "--datpath", type=str, required=True, help="path to output files"
    )
    parser.add_argument(
        "-m", "--mol", type=str, required=True, help="molecule label"
    )
    parser.add_argument(
        "-b", "--bs", type=str, required=False, nargs='+', default=None,
        help="basis sets to process. If left empty, all found basis sets will be processed."
    )
    parser.add_argument(
        "-u", "--unc", action=argparse.BooleanOptionalAction,
        default=False,
        help="create visualisations for uncontracted runs also if output files exist, optional"
    )
    parser.add_argument(
        "--plotfbyf", action=argparse.BooleanOptionalAction,
        default=False,
        help="plot function by funtion graphs, optional"
    )
    args = parser.parse_args()

    datadir = args.datpath
    mol = args.mol
    bsets = args.bs    
    handle_decontraction = args.unc
    plotfbyf = args.plotfbyf

    files = os.listdir(datadir)
    files_to_process = list(filter(lambda f: mol == f.split('.')[0], files))
    if handle_decontraction:
        # Doublecheck to see if any files have uncontraction output
        handle_decontraction = any('unc' in fname for fname in files_to_process)
    if bsets is not None:
        # If basis sets are provided filter files to be processed to only
        # go through selected basis set results
        files_to_process = list(filter(lambda f: any(bs in f for bs in bsets), files_to_process))
    bsets = list(set(map(lambda fp: fp.split('.')[1], files_to_process)))
    # Filter out uncontracted files if user requests not to draw them
    if not handle_decontraction:
        files_to_process = list(filter(lambda f: 'unc' not in f, files_to_process))
        bsets = list(set(map(lambda fp: fp.split('.')[1], files_to_process)))

    dat = [read_data(datadir + '/' + f) for f in files_to_process]
    dat.sort(key = operator.attrgetter('basis_set', 'init_guess'))
    
    for d in dat:
        print(d.name, d.basis_set, d.init_guess, end=' ')
        if d.init_guess == 'sap':
            print(d.sapbasisname)
        else:
            print()


    # Convergence panels
    ncontr = len(list(filter(lambda bs: 'unc' not in bs, bsets)))
    nuncontr = len(bsets) - ncontr

    ndat = len(dat)
    figs = []
    axs = []

    panelidx = 0

    for bset in list(filter(lambda bs: 'unc' not in bs, bsets)):
        # make 3 figures for uncontracted, contracted and projection panels,
        # or 2 for contracted and projection panels
        numpanels = (('unc-' + bset) in bsets) + 2
        figs.extend([
            plt.figure(panelidx + i, figsize=(10, 8), tight_layout=True)
            for i in range(numpanels)
        ])
        axs.extend([figs[i].add_subplot() for i in range(panelidx, panelidx + numpanels)])


        datalist = list(filter(lambda data: bset in data.basis_set, dat))
        colors = cm.managua(np.linspace(.2, .8, len(datalist)))

        for j,df in enumerate(datalist):
            current_panelidx = panelidx + (2 if 'unc' in df.basis_set else 0)
            if df.variant == 'enocc' and plotfbyf:
                axs[current_panelidx].semilogy(
                    df.fbfdat["nfunc"][1:],
                    -df.fbfdat["diff"][1:],
                    ".-",
                    label=r"fct by fct: $\Delta E_{orb}=$" + f', init_guess: {df.init_guess}',
                    # label=r"fct by fct: $\Delta E_{orb}=\sum E^{occ orb}_{n+1}-\sum E^{occ orb}_{n}$",
                )
            # print(bset, df.sbsdat.columns)
            x, y = df.sbsdat["nfunc"][1:], df.sbsdat["E_scf"][1:] - df.ehf
            if df.variant == 'elden':
                y *= -1
            axs[current_panelidx].semilogy(
                x, y,
                label=legend_labels[df.variant] + 
                (f', init_guess: {df.init_guess}' if df.init_guess != 'SCF' else f', {df.init_guess}'), c=colors[j], ls=ls[ j % numpanels ], marker='o', lw=lw
                )

            # Projection panels
            axs[panelidx + 1].semilogy(
                df.sbsdat["nfunc"],
                1 - df.sbsdat["Qsqrd"] / df.nocc,
                label=r"$\Delta Q_\sigma$, " + f"{df.basis_set}, nfunc: {df.nfunc}" +
                (f', init_guess: {df.init_guess}' if df.init_guess != 'SCF' else f', {df.init_guess}'),
                c=colors[j], ls=ls[ j % numpanels ], marker='o', lw=lw
            )
        
            axs[current_panelidx].set_title(
                f"${{{df.name}}}$\nBasis: {df.basis_set}, nfunc: {df.nfunc}",
                fontsize=24,
                fontweight="bold",
            )

            axs[current_panelidx].set_ylabel(ylabels[df.variant], fontsize=16)
            axs[current_panelidx].set_xlabel("Subbasis size N", fontsize=16, fontweight="bold")

            axs[panelidx + 1].set_title(
                f"${{{df.name}}}$, projection",#\nBasis: {df.basis_set}, nfunc: {df.nfunc}",
                fontsize=24,
                fontweight="bold",
            )
            axs[panelidx + 1].set_ylabel(
                r"$\Delta Q_\sigma = 1 - \frac{1}{N_{occ}}\sum_{i,j}^{N_{occ}}|<i^{subbasis}|j^{full basis}>|^2$",
                fontsize=16,
            )
            axs[panelidx + 1].set_xlabel("Subbasis size N", fontsize=16, fontweight="bold")

        panelidx += numpanels

    for ax in axs:
        ax.legend(fontsize=12)
        ax.grid(alpha=.5)

    # for i in range(len(axs)):
    #     for j in range(ndat):
    #         if dat[j].variant == 0:
    #             axs[i].semilogy(
    #                 dat[j].fbfdat["nfunc"][1:],
    #                 -dat[j].fbfdat["diff"][1:],
    #                 ".-",
    #                 label=r"fct by fct: $\Delta E_{orb}=\sum E^{occ orb}_{n+1}-\sum E^{occ orb}_{n}$",
    #             )
    #         x, y = dat[j].sbsdat["nfunc"][1:], -dat[j].sbsdat["diff"][1:]
    #         if dat[j].variant == 2:
    #             y *= -1
    #         axs[i].semilogy(x, y, ".-", label=legend_labels[dat[i].variant])
    #         axs[i].semilogy(
    #             dat[j].sbsdat["nfunc"],
    #             dat[j].sbsdat["E_scf"] - dat[j].ehf,
    #             ".-",
    #             label=r"$\Delta E^{scf}=E^{scf}_{subbasis}-E^{scf}_{full basis}$",
    #         )

    #         # Projection panels
    #         axs[-1].semilogy(
    #             dat[j].sbsdat["nfunc"],
    #             1 - dat[j].sbsdat["Qsqrd"] / dat[j].nocc,
    #             ".-",
    #             label=r"$\Delta Q_\sigma$, " + f"{dat[j].basis_set}, nfunc: {dat[j].nfunc}",
    #         )
        
    #     axs[i].set_title(
    #         f"${{{dat[i].name}}}$\nBasis: {dat[i].basis_set}, nfunc: {dat[i].nfunc}",
    #         fontsize=24,
    #         fontweight="bold",
    #     )
    #     axs[i].set_ylabel(ylabels[dat[i].variant], fontsize=16)
    #     axs[i].set_xlabel("Subbasis size N", fontsize=16, fontweight="bold")


    # for i in range(len(axs)):
    #     axs[i].grid(alpha=0.5)
    #     axs[i].legend()
    # axs[-1].set_title(
    #     f"${{{dat[0].name}}}$\nBasis: {dat[0].basis_set}, nfunc: {dat[0].nfunc}",
    #     fontsize=24,
    #     fontweight="bold",
    # )
    # axs[-1].set_ylabel(
    #     r"$\Delta Q_\sigma = 1 - \frac{1}{N_{occ}}\sum_{i,j}^{N_{occ}}|<i^{subbasis}|j^{full basis}>|^2$",
    #     fontsize=16,
    # )
    # axs[-1].set_xlabel("Subbasis size N", fontsize=16, fontweight="bold")

    plt.show()
