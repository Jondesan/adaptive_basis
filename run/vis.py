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

font = {"weight": "bold", "size": 18}
matplotlib.rc("font", **font)

ls = [
    'solid',
    'dashed',
    'dashdot'
]

lw = 3.0

ylabels = {
    'enocc': r"$E_{\rm{subscf}} - E_{\rm{fullscf}}$ [$E_h$]",
    'ecore': r"$\Delta \frac{1}{2}\sum_i^{occ}(\epsilon_i + h_{ii})$",
    'elden': r"$\Delta Q$",
}
legend_labels = {
    'enocc': r"$\Delta E_{orb}$",#=\sum E^{occ orb}_{n+1}-\sum E^{occ orb}_{n}$",
    'ecore': r"$\Delta (\frac{1}{2}\sum_i^{occ}(\epsilon_i + h_{ii}))_{n-1, n}$",
    'elden': r"$\Delta Q_{n-1, n}$",#=Q_{n}-Q_{n-1}$",
}


class MolData:
    name = ""
    basis_set = ""
    variant = None
    init_guess = ""
    sapbasisname = None
    link_status = True
    charge = 0
    spin = 0
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


def charge_state_label(label):
    if str(label) == '-1':
        return '-'
    elif str(label) == '0':
        return ''
    else:
        return label

def read_data(path: str):
    """Read data file from path"""
    dat = MolData()

    fbf_begin = 0
    sbs_begin = 0
    fbf_end = 0
    sbs_end = 0

    new_format_adder = 0

    f = open(path)
    avail_params = []
    for i, line in enumerate(f):
        if i == 0:
            avail_params = line.strip().split()
        if i == 1:
            baseinfo = line.strip().split()
            dat.name, dat.basis_set, dat.variant, dat.init_guess = baseinfo[:4]
            if 'sap_basis' in avail_params:
                dat.sapbasisname = baseinfo[avail_params.index('sap_basis')]
            if 'link_status' in avail_params:
                dat.link_status = baseinfo[avail_params.index('link_status')]
            dat.variant = str(dat.variant)
        if i == 2 and 'charge' in line:
            new_format_adder = 2
        if i == 3 and new_format_adder:
            dat.charge, dat.spin = map(int, line.split())
        if i == 6 + new_format_adder:
            dat.thf, dat.tfbf, dat.tsbs = list(
                map(lambda x: (float(x) if x != "-" else x), line.strip().split())
            )
        if i == 9 + new_format_adder:
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

def process_files(datadir, handle_decontraction, bsets):
    files = os.listdir(datadir)
    # list of all files corresponding to chosen molecule
    files_to_process = list(filter(lambda f: mol == f.split('.')[0], files))

    dashremoved_files = [fs.replace('-', '') for fs in files_to_process]
    dashremoved_files = sorted(list(set(dashremoved_files)))
    files_to_process  = sorted(list(set(files_to_process)))
    filedict = {
        x[0]: x[1] for x in zip(dashremoved_files, files_to_process)
    }
    if handle_decontraction:
        # Doublecheck to see if any files have uncontraction output
        handle_decontraction = any('unc' in fname for fname in dashremoved_files)
    if bsets is not None:
        # If basis sets are provided filter files to be processed to only
        # go through selected basis set results
        dashremoved_files = list(filter(lambda f: any(bs in f for bs in bsets), dashremoved_files))
    bsets = list(set(map(lambda fp: fp.split('.')[1], dashremoved_files)))
    # Filter out uncontracted files if user requests not to draw them
    if not handle_decontraction:
        dashremoved_files = list(filter(lambda f: 'unc' not in f, dashremoved_files))
        bsets = list(set(map(lambda fp: fp.split('.')[1], dashremoved_files)))
    files_to_process = [filedict[drf] for drf in dashremoved_files]
    dat = [read_data(datadir + '/' + f) for f in files_to_process]
    dat.sort(key = operator.attrgetter('basis_set', 'init_guess'))
    
    for d in dat:
        print(d.name, d.basis_set, d.init_guess, end=' ')
        if d.init_guess == 'sap':
            print(d.sapbasisname)
        else:
            print()

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
    parser.add_argument(
        "--show", action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to show drawn figures on screen, optional"
    )
    parser.add_argument(
        "--out", type=str, required=False, default=None,
        help="path to output folder where figures are saved, optional"
    )
    parser.add_argument(
        "--figset", type=str, required=False, default='default', choices=['default', 'comparison'],
        help="which set of figures to plot, optional"
    )
    args = parser.parse_args()

    datadir = args.datpath
    mol = args.mol
    bsets = args.bs    
    handle_decontraction = args.unc
    plotfbyf = args.plotfbyf
    out = args.out
    show = args.show
    figset = args.figset
    
    conv_tol = 1e-1

    bsets = [bset.replace('-','') for bset in bsets]

    if handle_decontraction:
        print('Handling decontractions!')

    dat = process_files(
        datadir=datadir,
        handle_decontraction=handle_decontraction,
        bsets=bsets
        )

    # print(files_to_process)


    if figset == 'default':
        # Draw convergence, projection and accuracy figures.
        # Convergence panels
        ncontr = len(list(filter(lambda bs: 'unc' not in bs, bsets)))
        nuncontr = len(bsets) - ncontr

        ndat = len(dat)
        figs = []
        axs = []

        panelidx = 0

        for bset in list(filter(lambda bs: 'unc' not in bs, bsets)):
            # make 4 figures for uncontracted, contracted, projection and accuracy panels,
            # or 3 for contracted, projection and accuracy panels
            numpanels = (any('unc' in bs for bs in bsets)) + 3
            figs.extend([
                plt.figure(panelidx + i, figsize=(10, 8), tight_layout=True)
                for i in range(numpanels)
            ])
            axs.extend([figs[i].add_subplot() for i in range(panelidx, panelidx + numpanels)])
            datalist = list(filter(lambda data: bset in data.basis_set.replace('-',''), dat))

            # colors = cm.managua(np.linspace(.2, .8, len(datalist)))
            colors = plt.get_cmap('tab20')(np.linspace(.2, .8, len(datalist)))

            convergences = []

            for j,df in enumerate(datalist):
                current_panelidx = panelidx + (3 if 'unc' in df.basis_set else 0)

                # function-by-function convergence
                if df.variant == 'enocc' and plotfbyf:
                    axs[current_panelidx].semilogy(
                        df.fbfdat["nfunc"][1:],
                        -df.fbfdat["diff"][1:],
                        ".-",
                        label=r"fct by fct: $\Delta E_{orb}=$" + f', init_guess: {df.init_guess}',
                    )
                
                # shell-by-shell convergence
                x, y = df.sbsdat["nfunc"][1:], df.sbsdat["E_scf"][1:] - df.ehf
                nfunc = df.sbsdat.loc[df.sbsdat.index[-1], 'nfunc']
                if df.variant == 'elden':
                    y *= -1
                axs[current_panelidx].semilogy(
                    x/nfunc, y,
                    label=( f'{df.init_guess}' + (f', basis: {df.sapbasisname}' if df.init_guess == 'sap' else '') ),
                    c=colors[j], ls=ls[ j % (numpanels-1) ], marker='o', lw=lw
                    )
                # Get index where convergence criteria is met and store
                # conv_idx = np.argwhere(y <= conv_tol)#[0]
                # if conv_idx.size == 0:
                #     conv_idx = -1
                # else:
                #     conv_idx = conv_idx[0]
                # convergences.append(x.values[conv_idx])#.values[0])

                # Projection panels
                axs[panelidx + 1].semilogy(
                        df.sbsdat["nfunc"] / nfunc,
                    1 - df.sbsdat["Qsqrd"] / df.nocc,
                    label=f"{df.basis_set}, nfunc: {df.nfunc}" +
                    # (f', init_guess: {df.init_guess}' if df.init_guess != 'SCF' else f', {df.init_guess}'),
                    f', {df.init_guess}' + ('' if df.init_guess != 'sap' else f' {df.sapbasisname}'),
                    c=colors[j], ls=ls[ j % (numpanels-1) ], marker='o', lw=lw
                )

                axs[current_panelidx].set_title(
                    f"${{{df.name}}}^{{{charge_state_label(df.charge)}}}$\nBasis: {df.basis_set}, nfunc: {df.nfunc}",
                    fontsize=24,
                    fontweight="bold",
                )

                axs[current_panelidx].set_ylabel(ylabels[df.variant], fontsize=24)
                axs[current_panelidx].set_xlabel(r"$N_{\rm{func,subbasis}}/N_{\rm{func,tot}}$", fontsize=24, fontweight="bold")

                axs[panelidx + 1].set_title(
                    f"${{{df.name}}}$, projection",
                    fontsize=24,
                    fontweight="bold",
                )
                axs[panelidx + 1].set_ylabel(
                    r"$\Delta Q_\sigma = 1 - \frac{1}{N_{occ}}\sum_{i,j}^{N_{occ}}|<i^{subbasis}|j^{full basis}>|^2$",
                    fontsize=16,
                )
                axs[panelidx + 1].set_xlabel(r"nfunc ratio $N/N_{\rm{func}}$", fontsize=16, fontweight="bold")

                # dEscf = df.ehf - df.sbsdat['E_scf']
                # dEscf = df.sbsdat['E_scf'][:-1].values - df.sbsdat['E_scf'][1:].values
                dEscf = df.ehf - df.sbsdat['E_scf'][1:]#.values
                dEcutoff = df.sbsdat['diff'][1:]#.values
                axs[panelidx + 2].plot(
                    np.log10(np.abs(dEscf)),#[(dEcutoff <= 0) & (dEscf != 0)])),
                    np.log10(np.abs(dEcutoff)),#[(dEcutoff <= 0) & (dEscf != 0)])),
                    # np.log10(np.abs(dEscf[(dEcutoff <= 0) & (dEscf != 0)])),
                    # np.log10(np.abs(dEcutoff[(dEcutoff <= 0) & (dEscf != 0)])),
                    label=( f'{df.init_guess}' + (f', basis: {df.sapbasisname}' if df.init_guess == 'sap' else '') ),
                    marker='o', c=colors[j], ls=ls[ j % (numpanels-1) ],
                    lw=lw
                )
                axs[panelidx + 2].set_xlabel(r'$\lg\Delta E_{\rm{cutoff}}$')
                axs[panelidx + 2].set_ylabel(r'$\lg\Delta E_{\rm{SCF}}$')
                nocc = df.nocc

                axs[panelidx].set_ylim(1e-6, 1e3)

            axs[panelidx].set_xlim(0.0, 1.0)

            xmin, xmax = axs[panelidx + 2].get_xlim()
            axs[panelidx + 2].autoscale(False, axis="both")
            axs[panelidx + 2].plot([xmin, xmax], [xmin, xmax], color='red', linestyle='--', alpha=.5)

            ymin, ymax = axs[current_panelidx].get_ylim()
            ### Vertical line indicating convergence
            # conv_line_x = np.max(convergences)
            # axs[current_panelidx].vlines(
            #     conv_line_x, ymin, ymax,
            #     linestyle='--', color='red',
            #     label=f'Convergence {conv_tol}: {conv_line_x} funcs')
            minimal_basis_line_x = nocc / 2
            axs[current_panelidx].axvline(
                minimal_basis_line_x/nfunc,# ymin, ymax,
                linestyle='--', color='red',
                label=f'Minimal basis: {minimal_basis_line_x} funcs',
                lw=lw)
            if np.any(['unc' in df.basis_set for df in datalist]):
                axs[panelidx].axvline(
                    minimal_basis_line_x/nfunc,# ymin, ymax,
                    linestyle='--', color='red',
                    label=f'Minimal basis: {minimal_basis_line_x} funcs',
                    lw=lw)
                axs[panelidx].set_ylim(ymin, ymax)
            axs[current_panelidx].set_ylim(ymin, ymax)
            panelidx += numpanels


            for ax in axs:
                ax.legend(fontsize=14)
                ax.grid(alpha=.5)

            if out is not None:
                panels = ['contracted', 'projection', 'accuracy', 'decontracted']
                for i,fig in enumerate(figs[-numpanels:]):
                    fname = f'{datalist[0].name}.{datalist[0].basis_set}.{datalist[0].variant}.{panels[i]}'
                    fig.savefig(
                        out + '/' + fname + '.pdf',
                        dpi=100, format='pdf')
    
    if figset == 'comparison':
        fig, ax = plt.subplots(figsize=(12,6))

        apc4_energy = 0.0
        xmax = 0
        for df in dat:
            if df.basis_set.replace('-', '') == 'augpc4':
                apc4_energy = df.ehf
            if df.basis_set.replace('-', '') == 'augpc3':
                xmax = df.nfunc
        for df in dat:
            if df.init_guess == 'SCF':# and df.basis_set.replace('-', '') != 'augpc4':
                x, y = df.sbsdat["nfunc"][1:], df.sbsdat["E_scf"][1:] - apc4_energy
                # if df.basis_set.replace('-', '') == 'augpc4':
                    # x, y = x[x <= xmax + 1], y[x <= xmax + 1]
                    # ax.autoscale(False, axis="both")
                ax.semilogy(
                    x,
                    y,
                    label=df.basis_set,
                    marker='.',
                    alpha=.8, lw=2
                )
        ax.set_xlabel('Number of basis functions in subbasis', fontweight='bold')
        ax.set_ylabel(r'$E_{\rm{fullbasis SCF}}^{\rm{aug-pc-4}}-E_{\rm{subbasis SCF}}$')
        ax.set_xlim(-2, xmax+2)
        ax.set_ylim(1e-6, 1e2)

        # ax.set_title(f'{dat[0].name}')

        ax.legend()
        ax.grid(alpha=.5)

        if out is not None:
            fname = f'{dat[0].name}.{dat[0].variant}.comparison'
            fig.savefig(
                out + '/' + fname + '.pdf',
                dpi=100, format='pdf')

    if show:
        plt.show()
