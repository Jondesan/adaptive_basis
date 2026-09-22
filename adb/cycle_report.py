"""Human-readable per-ADB-cycle report writer.

Standalone formatter for `adb.mask_analysis(cycle_report=True)`'s output
-- a separate, purpose-built file to help diagnose sub-basis SCF behavior
(added functions, converged energy, and per-irrep occupied/virtual
orbital energies) at a glance, one clearly delimited block per cycle,
rather than reading raw pyscf logging or reconstructing this by hand from
`dataframe`/CSV columns (as was done manually for the FeO investigation in
`adaptive_basis/untracked/jagged_convergence_check/REPORT.md`).

Pure post-processing: everything here is built from `mask_analysis`'s own
already-computed return values (`dataframe`, `cycle_report_history`) plus
already-existing utilities (`adb.smask_to_mask`,
`adb.function_labels_from_mask`) -- no new knowledge of the ADB search or
SCF machinery is needed in this module.
"""

import datetime

from .maskutil import smask_to_mask
from .ioutil import function_labels_from_mask


def _mask_of(mask_field, mol):
    """`dataframe`'s per-row mask field is a plain AO mask when
    `mask_analysis` was called in function-mode, or a shell mask (smask)
    in the far more common shell-mode -- distinguish by dtype (a plain
    mask is a uniform bool array; a smask is dtype=object, mixing bools,
    ints, and label tuples) and normalize to a plain AO mask either way.
    """
    if mask_field.dtype == bool:
        return mask_field
    return smask_to_mask(mask_field, mol.cart)


def _format_added_functions(mask, prev_mask, mol):
    added = mask & ~prev_mask
    if not added.any():
        return "  (none -- same basis as previous cycle)"
    by_atom = function_labels_from_mask(added, mol)
    lines = []
    for atom, shells in by_atom.items():
        lines.append(f"  {atom}: {', '.join(shells)}")
    return "\n".join(lines)


def _format_frontier_orbitals(orbitals):
    """`orbitals` is one `cycle_report_history[i]['orbitals']` list --
    (energy, irrep, is_occupied, spin_label) tuples. Groups by
    (irrep, spin_label), sorts, and flags occ/virt inversions."""
    groups: dict = {}
    for energy, irrep, is_occ, spin in orbitals:
        groups.setdefault((irrep, spin), {'occ': [], 'virt': []})
        groups[(irrep, spin)]['occ' if is_occ else 'virt'].append(energy)

    lines = []
    for (irrep, spin), data in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        occ = sorted(data['occ'])
        virt = sorted(data['virt'])
        label = irrep or '(no symmetry)'
        if spin is not None:
            label += f" ({spin})"
        occ_str = ', '.join(f"{e:.4f}" for e in occ) if occ else '(none)'
        virt_str = ', '.join(f"{e:.4f}" for e in virt) if virt else '(none)'
        line = f"  {label:<18s} occ [{occ_str}]   virt [{virt_str}]"
        if occ and virt and max(occ) > min(virt):
            line += "   <- INVERSION (occupied orbital above a virtual one in this irrep!)"
        lines.append(line)
    return "\n".join(lines) if lines else "  (no orbital data for this cycle)"


def write_cycle_report(path, dataframe, cycle_report_history, mol, header_info=None):
    """Write a human-readable per-cycle report combining `mask_analysis`'s
    `dataframe` (nfunc, SCF energy, basis composition) with
    `cycle_report_history` (per-irrep occupied/virtual orbital energies,
    from `cycle_report=True`) into one file, one clearly delimited block
    per ADB cycle.

    Parameters
    ----------
    path : str
        Output file path.
    dataframe : list
        `mask_analysis`'s returned dataframe (see its docstring for the
        row layout: nfunc, cursum, diff, E_scf, E_orb, Qsqrd, smask/mask,
        dE, E_HF_largebasis, conv_stat).
    cycle_report_history : list
        `mask_analysis(cycle_report=True)`'s returned history: one
        ``{'nfunc': int, 'orbitals': [...]}`` dict per cycle, in the same
        order as `dataframe` (see `adb.get_frontier_orbitals_from_scf`).
    mol : pyscf.gto.Mole
        The (shell-separated) molecule `dataframe`'s masks index into --
        pass the same `fullbasis_mol`/`shellsep_mol` `mask_analysis` itself
        used.
    header_info : dict, optional
        Extra ``key: value`` pairs (e.g. molecule name, basis, init guess)
        printed at the top of the file.

    Raises
    ------
    ValueError
        If `dataframe` and `cycle_report_history` don't have matching
        per-cycle `nfunc` sequences (they're expected to come from the
        same `mask_analysis` call).
    """
    nfuncs = [row[0] for row in dataframe]
    report_nfuncs = [entry['nfunc'] for entry in cycle_report_history]
    if nfuncs != report_nfuncs:
        raise ValueError(
            "dataframe and cycle_report_history have mismatched nfunc "
            f"sequences -- did they come from the same mask_analysis call? "
            f"{nfuncs} != {report_nfuncs}")

    lines = [f"# adb cycle report, generated {datetime.datetime.now()}"]
    if header_info:
        for key, val in header_info.items():
            lines.append(f"# {key}: {val}")
    lines.append("")

    prev_mask = None
    for i, (row, entry) in enumerate(zip(dataframe, cycle_report_history)):
        nfunc, _cursum, _diff, e_scf = row[0], row[1], row[2], row[3]
        conv_stat = row[9]
        mask = _mask_of(row[6], mol)
        if prev_mask is None:
            prev_mask = mask & False  # same shape, all False: first cycle "adds" everything

        lines.append("=" * 80)
        lines.append(f"Cycle {i}   nfunc={nfunc}   E_scf={e_scf:.6f} Ha   converged={bool(conv_stat)}")
        lines.append("=" * 80)
        lines.append("Added functions:")
        lines.append(_format_added_functions(mask, prev_mask, mol))
        lines.append("")
        lines.append("Frontier orbitals by irrep (occ / lowest virt) [Ha]:")
        lines.append(_format_frontier_orbitals(entry['orbitals']))
        lines.append("")

        prev_mask = mask

    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
