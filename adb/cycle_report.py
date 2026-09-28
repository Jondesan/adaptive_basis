"""Human-readable per-ADB-cycle report writer.

Standalone formatter for `adb.mask_analysis(cycle_report=True)`'s output
-- a separate, purpose-built file to help diagnose sub-basis SCF behavior
(added functions, converged energy, and per-irrep occupied/virtual
orbital energies) at a glance, one clearly delimited block per cycle,
rather than reading raw pyscf logging or reconstructing this by hand from
`dataframe`/CSV columns (as was done manually for the FeO investigation in
`adaptive_basis/untracked/jagged_convergence_check/REPORT.md`).

Pure post-processing: everything here is built from `mask_analysis`'s own
already-computed return values (`dataframe`, `cycle_report_history`) and
`mol.ao_labels()` -- no new knowledge of the ADB search or SCF machinery
is needed in this module.
"""

import datetime
import re

from pyscf.gto.mole import cart_labels, sph_labels

from .maskutil import smask_to_mask


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


# ---------------------------------------------------------------------------
# sigma/pi/delta/... axis-character classification, from raw AO labels.
# ---------------------------------------------------------------------------

# |m_l| -> name, for the shells pyscf labels by explicit signed m_l
# ('4f-2', '4f+0', ...  see _axis_character below).
_AXIS_CHARACTER_NAMES = ['sigma', 'pi', 'delta', 'phi', 'gamma', 'eta', 'iota']

# p/d shells (l=1,2) are labeled with descriptive Cartesian-combination
# suffixes rather than signed m_l -- mapped by hand, verified empirically
# against pyscf (see write_cycle_report's docstring): a z-axis-aligned
# lobe (z, z^2) is sigma; a lobe linear in x or y together with at most
# one z (x, y, xz, yz) is pi; a lobe quadratic in x/y alone (xy, x2-y2)
# is delta. This exactly reproduces the standard Coov -> C2v correlation
# table (Sigma+ -> A1, Pi -> B1+B2, Delta -> A1+A2).
_P_D_SIGMA = {'z', 'z^2'}
_P_D_PI = {'x', 'y', 'xz', 'yz'}
_P_D_DELTA = {'xy', 'x2-y2'}

_SHELL_TOKEN_RE = re.compile(r'^[0-9]+([spdfghik])(.*)$')


def _axis_character(shell_token: str) -> str | None:
    """Classify one shell token (e.g. '3dxz', '4f-2', '2s' -- the last
    whitespace-separated part of a raw `mol.ao_labels()`/`cart_labels()`
    entry, *not* shell-collapsed the way `adb.function_labels_from_mask`
    labels are) by its angular character along the z axis.

    Assumes z is the molecular axis, as pyscf orients a linear molecule
    for the abelian point groups this project forces them into (C2v/D2h,
    for FeO/GaO/CuH-style diatomics). Verified empirically: for FeO/C2v,
    Fe 2pz and 3dz^2 land in the fully symmetric irrep alongside s
    functions (sigma character); 2px/2py and 3dxz/3dyz pair up as B1/B2
    (pi); 3dx2-y2/3dxy pair up as A1/A2 (delta) -- exactly the standard
    Coov -> C2v correlation table. Only physically meaningful for a
    linear molecule; harmless (just not meaningful) otherwise.

    Returns one of 'sigma'/'pi'/'delta'/'phi'/... (sigma = |m_l| = 0,
    pi = |m_l| = 1, etc.), or `None` if `shell_token` can't be classified
    this way -- a Cartesian d/f/... component beyond xy/xz/yz (e.g.
    Cartesian 'xx'/'yy'/'zz', not individually pure angular-momentum
    eigenstates), or an unrecognized format.
    """
    m = _SHELL_TOKEN_RE.match(shell_token)
    if m is None:
        return None
    letter, suffix = m.group(1), m.group(2)
    if letter == 's':
        return 'sigma'
    if suffix[:1] in ('+', '-'):
        # f and beyond: pyscf labels these directly by signed m_l,
        # e.g. '4f-2', '4f+0', '4f+3'.
        try:
            abs_m = abs(int(suffix))
        except ValueError:
            return None
        if abs_m < len(_AXIS_CHARACTER_NAMES):
            return _AXIS_CHARACTER_NAMES[abs_m]
        return f'|m_l|={abs_m}'
    if suffix in _P_D_SIGMA:
        return 'sigma'
    if suffix in _P_D_PI:
        return 'pi'
    if suffix in _P_D_DELTA:
        return 'delta'
    return None  # e.g. Cartesian xx/yy/zz


def _ao_labels(mol):
    return cart_labels(mol) if mol.cart else sph_labels(mol)


def _character_breakdown(mask, ao_labels):
    """Group the AOs selected by `mask` by (atom, axis character).

    Returns ``{'<atom_id> <atom_symbol>': {character: [shell_token, ...]}}``,
    e.g. ``{'0 Fe': {'sigma': ['3dz^2', '4s'], 'pi': ['3dxz', '3dyz']}}``.
    Unclassifiable labels (see `_axis_character`) are grouped under the
    key ``'unclassified'``.
    """
    by_atom: dict = {}
    for label, selected in zip(ao_labels, mask):
        if not selected:
            continue
        parts = label.split()
        atom_key = f'{parts[0]} {parts[1]}'
        token = parts[-1]
        char = _axis_character(token) or 'unclassified'
        by_atom.setdefault(atom_key, {}).setdefault(char, []).append(token)
    return by_atom


def _format_added_functions(mask, prev_mask, ao_labels):
    added = mask & ~prev_mask
    if not added.any():
        return "  (none -- same basis as previous cycle)"
    breakdown = _character_breakdown(added, ao_labels)
    lines = []
    for atom, chars in breakdown.items():
        parts = [f"{char} [{', '.join(sorted(tokens))}]" for char, tokens in sorted(chars.items())]
        lines.append(f"  {atom}: {'; '.join(parts)}")
    return "\n".join(lines)


def _format_basis_composition(mask, ao_labels):
    """Compact per-atom character *counts* for the whole current
    sub-basis (not just what changed this cycle) -- deliberately counts
    rather than lists every function, so this stays readable once the
    sub-basis has grown large, unlike `_format_added_functions` (always
    small: only what's new this cycle)."""
    breakdown = _character_breakdown(mask, ao_labels)
    if not breakdown:
        return "  (empty)"
    lines = []
    for atom, chars in breakdown.items():
        counts = [f"{char}={len(tokens)}" for char, tokens in sorted(chars.items())]
        lines.append(f"  {atom}: {'  '.join(counts)}")
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

    Each cycle's block reports, per atom, the *axis character*
    (sigma/pi/delta/phi/...) of both the functions added that cycle and
    the full current sub-basis composition -- see `_axis_character`'s
    docstring for exactly how this is determined and its (linear-molecule)
    scope. This is a pure, additional interpretation of the raw AO labels
    already available; it needs no symmetry/irrep information and works
    the same whether or not the run used `--symmetry`/`--point_group_file`.

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

    ao_labels = _ao_labels(mol)

    lines = [f"# adb cycle report, generated {datetime.datetime.now()}"]
    if header_info:
        for key, val in header_info.items():
            lines.append(f"# {key}: {val}")
    lines.append("# Axis character (sigma/pi/delta/...) assumes z is the molecular")
    lines.append("# axis, as pyscf orients linear molecules for C2v/D2h; only")
    lines.append("# physically meaningful for a linear molecule.")
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
        lines.append("Added functions (by axis character):")
        lines.append(_format_added_functions(mask, prev_mask, ao_labels))
        lines.append("")
        lines.append("Current basis composition (axis character counts):")
        lines.append(_format_basis_composition(mask, ao_labels))
        lines.append("")
        lines.append("Frontier orbitals by irrep (occ / lowest virt) [Ha]:")
        lines.append(_format_frontier_orbitals(entry['orbitals']))
        lines.append("")

        prev_mask = mask

    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
