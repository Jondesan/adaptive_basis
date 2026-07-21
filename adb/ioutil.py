import os
from ast import literal_eval
from copy import deepcopy
from warnings import warn

import numpy as np
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .maskutil import init_smask
from .molutil import create_mol_from_file, create_shell_separated_mol

_console = Console()
_err_console = Console(stderr=True)


def get_files_in_folder(folder: str) -> list[str]:
    """List every file in a folder.

    Parameters
    ----------
    folder : str
        Folder to search.

    Returns
    -------
    list of str
        File names in `folder`.
    """
    return os.listdir(folder)


def get_molecules_in_dir(
        molpath:            str,
        basis_sets:         str | list[str],
        get_decontractions: bool                = False,
        unit:               str                 = 'Angstrom',
        symmetry:           bool | str          = False,
        symmetry_fname:     str | None          = None,
        ) -> list[list]:
    """Load every molecule geometry file in a directory (or a single file).

    For each geometry file and each requested basis set, builds a
    shell-separated `Mole` and its initial shell mask, printing a summary
    table at the end.

    Parameters
    ----------
    molpath : str
        Path to a directory of geometry files, or a single geometry file.
    basis_sets : str or list of str
        Basis set name(s) to build each molecule in. A space-separated
        string is split into a list.
    get_decontractions : bool, default False
        Whether to also build a fully-decontracted (``'unc-'``-prefixed)
        version of each basis set.
    unit : str, default 'Angstrom'
        Coordinate unit.
    symmetry : bool or str, default False
        Whether/which symmetry to enable, passed to `Mole(symmetry=...)`.
        Only consulted directly when `symmetry_fname` is not given, or
        when it is but has no matching entry for a given molecule (see
        `symmetry_fname`).
    symmetry_fname : str, optional
        Path to a point-group-label file (see `point_group_from_file`).
        When given (and `symmetry` is truthy), overrides `symmetry` with
        the point group looked up for each molecule's file name.

    Returns
    -------
    list of list
        One ``[filename, mol, shellsep_mol, smask, None, basis_name]``
        entry per (geometry file, basis set) combination, sorted by
        electron count, then basis name, then number of basis functions.
    """
    prefix = molpath

    if os.path.isdir(prefix):
        fs = [prefix + '/' + f for f in get_files_in_folder(prefix)]
    else:
        fs = [molpath]
    molecules = []
    for fn in fs:
        molfname = fn.split("/")[-1]
        if molfname[0] == "#":  # If mol fname starts with #, skip file
            continue
        _console.print(f"Reading file {fn}", style="cyan", markup=False)

        if symmetry and symmetry_fname is not None:
            symm = point_group_from_file(symmetry_fname, molfname)
        else:
            symm = symmetry

        if isinstance(basis_sets, str):
            basis_sets = basis_sets.split()
        for bs in basis_sets:
            for unc in (
                ["", "unc-"] if get_decontractions and "unc-" not in bs else [""]
            ):
                fnparts = fn.split('/')[-1].split('.')
                if len(fnparts) > 2:
                    charge = [int(substring.replace('charge', '')) for substring in fnparts if 'charge' in substring]
                    charge = charge[0] if len(charge) != 0 else 0
                    spin = [int(substring.replace('spin', '')) for substring in fnparts if 'spin' in substring]
                    spin = spin[0] if len(spin) != 0 else None
                else:
                    charge = 0
                    spin = None
                mol = create_mol_from_file(
                    fn, unc + bs, charge=charge, spin=spin, unit=unit, symmetry=symm)

                mol = create_shell_separated_mol(mol, verbose=mol.verbose)
                smask = init_smask(mol)
                _console.print(
                    f"Created molecule {molfname}, with charge {charge}, "
                    f"spin {spin} and symmetry set at {symm}",
                    style="green", markup=False)
                molecules.append(
                    [fn.split("/")[-1], mol, create_shell_separated_mol(mol), smask, None, bs]
                )

    # Sort by number of electrons, then by the basis, then by number of basis fcts
    molecules.sort(key=lambda x: (x[1].tot_electrons(), x[1].basis, x[1].nao_nr()))

    table = Table(title=f"Loaded {len(molecules)} molecular structures", box=box.ROUNDED)
    table.add_column("Molecule")
    table.add_column("# functions", justify="right")
    for name, mol_, *_rest in molecules:
        table.add_row(name, str(int(mol_.nao_nr())))
    _console.print(table)

    return molecules


def point_group_from_file(path: str, mol_filename: str) -> str | bool:
    """Look up a molecule's point group label from a file.

    Parameters
    ----------
    path : str
        Path to a whitespace-separated ``<filename> <point_group>`` file.
    mol_filename : str
        Molecule file name to search for.

    Returns
    -------
    str or bool
        The point group label if `mol_filename` is found in the file,
        else `True` (i.e. "enable symmetry, auto-detect the group").
    """
    pnt_grp = True
    _console.print(
        f"Reading file with point group information at path {path}",
        style="dim", markup=False)
    _console.print(
        f"Searching for point group match for molecule with filename {mol_filename}",
        style="dim", markup=False)
    with open(path, 'r') as file:
        for line in file:
            name, point_grp_label = line.split()
            if mol_filename == name:
                pnt_grp = point_grp_label
                _console.print(
                    f"Found point group {pnt_grp} for molecule with filename {mol_filename}",
                    style="green", markup=False)

    return pnt_grp


def read_symmetry_occs_from_file(fname: str, molfname: str) -> tuple[dict | None, str | None]:
    """Read a molecule's target per-irrep occupations from file.

    Parameters
    ----------
    fname : str
        Path to a ``;``-separated ``molfilename;occs;point_group`` file,
        e.g.::

            h2.charge0.spin0.xyz;{'Ag': 2, 'B1g': 0, ...};D2h
            ch4.charge0.spin0.xyz;{'A1': 6, 'A2': 0, 'B1': 2, 'B2': 2};Td

    molfname : str
        Molecule file name to search for.

    Returns
    -------
    irrep_nelec : dict or None
        The matching occupation dict, or `None` if `fname` doesn't exist
        or has no matching line.
    irrep_symb : str or None
        The matching point group label, or `None`.
    """
    if fname is None or not os.path.isfile(fname):
        return None, None

    with open(fname, 'r') as f:
        file = f.readlines()

    # Get lines from the file with matching molecule filename
    line = [x for x in file if molfname in x]
    # TODO: handle multiple lines with same name better
    line = None if len(line) < 1 else line[0]

    irrep_nelec = literal_eval(line.split(";")[1])
    irrep_symb = line.split(";")[2].rstrip()
    return irrep_nelec, irrep_symb


def write_orbital_history(
        orbital_history:    list,
        fn:                 str,
        molname:            str = "",
        basisname:          str = "",
        ) -> None:
    """Write a `find_subspace(track_orbitals=True)` history to a CSV file.

    One row per occupied orbital per ADB cycle: ``nfunc,energy,irrep``.

    Parameters
    ----------
    orbital_history : list
        List of ``{'nfunc': int, 'orbitals': [(energy, irrep), ...]}``
        dicts, as returned by `adb.find_subspace(track_orbitals=True)`.
    fn : str
        Output file name (``.csv`` is appended).
    molname : str, default ""
        Molecule name, recorded in a header comment if given.
    basisname : str, default ""
        Basis set name, recorded in a header comment if given.

    Notes
    -----
    `irrep` is left blank for symmetry-blind entries (`irrep_nelec`/
    `orbsym` not given to `find_subspace`).
    """
    with open(fn + ".csv", "w") as f:
        if molname or basisname:
            f.write(f"# molecule={molname} basis={basisname}\n")
        f.write("nfunc,energy,irrep\n")
        for entry in orbital_history:
            nfunc = entry["nfunc"]
            for energy, irrep in entry["orbitals"]:
                f.write(f"{nfunc},{energy:.12f},{irrep if irrep is not None else ''}\n")


# Column spec driving the streaming ADB-iteration table (print_data_header/
# print_data). Add a column by adding one tuple here and threading the
# value through print_data's `values` dict below -- no width arithmetic
# needs touching anywhere else.
_DATA_COLUMNS = [
    # (key, header label, width, justify, value format)
    ("nfunc",    "N_func",       10, ">", "d"),
    ("label",    "New funcs",    13, ">", ""),
    ("criteria", "Criteria val", 16, ">", ".9f"),
    ("diff",     "Difference",   15, ">", ".9f"),
    ("e_scf",    "E_subbasSCF",  15, ">", ".9f"),
    ("qsqrd",    "Q^2",          18, ">", ".12f"),
]


def _format_value(value, width: int, justify: str, fmt: str) -> str:
    """Format one `print_data` cell value, treating strings specially
    (skipping the numeric `fmt` spec, which doesn't apply to them)."""
    if isinstance(value, str):
        return f"{value:{justify}{width}}"
    return f"{value:{justify}{width}{fmt}}"


def print_data_header() -> None:
    """Print the streaming ADB-iteration table's header (see `_DATA_COLUMNS`)."""
    widths = [width for _, _, width, _, _ in _DATA_COLUMNS]
    top = "╭" + "┬".join("─" * (w + 2) for w in widths) + "╮"
    sep = "├" + "┼".join("─" * (w + 2) for w in widths) + "┤"
    header = "│" + "│".join(
        f" {label:{justify}{w}} " for _, label, w, justify, _ in _DATA_COLUMNS
    ) + "│"

    _console.print()
    _console.print(top, style="bold", markup=False, soft_wrap=True)
    _console.print(header, style="bold", markup=False, soft_wrap=True)
    _console.print(sep, style="bold", markup=False, soft_wrap=True)


def print_data_footer() -> None:
    """Print the streaming ADB-iteration table's closing border."""
    widths = [width for _, _, width, _, _ in _DATA_COLUMNS]
    bottom = "╰" + "┴".join("─" * (w + 2) for w in widths) + "╯"
    _console.print(bottom, style="bold", markup=False, soft_wrap=True)


def print_data(
        mask:               np.ndarray,
        criteria_value:     float,
        diff:               float,
        ao_or_shell_label:  str,
        E_scf:              float | str | None = "-",
        Qsqrd:              float | str | None = "-",
        print_header:       bool               = False,
        ) -> None:
    """Print one row of the streaming ADB-iteration table.

    Parameters
    ----------
    mask : ndarray
        The current mask (only its selected-count is shown).
    criteria_value : float
        The current step's search criterion value.
    diff : float
        The current step's criterion improvement.
    ao_or_shell_label : str
        Label of the function(s)/shell added this step.
    E_scf : float, str, or None, default "-"
        The subbasis SCF energy, or ``"-"``/`None` if not available.
    Qsqrd : float, str, or None, default "-"
        The squared projection onto the full-basis wavefunction, or
        ``"-"``/`None` if not available.
    print_header : bool, default False
        Whether to print the table header (`print_data_header`) first.
    """
    if print_header:
        print_data_header()

    if E_scf is None:
        E_scf = "-"
    if Qsqrd is None:
        Qsqrd = "-"

    values = {
        "nfunc":    sum(mask),
        "label":    ao_or_shell_label,
        "criteria": criteria_value,
        "diff":     diff,
        "e_scf":    E_scf,
        "qsqrd":    Qsqrd,
    }
    cells = [
        f" {_format_value(values[key], width, justify, fmt)} "
        for key, _, width, justify, fmt in _DATA_COLUMNS
    ]
    _console.print("│" + "│".join(cells) + "│", markup=False, soft_wrap=True)


def _print_atom_function_table(atom_dict: dict) -> None:
    """Print a two-column (atom, functions) table."""
    table = Table(box=box.SIMPLE)
    table.add_column("Atom")
    table.add_column("Functions")
    for key, elem in atom_dict.items():
        table.add_row(key, ", ".join(elem))
    _console.print(table)


def print_labels_of_functions_in_mask(
        mask:                       np.ndarray,
        mol,
        print_actual_minimal_basis: bool = False,
        ) -> None:
    """Print the AO labels selected by `mask`, grouped by atom.

    Parameters
    ----------
    mask : ndarray
        AO mask.
    mol : pyscf.gto.Mole
        Molecule object `mask` indexes into.
    print_actual_minimal_basis : bool, default False
        Whether to also print the functions of an actual STO-3G minimal
        basis on the same geometry, for comparison.
    """
    atom_dict = function_labels_from_mask(mask, mol)
    # Sort dictionary by the internal atom index
    atom_dict = dict(sorted(
        atom_dict.items(),
        key=lambda item: int(item[0].split()[0])))

    if print_actual_minimal_basis:
        minimal_mol = deepcopy(mol)
        minimal_mol.basis = 'sto3g'
        minimal_mol.build()
        minimal_mask = np.ones(minimal_mol.nao_nr(), dtype=bool)

        _console.print("\nFunctions in the 'actual' minimal basis:", style="bold", markup=False)
        minimal_atom_dict = function_labels_from_mask(minimal_mask, minimal_mol)
        # Sort dictionary by the internal atom index
        minimal_atom_dict = dict(sorted(
            minimal_atom_dict.items(),
            key=lambda item: int(item[0].split()[0])))
        _print_atom_function_table(minimal_atom_dict)

    _console.print("\nFunctions in the pseudominimal basis:", style="bold", markup=False)
    _print_atom_function_table(atom_dict)


def orbital_key(orb: str) -> tuple[int, int]:
    """Sort key for an orbital label (e.g. ``'2px'``) by shell then index.

    Parameters
    ----------
    orb : str
        Orbital label, e.g. ``'2px'``, ``'1s'``.

    Returns
    -------
    (int, int)
        ``(shell_order, n)`` -- angular-momentum rank then shell index.
        Malformed labels sort last (``(inf, inf)``).
    """
    import re

    shell_order = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5}
    match = re.match(r'^(\d+)([spdfghi])', orb.lower())
    if match:
        n, shell = match.groups()
        return (shell_order.get(shell, 99), int(n))
    return (float('inf'), int('inf'))


def function_labels_from_mask(mask: np.ndarray, mol) -> dict[str, list[str]]:
    """Group the AO labels selected by `mask` by atom.

    Parameters
    ----------
    mask : ndarray
        AO mask.
    mol : pyscf.gto.Mole
        Molecule object `mask` indexes into.

    Returns
    -------
    dict
        Maps ``'<atom_id> <atom_symbol>'`` to a sorted (by `orbital_key`)
        list of shell labels (e.g. ``['1s', '2s', '2px']``) selected on
        that atom.

    Raises
    ------
    ValueError
        If an AO label doesn't match the expected
        ``'<atom_id> <atom_symbol> <shell_label>'`` pattern.
    """
    from pyscf.gto.mole import cart_labels, sph_labels
    import re

    labels = []
    all_labels = np.array(cart_labels(mol)) if mol.cart else np.array(sph_labels(mol))
    for label in all_labels[mask]:
        # Split label strings of the form
        # 'Atom_idx Atom_symb sph/cart_label', e.g. '0 H 1s' or '1 O 2px'
        atom_num = label.split()[0]
        asymb = label.split()[1]
        pattern = re.compile(r'^([0-9]+[spdfgh])')
        match = pattern.match(label.split()[-1])
        if match is None:
            raise ValueError('No regex patter match found!')
        labels.append(' '.join([atom_num, asymb, match.group(1)]))
    labels = sorted(set(labels), key=lambda x: x)
    labels = [label.split() for label in labels]
    atom_dict = {}
    for label in labels:
        key = ' '.join(label[:2])
        atom_dict.setdefault(key, []).append(label[2])
    for key in atom_dict:
        atom_dict[key].sort(key=orbital_key)

    return atom_dict


def print_atomic_block_atom_header(atom: str, nfunc_per_minimal_atom: int, nfuncs: int) -> None:
    """Print one atom's header line in `adb.atomic_block_minimal_basis`'s verbose output."""
    _console.print(f"{atom=}", style="dim", markup=False)
    _console.print(f"{nfunc_per_minimal_atom=}", style="dim", markup=False)
    _console.print(f"{nfuncs=}", style="dim", markup=False)


def print_atomic_block_energies_debug(energies: np.ndarray) -> None:
    """Print one atom's local orbital energies in verbose ABD output."""
    _console.print(f"{energies=}", style="dim", markup=False)


def print_restricted_atom_orbital_summary(nocca: int, noccb: int, e_atom: np.ndarray) -> None:
    """Print a restricted atom's occupied-state count and HOMO energy in verbose ABD output."""
    _console.print(f"{nocca=}, {noccb=}", style="dim", markup=False)
    _console.print(
        f"Energy of highest orbital {e_atom[nocca-1]*27.2114} eV",
        style="dim", markup=False)


def print_unrestricted_atom_orbital_summary(nocca: int, noccb: int, e_atom: np.ndarray) -> None:
    """Print an unrestricted atom's per-spin HOMO energies in verbose ABD output."""
    _console.print(
        f"Energy of highest alpha orbital {e_atom[0, nocca-1]*27.2114} eV",
        style="dim", markup=False)
    _console.print(
        f"Energy of highest beta  orbital {e_atom[1, noccb-1]*27.2114} eV",
        style="dim", markup=False)


def print_atomic_block_state_energies(
        Qlim:       int,
        e_atom:     np.ndarray,
        nocca:      int,
        noccb:      int,
        restricted: bool,
        ) -> None:
    """Print an atom's bound/occupied state energies in verbose ABD output."""
    _console.print(f"{Qlim=}", style="dim", markup=False)
    with np.printoptions(precision=2, suppress=True):
        if restricted:
            _console.print(
                f"Bound state energies [eV]: {e_atom[e_atom<0]*27.2114}",
                style="dim", markup=False)
            _console.print(
                f"Occupied state energies [eV]: {e_atom[:nocca]*27.2114}",
                style="dim", markup=False)
        else:
            _console.print(
                f"Bound alpha state energies [eV]: {e_atom[0, e_atom[0,:]<0]*27.2114}",
                style="dim", markup=False)
            _console.print(
                f"Bound beta  state energies [eV]: {e_atom[1, e_atom[1,:]<0]*27.2114}",
                style="dim", markup=False)
            _console.print(
                f"Occupied alpha state energies [eV]: {e_atom[0, :nocca]*27.2114}",
                style="dim", markup=False)
            _console.print(
                f"Occupied beta  state energies [eV]: {e_atom[1, :noccb]*27.2114}",
                style="dim", markup=False)


def print_find_subspace_start(mol) -> None:
    """Print `find_subspace`'s startup banner."""
    _console.print(f"Running find_subspace for mol {mol.atom}", style="bold cyan", markup=False)


def warn_conflicting_initialization() -> None:
    """Warn that `abd_initialization` and `initialize_by_projection` were both requested."""
    warn(
        "Both 'abd_initialization' and 'initialize_by_projection' cannot be True "
        "simultaneously.\nInitialization by projection takes precedent.")


def print_projection_initialization_message() -> None:
    """Print the STO-3G projection initialization banner."""
    _console.print(
        "--- Initializing the dual basis by minimal basis projection ---",
        style="cyan", markup=False)


def print_mask_analysis_init_header(init_method: str) -> None:
    """Print `mask_analysis`'s initialization-history section header."""
    _console.print()
    _console.print(
        Panel(f"INITIALIZATION: {init_method.upper()}", style="bold", box=box.ROUNDED, expand=False))


def print_mask_history_label(label: str, index: int) -> None:
    """Print one initialization-history label, wrapping every 10 entries."""
    _console.print(f"{label},  ", style="dim", end="", markup=False)
    if index % 10 == 0:
        _console.print()


def print_minimal_basis_summary(minimal_mask: np.ndarray, mol) -> None:
    """Print the minimal basis's function count and per-atom breakdown."""
    atom_dict = function_labels_from_mask(minimal_mask, mol)
    atom_dict = dict(sorted(
        atom_dict.items(),
        key=lambda item: int(item[0].split()[0])))
    _console.print(
        f"\n{int(np.sum(minimal_mask))} functions in the initial basis:",
        style="bold", markup=False)
    _print_atom_function_table(atom_dict)


def print_initialization_footer(num_toggled: int) -> None:
    """Print `mask_analysis`'s initialization-history section footer."""
    _console.print(f"\nNumber of toggled functions: {num_toggled}", style="bold", markup=False)
    _console.print(Panel("INITIALIZATION END", style="bold", box=box.ROUNDED, expand=False))


def print_link_shells_notice() -> None:
    """Print a notice that shell linking may add extra functions."""
    _console.print()
    _console.print("Link shells: ON", style="bold cyan", markup=False)
    _console.print(
        "Additional functions may be added due to shell linking!",
        style="dim", markup=False)


def print_subbasis_scf_not_converged_warning() -> None:
    """Print a warning that a subbasis SCF failed to converge."""
    _err_console.print(
        "Warning: The SCF did not converge in the subbasis. Results may be unreliable.",
        style="bold red", markup=False)
