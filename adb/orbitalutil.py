"""Occupied-orbital extraction helpers shared by adb.search and adb.analysis."""

import numpy as np
from pyscf import symm


def get_occupied_orbitals(
        epsilon_i:      np.ndarray,
        nocc:           tuple,
        irrep_nelec:    dict | None         = None,
        orbsym:         np.ndarray | None   = None,
        restricted:     bool                = True,
        ) -> list[tuple[float, str | None]]:
    """Extract the occupied orbitals implied by the 'enocc' selection rule.

    Uses the same selection `adb.calculations.get_iteration_criteria_value`'s
    ``'enocc'`` branch (and `_enocc_by_irrep`) use to compute their
    criterion sum, but returns the individual selected ``(energy, irrep)``
    pairs instead of just their sum.

    Symmetry-blind (`irrep_nelec`/`orbsym` not given): the lowest
    ``nocc[0]`` (restricted) or ``nocc[0]``/``nocc[1]`` (unrestricted)
    eigenvalues overall.

    Symmetry-aware (`irrep_nelec`/`orbsym` given): the lowest
    target-count eigenvalues *within each irrep*, per pyscf's
    ``mf.irrep_nelec`` convention (int per irrep for restricted,
    ``(n_alpha, n_beta)`` tuple for unrestricted) -- mirroring
    `adb.calculations._enocc_by_irrep`'s selection exactly, minus its
    shortfall-penalty bookkeeping (irrelevant here: this is only ever
    called on an already-accepted mask/step, where by construction every
    targeted irrep has enough capacity).

    Parameters
    ----------
    epsilon_i : ndarray, shape (nmo,) or (2, nmo)
        Orbital energies, as returned by `adb.eig`/`adb.symmetrized_eig`.
    nocc : tuple
        ``(n_alpha, n_beta)`` occupied counts, used only when
        `irrep_nelec` is `None`.
    irrep_nelec : dict, optional
        Target occupation per irrep name (pyscf ``mf.irrep_nelec``
        format). When given, `orbsym` must be given too.
    orbsym : ndarray, optional
        Irrep name (string) for each entry along `epsilon_i`'s last axis,
        as returned by `adb.diagonalize_masked`/`adb.symmetrized_eig`
        (after translating irrep ids to names).
    restricted : bool, default True
        Whether `epsilon_i` is restricted-shaped.

    Returns
    -------
    list of (float, str or None)
        One ``(energy, irrep_label)`` tuple per occupied orbital.
        `irrep_label` is `None` throughout when `irrep_nelec`/`orbsym` are
        not given (symmetry-blind case).
    """
    occupied = []
    if irrep_nelec is not None:
        if orbsym is None:
            raise ValueError(
                "'orbsym' must be provided together with 'irrep_nelec'.")
        for irname, target in irrep_nelec.items():
            spin_targets = [(None, target // 2)] if restricted else \
                [(0, target[0]), (1, target[1])]
            for spin, n_need in spin_targets:
                if n_need == 0:
                    continue
                e_ir = epsilon_i[orbsym == irname] if spin is None \
                    else epsilon_i[spin][orbsym == irname]
                idx = np.argsort(np.real(e_ir))[:n_need]
                for e in np.real(e_ir)[idx]:
                    occupied.append((float(e), irname))
    elif restricted:
        for e in epsilon_i[:nocc[0]]:
            occupied.append((float(np.real(e)), None))
    else:
        for e in epsilon_i[0][:nocc[0]]:
            occupied.append((float(np.real(e)), None))
        for e in epsilon_i[1][:nocc[1]]:
            occupied.append((float(np.real(e)), None))
    return occupied


def get_occupied_orbitals_from_scf(mf) -> list[tuple[float, str | None]]:
    """Extract occupied orbital energies and symmetry labels from a converged SCF.

    Companion to `get_occupied_orbitals`: that one works from a raw
    ``(epsilon_i, orbsym)`` pair produced during the ADB search itself,
    before any SCF exists (a fixed guess Fock matrix, not self-consistent).
    This one instead reads ``mo_energy``/``mo_occ``/``mo_coeff`` straight
    off a *converged* mean-field object -- used by
    `adb.mask_analysis`'s ``track_orbitals`` to record the genuinely
    self-consistent occupied-orbital spectrum for each subbasis, as
    opposed to `adb.find_subspace`/`adb.expand_mask`'s guess-Fock-matrix
    spectrum.

    Parameters
    ----------
    mf : pyscf.scf.hf.SCF
        A converged mean-field object.

    Returns
    -------
    list of (float, str or None)
        One ``(energy, irrep_label)`` tuple per occupied MO. `irrep_label`
        is `None` throughout when ``mf.mol.symmetry`` is off/C1.
    """
    mol = mf.mol
    has_symmetry = bool(mol.symmetry) and mol.groupname != 'C1'
    restricted = (np.asarray(mf.mo_occ, dtype=object).ndim == 1)

    def _labels(mo_coeff):
        if not has_symmetry:
            return [None] * mo_coeff.shape[1]
        return list(symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo_coeff))

    occupied = []
    if restricted:
        for e, occ, lbl in zip(mf.mo_energy, mf.mo_occ, _labels(mf.mo_coeff)):
            if occ > 0:
                occupied.append((float(e), lbl))
    else:
        for spin in (0, 1):
            for e, occ, lbl in zip(
                    mf.mo_energy[spin], mf.mo_occ[spin], _labels(mf.mo_coeff[spin])):
                if occ > 0:
                    occupied.append((float(e), lbl))
    return occupied


def get_frontier_orbitals_from_scf(
        mf, n_virtual: int = 3
        ) -> list[tuple[float, str | None, bool, str | None]]:
    """Extract occupied orbitals *and* the lowest few virtual orbitals of
    each irrep from a converged SCF -- for spotting occupied/virtual
    energy inversions within an irrep (a direct signature of a sub-basis
    SCF converged to a non-ground configuration; see
    `adaptive_basis/untracked/jagged_convergence_check/REPORT.md`'s FeO
    case for the by-hand version of exactly this check).

    Independent sibling of `get_occupied_orbitals_from_scf`, which it does
    not modify: that function's `(energy, irrep)` return shape is relied
    on by `write_orbital_history`'s CSV format, so extending it in place
    would be a breaking change. This one returns a 4th field per orbital
    instead, and is used only by the separate `cycle_report` feature.

    Parameters
    ----------
    mf : pyscf.scf.hf.SCF
        A converged mean-field object.
    n_virtual : int, default 3
        How many of the lowest-energy virtual orbitals to keep per irrep
        (or overall, if `mf.mol` has no symmetry). All occupied orbitals
        are always kept.

    Returns
    -------
    list of (float, str or None, bool, str or None)
        One ``(energy, irrep_label, is_occupied, spin_label)`` tuple per
        kept MO. `irrep_label` is `None` throughout when
        ``mf.mol.symmetry`` is off/C1. `spin_label` is `None` for a
        restricted `mf`, else ``'alpha'``/``'beta'``.
    """
    mol = mf.mol
    has_symmetry = bool(mol.symmetry) and mol.groupname != 'C1'
    restricted = (np.asarray(mf.mo_occ, dtype=object).ndim == 1)

    def _labels(mo_coeff):
        if not has_symmetry:
            return [None] * mo_coeff.shape[1]
        return list(symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo_coeff))

    def _channel(mo_energy, mo_occ, mo_coeff, spin_label):
        by_irrep_occ: dict = {}
        by_irrep_virt: dict = {}
        for e, occ, lbl in zip(mo_energy, mo_occ, _labels(mo_coeff)):
            bucket = by_irrep_occ if occ > 0 else by_irrep_virt
            bucket.setdefault(lbl, []).append(float(np.real(e)))

        kept = []
        for lbl, energies in by_irrep_occ.items():
            for e in energies:
                kept.append((e, lbl, True, spin_label))
        for lbl, energies in by_irrep_virt.items():
            for e in sorted(energies)[:n_virtual]:
                kept.append((e, lbl, False, spin_label))
        return kept

    if restricted:
        return _channel(mf.mo_energy, mf.mo_occ, mf.mo_coeff, None)
    return (_channel(mf.mo_energy[0], mf.mo_occ[0], mf.mo_coeff[0], 'alpha')
            + _channel(mf.mo_energy[1], mf.mo_occ[1], mf.mo_coeff[1], 'beta'))
