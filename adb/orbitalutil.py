"""Occupied-orbital extraction helpers shared by adb.search and adb.analysis"""

import numpy as np
from pyscf import symm


def get_occupied_orbitals(
        epsilon_i:      np.ndarray,
        nocc:           tuple,
        irrep_nelec:    dict | None         = None,
        orbsym:         np.ndarray | None   = None,
        restricted:     bool                = True,
        ) -> list:
    """Extract the occupied orbital set implied by the same selection rule
    get_iteration_criteria_value's 'enocc' branch (and _enocc_by_irrep)
    use to compute their criterion sum -- but returning the individual
    selected (energy, irrep) pairs instead of just their sum.

    Symmetry-blind (irrep_nelec/orbsym not given): the lowest nocc[0]
    (restricted) or nocc[0]/nocc[1] (unrestricted) eigenvalues overall.

    Symmetry-aware (irrep_nelec/orbsym given): the lowest target-count
    eigenvalues *within each irrep*, per pyscf's mf.irrep_nelec convention
    (int per irrep for restricted, (n_alpha, n_beta) tuple for
    unrestricted) -- mirroring _enocc_by_irrep's selection exactly, minus
    its shortfall-penalty bookkeeping (irrelevant here: this is only ever
    called on an already-accepted mask/step, where by construction every
    targeted irrep has enough capacity).

    Args:
        epsilon_i : ndarray
            Orbital energies, as returned by adb.eig/symmetrized_eig.
            Shape (nmo,) for restricted, (2, nmo) for unrestricted.
        nocc : tuple
            (n_alpha, n_beta) occupied counts, used only when irrep_nelec
            is None.
        irrep_nelec : dict | None
            Target occupation per irrep name (pyscf mf.irrep_nelec
            format). When given, `orbsym` must be given too.
        orbsym : ndarray | None
            Irrep name (string) for each entry along epsilon_i's last
            axis, as returned by diagonalize_masked/symmetrized_eig (after
            translating irrep ids to names).
        restricted : bool
            Whether epsilon_i is restricted (2D h) or unrestricted-shaped.

    Returns:
        List of (energy, irrep_label) tuples, one per occupied orbital.
        irrep_label is None throughout when irrep_nelec/orbsym are not
        given (symmetry-blind case).
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


def get_occupied_orbitals_from_scf(mf) -> list:
    """Extract the occupied orbital energies and (if mf.mol.symmetry is
    enabled) their symmetry labels from a converged pyscf SCF object.

    Companion to get_occupied_orbitals: that one works from a raw
    (epsilon_i, orbsym) pair produced during the ADB search itself, before
    any SCF exists (a fixed guess Fock matrix, not self-consistent). This
    one instead reads mo_energy/mo_occ/mo_coeff straight off a *converged*
    mf object -- used by mask_analysis's track_orbitals to record the
    genuinely self-consistent occupied-orbital spectrum for each subbasis,
    as opposed to find_subspace/expand_mask's guess-Fock-matrix spectrum.

    Returns a list of (energy, irrep_label) tuples, one per occupied MO.
    irrep_label is None throughout when mf.mol.symmetry is off/C1.
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
