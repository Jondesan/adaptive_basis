"""Workarounds for pyscf bugs affecting this project's SCF usage.

Currently contains one fix: pyscf's Newton/SOSCF solver (`mf.newton()`)
crashes when `mol.symmetry` is enabled together with a mean-field object
whose basis has been reduced relative to `mol.nao` -- e.g. via pyscf's built-in
linear dependency removal on a large/diffuse basis where some AO
combinations are genuinely (near-)linearly dependent.

Root cause (traced directly in `pyscf/soscf/newton_ah.py`'s
`_rotate_orb_cc`): its periodic "keyframe" bookkeeping composes two
sequential orbital-rotation matrices, each shape (nmo, nmo), by reusing
`rotate_mo` purely as a `numpy.dot` convenience:

    u = mf.update_rotate_matrix(dr, mo_occ, mo_coeff=mo_coeff)  # (nmo, nmo)
    if ukf is not None:
        u = mf.rotate_mo(ukf, u)                                # composes rotations

But `_CIAH_SOSCF.rotate_mo` unconditionally tries to symmetry-label its
first argument as if it were a real AO-basis MO coefficient matrix
whenever `mol.symmetry` is set:

    def rotate_mo(self, mo_coeff, u, log=None):
        mo = numpy.dot(mo_coeff, u)
        if self._scf.mol.symmetry:
            orbsym = hf_symm.get_orbsym(self._scf.mol, mo_coeff)  # <-- bug

When nao == nmo (no basis reduction) this happens to not crash, since the
overlap matrix and the bare rotation matrix are coincidentally the same
shape -- it's still conceptually wrong (labeling a rotation matrix's
"orbital symmetry" is meaningless), just not visibly so. It only raises
once nao != nmo: `pyscf.symm.label_orb_symm` tries `s @ mo_coeff` with the
full (nao, nao) overlap against the (nmo, nmo) rotation and gets a shape
mismatch.

This module patches only that one method on the object `pyscf.scf.newton`
returns, reusing all of pyscf's actual second-order optimizer machinery
unchanged -- it does not reimplement any SCF/optimization logic.
"""

import types
from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf_symm
from pyscf.soscf import newton_ah
from pyscf.soscf.newton_ah import _effective_svd


def _symmetry_safe_rotate_mo(self, mo_coeff, u, log=None):
    """Same as pyscf's `_CIAH_SOSCF.rotate_mo`, except the symmetry-label
    (and debug-overlap) steps only run when `mo_coeff` is actually a full
    AO-basis MO coefficient matrix (`mo_coeff.shape[0] == mol.nao`) rather
    than a bare (nmo, nmo) rotation matrix being composed with another one.
    See this module's docstring for why that distinction matters.
    """
    mo = numpy.dot(mo_coeff, u)
    is_ao_basis = mo_coeff.shape[0] == self._scf.mol.nao

    if is_ao_basis and self._scf.mol.symmetry:
        orbsym = hf_symm.get_orbsym(self._scf.mol, mo_coeff)
        mo = lib.tag_array(mo, orbsym=orbsym)

    if is_ao_basis and isinstance(log, logger.Logger) and log.verbose >= logger.DEBUG:
        idx = self.mo_occ > 0
        s = reduce(numpy.dot, (mo[:, idx].conj().T, self._scf.get_ovlp(),
                               self.mo_coeff[:, idx]))
        log.debug('Overlap to initial guess, SVD = %s',
                  _effective_svd(s, 1e-5))
        log.debug('Overlap to last step, SVD = %s',
                  _effective_svd(u[idx][:, idx], 1e-5))
    return mo


def symmetry_safe_newton(mf):
    """Drop-in replacement for `mf.newton()` / `pyscf.scf.newton(mf)`.

    Use this instead of calling `.newton()` directly whenever `mf.mol` may
    have `symmetry` enabled together with a basis-reducing linear dependency
    removal built-into pyscf -- that combination is exactly what
    triggers the pyscf bug this module's docstring describes.

    Args:
        mf : pyscf.scf.hf.SCF
            The mean-field object to wrap, exactly as you would pass to
            `mf.newton()`.

    Returns:
        The same SOSCF-decorated object `pyscf.scf.newton(mf)` would
        return, with `rotate_mo` patched to the corrected version above.
    """
    mf1 = newton_ah.newton(mf)
    mf1.rotate_mo = types.MethodType(_symmetry_safe_rotate_mo, mf1)
    return mf1
