"""Workarounds for pyscf bugs affecting this project's SCF usage.

Contains two fixes, both applied by `symmetry_safe_newton` to whatever
`pyscf.scf.newton()` returns:

1. Restricted (RHF-like): pyscf's Newton/SOSCF solver (`mf.newton()`)
   crashes when `mol.symmetry` is enabled together with a mean-field object
   whose basis has been reduced relative to `mol.nao` -- e.g. via pyscf's
   built-in linear dependency removal on a large/diffuse basis where some AO
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
   mismatch. `_symmetry_safe_rotate_mo` fixes this by only symmetry-labeling
   when `mo_coeff` is actually AO-basis-shaped.

2. Unrestricted (UHF): pyscf's own `_SecondOrderUHF.rotate_mo` already
   composes each spin channel separately --

       mo = numpy.asarray((numpy.dot(mo_coeff[0], u[0]),
                            numpy.dot(mo_coeff[1], u[1])))

   -- since `mo_coeff`/`u` are spin-stacked `(2, n, n)` arrays for UHF, and a
   plain `numpy.dot` on those does *not* perform the intended per-spin
   matrix product (it produces a `(2, n, 2, n)` outer-product-shaped result
   instead). Before this fix, `symmetry_safe_newton` unconditionally
   installed the restricted-only `_symmetry_safe_rotate_mo` (plain
   `numpy.dot`) regardless of SCF flavor, discarding pyscf's own correct
   UHF `rotate_mo` and silently corrupting -- or, once `nao != nmo`, flatly
   crashing -- every orbital rotation step for any UHF+symmetry Newton run
   (e.g. any open-shell system with `mol.symmetry` enabled going through
   `adb.symmetry_safe_newton`, such as `run.py`'s full-basis warmup).
   `_symmetry_safe_rotate_mo_uhf` mirrors pyscf's own per-spin composition
   and `uhf_symm.get_orbsym` labeling, with the same is_ao_basis guard as
   fix 1.

This module patches only `rotate_mo` on the object `pyscf.scf.newton`
returns, reusing all of pyscf's actual second-order optimizer machinery
unchanged -- it does not reimplement any SCF/optimization logic.
"""

import types
from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf_symm, uhf_symm
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


def _symmetry_safe_rotate_mo_uhf(self, mo_coeff, u, log=None):
    """UHF counterpart of `_symmetry_safe_rotate_mo` -- see fix 2 in this
    module's docstring. Mirrors pyscf's own `_SecondOrderUHF.rotate_mo`
    (per-spin-channel composition, `uhf_symm.get_orbsym` labeling), adding
    the same is_ao_basis guard as the restricted fix so symmetry-labeling
    is skipped for bare (2, nmo, nmo) rotation-composition matrices instead
    of crashing/mislabeling them.
    """
    mo = numpy.asarray((numpy.dot(mo_coeff[0], u[0]),
                         numpy.dot(mo_coeff[1], u[1])))
    is_ao_basis = mo_coeff[0].shape[0] == self._scf.mol.nao

    if is_ao_basis and self._scf.mol.symmetry:
        orbsym = uhf_symm.get_orbsym(self._scf.mol, mo_coeff)
        mo = lib.tag_array(mo, orbsym=orbsym)

    return mo


def symmetry_safe_newton(mf):
    """Drop-in replacement for `mf.newton()` / `pyscf.scf.newton(mf)`.

    Use this instead of calling `.newton()` directly whenever `mf.mol` may
    have `symmetry` enabled together with a basis-reducing linear dependency
    removal built-into pyscf (fix 1), or whenever `mf` is UHF and
    `mf.mol.symmetry` is enabled (fix 2) -- see this module's docstring.

    Args:
        mf : pyscf.scf.hf.SCF
            The mean-field object to wrap, exactly as you would pass to
            `mf.newton()`.

    Returns:
        The same SOSCF-decorated object `pyscf.scf.newton(mf)` would
        return, with `rotate_mo` patched to the corrected version above
        (dispatched by SCF flavor: `_symmetry_safe_rotate_mo_uhf` for UHF,
        `_symmetry_safe_rotate_mo` otherwise).
    """
    mf1 = newton_ah.newton(mf)
    if mf1.istype('UHF'):
        mf1.rotate_mo = types.MethodType(_symmetry_safe_rotate_mo_uhf, mf1)
    else:
        mf1.rotate_mo = types.MethodType(_symmetry_safe_rotate_mo, mf1)
    return mf1
