import psi4
import subprocess
from contextlib import redirect_stdout
from contextlib import contextmanager
import ctypes
import io
import os, sys
import tempfile
import numpy as np

libc = ctypes.CDLL(None)
c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')

@contextmanager
def stdout_redirector(stream):
    # The original fd stdout points to. Usually 1 on POSIX systems.
    original_stdout_fd = sys.stdout.fileno()

    def _redirect_stdout(to_fd):
        """Redirect stdout to the given file descriptor."""
        # Flush the C-level buffer stdout
        libc.fflush(c_stdout)
        # Flush and close sys.stdout - also closes the file descriptor (fd)
        sys.stdout.close()
        # Make original_stdout_fd point to the same file as to_fd
        os.dup2(to_fd, original_stdout_fd)
        # Create a new sys.stdout that points to the redirected fd
        sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))

    # Save a copy of the original stdout fd in saved_stdout_fd
    saved_stdout_fd = os.dup(original_stdout_fd)
    try:
        # Create a temporary fileand redirect stdout to it
        tfile = tempfile.TemporaryFile(mode='w+b')
        _redirect_stdout(tfile.fileno())
        # Yield to caller, then redirect stdout back to the saved fd
        yield
        _redirect_stdout(saved_stdout_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        stream.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_stdout_fd)

if __name__ == '__main__':
    prefix = '/home/joonahuh/uni/electronic_structure/'
    with open(prefix + 'benchmarks/pom_geom/weigend_ahlrichs/new_geoms/li2o.spin0.init_atom.xyz') as f:
        xyz = f.read()
    # with open(prefix + 'abs/bas_output_basis.gbs') as f:
    #     basis = f.read()
    mol = psi4.geometry(xyz)
    mol.set_units(psi4.core.GeometryUnits(1))
    mol.set_multiplicity(1)

    #psi4.set_options({'basis': prefix + 'abs/bas_output_basis.gbs'})

    f = io.BytesIO()
    converged = False
    e_tot, wfn = 0.0, None
    with stdout_redirector(f):
        try:
            e_tot, wfn = psi4.energy(
                'PBE',
                basis='li2o_subbasis',
                # basis='def2-TZVP',
                return_wfn=True)
            converged = True
        except:
            pass

    if not converged:
        psi4output = f.getvalue().decode('utf-8').split('\n')
        # Filter lines with DOCC
        psi4output = list(filter(lambda x: 'DOCC' in x, psi4output))
        unique_docc = list(set(psi4output))

        print('Found the following symmetries:\n','\n'.join(unique_docc))
        print('Testing which provides lowest converged energy...')
        doccs = []
        for docc in unique_docc:
            docc = ''.join(docc.split()[1:])
            docc = docc.translate({ord(c): None for c in '[]'}) # Remove '[' and ']'
            
            docc = np.fromstring(docc, dtype=int, sep=',')

            psi4.set_options({'DOCC': list(docc)})
            f = io.BytesIO()
            with stdout_redirector(f):
                e_tot_docc, wfn_docc = psi4.energy(
                    'PBE',
                    basis='li2o_subbasis',
                    # basis='def2-TZVP',
                    return_wfn=True)
            doccs.append((docc, e_tot_docc))
            if e_tot_docc < e_tot:
                e_tot = e_tot_docc
                wfn = wfn_docc
        doccs.sort(key=lambda x: x[1])
        print(doccs)

    print(e_tot)