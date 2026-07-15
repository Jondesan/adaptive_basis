"""adb: adaptive basis set method.

Core algorithmic API is available directly off this package, e.g.
adb.find_subspace(...). Auxiliary functionality (psi4/basis-exchange
I/O helpers, the legacy atomic-block workflow) is not hoisted here --
import the submodule directly: import adb.adbutils, import
adb.atomic_block_util.
"""

from .core import (
    get_sub_scf_attributes,
    get_occupied_orbitals,
    get_occupied_orbitals_from_scf,
    atomic_block_minimal_basis,
    find_projected_minimal_basis_mask,
    find_subspace,
    expand_mask,
    mask_analysis,
)

from .calculations import (
    eig,
    canonical_orth,
    symmetrized_eig,
    get_iteration_criteria_value,
    get_q_sqrd,
    diagonalize_masked,
    spherical_average,
    sph_avg,
    dual_basis_energy_correction,
)

from .maskutil import (
    init_smask,
    smask_to_mask,
    mask_to_smask,
    linked_shell_idx,
    get_all_shell_labels,
    link_shells,
    get_atom_shell_label,
    print_shells,
    maskidx_to_smaskidx,
    set_linked_shells,
    mask_matrix,
)

from .molutil import (
    create_shell_separated_mol,
    get_shells,
    basis_functions_per_atom,
    create_subbasis_mol,
    get_array_of_angular_momenta_and_atom_id,
    funcs_on_shell,
)

from .basisutil import (
    get_uncontracted_basis,
    get_basis_dict,
    basis_to_file_nwchem,
    extract_basis,
)

from .ioutil import (
    write_orbital_history,
    print_data_header,
    print_data,
    print_labels_of_functions_in_mask,
    orbital_key,
    function_labels_from_mask,
)

from . import CONSTANTS
from .CONSTANTS import (
    VARIANTS, NFUNCS, ANGULAR, ELEMENTS,
    SYMMETRY_SHORTFALL_PENALTY, EXPAND_MASK_EPS,
)

# adbutils / atomic_block_util are intentionally NOT imported here --
# reach them via `import adb.adbutils` / `import adb.atomic_block_util`.
