
# Available criterion variants
VARIANTS = [
    'enocc', # Energy sum of occupied orbitals
    'elden', # Electron density
]

# Number of functions per spherical shell
NFUNCS = {
    'S': 1,
    'P': 3,
    'D': 5,
    'F': 7,
    'G': 9,
    'H': 11,
    'I': 13,
    'J': 15,
}

# Penalty (in Hartree) added per electron slot that a target irrep cannot
# yet hold in a partially-grown trial subbasis, used by the optional
# symmetry-aware 'enocc' criterion in get_iteration_criteria_value. Chosen
# to be many orders of magnitude larger than any physically meaningful
# orbital-energy sum, so the greedy search always prioritises adding shells
# of an under-represented irrep before anything else.
SYMMETRY_SHORTFALL_PENALTY = 1e3