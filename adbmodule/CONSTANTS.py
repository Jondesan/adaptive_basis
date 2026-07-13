
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

ANGULAR = "spdfghik"

ELEMENTS = [
    'X',  # Ghost
    'H' , 'He', 'Li', 'Be', 'B' , 'C' , 'N' , 'O' , 'F' , 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P' , 'S' , 'Cl', 'Ar', 'K' , 'Ca',
    'Sc', 'Ti', 'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y' , 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I' , 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U' , 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',
]


# Penalty (in Hartree) added per electron slot that a target irrep cannot
# yet hold in a partially-grown trial subbasis, used by the optional
# symmetry-aware 'enocc' criterion in get_iteration_criteria_value. Chosen
# to be many orders of magnitude larger than any physically meaningful
# orbital-energy sum, so the greedy search always prioritises adding shells
# of an under-represented irrep before anything else.
SYMMETRY_SHORTFALL_PENALTY = 1e3