"""Optional geometry loading for the 3D bottleneck-visualization notebook cell.

Requires the optional ``MDAnalysis`` dependency (``pip install gbctp[geometry]``).
Not needed for the classification pipeline itself.
"""

from pathlib import Path
from typing import Dict, Tuple


def read_gro_file(gro_path: Path) -> Dict[int, Tuple[float, float, float]]:
    """Read a .gro file and return each residue's center-of-mass coordinates (nm).

    Parameters
    ----------
    gro_path : Path
        Path to a GROMACS .gro structure file whose residue numbers match
        the ``site_i`` / ``site_j`` IDs used in the pathway coupling CSVs.

    Returns
    -------
    dict[int, tuple[float, float, float]]
        Maps residue number (site ID) -> (x, y, z) center of mass, in nm.
    """
    try:
        import MDAnalysis as mda
    except ImportError as exc:
        raise ImportError(
            "read_gro_file() requires MDAnalysis. Install it with `pip install MDAnalysis` "
            "or `pip install gbctp[geometry]`."
        ) from exc

    u = mda.Universe(str(gro_path))

    com_coords = {}
    for residue in u.residues:
        com_angstrom = residue.atoms.center_of_mass()
        com_coords[residue.resid] = tuple(com_angstrom / 10.0)  # Angstrom -> nm

    return com_coords
