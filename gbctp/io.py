"""Reading per-pathway coupling time series from disk.

Expected input file (one CSV per sampled pathway)
---------------------------------------------------
Row 1   : "sites", followed by the site/dimer-endpoint IDs along the path,
          in source -> target order (length n).
Row 2   : column header: "time" followed by one "cpl(i,i+1)"-style column
          per dimer (n - 1 columns total).
Row 3+  : the numeric coupling time series -- one row per MD/QM snapshot.

Example (n = 4 sites -> 3 dimers)::

    sites,1276,1834,2210,3007
    time,"cpl(1,2)","cpl(2,3)","cpl(3,4)"
    0.0,0.021,-0.034,0.018
    0.5,0.019,-0.031,0.020
    ...

Couplings may be signed; this module returns them as-is and leaves taking
the absolute value to the descriptor functions.
"""

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


def read_path_csv(csv_path: Path) -> Tuple[List[int], np.ndarray, np.ndarray, List[str]]:
    """Read one path-coupling CSV file (see module docstring for the format).

    Parameters
    ----------
    csv_path : Path
        Path to a single pathway's coupling time series CSV.

    Returns
    -------
    sites : list[int]
        Site IDs along the path (length n).
    time : np.ndarray, shape (Nt,)
        Time (or snapshot) values.
    X : np.ndarray, shape (Nt, n - 1)
        Signed pairwise couplings, one column per dimer.
    cpl_cols : list[str]
        Names of the coupling columns, in path order.
    """
    csv_path = Path(csv_path)

    first = pd.read_csv(csv_path, header=None, nrows=1).iloc[0].tolist()
    if len(first) < 3 or str(first[0]).strip().lower() != "sites":
        raise ValueError(f"{csv_path}: first row must start with 'sites,'")

    sites = [int(x) for x in first[1:]]

    df = pd.read_csv(csv_path, skiprows=1)
    if "time" not in df.columns:
        raise ValueError(f"{csv_path}: second row must include a 'time' column")

    time = df["time"].to_numpy(float)
    cpl_cols = [c for c in df.columns if c != "time"]
    X = df[cpl_cols].to_numpy(float)

    n = len(sites)
    if X.shape[1] != n - 1:
        raise ValueError(
            f"{csv_path}: sites={n} implies {n - 1} dimers, "
            f"but found {X.shape[1]} coupling columns"
        )

    return sites, time, X, cpl_cols


def find_path_files(root_dir: Path, pattern: str) -> List[Path]:
    """Glob for per-pathway coupling CSVs under ``root_dir``.

    Parameters
    ----------
    root_dir : Path
        Directory containing one subfolder (or file) per sampled pathway.
    pattern : str
        Glob pattern relative to ``root_dir``, e.g.
        ``"*/TRAJ1/path_couplings_timeseries.csv"``.

    Returns
    -------
    list[Path]
        Sorted list of matching files. Raises ValueError if none are found.
    """
    root_dir = Path(root_dir)
    csv_files = sorted(root_dir.glob(pattern))
    if not csv_files:
        raise ValueError(f"No CSV files found under {root_dir} matching pattern {pattern!r}")
    return csv_files


def write_per_dimer_vectors(csv_path: Path, sites: Sequence[int], J, C, E) -> Path:
    """Save per-dimer J/C/E next to the source CSV, as ``<name>.JCE_vectors.csv``.

    Consumed downstream by ``gbctp.reliability.load_E_profile`` and by the
    optional 3D bottleneck-visualization notebook cell.
    """
    csv_path = Path(csv_path)
    out = pd.DataFrame(
        {
            "dimer_i": np.arange(1, len(J) + 1),
            "site_i": list(sites[:-1]),
            "site_j": list(sites[1:]),
            "J_abs_mean": J,
            "C_coherence": C,
            "E_JxC": E,
        }
    )
    out_path = csv_path.with_suffix(".JCE_vectors.csv")
    out.to_csv(out_path, index=False)
    return out_path
