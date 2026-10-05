#!/usr/bin/env python3
"""Generate the classification input: one coupling time series CSV per pathway.

Run this before scripts/run_classification.py.

For every sampled pathway, a fixed-nuclei (jobtype = NOM) tight-binding run
along the MD trajectory writes the full site Hamiltonian at every time step.
This script extracts the couplings between consecutive sites of the pathway
and writes them in the format expected by gbctp.io.read_path_csv.

Input (per pathway, in <base_dir>/subdir_*/TRAJ1/):
    charge-transfer.dat    its "sites = ..." line gives the site IDs along the
                           pathway, source -> target
    TB_HAMILTONIAN.xvg     column 1 = time; then, for each site k = 1..N, the
                           site energy H_kk followed by the couplings
                           H_k,k+1 ... H_k,N

Output (same directory):
    path_couplings_timeseries.csv
        row 1   : "sites", followed by the N site IDs
        row 2   : "time", "cpl(1,2)", "cpl(2,3)", ..., "cpl(N-1,N)"
        rows 3+ : time and the signed coupling of each consecutive pair,
                  in the units of TB_HAMILTONIAN.xvg

Usage:
    python scripts/generate_coupling_csvs.py /path/to/base_dir
    python scripts/generate_coupling_csvs.py --base-dir /path/to/base_dir --workers 8
"""

import os
import csv
import glob
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed


# ============================================================================
# FILE I/O FUNCTIONS
# ============================================================================

def read_charge_transfer(path):
    """
    Read charge_transfer.dat as raw non-empty lines.
    
    Parameters
    ----------
    path : str
        Path to charge-transfer.dat file
        
    Returns
    -------
    list of str
        Non-empty lines from the file
    """
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def read_tb_hamiltonian(path):
    """
    Read TB_HAMILTONIAN.xvg as raw lines.
    Keeps all lines (including comments starting with @ or #).
    
    Parameters
    ----------
    path : str
        Path to TB_HAMILTONIAN.xvg file
        
    Returns
    -------
    list of str
        All lines from the file
    """
    with open(path, "r") as f:
        lines = [line.rstrip("\n") for line in f]
    return lines


# ============================================================================
# PARSING FUNCTIONS
# ============================================================================

def parse_sites_from_charge_transfer(ct_lines):
    """
    Find the line starting with 'sites' and return the list of site indices
    in the order they appear.

    Example line:
        sites = 1276 1834 2150 ...

    Parameters
    ----------
    ct_lines : list of str
        Lines from charge_transfer.dat
        
    Returns
    -------
    list of int
        Site indices [1276, 1834, 2150, ...]
        
    Raises
    ------
    ValueError
        If no 'sites' line is found
    """
    for line in ct_lines:
        if line.strip().startswith("sites"):
            parts = line.split("=")
            if len(parts) < 2:
                continue
            right = parts[1]
            site_ids = [int(x) for x in right.split()]
            return site_ids
    
    raise ValueError("No 'sites' line found in charge_transfer.dat")


def get_coupling_column_index(N, i, j):
    """
    Given total number of sites N and a pair (i, j) with 1-based indices and j > i,
    return the 0-based column index in the TB_HAMILTONIAN.xvg data array.

    Column layout:
      col 0: time
      then for each row index k = 1..N:
        energy kk
        couplings k(k+1), k(k+2), ..., kN
        
    Parameters
    ----------
    N : int
        Total number of sites
    i : int
        First site index (1-based)
    j : int
        Second site index (1-based), must be > i
        
    Returns
    -------
    int
        0-based column index for coupling (i,j)
    """
    assert 1 <= i < j <= N, "Require 1 <= i < j <= N"

    col = 1  # start after time column

    # Skip whole blocks for rows k < i
    for k in range(1, i):
        block_size = 1 + (N - k)  # energy + couplings to higher indices
        col += block_size

    # Now at start of row 'i' block:
    # energy ii at 'col', couplings i(i+1)...iN follow
    # coupling i-j is offset (j - i) from the energy column
    col_ij = col + (j - i)
    return col_ij


def get_consecutive_coupling_columns(N):
    """
    Return the list of column indices for consecutive couplings (1,2), (2,3), ..., (N-1,N).
    
    Parameters
    ----------
    N : int
        Total number of sites
        
    Returns
    -------
    list of int
        Column indices for consecutive couplings
    """
    cols = []
    for i in range(1, N):
        j = i + 1
        cols.append(get_coupling_column_index(N, i, j))
    return cols


def parse_xvg_numeric(ham_lines):
    """
    Parse TB_HAMILTONIAN.xvg lines into a numeric numpy array.
    Skips comment/metadata lines starting with '@' or '#'.
    
    Parameters
    ----------
    ham_lines : list of str
        Lines from TB_HAMILTONIAN.xvg
        
    Returns
    -------
    np.ndarray
        Numeric data array
        
    Raises
    ------
    ValueError
        If no numeric data is found
    """
    data = []
    for line in ham_lines:
        line = line.strip()
        if not line or line.startswith(("#", "@")):
            continue
        parts = line.split()
        row = [float(x) for x in parts]
        data.append(row)
    
    if not data:
        raise ValueError("No numeric data found in TB_HAMILTONIAN.xvg")
    
    return np.array(data)


def extract_time_and_consecutive_couplings(ham_lines, n_sites):
    """
    Given the raw TB_HAMILTONIAN.xvg lines and number of sites in this path,
    extract time and consecutive couplings.
    
    Parameters
    ----------
    ham_lines : list of str
        Lines from TB_HAMILTONIAN.xvg
    n_sites : int
        Number of sites in the path
        
    Returns
    -------
    time : np.ndarray
        Time values, shape (n_frames,)
    cpls : np.ndarray
        Coupling values, shape (n_frames, n_sites-1) for (1-2, 2-3, ..., (N-1)-N)
    """
    data = parse_xvg_numeric(ham_lines)
    time = data[:, 0]

    cpl_cols = get_consecutive_coupling_columns(n_sites)
    cpls = data[:, cpl_cols]

    return time, cpls


# ============================================================================
# TRAJECTORY DISCOVERY
# ============================================================================

def find_traj_dirs(base_dir):
    """
    Search for TRAJ* directories inside subdir_* folders
    that contain BOTH:
      - charge-transfer.dat
      - TB_HAMILTONIAN.xvg

    Parameters
    ----------
    base_dir : str
        Base directory containing subdir_* folders
        
    Returns
    -------
    list of dict
        Each dict contains:
        - 'traj_dir': path to TRAJ directory
        - 'charge_transfer': path to charge-transfer.dat
        - 'hamiltonian': path to TB_HAMILTONIAN.xvg
    """
    traj_entries = []

    # Find all subdir_* directories
    subdirs = sorted(glob.glob(os.path.join(base_dir, "subdir_*")))
    
    for subdir in subdirs:
        # Look for TRAJ* directories inside each subdir
        traj_dirs = glob.glob(os.path.join(subdir, "TRAJ1"))
        
        for traj_dir in traj_dirs:
            ct_file = os.path.join(traj_dir, "charge-transfer.dat")
            ham_file = os.path.join(traj_dir, "TB_HAMILTONIAN.xvg")

            if os.path.isfile(ct_file) and os.path.isfile(ham_file):
                traj_entries.append({
                    "traj_dir": traj_dir,
                    "charge_transfer": ct_file,
                    "hamiltonian": ham_file,
                })

    return traj_entries


# ============================================================================
# CSV GENERATION (PARALLEL)
# ============================================================================

def save_path_couplings_csv(traj_entry):
    """
    For a given TRAJ entry dict, parse sites and TB_HAMILTONIAN.xvg,
    then write a CSV: traj_dir/path_couplings_timeseries.csv
    
    This function is designed to be called in parallel by ProcessPoolExecutor.
    Each trajectory writes to its own unique file, so no data mixing occurs.
    
    Parameters
    ----------
    traj_entry : dict
        Dictionary with keys:
        - 'traj_dir': path to TRAJ directory
        - 'charge_transfer': path to charge-transfer.dat
        - 'hamiltonian': path to TB_HAMILTONIAN.xvg
        
    Returns
    -------
    str
        Path to the generated CSV file
        
    Raises
    ------
    Exception
        If parsing or file writing fails
    """
    # Read raw files
    ct_lines = read_charge_transfer(traj_entry["charge_transfer"])
    ham_lines = read_tb_hamiltonian(traj_entry["hamiltonian"])

    # Parse sites from charge_transfer.dat
    sites = parse_sites_from_charge_transfer(ct_lines)
    n_sites = len(sites)

    # Extract time and couplings for (1-2, 2-3, ..., (N-1)-N)
    time, cpls = extract_time_and_consecutive_couplings(ham_lines, n_sites)

    # Prepare CSV path (unique per trajectory)
    csv_path = os.path.join(traj_entry["traj_dir"], "path_couplings_timeseries.csv")

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Row 1: sites
        writer.writerow(["sites"] + sites)

        # Row 2: header for time and couplings
        header = ["time"]
        for i in range(1, n_sites):
            header.append(f"cpl({i},{i+1})")
        writer.writerow(header)

        # Data rows
        for t_val, cpl_row in zip(time, cpls):
            writer.writerow([t_val] + list(cpl_row))

    return csv_path


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Main execution function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate one coupling time series CSV per pathway (input of run_classification.py).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/base_dir
  %(prog)s --base-dir /path/to/base_dir --workers 8
  %(prog)s -d /path/to/base_dir -w 4

Note: Each trajectory writes to its own unique CSV file, ensuring no data mixing.
        """
    )
    parser.add_argument(
        "base_dir",
        nargs="?",
        type=str,
        help="Base directory containing subdir_*/TRAJ1 folders"
    )
    parser.add_argument(
        "-d", "--base-dir",
        dest="base_dir_flag",
        type=str,
        help="Base directory (alternative to positional argument)"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )
    
    args = parser.parse_args()
    
    # Get base_dir from either positional or flag argument
    base_dir = args.base_dir or args.base_dir_flag
    
    if not base_dir:
        parser.error("base_dir is required (either as positional argument or with --base-dir)")
    
    if not os.path.isdir(base_dir):
        parser.error(f"Directory does not exist: {base_dir}")
    
    n_workers = args.workers
    
    print("=" * 70)
    print("COUPLING TIMESERIES CSV GENERATOR")
    print("=" * 70)
    print(f"Base directory: {base_dir}")
    print(f"Number of workers: {n_workers}")
    print()
    
    # ========================================================================
    # STEP 1: Find TRAJ directories
    # ========================================================================
    print("Step 1: Finding TRAJ directories...")
    subdirs = sorted(glob.glob(os.path.join(base_dir, "subdir_*")))
    print(f"  Found {len(subdirs)} subdir_* directories")
    
    traj_entries = find_traj_dirs(base_dir)
    print(f"  Found {len(traj_entries)} TRAJ directories with both required files")
    
    if not traj_entries:
        print("\nNo valid TRAJ directories found. Exiting.")
        return
    
    if traj_entries:
        print(f"  Example: {traj_entries[0]['traj_dir']}")
    print()
    
    # ========================================================================
    # STEP 2: Generate CSV files using parallel processing
    # ========================================================================
    print(f"Step 2: Generating CSV files (using {n_workers} workers)...")
    csv_paths = []
    failed_count = 0
    
    n_workers = min(n_workers, len(traj_entries))
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(save_path_couplings_csv, entry): entry 
            for entry in traj_entries
        }
        
        for i, future in enumerate(as_completed(futures), 1):
            entry = futures[future]
            try:
                csv_path = future.result()
                csv_paths.append(csv_path)
                if i % 50 == 0 or i == len(traj_entries):
                    print(f"  Processed {i}/{len(traj_entries)} trajectories...")
            except Exception as e:
                failed_count += 1
                print(f"  ERROR in {entry['traj_dir']}: {e}")
    
    print(f"  Successfully wrote {len(csv_paths)} CSV files")
    if failed_count > 0:
        print(f"  Failed: {failed_count} trajectories")
    print()
    
    # ========================================================================
    # DONE
    # ========================================================================
    print("=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total CSVs generated: {len(csv_paths)}")
    if failed_count > 0:
        print(f"Failed: {failed_count} trajectories")
    print()
    print("Each trajectory now has a 'path_couplings_timeseries.csv' file.")
    print()


if __name__ == "__main__":
    main()
