#!/usr/bin/env python
"""CLI entry point for the subsampling-convergence and threshold-sensitivity
checks (gbctp.reliability), matching notebooks/02_convergence_and_robustness.ipynb.

Reads the paths_order_descriptors_with_archetype_labels.csv produced by
run_classification.py / gbctp.pipeline.run_pipeline, and writes the
convergence and sensitivity result CSVs used by that notebook's plotting
cells (this script does not produce any plots itself).

Example
-------
    python scripts/run_convergence.py \\
        --order-descriptor-file /path/to/outputs/paths_order_descriptors_with_archetype_labels.csv \\
        --output-dir /path/to/outputs/classification_reliability_figures
"""

import argparse
from pathlib import Path

import pandas as pd

from gbctp.reliability import run_subsampling_convergence, run_threshold_sensitivity


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--order-descriptor-file", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--random-seed", type=int, default=123)
    p.add_argument("--n-boot", type=int, default=300)
    p.add_argument("--n-min", type=int, default=30)
    p.add_argument("--n-step", type=int, default=25)
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_all = pd.read_csv(args.order_descriptor_file)

    print("Running subsampling convergence...")
    df_conv, df_full, labels, full_frac, full_gb_frac = run_subsampling_convergence(
        df_all, n_min=args.n_min, n_step=args.n_step, n_boot=args.n_boot, random_seed=args.random_seed,
    )
    df_conv.to_csv(args.output_dir / "subsampling_convergence_results.csv", index=False)
    print(f"Full-dataset label counts:\n{df_full['label_primary'].value_counts()}")

    print("\nRunning threshold sensitivity...")
    df_sens, df_agree = run_threshold_sensitivity(df_all)
    df_sens.to_csv(args.output_dir / "threshold_sensitivity_class_fractions.csv", index=False)
    df_agree.to_csv(args.output_dir / "threshold_sensitivity_label_agreement.csv", index=False)

    print(f"\nResults written to: {args.output_dir}")


if __name__ == "__main__":
    main()
