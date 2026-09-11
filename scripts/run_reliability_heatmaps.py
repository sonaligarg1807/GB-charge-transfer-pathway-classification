#!/usr/bin/env python
"""CLI entry point for the bootstrap/threshold-perturbation reliability checks
(gbctp.reliability), matching notebooks/03_reliability_heatmaps.ipynb.

Reads the paths_order_descriptors_with_archetype_labels.csv produced by
run_classification.py / gbctp.pipeline.run_pipeline, and writes the
label-retention matrices used by that notebook's heat-map plotting cells
(this script does not produce any plots itself).

Example
-------
    python scripts/run_reliability_heatmaps.py \\
        --order-descriptor-file /path/to/outputs/paths_order_descriptors_with_archetype_labels.csv \\
        --output-dir /path/to/outputs/results
"""

import argparse
from pathlib import Path

import pandas as pd

from gbctp.reliability import bootstrap_classification_convergence, run_combined_bootstrap_threshold_analysis


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--order-descriptor-file", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--n-bootstrap", type=int, default=300)
    p.add_argument("--n-perturb", type=int, default=30, help="Threshold perturbations per bootstrap sample (Part B).")
    p.add_argument(
        "--skip-part-b", action="store_true",
        help="Only run Part A (bootstrap convergence); skip the slower Part B (bootstrap + threshold perturbation).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.order_descriptor_file)

    print("Part A: bootstrap convergence...")
    stability_df = bootstrap_classification_convergence(df, n_bootstrap=args.n_bootstrap, random_seed=args.random_seed)
    stability_df.to_csv(args.output_dir / "classification_convergence_values.csv", index=False)
    print(stability_df.round(3).to_string(index=False))

    if not args.skip_part_b:
        print("\nPart B: bootstrap + threshold perturbation (slower)...")
        stability_mean_df, stability_std_df, all_results_df = run_combined_bootstrap_threshold_analysis(
            df, n_bootstrap=args.n_bootstrap, n_perturb=args.n_perturb, random_seed=args.random_seed,
        )
        stability_mean_df.to_csv(args.output_dir / "combined_stability_mean_matrix.csv")
        stability_std_df.to_csv(args.output_dir / "combined_stability_std_matrix.csv")
        all_results_df.to_csv(args.output_dir / "bootstrap_threshold_all_retention_values.csv", index=False)

    print(f"\nResults written to: {args.output_dir}")


if __name__ == "__main__":
    main()
