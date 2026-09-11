#!/usr/bin/env python
"""CLI entry point for the pathway classification pipeline (gbctp.pipeline).

Reads one coupling-time-series CSV per sampled pathway (see gbctp.io for the
expected format), computes symmetric + order-aware descriptors, assigns
primary/secondary labels, selects representative pathways, and writes all
output CSVs to --output-dir. Purely a thin wrapper around
gbctp.pipeline.run_pipeline -- import that function directly instead if you
need more control than these flags expose.

Example
-------
    python scripts/run_classification.py \\
        --root-dir /path/to/pathway_csvs \\
        --output-dir /path/to/outputs \\
        --csv-pattern "*/TRAJ1/path_couplings_timeseries.csv"
"""

import argparse
from pathlib import Path

from gbctp.pipeline import PipelineConfig, run_pipeline


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root-dir", required=True, type=Path, help="Directory containing per-pathway coupling CSVs.")
    p.add_argument("--output-dir", required=True, type=Path, help="Directory to write output CSVs to.")
    p.add_argument(
        "--csv-pattern", default="*/TRAJ1/path_couplings_timeseries.csv",
        help="Glob pattern (relative to --root-dir) matching one coupling CSV per pathway.",
    )
    p.add_argument("--n-representatives-per-label", type=int, default=1)
    p.add_argument("--q-source", type=float, default=20.0)
    p.add_argument("--q-target", type=float, default=80.0)
    p.add_argument("--q-high", type=float, default=75.0)
    p.add_argument("--q-low", type=float, default=25.0)
    p.add_argument("--rec-hi", type=float, default=1.05)
    p.add_argument("--rec-lo", type=float, default=0.95)
    p.add_argument(
        "--no-diffuse-label", action="store_false", dest="include_diffuse_as_5th_label",
        help="Fold 'diffuse/weakly structured' pathways into the 4 main labels instead of a 5th label.",
    )
    p.add_argument(
        "--no-save-per-dimer-vectors", action="store_false", dest="save_per_dimer_vectors",
        help="Skip writing <name>.JCE_vectors.csv sidecar files (needed by the reliability notebooks).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    config = PipelineConfig(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        csv_pattern=args.csv_pattern,
        n_representatives_per_label=args.n_representatives_per_label,
        q_source=args.q_source,
        q_target=args.q_target,
        q_high=args.q_high,
        q_low=args.q_low,
        rec_hi=args.rec_hi,
        rec_lo=args.rec_lo,
        include_diffuse_as_5th_label=args.include_diffuse_as_5th_label,
        save_per_dimer_vectors=args.save_per_dimer_vectors,
    )
    result = run_pipeline(config)

    print("Primary label counts:")
    print(result.summary_primary.to_string(index=False))
    print(f"\nOutputs written to: {config.output_dir}")


if __name__ == "__main__":
    main()
