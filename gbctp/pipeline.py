"""End-to-end pathway-classification pipeline.

Turns a directory of per-pathway coupling CSVs (see :mod:`gbctp.io` for the
expected format) into symmetric + order-aware descriptors, primary/secondary
labels, and representative ("medoid") pathways per label -- and writes all
of the intermediate CSVs used by ``notebooks/01_pathway_classification.ipynb``
and by the downstream convergence/reliability analyses.

Typical usage::

    from pathlib import Path
    from gbctp.pipeline import PipelineConfig, run_pipeline

    config = PipelineConfig(
        root_dir=Path("/path/to/pathway_csvs"),
        output_dir=Path("/path/to/outputs"),
    )
    result = run_pipeline(config)
    print(result.summary_primary)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

from . import io as gio
from .descriptors import DEFAULT_EPS, order_descriptors_full, per_dimer_vectors_abs, summarize_symmetric
from .labeling import add_primary_label, label_archetypes, label_from_symmetric, secondary_composition_tables
from .representatives import DEFAULT_FEATURE_COLS_FOR_MEDOID, select_representative_paths


@dataclass
class PipelineConfig:
    """Configuration for :func:`run_pipeline`.

    Parameters
    ----------
    root_dir : Path
        Directory containing one subfolder (or file) per sampled pathway.
        See :mod:`gbctp.io` for the expected per-pathway CSV format.
    output_dir : Path
        Directory to write output CSVs to (created if missing).
    csv_pattern : str
        Glob pattern (relative to ``root_dir``) matching one coupling CSV
        per pathway, e.g. ``"*/TRAJ1/path_couplings_timeseries.csv"``.
    save_per_dimer_vectors : bool
        If True, also write a ``<name>.JCE_vectors.csv`` file per pathway
        with its per-dimer J/C/E values (needed by the convergence/
        reliability notebooks and by the optional 3D bottleneck plot).
    eps : float
        Small constant to avoid division by zero in descriptor formulas.
    q_source, q_target, q_high, q_low, rec_hi, rec_lo : float
        Quantile levels / recovery cutoffs for the archetype classification;
        see :func:`gbctp.labeling.compute_archetype_thresholds`.
    include_diffuse_as_5th_label : bool
        Add a separate "Diffuse/weakly structured" primary label for
        pathways with no clear structure, instead of folding them in.
    flat_counts_as_non_recovering : bool
        Treat "flat" recovery (neither clearly recovering nor degrading) as
        non-recovering when merging into the primary label.
    n_representatives_per_label : int
        Number of representative ("medoid") pathways to select per primary label.
    feature_cols_for_medoid : list[str]
        Descriptor columns used to measure closeness to each class centroid.
    """

    root_dir: Path
    output_dir: Path
    csv_pattern: str = "*/TRAJ1/path_couplings_timeseries.csv"
    save_per_dimer_vectors: bool = True
    eps: float = DEFAULT_EPS

    q_source: float = 20.0
    q_target: float = 80.0
    q_high: float = 75.0
    q_low: float = 25.0
    rec_hi: float = 1.05
    rec_lo: float = 0.95

    include_diffuse_as_5th_label: bool = True
    flat_counts_as_non_recovering: bool = True

    n_representatives_per_label: int = 1
    feature_cols_for_medoid: List[str] = field(default_factory=lambda: list(DEFAULT_FEATURE_COLS_FOR_MEDOID))

    def __post_init__(self):
        self.root_dir = Path(self.root_dir)
        self.output_dir = Path(self.output_dir)

    def output_files(self) -> dict:
        d = self.output_dir
        return {
            "symmetric_descriptors": d / "paths_symmetric_descriptors_labeled.csv",
            "order_descriptors": d / "paths_order_descriptors_with_archetype_labels.csv",
            "primary_labels": d / "paths_with_primary_labels.csv",
            "primary_label_counts": d / "primary_label_counts.csv",
            "secondary_cause": d / "secondary_cause_within_primary.csv",
            "secondary_shape": d / "secondary_shape_within_primary.csv",
            "secondary_recovery": d / "secondary_recovery_within_primary.csv",
            "secondary_joint": d / "secondary_joint_cause_shape_recovery_within_primary.csv",
            "representative_paths": d / "representative_paths_for_dynamics.csv",
            "rep_with_symmetric": d / "representative_paths_with_symmetric_labels.csv",
            "all_paths_both_labels": d / "all_paths_with_both_labels.csv",
            "crosstab_full": d / "primary_symmetric_crosstab_full_dataset.csv",
            "crosstab_pct": d / "primary_symmetric_percentage_full_dataset.csv",
        }


@dataclass
class PipelineResult:
    labeled_symmetric: pd.DataFrame
    labeled_order: pd.DataFrame
    df_primary: pd.DataFrame
    summary_primary: pd.DataFrame
    comp_cause: Optional[pd.DataFrame]
    comp_shape: Optional[pd.DataFrame]
    comp_recovery: Optional[pd.DataFrame]
    comp_joint: Optional[pd.DataFrame]
    rep_paths: pd.DataFrame
    rep_with_symmetric: pd.DataFrame
    comprehensive: pd.DataFrame
    crosstab: pd.DataFrame
    crosstab_pct: pd.DataFrame
    output_files: dict


def build_symmetric_dataset(
    root_dir: Path, pattern: str, eps: float = DEFAULT_EPS, save_vectors: bool = False
) -> pd.DataFrame:
    """Compute symmetric (order-independent) descriptors for every pathway CSV found."""
    rows = []
    for csv_path in gio.find_path_files(root_dir, pattern):
        sites, _time, X, _cols = gio.read_path_csv(csv_path)
        J, C, _A2, _Astd, E = per_dimer_vectors_abs(X, eps=eps)
        desc = summarize_symmetric(J, C, eps=eps)
        desc["path"] = str(csv_path)
        desc["n_sites"] = len(sites)
        desc["n_dimers"] = len(J)
        rows.append(desc)

        if save_vectors:
            gio.write_per_dimer_vectors(csv_path, sites, J, C, E)

    return pd.DataFrame(rows)


def build_order_dataset(
    root_dir: Path, pattern: str, eps: float = DEFAULT_EPS, save_vectors: bool = False
) -> pd.DataFrame:
    """Compute order-aware (unlabeled) descriptors for every pathway CSV found."""
    rows = []
    for csv_path in gio.find_path_files(root_dir, pattern):
        sites, _time, X, _cols = gio.read_path_csv(csv_path)
        J, C, _A2, _Astd, E = per_dimer_vectors_abs(X, eps=eps)

        desc = order_descriptors_full(J, C, E, eps=eps)
        desc["path"] = str(csv_path)
        desc["n_sites"] = len(sites)
        desc["n_dimers"] = len(J)
        rows.append(desc)

        if save_vectors:
            gio.write_per_dimer_vectors(csv_path, sites, J, C, E)

    return pd.DataFrame(rows)


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run the full classification pipeline and write all output CSVs.

    See :class:`PipelineConfig` for the input options and
    ``config.output_files()`` for where each output is written.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    out = config.output_files()

    # --- Symmetric descriptors + coarse cross-check label ---
    # (per-dimer vectors are written once, below, from the order-aware pass)
    desc_symmetric = build_symmetric_dataset(
        config.root_dir, config.csv_pattern, eps=config.eps, save_vectors=False
    )
    labeled_symmetric = label_from_symmetric(desc_symmetric)
    labeled_symmetric.to_csv(out["symmetric_descriptors"], index=False)

    # --- Order-aware descriptors + primary archetype classification ---
    desc_order = build_order_dataset(
        config.root_dir, config.csv_pattern, eps=config.eps, save_vectors=config.save_per_dimer_vectors
    )
    labeled_order, _thresholds = label_archetypes(
        desc_order,
        q_source=config.q_source, q_target=config.q_target,
        q_high=config.q_high, q_low=config.q_low,
        rec_hi=config.rec_hi, rec_lo=config.rec_lo,
    )
    labeled_order.to_csv(out["order_descriptors"], index=False)

    df_primary = add_primary_label(
        labeled_order,
        include_diffuse_as_5th=config.include_diffuse_as_5th_label,
        flat_counts_as_nonrecovering=config.flat_counts_as_non_recovering,
    )
    summary_primary, comp_cause, comp_shape, comp_recovery, comp_joint = secondary_composition_tables(df_primary)

    df_primary.to_csv(out["primary_labels"], index=False)
    summary_primary.to_csv(out["primary_label_counts"], index=False)
    if comp_cause is not None:
        comp_cause.to_csv(out["secondary_cause"], index=False)
    if comp_shape is not None:
        comp_shape.to_csv(out["secondary_shape"], index=False)
    if comp_recovery is not None:
        comp_recovery.to_csv(out["secondary_recovery"], index=False)
    if comp_joint is not None:
        comp_joint.to_csv(out["secondary_joint"], index=False)

    # --- Representative ("medoid") pathways per primary label ---
    rep_paths = select_representative_paths(
        df_primary,
        n_per_label=config.n_representatives_per_label,
        feature_cols=config.feature_cols_for_medoid,
    )
    rep_paths.to_csv(out["representative_paths"], index=False)

    rep_with_symmetric = rep_paths.merge(labeled_symmetric[["path", "label"]], on="path", how="left")
    rep_with_symmetric = rep_with_symmetric.rename(columns={"label": "label_symmetric"})
    rep_with_symmetric.to_csv(out["rep_with_symmetric"], index=False)

    # --- Comprehensive view: primary + symmetric labels for every pathway ---
    comprehensive = df_primary.merge(labeled_symmetric[["path", "label"]], on="path", how="left")
    comprehensive = comprehensive.rename(columns={"label": "label_symmetric"})
    comprehensive.to_csv(out["all_paths_both_labels"], index=False)

    crosstab = pd.crosstab(comprehensive["label_primary"], comprehensive["label_symmetric"], margins=True)
    crosstab.to_csv(out["crosstab_full"])

    crosstab_pct = pd.crosstab(
        comprehensive["label_primary"], comprehensive["label_symmetric"], normalize="index"
    ) * 100
    crosstab_pct.to_csv(out["crosstab_pct"])

    return PipelineResult(
        labeled_symmetric=labeled_symmetric,
        labeled_order=labeled_order,
        df_primary=df_primary,
        summary_primary=summary_primary,
        comp_cause=comp_cause,
        comp_shape=comp_shape,
        comp_recovery=comp_recovery,
        comp_joint=comp_joint,
        rep_paths=rep_paths,
        rep_with_symmetric=rep_with_symmetric,
        comprehensive=comprehensive,
        crosstab=crosstab,
        crosstab_pct=crosstab_pct,
        output_files=out,
    )
