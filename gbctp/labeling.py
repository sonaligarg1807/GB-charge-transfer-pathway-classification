"""Turning descriptors into labels.

Two independent labeling schemes are provided:

- :func:`label_from_symmetric` -- a coarse strong/weak x coherent/fluctuating
  label from the symmetric (order-independent) descriptors, used only as a
  cross-check against the primary classification.
- :func:`compute_archetype_thresholds` / :func:`apply_archetype_labels` /
  :func:`label_archetypes` -- the order-aware archetype classification
  (location / cause / shape / recovery), merged by :func:`add_primary_label`
  into one of four (or five) primary transport labels.

The threshold/apply split exists so that thresholds can be learned from one
dataset (e.g. a bootstrap subsample) and applied to another -- this is what
the convergence and reliability analyses in :mod:`gbctp.reliability` need.
"""

from typing import Optional

import numpy as np
import pandas as pd

#: Percentile levels and recovery cutoffs used by default everywhere in the
#: pipeline. Override via the corresponding keyword arguments to test
#: sensitivity to these choices (see gbctp.reliability).
DEFAULT_Q_SOURCE = 20.0
DEFAULT_Q_TARGET = 80.0
DEFAULT_Q_HIGH = 75.0
DEFAULT_Q_LOW = 25.0
DEFAULT_REC_HI = 1.05
DEFAULT_REC_LO = 0.95


def label_from_symmetric(desc_df: pd.DataFrame) -> pd.DataFrame:
    """Assign coarse strong/weak x coherent/fluctuating labels from symmetric descriptors.

    Adds a ``label`` column and stores the data-driven thresholds used in
    ``df.attrs["thresholds"]``.
    """
    df = desc_df.copy()

    Jm_med = float(np.percentile(df["J_mean"], 50))
    Cm_med = float(np.percentile(df["C_mean"], 50))
    Jstd_q75 = float(np.percentile(df["J_std"], 75))
    Jminratio_q10 = float(np.percentile(df["J_min_over_med"], 10))

    labels = []
    for _, r in df.iterrows():
        strength = "strong" if r["J_mean"] >= Jm_med else "weak"
        stability = "coherent" if r["C_mean"] >= Cm_med else "fluctuating"
        hetero = (r["J_std"] >= Jstd_q75) or (r["J_min_over_med"] <= Jminratio_q10)

        if hetero:
            labels.append(f"{strength}-{stability}-heterogeneous")
        else:
            labels.append(f"{strength}-{stability}")

    df["label"] = labels
    df.attrs["thresholds"] = {
        "J_mean_median": Jm_med,
        "C_mean_median": Cm_med,
        "J_std_q75": Jstd_q75,
        "J_min_over_med_q10": Jminratio_q10,
    }
    return df


def _percentile(x, q: float) -> float:
    """Percentile with q clipped to [0, 100] (safe under quantile-level perturbation)."""
    return float(np.percentile(x, float(np.clip(q, 0.0, 100.0))))


def compute_archetype_thresholds(
    df: pd.DataFrame,
    q_source: float = DEFAULT_Q_SOURCE,
    q_target: float = DEFAULT_Q_TARGET,
    q_high: float = DEFAULT_Q_HIGH,
    q_low: float = DEFAULT_Q_LOW,
    rec_hi: float = DEFAULT_REC_HI,
    rec_lo: float = DEFAULT_REC_LO,
) -> dict:
    """Compute quantile-based archetype-classification thresholds from ``df``.

    ``df`` must contain the order-aware descriptor columns produced by
    :func:`gbctp.descriptors.order_descriptors_full` (as a DataFrame of rows,
    one per pathway): pE, E_dip, C_dip, rE, rJ, rC, Lweak_E, Lweak_C, drop_E,
    rise_E, drop_C, rise_C.

    ``df`` can be the full dataset or any subset (e.g. a bootstrap sample) --
    thresholds learned here can then be applied to a *different* dataset via
    :func:`apply_archetype_labels`.
    """
    required = [
        "pE", "E_dip", "C_dip", "rE", "rJ", "rC",
        "Lweak_E", "Lweak_C", "drop_E", "rise_E", "drop_C", "rise_C",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required descriptor columns: {missing}")

    return {
        "pE_q_source": _percentile(df["pE"], q_source),
        "pE_q_target": _percentile(df["pE"], q_target),
        "E_dip_q_high": _percentile(df["E_dip"], q_high),
        "C_dip_q_high": _percentile(df["C_dip"], q_high),
        "rE_q_low": _percentile(df["rE"], q_low),
        "rJ_q_low": _percentile(df["rJ"], q_low),
        "rC_q_low": _percentile(df["rC"], q_low),
        "Lweak_E_q_high": _percentile(df["Lweak_E"], q_high),
        "Lweak_C_q_high": _percentile(df["Lweak_C"], q_high),
        "drop_E_q_high": _percentile(df["drop_E"], q_high),
        "rise_E_q_high": _percentile(df["rise_E"], q_high),
        "drop_C_q_high": _percentile(df["drop_C"], q_high),
        "rise_C_q_high": _percentile(df["rise_C"], q_high),
        "recovery_hi": float(rec_hi),
        "recovery_lo": float(rec_lo),
        "q_source": float(q_source),
        "q_target": float(q_target),
        "q_high": float(q_high),
        "q_low": float(q_low),
    }


def apply_archetype_labels(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Apply order-aware archetype labels to ``df`` using externally supplied thresholds.

    Adds ``label_location``, ``label_cause``, ``label_shape``,
    ``label_recovery``, ``label_combined``. See
    :func:`compute_archetype_thresholds` for the thresholds dict format.
    """
    out = df.copy()

    loc_labels, cause_labels, shape_labels, rec_labels, combined = [], [], [], [], []

    for _, r in out.iterrows():
        # Location: where along the path the coupling bottleneck sits.
        if r["pE"] <= thresholds["pE_q_source"]:
            location = "source-limited"
        elif r["pE"] >= thresholds["pE_q_target"]:
            location = "target-limited"
        else:
            location = "GB-limited"

        # Cause: is the bottleneck driven by weak coupling, low coherence, or both.
        coupling_signal = (
            (r["E_dip"] >= thresholds["E_dip_q_high"])
            or (r["drop_E"] >= thresholds["drop_E_q_high"])
            or (r["Lweak_E"] >= thresholds["Lweak_E_q_high"])
            or (r["rJ"] <= thresholds["rJ_q_low"])
        )
        coherence_signal = (
            (r["C_dip"] >= thresholds["C_dip_q_high"])
            or (r["drop_C"] >= thresholds["drop_C_q_high"])
            or (r["Lweak_C"] >= thresholds["Lweak_C_q_high"])
            or (r["rC"] <= thresholds["rC_q_low"])
        )
        if coupling_signal and not coherence_signal:
            cause = "coupling-limited"
        elif coherence_signal and not coupling_signal:
            cause = "coherence-limited"
        else:
            cause = "mixed-limited"

        # Shape: sharp (single steep drop+rise) vs. extended weak region vs. neither.
        sharp = (
            ((r["drop_E"] >= thresholds["drop_E_q_high"]) and (r["rise_E"] >= thresholds["rise_E_q_high"]))
            or ((r["drop_C"] >= thresholds["drop_C_q_high"]) and (r["rise_C"] >= thresholds["rise_C_q_high"]))
        )
        extended = (
            (r["Lweak_E"] >= thresholds["Lweak_E_q_high"])
            or (r["Lweak_C"] >= thresholds["Lweak_C_q_high"])
        )
        if sharp and extended:
            shape = "sharp+extended"
        elif sharp:
            shape = "sharp"
        elif extended:
            shape = "extended"
        else:
            shape = "diffuse"

        # Recovery: does the effective link recover past the bottleneck (late > early).
        if r["RE"] > thresholds["recovery_hi"]:
            recovery = "recovering"
        elif r["RE"] < thresholds["recovery_lo"]:
            recovery = "degrading"
        else:
            recovery = "flat"

        loc_labels.append(location)
        cause_labels.append(cause)
        shape_labels.append(shape)
        rec_labels.append(recovery)
        combined.append(f"{location}, {cause}, {shape}, {recovery}")

    out["label_location"] = loc_labels
    out["label_cause"] = cause_labels
    out["label_shape"] = shape_labels
    out["label_recovery"] = rec_labels
    out["label_combined"] = combined
    out.attrs["thresholds"] = thresholds

    return out


def label_archetypes(
    df: pd.DataFrame,
    q_source: float = DEFAULT_Q_SOURCE,
    q_target: float = DEFAULT_Q_TARGET,
    q_high: float = DEFAULT_Q_HIGH,
    q_low: float = DEFAULT_Q_LOW,
    rec_hi: float = DEFAULT_REC_HI,
    rec_lo: float = DEFAULT_REC_LO,
):
    """Convenience wrapper: learn thresholds from ``df`` and apply them to ``df``.

    Returns (labeled_df, thresholds). Use :func:`compute_archetype_thresholds`
    + :func:`apply_archetype_labels` directly when thresholds should be
    learned from one dataset and applied to another.
    """
    thresholds = compute_archetype_thresholds(
        df, q_source=q_source, q_target=q_target, q_high=q_high, q_low=q_low,
        rec_hi=rec_hi, rec_lo=rec_lo,
    )
    labeled = apply_archetype_labels(df, thresholds)
    return labeled, thresholds


def add_primary_label(
    df: pd.DataFrame,
    include_diffuse_as_5th: bool = True,
    flat_counts_as_nonrecovering: bool = True,
) -> pd.DataFrame:
    """Merge label_location/label_cause/label_shape/label_recovery into one primary label.

    Primary labels: "Source-limited", "Target-limited",
    "GB-crossing (recovering)", "GB-blocked (non-recovering)", and
    optionally "Diffuse/weakly structured" for pathways with no clear
    structure (``label_shape == "diffuse"``).
    """
    out = df.copy()

    required = ["label_location", "label_recovery", "label_shape"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing columns for merging primary labels: {missing}")

    if flat_counts_as_nonrecovering:
        nonrec = out["label_recovery"].isin(["degrading", "flat"])
    else:
        nonrec = out["label_recovery"].eq("degrading")

    out["label_primary"] = "Unassigned"

    if include_diffuse_as_5th:
        out.loc[out["label_shape"].eq("diffuse"), "label_primary"] = "Diffuse/weakly structured"

    out.loc[out["label_location"].eq("source-limited"), "label_primary"] = "Source-limited"
    out.loc[out["label_location"].eq("target-limited"), "label_primary"] = "Target-limited"

    gb = out["label_location"].eq("GB-limited")
    out.loc[gb & out["label_recovery"].eq("recovering"), "label_primary"] = "GB-crossing (recovering)"
    out.loc[gb & nonrec, "label_primary"] = "GB-blocked (non-recovering)"

    return out


def secondary_composition_tables(df: pd.DataFrame, primary_col: str = "label_primary"):
    """Composition tables showing secondary label distributions within each primary label.

    Returns
    -------
    summary_primary, comp_cause, comp_shape, comp_recovery, comp_joint : DataFrame
        Counts/percentages for the primary labels, and for label_cause,
        label_shape, label_recovery and their joint combination within each
        primary label (comp_* are None if the required columns are absent).
    """
    if primary_col not in df.columns:
        raise ValueError(f"'{primary_col}' not found. Run add_primary_label() first.")

    summary_primary = (
        df[primary_col]
        .value_counts(dropna=False)
        .rename_axis(primary_col)
        .reset_index(name="count")
    )
    summary_primary["percent"] = 100.0 * summary_primary["count"] / summary_primary["count"].sum()

    def within_dist(col: str) -> Optional[pd.DataFrame]:
        if col not in df.columns:
            return None
        tmp = (
            df.groupby([primary_col, col], dropna=False)
            .size()
            .reset_index(name="count")
        )
        tmp["percent_within_primary"] = (
            tmp["count"] / tmp.groupby(primary_col)["count"].transform("sum") * 100.0
        )
        return tmp.sort_values([primary_col, "count"], ascending=[True, False])

    comp_cause = within_dist("label_cause")
    comp_shape = within_dist("label_shape")
    comp_recovery = within_dist("label_recovery")

    cols_joint = [c for c in ["label_cause", "label_shape", "label_recovery"] if c in df.columns]
    if len(cols_joint) == 3:
        comp_joint = (
            df.groupby([primary_col] + cols_joint, dropna=False)
            .size()
            .reset_index(name="count")
        )
        comp_joint["percent_within_primary"] = (
            comp_joint["count"] / comp_joint.groupby(primary_col)["count"].transform("sum") * 100.0
        )
        comp_joint = comp_joint.sort_values([primary_col, "count"], ascending=[True, False])
    else:
        comp_joint = None

    return summary_primary, comp_cause, comp_shape, comp_recovery, comp_joint
