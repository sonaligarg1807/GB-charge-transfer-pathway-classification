"""Convergence and reliability checks for the primary pathway classification.

These re-run the archetype classification (see :mod:`gbctp.labeling`) many
times under random subsampling and/or quantile-threshold perturbation, to
test whether the primary-label populations are an artifact of the number of
sampled pathways or of the exact thresholds chosen. Used by
``notebooks/02_convergence_and_robustness.ipynb`` and
``notebooks/03_reliability_heatmaps.ipynb`` -- this module has no plotting
code; it only produces DataFrames for the notebooks to plot and save.
"""

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .labeling import (
    add_primary_label,
    apply_archetype_labels,
    compute_archetype_thresholds,
)

#: Canonical primary-label order used for consistent plot legends/axes.
DEFAULT_LABEL_ORDER = [
    "Source-limited",
    "GB-crossing (recovering)",
    "GB-blocked (non-recovering)",
    "Target-limited",
    "Diffuse/weakly structured",
]


def ordered_present_labels(labels: Sequence[str], label_order: Sequence[str] = DEFAULT_LABEL_ORDER) -> List[str]:
    """Labels actually present in ``labels``, ordered per ``label_order`` (extras appended)."""
    present = list(pd.Series(labels).dropna().unique())
    ordered = [x for x in label_order if x in present]
    ordered += [x for x in present if x not in ordered]
    return ordered


# ---------------------------------------------------------------------------
# Classification wrappers (learn thresholds from one dataset, apply to another)
# ---------------------------------------------------------------------------


def classify_dataframe(
    df: pd.DataFrame,
    q_source: float = 20.0,
    q_target: float = 80.0,
    q_high: float = 75.0,
    q_low: float = 25.0,
    rec_hi: float = 1.05,
    rec_lo: float = 0.95,
    include_diffuse_as_5th_label: bool = True,
    flat_counts_as_non_recovering: bool = True,
):
    """Classify ``df`` using thresholds learned from ``df`` itself.

    Returns (labeled_df, thresholds).
    """
    thresholds = compute_archetype_thresholds(
        df, q_source=q_source, q_target=q_target, q_high=q_high, q_low=q_low, rec_hi=rec_hi, rec_lo=rec_lo
    )
    labeled = apply_archetype_labels(df, thresholds)
    labeled = add_primary_label(
        labeled,
        include_diffuse_as_5th=include_diffuse_as_5th_label,
        flat_counts_as_nonrecovering=flat_counts_as_non_recovering,
    )
    return labeled, thresholds


def classify_with_reference_thresholds(
    df_all: pd.DataFrame,
    df_threshold_reference: pd.DataFrame,
    q_source: float = 20.0,
    q_target: float = 80.0,
    q_high: float = 75.0,
    q_low: float = 25.0,
    rec_hi: float = 1.05,
    rec_lo: float = 0.95,
    rng: Optional[np.random.Generator] = None,
    perturb_quantiles: bool = False,
    perturb_recovery: bool = False,
    q_perturb_width: float = 5.0,
    rec_perturb_width: float = 0.02,
    include_diffuse_as_5th_label: bool = True,
    flat_counts_as_non_recovering: bool = True,
):
    """Learn thresholds from ``df_threshold_reference`` and classify ``df_all``.

    This is the building block for both reliability checks in
    ``03_reliability_heatmaps.ipynb``:

    - Part A (bootstrap convergence): call with ``df_threshold_reference`` a
      random subsample of ``df_all`` and ``perturb_quantiles=perturb_recovery=False``.
    - Part B (bootstrap + threshold perturbation): same, but with
      ``perturb_quantiles=perturb_recovery=True`` so the quantile *levels*
      themselves (e.g. the 75th percentile) are jittered by up to
      ``q_perturb_width`` percentile points, and the recovery cutoffs by up
      to ``rec_perturb_width``.

    Returns (labeled_df, thresholds).
    """
    if rng is None:
        rng = np.random.default_rng()

    if perturb_quantiles:
        q_source = float(np.clip(rng.uniform(q_source - q_perturb_width, q_source + q_perturb_width), 0.0, 100.0))
        q_target = float(np.clip(rng.uniform(q_target - q_perturb_width, q_target + q_perturb_width), 0.0, 100.0))
        q_high = float(np.clip(rng.uniform(q_high - q_perturb_width, q_high + q_perturb_width), 0.0, 100.0))
        q_low = float(np.clip(rng.uniform(q_low - q_perturb_width, q_low + q_perturb_width), 0.0, 100.0))

    if perturb_recovery:
        rec_hi_p = rec_hi + rng.uniform(-rec_perturb_width, rec_perturb_width)
        rec_lo_p = rec_lo + rng.uniform(-rec_perturb_width, rec_perturb_width)
        if rec_lo_p < rec_hi_p:
            rec_hi, rec_lo = rec_hi_p, rec_lo_p

    thresholds = compute_archetype_thresholds(
        df_threshold_reference,
        q_source=q_source, q_target=q_target, q_high=q_high, q_low=q_low,
        rec_hi=rec_hi, rec_lo=rec_lo,
    )
    labeled = apply_archetype_labels(df_all, thresholds)
    labeled = add_primary_label(
        labeled,
        include_diffuse_as_5th=include_diffuse_as_5th_label,
        flat_counts_as_nonrecovering=flat_counts_as_non_recovering,
    )
    return labeled, thresholds


# ---------------------------------------------------------------------------
# Effective-link (E) profile loading, for class-fingerprint plots
# ---------------------------------------------------------------------------


def get_jce_file(path_string: str) -> Path:
    """Path to the per-dimer J/C/E vectors file for a given source pathway CSV."""
    return Path(path_string).with_suffix(".JCE_vectors.csv")


def load_E_profile(path_string: str):
    """Load a pathway's E_i = J_i * C_i profile from its saved JCE-vectors file.

    Accepts either ``J, C, E`` or ``J_abs_mean, C_coherence, E_JxC`` column
    naming. Returns (x, E) where x is the normalized position (0..1) along
    the path.
    """
    jce_file = get_jce_file(path_string)
    if not jce_file.exists():
        raise FileNotFoundError(f"JCE file not found: {jce_file}")

    df = pd.read_csv(jce_file)
    df.columns = [c.strip() for c in df.columns]

    if "E" in df.columns:
        E = df["E"].to_numpy(float)
    elif "E_JxC" in df.columns:
        E = df["E_JxC"].to_numpy(float)
    elif "J" in df.columns and "C" in df.columns:
        E = df["J"].to_numpy(float) * df["C"].to_numpy(float)
    elif "J_abs_mean" in df.columns and "C_coherence" in df.columns:
        E = df["J_abs_mean"].to_numpy(float) * df["C_coherence"].to_numpy(float)
    else:
        raise ValueError(f"Cannot find E or J/C columns in {jce_file}. Available columns: {list(df.columns)}")

    x = np.array([0.5]) if len(E) == 1 else np.linspace(0.0, 1.0, len(E))
    return x, E


def normalize_profile(E: np.ndarray, method: str = "median") -> np.ndarray:
    """Rescale an E profile so different pathways can be overlaid.

    ``method`` is one of "raw", "median", "max", "first_last".
    """
    E = np.asarray(E, dtype=float)

    if method == "raw":
        return E
    elif method == "median":
        scale = np.median(E)
    elif method == "max":
        scale = np.max(E)
    elif method == "first_last":
        scale = 0.5 * (E[0] + E[-1])
    else:
        raise ValueError(f"Unknown normalization: {method}")

    if scale <= 0 or not np.isfinite(scale):
        return E
    return E / scale


def interpolate_profile(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolate a (possibly short) profile onto a common normalized-position grid."""
    if len(x) == 1:
        return np.ones_like(grid) * y[0]
    return np.interp(grid, x, y)


def collect_profiles_for_label(
    df_primary: pd.DataFrame, label: str, grid: np.ndarray, normalization: str = "median"
) -> Optional[np.ndarray]:
    """Stack the normalized, grid-interpolated E profiles of every pathway with a given primary label.

    Requires the ``.JCE_vectors.csv`` sidecar files (see :mod:`gbctp.io`) to
    already exist next to each pathway's source CSV. Returns None if no
    profile could be loaded for this label.
    """
    sub = df_primary[df_primary["label_primary"] == label]

    profiles, skipped = [], []
    for _, row in sub.iterrows():
        try:
            x, E = load_E_profile(row["path"])
            E_norm = normalize_profile(E, method=normalization)
            profiles.append(interpolate_profile(x, E_norm, grid))
        except Exception as err:  # noqa: BLE001 - report and continue
            skipped.append((row["path"], str(err)))

    if skipped:
        print(f"{label}: skipped {len(skipped)} paths.")
        for p, err in skipped[:3]:
            print(p, err)

    return np.vstack(profiles) if profiles else None


# ---------------------------------------------------------------------------
# Subsampling convergence (notebook 02)
# ---------------------------------------------------------------------------


def summarize_bootstrap(df: pd.DataFrame, quantity: str) -> pd.DataFrame:
    """Mean/95%-interval/std of ``value`` by (n_sample, label) for one ``quantity`` slice."""
    sub = df[df["quantity"] == quantity]
    return (
        sub.groupby(["n_sample", "label"])["value"]
        .agg(mean="mean", q025=lambda x: np.percentile(x, 2.5), q975=lambda x: np.percentile(x, 97.5), std="std")
        .reset_index()
    )


#: Threshold keys tracked by run_subsampling_convergence / plotted by threshold-convergence plots.
DEFAULT_THRESHOLD_KEYS = [
    "pE_q_source", "pE_q_target", "E_dip_q_high", "C_dip_q_high",
    "rE_q_low", "Lweak_E_q_high", "drop_E_q_high", "rise_E_q_high",
]


def run_subsampling_convergence(
    df_all: pd.DataFrame,
    n_min: int = 30,
    n_step: int = 25,
    n_boot: int = 300,
    random_seed: int = 123,
    label_order: Sequence[str] = DEFAULT_LABEL_ORDER,
    threshold_keys: Sequence[str] = DEFAULT_THRESHOLD_KEYS,
    **classify_kwargs,
):
    """Re-classify random subsets of increasing size to check class-fraction convergence.

    Parameters
    ----------
    df_all : DataFrame
        Full order-aware descriptor dataset (one row per pathway).
    n_min, n_step : int
        Smallest sample size and step between sample sizes (up to the full
        dataset size).
    n_boot : int
        Number of random subsamples drawn per sample size.
    random_seed : int
        Seed for reproducibility.
    **classify_kwargs
        Extra keyword arguments forwarded to :func:`classify_dataframe`
        (e.g. ``q_source``, ``include_diffuse_as_5th_label``).

    Returns
    -------
    df_conv : DataFrame
        Long-format results: one row per (n_sample, bootstrap, quantity, label).
    df_full : DataFrame
        Full-dataset reference classification.
    labels : list[str]
        Primary labels present, in canonical order.
    full_frac, full_gb_frac : Series
        Full-dataset primary-label fractions overall, and within GB-limited pathways.
    """
    df_full, _ = classify_dataframe(df_all, **classify_kwargs)
    labels = ordered_present_labels(df_full["label_primary"], label_order)

    rng = np.random.default_rng(random_seed)
    n_total = len(df_all)

    n_values = list(range(n_min, n_total + 1, n_step))
    if n_values[-1] != n_total:
        n_values.append(n_total)

    results = []
    for n_sample in n_values:
        for boot in range(n_boot):
            idx = rng.choice(n_total, size=n_sample, replace=False)
            df_lab, thresholds = classify_dataframe(df_all.iloc[idx], **classify_kwargs)

            base = {
                "n_sample": n_sample,
                "bootstrap": boot,
                "n_total": len(df_lab),
                "n_GB_limited": int((df_lab["label_location"] == "GB-limited").sum()),
            }

            frac = df_lab["label_primary"].value_counts(normalize=True)
            for lab in labels:
                results.append({**base, "quantity": "primary_fraction", "label": lab, "value": float(frac.get(lab, 0.0))})

            df_gb = df_lab[df_lab["label_location"] == "GB-limited"]
            if len(df_gb) > 0:
                gb_frac = df_gb["label_primary"].value_counts(normalize=True)
                for lab in ["GB-crossing (recovering)", "GB-blocked (non-recovering)"]:
                    results.append({**base, "quantity": "GB_fraction", "label": lab, "value": float(gb_frac.get(lab, 0.0))})

            for key in threshold_keys:
                results.append({**base, "quantity": "threshold", "label": key, "value": float(thresholds[key])})

    df_conv = pd.DataFrame(results)

    full_frac = df_full["label_primary"].value_counts(normalize=True)
    full_gb_frac = df_full[df_full["label_location"] == "GB-limited"]["label_primary"].value_counts(normalize=True)

    return df_conv, df_full, labels, full_frac, full_gb_frac


# ---------------------------------------------------------------------------
# Threshold sensitivity (notebook 02)
# ---------------------------------------------------------------------------

#: Low/default/high quantile-cutoff settings used by run_threshold_sensitivity.
DEFAULT_SENSITIVITY_SETTINGS = [
    {"name": "low_cutoff", "q_source": 15, "q_target": 85, "q_high": 80, "q_low": 20},
    {"name": "default", "q_source": 20, "q_target": 80, "q_high": 75, "q_low": 25},
    {"name": "high_cutoff", "q_source": 25, "q_target": 75, "q_high": 70, "q_low": 30},
]


def run_threshold_sensitivity(
    df_all: pd.DataFrame,
    settings: Sequence[dict] = DEFAULT_SENSITIVITY_SETTINGS,
    **classify_kwargs,
):
    """Re-classify the full dataset under a few quantile-cutoff settings.

    Checks whether primary-label fractions (and per-pathway label
    agreement, relative to the "default" setting) are sensitive to the
    exact quantile levels chosen.

    Returns (df_sens, df_agree): per-setting class fractions, and
    per-setting agreement with the "default" setting's labels.
    """
    rows = []
    label_reference = None

    for setting in settings:
        df_lab, _ = classify_dataframe(
            df_all,
            q_source=setting["q_source"], q_target=setting["q_target"],
            q_high=setting["q_high"], q_low=setting["q_low"],
            **classify_kwargs,
        )
        counts = df_lab["label_primary"].value_counts()
        frac = df_lab["label_primary"].value_counts(normalize=True)

        if setting["name"] == "default":
            label_reference = df_lab[["path", "label_primary"]].rename(columns={"label_primary": "label_default"})

        for lab, val in frac.items():
            rows.append({
                "setting": setting["name"], "label_primary": lab, "fraction": float(val), "count": int(counts[lab]),
                "q_source": setting["q_source"], "q_target": setting["q_target"],
                "q_high": setting["q_high"], "q_low": setting["q_low"],
            })

    df_sens = pd.DataFrame(rows)

    agreement_rows = []
    if label_reference is not None:
        for setting in settings:
            df_lab, _ = classify_dataframe(
                df_all,
                q_source=setting["q_source"], q_target=setting["q_target"],
                q_high=setting["q_high"], q_low=setting["q_low"],
                **classify_kwargs,
            )
            tmp = df_lab[["path", "label_primary"]].merge(label_reference, on="path", how="left")
            agreement = float(np.mean(tmp["label_primary"] == tmp["label_default"]))
            agreement_rows.append({"setting": setting["name"], "agreement_with_default": agreement})

    df_agree = pd.DataFrame(agreement_rows)
    return df_sens, df_agree


# ---------------------------------------------------------------------------
# Bootstrap convergence / combined reliability heat maps (notebook 03)
# ---------------------------------------------------------------------------


def bootstrap_classification_convergence(
    df: pd.DataFrame,
    sample_fractions: Sequence[float] = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00),
    n_bootstrap: int = 300,
    random_seed: int = 42,
    primary_order: Sequence[str] = (
        "Source-limited", "GB-crossing (recovering)", "GB-blocked (non-recovering)", "Target-limited",
    ),
    min_sample_size: int = 5,
    **classify_kwargs,
) -> pd.DataFrame:
    """Part A reliability check: label retention when thresholds are learned from a subsample.

    For each sample size (as a fraction of the full dataset), repeatedly:
    sample pathways, learn thresholds from the sample, classify the *full*
    dataset with those thresholds, and measure the fraction of each
    reference-class pathway that keeps its full-dataset label.

    Returns a DataFrame with one row per sample size and one column per
    primary label (mean retention fraction).
    """
    rng = np.random.default_rng(random_seed)
    n_total = len(df)

    reference_labeled, _ = classify_with_reference_thresholds(df, df, rng=rng, **classify_kwargs)
    reference_labels = reference_labeled["label_primary"].to_numpy()

    sample_sizes = sorted({max(min_sample_size, min(int(round(f * n_total)), n_total)) for f in sample_fractions})

    rows = []
    for n_sample in sample_sizes:
        stability_by_label = {label: [] for label in primary_order}

        for _ in range(n_bootstrap):
            sampled_idx = rng.choice(n_total, size=n_sample, replace=False)
            df_sample = df.iloc[sampled_idx]

            boot_labeled, _ = classify_with_reference_thresholds(df, df_sample, rng=rng, **classify_kwargs)
            boot_labels = boot_labeled["label_primary"].to_numpy()

            for label in primary_order:
                mask = reference_labels == label
                stability = np.nan if mask.sum() == 0 else np.mean(boot_labels[mask] == reference_labels[mask])
                stability_by_label[label].append(stability)

        row = {"n_sampled_paths": n_sample, "sample_fraction": n_sample / n_total}
        row.update({label: np.nanmean(vals) for label, vals in stability_by_label.items()})
        rows.append(row)

    return pd.DataFrame(rows)


def run_combined_bootstrap_threshold_analysis(
    df: pd.DataFrame,
    sample_sizes: Optional[Sequence[int]] = None,
    n_bootstrap: int = 300,
    n_perturb: int = 30,
    random_seed: int = 42,
    primary_order: Sequence[str] = (
        "Source-limited", "GB-crossing (recovering)", "GB-blocked (non-recovering)", "Target-limited",
    ),
    q_perturb_width: float = 5.0,
    rec_perturb_width: float = 0.02,
    max_sample_size: int = 500,
    sample_step: int = 50,
    **classify_kwargs,
):
    """Part B reliability check: as bootstrap convergence, plus quantile/recovery-cutoff jitter.

    For each sample size, draws ``n_bootstrap`` subsamples and, for each,
    ``n_perturb`` random perturbations of the classification thresholds
    themselves -- a stronger test that also probes sensitivity to the exact
    threshold *definitions*, not just to which pathways were used to set them.

    Returns (stability_mean_df, stability_std_df, all_results_df): pivoted
    mean/std retention-fraction matrices (rows = sample size, columns =
    primary label) and the full long-format results.
    """
    rng = np.random.default_rng(random_seed)
    n_total = len(df)

    reference_labeled, _ = classify_with_reference_thresholds(
        df, df, rng=rng, perturb_quantiles=False, perturb_recovery=False, **classify_kwargs
    )
    reference_labels = reference_labeled["label_primary"].to_numpy()

    if sample_sizes is None:
        max_sample = min(max_sample_size, n_total)
        sample_sizes = list(range(sample_step, max_sample + 1, sample_step))

    all_results = []
    for n_sample in sample_sizes:
        for b in range(n_bootstrap):
            sampled_idx = rng.choice(n_total, size=n_sample, replace=False)
            df_sample = df.iloc[sampled_idx]

            for p in range(n_perturb):
                perturbed_labeled, _ = classify_with_reference_thresholds(
                    df, df_sample, rng=rng,
                    perturb_quantiles=True, perturb_recovery=True,
                    q_perturb_width=q_perturb_width, rec_perturb_width=rec_perturb_width,
                    **classify_kwargs,
                )
                perturbed_labels = perturbed_labeled["label_primary"].to_numpy()

                for primary_label in primary_order:
                    mask = reference_labels == primary_label
                    retention = np.nan if mask.sum() == 0 else np.mean(perturbed_labels[mask] == reference_labels[mask])
                    all_results.append({
                        "n_sampled_paths": n_sample, "sample_fraction": n_sample / n_total,
                        "bootstrap_id": b + 1, "perturb_id": p + 1,
                        "primary_label": primary_label, "retention_fraction": retention,
                    })

    all_results_df = pd.DataFrame(all_results)

    grouped = all_results_df.groupby(["n_sampled_paths", "sample_fraction", "primary_label"])["retention_fraction"]
    mean_df = grouped.mean().reset_index()
    std_df = grouped.std().reset_index()

    stability_mean_df = mean_df.pivot(index="n_sampled_paths", columns="primary_label", values="retention_fraction")
    stability_std_df = std_df.pivot(index="n_sampled_paths", columns="primary_label", values="retention_fraction")

    stability_mean_df = stability_mean_df.reindex(columns=list(primary_order))
    stability_std_df = stability_std_df.reindex(columns=list(primary_order))

    return stability_mean_df, stability_std_df, all_results_df
