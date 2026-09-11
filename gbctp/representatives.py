"""Selecting representative ("medoid") pathways per primary label."""

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#: Default descriptor columns used to define "closeness to the class centroid".
DEFAULT_FEATURE_COLS_FOR_MEDOID = ["E_dip", "C_dip", "rE", "Lweak_E", "pweak_E", "RE"]


def select_representative_paths(
    df: pd.DataFrame,
    primary_col: str = "label_primary",
    feature_cols: Optional[List[str]] = None,
    n_per_label: int = 1,
    within_class_standardize: bool = True,
) -> pd.DataFrame:
    """Select the pathway(s) closest to each primary label's class centroid.

    Parameters
    ----------
    df : DataFrame
        Dataset with a primary-label column and the descriptor columns in
        ``feature_cols``.
    primary_col : str
        Column holding the primary label.
    feature_cols : list[str] or None
        Descriptor columns defining representativeness; defaults to
        :data:`DEFAULT_FEATURE_COLS_FOR_MEDOID`.
    n_per_label : int
        Number of representative pathways to select per label.
    within_class_standardize : bool
        Standardize features within each class before computing the centroid.

    Returns
    -------
    DataFrame
        Rows from ``df`` for the selected pathways, with added ``rep_rank``
        (1 = closest to centroid) and ``rep_dist`` columns.
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS_FOR_MEDOID

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if primary_col not in df.columns:
        raise ValueError(f"Missing primary label column: {primary_col}")
    if n_per_label < 1:
        raise ValueError("n_per_label must be >= 1")

    reps = []

    for _, sub in df.groupby(primary_col, sort=True):
        sub = sub.copy()

        good = sub[feature_cols].notna().all(axis=1)
        sub_good = sub.loc[good].copy()

        if len(sub_good) == 0:
            take = sub.head(min(n_per_label, len(sub))).copy()
            take["rep_dist"] = np.nan
            take["rep_rank"] = np.arange(1, len(take) + 1)
            reps.append(take)
            continue

        X = sub_good[feature_cols].to_numpy(float)

        if within_class_standardize and len(sub_good) > 1:
            Xs = StandardScaler().fit_transform(X)
        else:
            Xs = X

        centroid = Xs.mean(axis=0)
        dists = np.linalg.norm(Xs - centroid, axis=1)
        sub_good["rep_dist"] = dists

        sub_good = sub_good.sort_values("rep_dist", ascending=True)
        k = min(n_per_label, len(sub_good))
        chosen = sub_good.head(k).copy()
        chosen["rep_rank"] = np.arange(1, k + 1)

        reps.append(chosen)

    return pd.concat(reps, ignore_index=True)
