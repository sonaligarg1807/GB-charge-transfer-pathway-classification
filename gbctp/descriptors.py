"""Per-dimer coupling metrics and the two descriptor families built from them.

For each dimer i along a pathway, with time-resolved (signed) coupling
T_i(t) and A_i(t) = |T_i(t)|:

    J_i = <A_i>              time-averaged absolute coupling
    C_i = <A_i>^2 / <A_i^2>  coherence in (0, 1]
    E_i = J_i * C_i          effective link (coupling weighted by coherence)

Two descriptor families are built from the (J_i, C_i, E_i) profile of a path:

- Symmetric (order-independent): permutation-invariant summary statistics,
  see :func:`summarize_symmetric`.
- Non-symmetric (order-aware): bottleneck position/depth, recovery ratios,
  mid-path dip, weak-region width, etc., see :func:`order_descriptors_full`.
"""

from typing import Tuple

import numpy as np

DEFAULT_EPS = 1e-12


def per_dimer_vectors_abs(X: np.ndarray, eps: float = DEFAULT_EPS):
    """Compute per-dimer J, C, E (and raw moments) from signed couplings.

    Parameters
    ----------
    X : np.ndarray, shape (Nt, m)
        Signed couplings for m = n - 1 dimers, Nt time snapshots.
    eps : float
        Small constant added to denominators to avoid division by zero.

    Returns
    -------
    J, C, A2, Astd, E : np.ndarray, each shape (m,)
        Mean absolute coupling, coherence, mean squared absolute coupling,
        std of absolute coupling, and effective link J * C.
    """
    A = np.abs(X)
    J = np.mean(A, axis=0)
    A2 = np.mean(A * A, axis=0)
    C = (J * J) / (A2 + eps)
    Astd = np.std(A, axis=0, ddof=1) if A.shape[0] > 1 else np.zeros_like(J)
    E = J * C
    return J, C, A2, Astd, E


def summarize_symmetric(J: np.ndarray, C: np.ndarray, eps: float = DEFAULT_EPS) -> dict:
    """Permutation-invariant descriptors for one path (coupling/coherence distribution)."""
    E = J * C

    def q(x, p):
        return float(np.percentile(x, p))

    return {
        "J_mean": float(np.mean(J)),
        "J_std": float(np.std(J, ddof=1)) if len(J) > 1 else 0.0,
        "J_min": float(np.min(J)),
        "J_med": q(J, 50),
        "J_q10": q(J, 10),
        "J_q90": q(J, 90),
        "J_min_over_med": float(np.min(J) / (q(J, 50) + eps)),
        "C_mean": float(np.mean(C)),
        "C_std": float(np.std(C, ddof=1)) if len(C) > 1 else 0.0,
        "C_min": float(np.min(C)),
        "C_med": q(C, 50),
        "C_q10": q(C, 10),
        "C_q90": q(C, 90),
        "frac_C_lt_0p5": float(np.mean(C < 0.5)),
        "E_mean": float(np.mean(E)),
        "E_min": float(np.min(E)),
        "E_med": q(E, 50),
        "E_q10": q(E, 10),
        "E_min_over_med": float(np.min(E) / (q(E, 50) + eps)),
    }


def longest_run_true(b: np.ndarray) -> int:
    """Longest consecutive run of True in a boolean 1D array."""
    best = cur = 0
    for x in b:
        if bool(x):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def biggest_drop_rise(x: np.ndarray) -> Tuple[float, float, float, float]:
    """Position (0, 1] and magnitude of the biggest step-to-step drop and rise in x."""
    m = len(x)
    if m < 3:
        return np.nan, np.nan, np.nan, np.nan

    dx = np.diff(x)
    i_drop = int(np.argmin(dx)) + 1
    i_rise = int(np.argmax(dx)) + 1

    p_drop = i_drop / (m - 1)
    p_rise = i_rise / (m - 1)
    drop_mag = float(-np.min(dx))
    rise_mag = float(np.max(dx))

    return p_drop, drop_mag, p_rise, rise_mag


def end_mid_end_dip_score(x: np.ndarray, eps: float = DEFAULT_EPS) -> Tuple[float, float, float]:
    """'Dip in the middle, recovers at both ends' score, using quarters of the path.

    Returns (end_mean, mid_mean, dip_frac); dip_frac > 0 means the middle
    quantile is weaker than the average of the first/last quantile (GB-like dip).
    """
    m = len(x)
    q = max(1, int(0.25 * m))
    first = x[:q]
    last = x[-q:]
    mid = x[q:-q] if (m - 2 * q) >= 1 else x

    end_mean = 0.5 * (np.mean(first) + np.mean(last))
    mid_mean = np.mean(mid)
    dip_frac = (end_mean - mid_mean) / (end_mean + eps)

    return float(end_mean), float(mid_mean), float(dip_frac)


def order_descriptors_full(J: np.ndarray, C: np.ndarray, E: np.ndarray, eps: float = DEFAULT_EPS) -> dict:
    """Order-aware descriptors that respect the source -> target ordering along a path.

    Captures bottleneck position/depth, early-vs-late recovery, mid-path
    dip relative to the ends, drop/rise magnitudes, and weak-region width.
    """
    m = len(J)

    pJ = (int(np.argmin(J)) + 1) / m
    pC = (int(np.argmin(C)) + 1) / m
    pE = (int(np.argmin(E)) + 1) / m

    rJ = float(np.min(J) / (np.median(J) + eps))
    rC = float(np.min(C) / (np.median(C) + eps))
    rE = float(np.min(E) / (np.median(E) + eps))

    mid = m // 2
    Je, Jl = np.mean(J[:mid]), np.mean(J[mid:])
    Ce, Cl = np.mean(C[:mid]), np.mean(C[mid:])
    Ee, El = np.mean(E[:mid]), np.mean(E[mid:])

    RJ = float(Jl / (Je + eps))
    RC = float(Cl / (Ce + eps))
    RE = float(El / (Ee + eps))

    J_end, J_mid, J_dip = end_mid_end_dip_score(J, eps=eps)
    C_end, C_mid, C_dip = end_mid_end_dip_score(C, eps=eps)
    E_end, E_mid, E_dip = end_mid_end_dip_score(E, eps=eps)

    pdJ, dJ, prJ, rJ_val = biggest_drop_rise(J)
    pdC, dC, prC, rC_val = biggest_drop_rise(C)
    pdE, dE, prE, rE_val = biggest_drop_rise(E)

    J_thr = 0.5 * (J.min() + J.max())
    C_thr = 0.5 * (C.min() + C.max())
    E_thr = 0.5 * (E.min() + E.max())

    Lweak_J = longest_run_true(J < J_thr)
    Lweak_C = longest_run_true(C < C_thr)
    Lweak_E = longest_run_true(E < E_thr)

    pweak_J = (np.argmin(J) + 1) / m
    pweak_C = (np.argmin(C) + 1) / m
    pweak_E = (np.argmin(E) + 1) / m

    return {
        "pJ": pJ, "pC": pC, "pE": pE,
        "rJ": rJ, "rC": rC, "rE": rE,
        "RJ": RJ, "RC": RC, "RE": RE,
        "J_dip": J_dip, "C_dip": C_dip, "E_dip": E_dip,
        "drop_J": dJ, "rise_J": rJ_val, "pdrop_J": pdJ, "prise_J": prJ,
        "drop_C": dC, "rise_C": rC_val, "pdrop_C": pdC, "prise_C": prC,
        "drop_E": dE, "rise_E": rE_val, "pdrop_E": pdE, "prise_E": prE,
        "Lweak_J": Lweak_J, "Lweak_C": Lweak_C, "Lweak_E": Lweak_E,
        "pweak_J": pweak_J, "pweak_C": pweak_C, "pweak_E": pweak_E,
    }
