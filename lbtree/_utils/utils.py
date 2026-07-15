"""
lbtree/_utils/utils.py
======================
Contingency-matrix utilities used by both SCTree and SLBT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _contingency_matrix(x: pd.Series, y: pd.Series) -> np.ndarray:
    """
    Build a row-conditional contingency matrix F (shape I × J).

    Each row i is the conditional distribution  P(Y=j | X=x_i).
    Rows with zero marginal are set to 0 (via fillna).

    Parameters
    ----------
    x : pd.Series   — predictor column (categorical)
    y : pd.Series   — target column (categorical)

    Returns
    -------
    F : np.ndarray, shape (I, J), dtype float64
    """
    F = pd.crosstab(x, y, dropna=False)
    F_cond = F.div(F.sum(axis=1), axis=0).fillna(0)
    return F_cond.to_numpy(dtype=np.float64)


def _stratified_contingency_weighted(
    x: pd.Series,
    y: pd.Series,
    x_s: np.ndarray,
    w: np.ndarray,
    norm: bool = False,
) -> np.ndarray:
    """
    Weighted version of ``_stratified_contingency``.

    Each cell (k, i, j) is the sum of sample weights where
    ``x_s == s_k``, ``x == x_i``, ``y == y_j``, normalised by the
    total weight sum.

    Parameters
    ----------
    x    : pd.Series
    y    : pd.Series
    x_s  : np.ndarray  — stratum indicator
    w    : np.ndarray  — per-sample weights (should sum to 1)
    norm : bool, default False
        Same semantics as in ``_stratified_contingency``:
        False → raw joint weight proportions (K, I, J)
        True  → row-conditional within-stratum (K, I, J)

    Returns
    -------
    Fs : np.ndarray, shape (K, I, J), dtype float64
    """
    x_levels = sorted(np.unique(x))
    y_levels = sorted(np.unique(y))
    s_levels = sorted(np.unique(x_s))

    I = len(x_levels)
    J = len(y_levels)
    K = len(s_levels)

    w = np.asarray(w, dtype=np.float64)

    Fs_raw = np.zeros((K, I, J), dtype=np.float64)
    for k, s in enumerate(s_levels):
        mask  = x_s == s
        x_sub = x[mask]
        y_sub = y[mask]
        w_sub = w[mask]
        ct = pd.crosstab(
            pd.Categorical(x_sub, categories=x_levels),
            pd.Categorical(y_sub, categories=y_levels),
            values=w_sub,
            aggfunc="sum",
            dropna=False,
        ).reindex(index=x_levels, columns=y_levels, fill_value=0).fillna(0)
        Fs_raw[k] = ct.to_numpy(dtype=np.float64)

    W_total = Fs_raw.sum()
    if W_total > 0:
        Fs_raw /= W_total

    if not norm:
        return Fs_raw

    Fs = np.zeros((K, I, J), dtype=np.float64)
    for k in range(K):
        for i in range(I):
            row_sum = Fs_raw[k, i, :].sum()
            if row_sum > 0:
                Fs[k, i, :] = Fs_raw[k, i, :] / row_sum
    return Fs


def _stratified_contingency(
    x: pd.Series,
    y: pd.Series,
    x_s: np.ndarray,
    norm: bool = False,
) -> np.ndarray:
    """
    Build an array of contingency matrices stratified by ``x_s``.

    Parameters
    ----------
    x    : pd.Series    — predictor column
    y    : pd.Series    — target column
    x_s  : np.ndarray  — stratum indicator (one integer per observation)
    norm : bool, default False
        If False  → returns raw joint frequency array  Fs[k, i, j]
                    (proportions, summing to 1 over all k, i, j)
        If True   → returns row-conditional within-stratum array
                    Fs[k, i, j] = P(Y=j | X=i) * stratum_weight

    Returns
    -------
    Fs : np.ndarray, shape (K, I, J), dtype float64
    """
    x_levels = sorted(np.unique(x))
    y_levels = sorted(np.unique(y))
    s_levels = sorted(np.unique(x_s))

    I = len(x_levels)
    J = len(y_levels)
    K = len(s_levels)

    # Overall row-conditional distribution  (used when norm=True)
    F_total      = pd.crosstab(x, y, dropna=False)
    F_total_cond = F_total.div(F_total.sum(axis=1), axis=0).fillna(0)
    F_total_np   = F_total_cond.to_numpy(dtype=np.float64)

    # Joint frequency array
    Fs_raw = np.zeros((K, I, J), dtype=np.float64)
    for k, s in enumerate(s_levels):
        mask = (x_s == s)
        ct = pd.crosstab(
            pd.Categorical(x[mask], categories=x_levels),
            pd.Categorical(y[mask], categories=y_levels),
            dropna=False,
        ).reindex(index=x_levels, columns=y_levels, fill_value=0)
        Fs_raw[k] = ct.values

    N_total = Fs_raw.sum()
    if N_total > 0:
        Fs_raw /= N_total

    if not norm:
        return Fs_raw

    # Row-conditional within-stratum array
    Fs = np.zeros((K, I, J), dtype=np.float64)
    for k in range(K):
        for i in range(I):
            row_sum = Fs_raw[k, i, :].sum()
            if row_sum > 0:
                Fs[k, i, :] = Fs_raw[k, i, :] / row_sum
            else:
                Fs[k, i, :] = 0.0

    return Fs
