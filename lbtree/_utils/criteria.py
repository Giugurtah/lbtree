"""
lbtree/_utils/criteria.py
=========================
Impurity and GPI evaluation functions.

Provides two ``_gpi`` variants:
  - ``_gpi``                    → for SCTree (non-stratified)
  - ``_gpi_stratified``         → for SLBT  (stratified)
  - ``_gpi_stratified_weighted`` → for SLBTWeighted (stratified + sample weights)

Both use the C backends via lbtree._backend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import (
    _contingency_matrix,
    _stratified_contingency,
    _stratified_contingency_weighted,
)


# ============================================================
#  Impurity
# ============================================================

def _impurity(y: pd.Series) -> float:
    """Gini impurity of a label array."""
    dist = np.unique(y, return_counts=True)[1] / len(y)
    return float(1.0 - np.sum(dist ** 2))


def _variance(y: pd.Series) -> float:
    """Variance of a numeric series (impurity measure for twoClass model)."""
    return float(np.var(y.to_numpy(dtype=np.float64)))


def _y_stats(y: pd.Series) -> dict:
    """
    Compute boxplot statistics on the original numeric target y.
    Used by twoClass nodes to drive the gradient boxplot visualisation.

    Returns
    -------
    dict with keys: median, q1, q3, mean, wlo, whi, ymin, ymax
    """
    arr          = y.to_numpy(dtype=np.float64)
    q1, median, q3 = np.percentile(arr, [25, 50, 75])
    iqr          = q3 - q1
    return {
        "median": float(median),
        "q1":     float(q1),
        "q3":     float(q3),
        "mean":   float(arr.mean()),
        "wlo":    float(max(arr.min(), q1 - 1.5 * iqr)),
        "whi":    float(min(arr.max(), q3 + 1.5 * iqr)),
        "ymin":   float(arr.min()),
        "ymax":   float(arr.max()),
    }


def _impurity_weighted(y: pd.Series, w: np.ndarray) -> float:
    """Weighted Gini impurity: 1 - sum(p_j^2) where p_j = sum(w[y==j]) / sum(w)."""
    y_arr = np.asarray(y)
    w_arr = np.asarray(w, dtype=np.float64)
    W     = w_arr.sum()
    if W == 0:
        return 0.0
    classes = np.unique(y_arr)
    p = np.array([w_arr[y_arr == c].sum() / W for c in classes])
    return float(1.0 - np.sum(p ** 2))


def _distribution_weighted(y: pd.Series, w: np.ndarray) -> np.ndarray:
    """Weighted class distribution: p_j = sum(w[y==j]) / sum(w)."""
    y_arr = np.asarray(y)
    w_arr = np.asarray(w, dtype=np.float64)
    W     = w_arr.sum()
    classes = np.unique(y_arr)
    if W == 0:
        return np.ones(len(classes)) / len(classes)
    return np.array([w_arr[y_arr == c].sum() / W for c in classes])


def _get_sizes_weighted_slbt(X: pd.DataFrame, y: pd.Series, w: np.ndarray):
    """
    Weighted version of ``_get_sizes`` for SLBTWeighted.

    Returns
    -------
    n_samples    : int   — unweighted observation count
    n_feats      : int
    n_labels     : int
    impurity     : float — weighted Gini
    distribution : np.ndarray — weighted class frequencies
    """
    n_samples, n_feats = X.shape
    n_labels     = len(np.unique(y))
    impurity     = _impurity_weighted(y, w)
    distribution = _distribution_weighted(y, w)
    return n_samples, n_feats, n_labels, impurity, distribution


def _get_sizes(X: pd.DataFrame, y: pd.Series):
    """
    Return basic dataset statistics used at each tree node.

    Returns
    -------
    n_samples    : int
    n_feats      : int
    n_labels     : int
    impurity     : float
    distribution : np.ndarray — relative class frequencies
    """
    n_samples, n_feats = X.shape
    n_labels    = len(np.unique(y))
    impurity    = _impurity(y)
    distribution = np.unique(y, return_counts=True)[1] / len(y)
    return n_samples, n_feats, n_labels, impurity, distribution


# ============================================================
#  GPI — SCTree (non-stratified)
# ============================================================

def _gpi(X: pd.DataFrame, y: pd.Series):
    """
    Compute the GPI for every feature column (non-stratified) and
    return them sorted descending.

    Uses the lbtree C backend (``liblbtree``).

    Returns
    -------
    gpi_vals  : tuple of floats, sorted descending
    gpi_index : tuple of column names, in the same order
    """
    from lbtree._backend._lbtree import gpi as _c_gpi

    gpi_vals  = []
    gpi_index = []

    for col in X.columns:
        F      = _contingency_matrix(X[col], y)
        I, J   = F.shape
        F_flat = F.ravel().astype(np.float64)
        gpi_val = _c_gpi(I, J, F_flat)
        gpi_vals.append(gpi_val)
        gpi_index.append(col)

    gpi_vals, gpi_index = zip(
        *sorted(zip(gpi_vals, gpi_index), reverse=True)
    )
    return gpi_vals, gpi_index


# ============================================================
#  GPI — SLBT (stratified)
# ============================================================

def _gpi_stratified(X: pd.DataFrame, y: pd.Series, x_s: np.ndarray):
    """
    Compute the stratified GPI for every feature column and return
    them sorted descending.

    Uses the slbt C backend (``libslbt``).

    Returns
    -------
    gpi_vals  : tuple of floats, sorted descending
    gpi_index : tuple of column names, in the same order
    """
    from lbtree._backend._slbt import gpi as _c_gpi_slbt

    gpi_vals  = []
    gpi_index = []

    for col in X.columns:
        Fs     = _stratified_contingency(X[col], y, x_s, norm=False)
        K, I, J = Fs.shape
        Fs_flat = Fs.ravel().astype(np.float64)
        gpi_val = _c_gpi_slbt(K, I, J, Fs_flat)
        gpi_vals.append(gpi_val)
        gpi_index.append(col)

    gpi_vals, gpi_index = zip(
        *sorted(zip(gpi_vals, gpi_index), reverse=True)
    )
    return gpi_vals, gpi_index


def _gpi_stratified_weighted(
    X: pd.DataFrame,
    y: pd.Series,
    x_s: np.ndarray,
    w: np.ndarray,
):
    """
    Weighted stratified GPI ranking.

    Builds weighted joint-frequency matrices (sum of weights per cell)
    and passes them to the same C ``gpi`` function used by SLBT.
    The C function is agnostic to whether the matrix contains raw counts
    or weight sums — it only sees proportions.

    Returns
    -------
    gpi_vals  : tuple of floats, sorted descending
    gpi_index : tuple of column names, in the same order
    """
    from lbtree._backend._slbt import gpi as _c_gpi_slbt

    gpi_vals  = []
    gpi_index = []

    for col in X.columns:
        Fs      = _stratified_contingency_weighted(X[col], y, x_s, w, norm=False)
        K, I, J = Fs.shape
        Fs_flat = Fs.ravel().astype(np.float64)
        gpi_val = _c_gpi_slbt(K, I, J, Fs_flat)
        gpi_vals.append(gpi_val)
        gpi_index.append(col)

    gpi_vals, gpi_index = zip(
        *sorted(zip(gpi_vals, gpi_index), reverse=True)
    )
    return gpi_vals, gpi_index
