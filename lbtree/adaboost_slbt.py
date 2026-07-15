"""
lbtree/adaboost_slbt.py
=======================
SLBTAdaBoostForest — AdaBoost.M1 ensemble of SLBTWeighted.

Algorithm (AdaBoost.M1 — classification only)
----------------------------------------------
  - Error per sample : binary  0 / 1
  - Learner weight   : α = ½ · ln((1 − ε) / ε)
  - Weight update    : wᵢ ∝ wᵢ · exp(α · (2 · 𝟙[wrong] − 1))
  - Final prediction : weighted majority vote

The stratum vector x_s is optional.  When omitted, every observation
is assigned to stratum 0 and each SLBTWeighted tree uses
homogeneity="AB", which is equivalent to a classic weighted LBT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .slbt_weighted import SLBTWeighted


class SLBTAdaBoostForest:
    """
    AdaBoost.M1 ensemble of SLBTWeighted.

    Parameters
    ----------
    homogeneity : {"none", "A", "B", "AB"}, default "none"
        Homogeneity constraint forwarded to each SLBTWeighted.
        Ignored (forced to "AB") when x_s is not supplied to fit().
    n_estimators : int, default 50
        Maximum number of boosting iterations.
    max_depth : int, default 3
        Maximum depth of each weak learner.
    learning_rate : float, default 1.0
        Shrinkage applied to each α_t.
    random_state : int | None
        Unused (kept for API consistency with AdaBoostForest).
    **tree_kwargs
        Additional keyword arguments forwarded to SLBTWeighted
        (e.g. min_ppi, min_gpi, feats_viewed, …).

    Attributes (set after fit)
    --------------------------
    estimators_ : list[SLBTWeighted]
    estimator_weights_ : list[float]   — α_t for each weak learner
    estimator_errors_  : list[float]   — ε_t for each weak learner
    classes_ : np.ndarray
    """

    def __init__(
        self,
        homogeneity: str     = "none",
        n_estimators: int    = 50,
        max_depth: int       = 3,
        learning_rate: float = 1.0,
        random_state         = None,
        **tree_kwargs,
    ):
        if homogeneity not in ("none", "A", "B", "AB"):
            raise ValueError(
                f"homogeneity='{homogeneity}' not supported. "
                "Choose from: 'none', 'A', 'B', 'AB'."
            )
        self.homogeneity   = homogeneity
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate
        self.random_state  = random_state
        self.tree_kwargs   = tree_kwargs

        self.estimators_        : list[SLBTWeighted] = []
        self.estimator_weights_ : list[float]        = []
        self.estimator_errors_  : list[float]        = []
        self.classes_           : np.ndarray | None  = None

    # ================================================================
    #  PUBLIC — fit / predict / predict_proba / staged_predict
    # ================================================================

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        x_s: np.ndarray | None = None,
    ) -> "SLBTAdaBoostForest":
        """
        Train weak learners with AdaBoost.M1.

        Parameters
        ----------
        X   : pd.DataFrame — all-categorical predictors.
        y   : pd.Series    — categorical target.
        x_s : np.ndarray, optional — stratum indicator.
              If None, all observations share stratum 0 and each tree
              uses homogeneity="AB" (classic LBT).

        Returns
        -------
        self
        """
        self.classes_           = np.unique(y)
        self.estimators_        = []
        self.estimator_weights_ = []
        self.estimator_errors_  = []

        n_samples     = len(y)
        sample_weight = np.ones(n_samples, dtype=np.float64) / n_samples

        for t in range(self.n_estimators):
            # ── 1. Fit weak learner ──────────────────────────────────
            tree = SLBTWeighted(
                homogeneity=self.homogeneity,
                max_depth=self.max_depth,
                **self.tree_kwargs,
            )
            tree.fit(X, y, x_s=x_s, sample_weight=sample_weight)
            preds = tree.predict(X)

            # ── 2. Weighted error ε_t ────────────────────────────────
            incorrect = (preds != y.values).astype(np.float64)
            err = float(
                (sample_weight * incorrect).sum() / sample_weight.sum()
            )

            # ── 3. Early stop: learner no better than chance ─────────
            if err >= 0.5:
                if t == 0:
                    self.estimators_.append(tree)
                    self.estimator_weights_.append(1e-10)
                    self.estimator_errors_.append(err)
                    print(f"  AdaBoost-SLBT iter 1: err={err:.4f} >= 0.5, "
                          f"saved with weight~0")
                break

            # ── 4. Learner weight α_t ────────────────────────────────
            err_clip = np.clip(err, 1e-10, 1 - 1e-10)
            alpha    = float(0.5 * np.log((1 - err_clip) / err_clip))
            alpha   *= self.learning_rate

            if (t + 1) % 10 == 0 or t == 0 or t == self.n_estimators - 1:
                print(f"  AdaBoost-SLBT iter {t + 1}/{self.n_estimators}: "
                      f"err={err:.4f}, alpha={alpha:.4f}")

            # ── 5. Update sample weights ─────────────────────────────
            sample_weight = sample_weight * np.exp(alpha * (2 * incorrect - 1))
            total = sample_weight.sum()
            if total > 0:
                sample_weight /= total

            self.estimators_.append(tree)
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(err)

        print(f"AdaBoost-SLBT: {len(self.estimators_)} weak learners fitted.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted majority-vote prediction across all weak learners."""
        self._check_fitted()
        votes = self._compute_vote_matrix(X)
        return self.classes_[np.argmax(votes, axis=1)]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Class probability as normalised weighted votes.

        Returns
        -------
        proba : np.ndarray, shape (n_samples, n_classes)
            Columns ordered as ``self.classes_``.
        """
        self._check_fitted()
        votes    = self._compute_vote_matrix(X)
        row_sums = votes.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return votes / row_sums

    def staged_predict(self, X: pd.DataFrame):
        """
        Generator that yields predictions after each boosting stage.
        Useful for selecting the optimal number of estimators.
        """
        self._check_fitted()
        n_samples    = len(X)
        n_classes    = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        votes        = np.zeros((n_samples, n_classes))
        for tree, alpha in zip(self.estimators_, self.estimator_weights_):
            preds = tree.predict(X)
            for i, pred in enumerate(preds):
                if pred in class_to_idx:
                    votes[i, class_to_idx[pred]] += alpha
            yield self.classes_[np.argmax(votes, axis=1)]

    def get_params(self, deep: bool = True) -> dict:
        return {
            "homogeneity":   self.homogeneity,
            "n_estimators":  self.n_estimators,
            "max_depth":     self.max_depth,
            "learning_rate": self.learning_rate,
            "random_state":  self.random_state,
            **self.tree_kwargs,
        }

    def set_params(self, **params) -> "SLBTAdaBoostForest":
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                self.tree_kwargs[k] = v
        return self

    # ================================================================
    #  PRIVATE
    # ================================================================

    def _compute_vote_matrix(self, X: pd.DataFrame) -> np.ndarray:
        n_samples    = len(X)
        n_classes    = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        votes        = np.zeros((n_samples, n_classes))
        for tree, alpha in zip(self.estimators_, self.estimator_weights_):
            preds = tree.predict(X)
            for i, pred in enumerate(preds):
                if pred in class_to_idx:
                    votes[i, class_to_idx[pred]] += alpha
        return votes

    def _check_fitted(self) -> None:
        if not self.estimators_:
            raise RuntimeError(
                "SLBTAdaBoostForest not fitted. Call fit() first."
            )

    # ================================================================
    #  REPR
    # ================================================================

    def __repr__(self) -> str:
        status = (f"{len(self.estimators_)} weak learners fitted"
                  if self.estimators_ else "not fitted")
        return (
            f"SLBTAdaBoostForest("
            f"homogeneity='{self.homogeneity}', "
            f"n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, "
            f"learning_rate={self.learning_rate}, "
            f"[{status}])"
        )
