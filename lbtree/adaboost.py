"""
lbtree/adaboost.py
==================
AdaBoostForest — ensemble of SCTreeWeighted.

Supports two boosting algorithms selected via ``model``:

AdaBoost.M1  (model="twoStage" | "twoing") — classification
-------------------------------------------------------------
  - Error per sample : binary  0 / 1
  - Learner weight   : α = ½ · ln((1 − ε) / ε)
  - Weight update    : wᵢ ∝ wᵢ · exp(α · (2 · 𝟙[wrong] − 1))
  - Final prediction : weighted majority vote

AdaBoost.R2  (model="twoClass") — regression
---------------------------------------------
  - Error per sample : Lᵢ = |yᵢ − ĥ(xᵢ)| / max|yⱼ − ĥ(xⱼ)|  ∈ [0, 1]
  - Learner weight   : β = ε / (1 − ε)          (small β = good learner)
  - Weight update    : wᵢ ∝ wᵢ · β^(1 − Lᵢ)
  - Final prediction : weighted median of tree predictions
                       with per-tree weights log(1 / β)

Note on categorical data
------------------------
AdaBoost with depth-1 stumps performs poorly on multi-class categorical
predictors.  Prefer max_depth >= 2 (recommended: 3–4) or use
SCTreeForest (Random Forest) which is more robust.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .sctree_weighted import SCTreeWeighted


class AdaBoostForest:
    """
    AdaBoost ensemble of SCTreeWeighted.

    Uses AdaBoost.M1 for classification (model="twoStage" / "twoing") and
    AdaBoost.R2 for regression (model="twoClass").

    Parameters
    ----------
    model : {"twoStage", "twoing", "twoClass"}, default "twoStage"
        Splitting algorithm forwarded to each SCTreeWeighted.
        Selects AdaBoost.M1 (classification) or AdaBoost.R2 (regression).
    n_estimators : int, default 50
        Maximum number of boosting iterations (weak learners).
    max_depth : int, default 3
        Maximum depth of each weak learner.
        For multi-class categorical data, use >= 2.
    learning_rate : float, default 1.0
        Shrinkage applied to each alpha_t / log(1/beta_t).
    random_state : int | None
        Seed for reproducibility.
    **tree_kwargs
        Additional keyword arguments forwarded to SCTreeWeighted
        (e.g. min_ppi, min_gpi, feats_viewed, …).

    Attributes (set after fit)
    --------------------------
    estimators_ : list[SCTreeWeighted]
    estimator_weights_ : list[float]
        α_t  (M1) or log(1/β_t) (R2) for each weak learner.
    estimator_errors_ : list[float]
        Weighted error ε_t of each weak learner.
    classes_ : np.ndarray
        Unique class labels (M1) or unique target values (R2).
    """

    def __init__(
        self,
        model: str           = "twoStage",
        n_estimators: int    = 50,
        max_depth: int       = 3,
        learning_rate: float = 1.0,
        random_state         = None,
        **tree_kwargs,
    ):
        if model not in ("twoStage", "twoing", "twoClass"):
            raise ValueError(
                f"model='{model}' not supported. "
                "Choose 'twoStage', 'twoing', or 'twoClass'."
            )
        self.model         = model
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate
        self.random_state  = random_state
        self.tree_kwargs   = tree_kwargs

        self.estimators_        : list[SCTreeWeighted] = []
        self.estimator_weights_ : list[float]          = []
        self.estimator_errors_  : list[float]          = []
        self.classes_           : np.ndarray | None    = None

        self._rng = np.random.default_rng(random_state)

    # ================================================================
    #  PUBLIC — fit / predict / predict_proba / staged_predict
    # ================================================================

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AdaBoostForest":
        """
        Train weak learners with AdaBoost.M1 (classification) or
        AdaBoost.R2 (twoClass / regression).

        Returns
        -------
        self
        """
        self.classes_           = np.unique(y)
        self.estimators_        = []
        self.estimator_weights_ = []
        self.estimator_errors_  = []

        n_samples     = len(y)
        sample_weight = np.ones(n_samples) / n_samples

        for t in range(self.n_estimators):
            # ── 1. Fit weak learner ──────────────────────────────────
            tree = SCTreeWeighted(
                model=self.model, max_depth=self.max_depth, **self.tree_kwargs
            )
            tree.fit(X, y, sample_weight=sample_weight)
            preds = tree.predict(X)

            # ── 2. Weighted error ε_t ────────────────────────────────
            err = self._compute_weighted_error(preds, y, sample_weight)

            # ── 3. Early stop: learner no better than chance ─────────
            if err >= 0.5:
                if t == 0:
                    self.estimators_.append(tree)
                    self.estimator_weights_.append(1e-10)
                    self.estimator_errors_.append(err)
                    print(f"  AdaBoost iter 1: err={err:.4f} >= 0.5, "
                          f"saved with weight~0")
                break

            # ── 4. Learner weight ────────────────────────────────────
            weight = self._compute_learner_weight(err)
            weight *= self.learning_rate

            if (t + 1) % 10 == 0 or t == 0 or t == self.n_estimators - 1:
                label = "alpha" if self.model != "twoClass" else "log(1/β)"
                print(f"  AdaBoost iter {t + 1}/{self.n_estimators}: "
                      f"err={err:.4f}, {label}={weight:.4f}")

            # ── 5. Update sample weights ─────────────────────────────
            sample_weight = self._update_sample_weights(
                sample_weight, preds, y, weight
            )

            self.estimators_.append(tree)
            self.estimator_weights_.append(weight)
            self.estimator_errors_.append(err)

        print(f"AdaBoost: {len(self.estimators_)} weak learners fitted.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prediction across all weak learners.

        - Classification (twoStage / twoing): weighted majority vote.
        - Regression (twoClass): weighted median of leaf predictions.
        """
        if not self.estimators_:
            raise RuntimeError("AdaBoostForest not fitted. Call fit() first.")
        if self.model == "twoClass":
            return self._predict_regression(X)
        votes = self._compute_vote_matrix(X)
        return self.classes_[np.argmax(votes, axis=1)]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Class probability as normalised weighted votes (classification only).

        Raises
        ------
        NotImplementedError
            When ``model="twoClass"``.

        Returns
        -------
        proba : np.ndarray, shape (n_samples, n_classes)
        """
        if not self.estimators_:
            raise RuntimeError("AdaBoostForest not fitted.")
        if self.model == "twoClass":
            raise NotImplementedError(
                "predict_proba() is not available for model='twoClass' "
                "(regression mode). Use predict() to get the weighted-median "
                "numeric prediction."
            )
        votes    = self._compute_vote_matrix(X)
        row_sums = votes.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return votes / row_sums

    def staged_predict(self, X: pd.DataFrame):
        """
        Generator that yields predictions after each boosting stage.

        For twoClass yields the weighted-median after each stage.
        Useful for selecting the optimal number of estimators.
        """
        if not self.estimators_:
            raise RuntimeError("AdaBoostForest not fitted.")

        if self.model == "twoClass":
            # Accumulate (prediction, weight) pairs per sample, yield
            # weighted median after each stage.
            n_samples    = len(X)
            all_preds    = []   # list of np.ndarray, one per stage
            all_weights  = []   # list of float
            for tree, w in zip(self.estimators_, self.estimator_weights_):
                all_preds.append(tree.predict(X).astype(np.float64))
                all_weights.append(w)
                yield self._weighted_median(
                    np.column_stack(all_preds),
                    np.array(all_weights),
                )
        else:
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
            "model":        self.model,
            "n_estimators": self.n_estimators,
            "max_depth":    self.max_depth,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
            **self.tree_kwargs,
        }

    def set_params(self, **params) -> "AdaBoostForest":
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                self.tree_kwargs[k] = v
        return self

    # ================================================================
    #  PRIVATE — error / weight dispatch (M1 vs R2)
    # ================================================================

    def _compute_weighted_error(
        self,
        preds: np.ndarray,
        y: pd.Series,
        sample_weight: np.ndarray,
    ) -> float:
        """
        Compute the weighted error ε_t.

        M1 (classification): ε = Σ wᵢ · 𝟙[ĥ(xᵢ) ≠ yᵢ]
        R2 (twoClass)       : ε = Σ wᵢ · Lᵢ
                              where Lᵢ = |yᵢ − ĥ(xᵢ)| / max|yⱼ − ĥ(xⱼ)|
        """
        if self.model == "twoClass":
            loss = np.abs(preds.astype(np.float64) - y.to_numpy(np.float64))
            max_loss = loss.max()
            if max_loss > 0:
                loss /= max_loss
            # sample_weight is already normalised (sums to 1)
            return float((sample_weight * loss).sum())
        else:
            incorrect = (preds != y.values).astype(np.float64)
            return float(
                (sample_weight * incorrect).sum() / sample_weight.sum()
            )

    def _compute_learner_weight(self, err: float) -> float:
        """
        Compute the learner weight from ε_t.

        M1: α = ½ · ln((1 − ε) / ε)
        R2: log(1/β) = log((1 − ε) / ε)   [note: twice the M1 alpha]
            stored as log(1/β) so it can be used directly as the
            weight in the weighted-median aggregation.
        """
        err_clip = np.clip(err, 1e-10, 1 - 1e-10)
        if self.model == "twoClass":
            # β = ε / (1 − ε)  →  log(1/β) = log((1−ε)/ε)
            return float(np.log((1 - err_clip) / err_clip))
        else:
            return float(0.5 * np.log((1 - err_clip) / err_clip))

    def _update_sample_weights(
        self,
        sample_weight: np.ndarray,
        preds: np.ndarray,
        y: pd.Series,
        weight: float,
    ) -> np.ndarray:
        """
        Update and renormalise sample weights.

        M1: wᵢ ∝ wᵢ · exp(α · (2 · 𝟙[wrong] − 1))
            → wrong samples multiplied by e^α, correct by e^{−α}

        R2: wᵢ ∝ wᵢ · β^(1 − Lᵢ)
            where β = e^{−log(1/β)} = e^{−weight}
            → wᵢ ∝ wᵢ · exp(−weight · (1 − Lᵢ))
            → well-predicted samples (Lᵢ ≈ 0) get weight reduced,
               poorly-predicted (Lᵢ ≈ 1) get weight unchanged.
        """
        if self.model == "twoClass":
            loss = np.abs(preds.astype(np.float64) - y.to_numpy(np.float64))
            max_loss = loss.max()
            if max_loss > 0:
                loss /= max_loss
            # β^(1−Lᵢ) = exp(−weight · (1 − Lᵢ))
            sample_weight = sample_weight * np.exp(-weight * (1.0 - loss))
        else:
            incorrect     = (preds != y.values).astype(np.float64)
            sample_weight = sample_weight * np.exp(weight * (2 * incorrect - 1))

        total = sample_weight.sum()
        if total > 0:
            sample_weight /= total
        return sample_weight

    # ================================================================
    #  PRIVATE — aggregation
    # ================================================================

    def _compute_vote_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted vote matrix for classification (M1)."""
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

    def _predict_regression(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted median of leaf predictions across all trees (R2)."""
        n_samples = len(X)
        T         = len(self.estimators_)
        preds_mat = np.zeros((n_samples, T), dtype=np.float64)
        for j, (tree, _) in enumerate(
            zip(self.estimators_, self.estimator_weights_)
        ):
            preds_mat[:, j] = tree.predict(X).astype(np.float64)
        weights = np.array(self.estimator_weights_, dtype=np.float64)
        return self._weighted_median(preds_mat, weights)

    @staticmethod
    def _weighted_median(preds_mat: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Compute the weighted median row-wise.

        Parameters
        ----------
        preds_mat : np.ndarray, shape (n_samples, T)
            Predictions of T weak learners for each sample.
        weights : np.ndarray, shape (T,)
            Per-learner weights log(1/β_t).

        Returns
        -------
        medians : np.ndarray, shape (n_samples,)
        """
        weights  = weights / weights.sum()          # normalise
        n_samples = preds_mat.shape[0]
        medians   = np.empty(n_samples, dtype=np.float64)

        for i in range(n_samples):
            row      = preds_mat[i]
            sort_idx = np.argsort(row)
            sorted_p = row[sort_idx]
            sorted_w = weights[sort_idx]
            cumw     = np.cumsum(sorted_w)
            # weighted median: smallest value whose cumulative weight >= 0.5
            idx      = np.searchsorted(cumw, 0.5)
            idx      = min(idx, len(sorted_p) - 1)
            medians[i] = sorted_p[idx]

        return medians

    # ================================================================
    #  REPR
    # ================================================================

    def __repr__(self) -> str:
        status = (f"{len(self.estimators_)} weak learners fitted"
                  if self.estimators_ else "not fitted")
        return (
            f"AdaBoostForest("
            f"model='{self.model}', "
            f"n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, "
            f"learning_rate={self.learning_rate}, "
            f"[{status}])"
        )
