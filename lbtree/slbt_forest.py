"""
lbtree/slbt_forest.py
=====================
SLBTForest — Random Forest of SLBT with PPI-based feature importance.

Algorithm overview
------------------
Each tree is trained on a bootstrap sample of the data (rows and stratum
labels sampled jointly) with a random feature subset.  The final prediction
is a majority vote across all trees (SLBT is classification-only).

Feature sampling modes
----------------------
"tree" (random subspace): a fixed feature subset is drawn once per tree and
    the tree only ever sees those columns (existing behaviour).
"node" (true RF): all columns reach each tree, but at every node a fresh
    random subset of ``max_features`` columns is drawn before the stratified
    GPI ranking.  This matches the canonical Random Forest algorithm.

PPI importance
--------------
For each internal node *t* that splits on feature *x_j*:

    contribution(t, x_j) = (N_t / N_root) * pi_t

The importance of x_j is the sum of contributions across all nodes that
use it, averaged over all trees and normalised to sum to 1.

OOB score
---------
Each bootstrap sample leaves ~36.8 % of observations out-of-bag.
OOB score = accuracy of majority-vote predictions on OOB samples.
The stratum vector x_s is sliced consistently with X and y for OOB
predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from collections import Counter

from .slbt import SLBT


class SLBTForest:
    """
    Random Forest of SLBT with PPI-weighted feature importance.

    Parameters
    ----------
    n_estimators : int, default 10
        Number of trees.
    homogeneity : {"none", "A", "B", "AB"}, default "none"
        Homogeneity constraint forwarded to each SLBT.
    max_features : int | float | "sqrt" | "log2" | None, default "sqrt"
        Number of features considered at each split (node mode) or per tree
        (tree mode):
        - int   → exact count
        - float → fraction of total columns
        - "sqrt"→ floor(sqrt(p))
        - "log2"→ floor(log2(p))
        - None  → all features (no subsampling)
    feature_sampling : {"tree", "node"}, default "tree"
        "tree" — random subspace: columns are subsampled once per tree.
        "node" — true RF: all columns reach each tree, a fresh random subset
                 is drawn at every node before the stratified GPI ranking.
    bootstrap : bool, default True
        If True, each tree is trained on a bootstrap sample of rows.
        If False, all rows are used (only feature subsampling applies).
    random_state : int | None
        Seed for reproducibility.
    **tree_kwargs
        Additional keyword arguments forwarded to each SLBT
        (e.g. min_ppi, max_depth, feats_viewed, …).

    Attributes (set after fit)
    --------------------------
    estimators_ : list[SLBT]
    estimators_features_ : list[list[str]]
    ppi_importance_ : dict[str, float]
        Feature importance normalised to sum 1, sorted descending.
    oob_score_ : float | None
        OOB accuracy (only when bootstrap=True).
    classes_ : np.ndarray
        Unique class labels seen during fit.
    """

    def __init__(
        self,
        n_estimators: int      = 10,
        homogeneity: str       = "none",
        max_features           = "sqrt",
        feature_sampling: str  = "tree",
        bootstrap: bool        = True,
        random_state           = None,
        **tree_kwargs,
    ):
        if homogeneity not in ("none", "A", "B", "AB"):
            raise ValueError(
                f"homogeneity='{homogeneity}' not supported. "
                "Choose from: 'none', 'A', 'B', 'AB'."
            )
        if feature_sampling not in ("tree", "node"):
            raise ValueError(
                f"feature_sampling='{feature_sampling}' not supported. "
                "Choose 'tree' (random subspace, default) or 'node' (true RF)."
            )
        self.n_estimators      = n_estimators
        self.homogeneity       = homogeneity
        self.max_features      = max_features
        self.feature_sampling  = feature_sampling
        self.bootstrap         = bootstrap
        self.random_state      = random_state
        self.tree_kwargs       = tree_kwargs

        self.estimators_          : list[SLBT]          = []
        self.estimators_features_ : list[list[str]]     = []
        self.ppi_importance_      : dict                = {}
        self.oob_score_           : float | None        = None
        self.classes_             : np.ndarray | None   = None

        self._rng = np.random.default_rng(random_state)

    # ================================================================
    #  PUBLIC — fit / predict / predict_proba
    # ================================================================

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        x_s: np.ndarray | None = None,
    ) -> "SLBTForest":
        """
        Train ``n_estimators`` SLBT trees on bootstrap samples of (X, y, x_s).

        Parameters
        ----------
        X   : pd.DataFrame — all-categorical predictor matrix.
        y   : pd.Series    — categorical target.
        x_s : np.ndarray, shape (n_samples,), optional
            Stratum indicator (integer per observation).
            If None, all observations are assigned to stratum 0 and each
            tree uses homogeneity="AB" (non-stratified LBA), exactly as
            SLBT.fit() does when x_s is None.

        Returns
        -------
        self
        """
        n_samples = len(y)
        self.classes_             = np.unique(y)
        self.estimators_          = []
        self.estimators_features_ = []

        # Normalise x_s: if None, use a single stratum (SLBT will set AB)
        if x_s is None:
            x_s_arr = np.zeros(n_samples, dtype=int)
        else:
            x_s_arr = np.asarray(x_s)

        n_feats           = X.shape[1]
        n_features_to_use = self._resolve_max_features(n_feats)
        all_columns       = list(X.columns)

        # OOB: per-sample list of predictions from trees that did not see it
        oob_preds = [[] for _ in range(n_samples)]

        for b in range(self.n_estimators):
            if (b + 1) % 10 == 0 or b == 0:
                print(f"  SLBTForest: {b + 1}/{self.n_estimators} trees fitted")

            # 1. Bootstrap row sampling
            if self.bootstrap:
                boot_idx = self._rng.integers(0, n_samples, size=n_samples)
            else:
                boot_idx = np.arange(n_samples)

            oob_idx = np.setdiff1d(np.arange(n_samples), boot_idx)

            # 2. Feature subsampling + tree construction
            if self.feature_sampling == "tree":
                # Random subspace: subsample columns once per tree
                feat_idx   = self._rng.choice(n_feats, size=n_features_to_use, replace=False)
                feat_names = list(X.columns[feat_idx])
                X_boot     = X.iloc[boot_idx][feat_names].reset_index(drop=True)
                y_boot     = y.iloc[boot_idx].reset_index(drop=True)
                xs_boot    = x_s_arr[boot_idx]
                tree       = SLBT(homogeneity=self.homogeneity, **self.tree_kwargs)
            else:
                # True RF: all columns reach the tree; each node subsamples independently
                feat_names = list(X.columns)
                X_boot     = X.iloc[boot_idx].reset_index(drop=True)
                y_boot     = y.iloc[boot_idx].reset_index(drop=True)
                xs_boot    = x_s_arr[boot_idx]
                tree_seed  = int(self._rng.integers(0, 2**31))
                tree_rng   = np.random.default_rng(tree_seed)
                tree       = SLBT(
                    homogeneity=self.homogeneity,
                    node_max_features=n_features_to_use,
                    rng=tree_rng,
                    **self.tree_kwargs,
                )

            # 3. Fit single SLBT
            tree.fit(X_boot, y_boot, x_s=xs_boot)
            self.estimators_.append(tree)
            self.estimators_features_.append(feat_names)

            # 4. OOB predictions
            if self.bootstrap and len(oob_idx) > 0:
                X_oob  = X.iloc[oob_idx][feat_names].reset_index(drop=True)
                preds  = tree.predict(X_oob)
                for local_i, global_i in enumerate(oob_idx):
                    oob_preds[global_i].append(preds[local_i])

        # 5. OOB score
        if self.bootstrap:
            self.oob_score_ = self._compute_oob_score(y, oob_preds)
            print(f"\nOOB Score: {self.oob_score_:.4f}")

        # 6. PPI importance
        self.ppi_importance_ = self._compute_ppi_importance(all_columns)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Majority-vote prediction across all trees.
        """
        self._check_fitted()
        votes = [Counter() for _ in range(len(X))]
        for tree, feat_names in zip(self.estimators_, self.estimators_features_):
            available = [f for f in feat_names if f in X.columns]
            preds     = tree.predict(X[available].reset_index(drop=True))
            for i, pred in enumerate(preds):
                votes[i][pred] += 1
        return np.array([v.most_common(1)[0][0] for v in votes])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Class probability as vote fraction.

        Returns
        -------
        proba : np.ndarray, shape (n_samples, n_classes)
            Columns ordered as ``self.classes_``.
        """
        self._check_fitted()
        n_samples    = len(X)
        n_classes    = len(self.classes_)
        proba        = np.zeros((n_samples, n_classes))
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        for tree, feat_names in zip(self.estimators_, self.estimators_features_):
            available = [f for f in feat_names if f in X.columns]
            preds     = tree.predict(X[available].reset_index(drop=True))
            for i, pred in enumerate(preds):
                if pred in class_to_idx:
                    proba[i, class_to_idx[pred]] += 1

        proba /= len(self.estimators_)
        return proba

    def get_params(self, deep: bool = True) -> dict:
        return {
            "n_estimators":    self.n_estimators,
            "homogeneity":     self.homogeneity,
            "max_features":    self.max_features,
            "feature_sampling": self.feature_sampling,
            "bootstrap":       self.bootstrap,
            "random_state":    self.random_state,
            **self.tree_kwargs,
        }

    def set_params(self, **params) -> "SLBTForest":
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                self.tree_kwargs[k] = v
        return self

    # ================================================================
    #  PRIVATE — PPI importance
    # ================================================================

    def _compute_ppi_importance(self, all_columns: list) -> dict:
        importances = {col: 0.0 for col in all_columns}
        for tree in self.estimators_:
            if tree.root is None:
                continue
            root_N = tree.root.N
            self._accumulate_ppi_importance(tree.root, importances, root_N)
        for col in importances:
            importances[col] /= self.n_estimators
        total = sum(importances.values())
        if total > 0:
            importances = {k: v / total for k, v in importances.items()}
        return dict(sorted(importances.items(), key=lambda x: -x[1]))

    def _accumulate_ppi_importance(self, node, importances: dict, root_N: int) -> None:
        if node is None or node._is_leaf_node():
            return
        if (node.feature in importances
                and node.pi is not None
                and node.N  is not None):
            importances[node.feature] += (node.N / root_N) * node.pi
        self._accumulate_ppi_importance(node.left,  importances, root_N)
        self._accumulate_ppi_importance(node.right, importances, root_N)

    # ================================================================
    #  PRIVATE — OOB
    # ================================================================

    @staticmethod
    def _compute_oob_score(y: pd.Series, oob_preds: list) -> float:
        correct = counted = 0
        for i, preds in enumerate(oob_preds):
            if preds:
                majority = Counter(preds).most_common(1)[0][0]
                if majority == y.iloc[i]:
                    correct += 1
                counted += 1
        return correct / counted if counted > 0 else float("nan")

    # ================================================================
    #  PRIVATE — helpers
    # ================================================================

    def _resolve_max_features(self, n_feats: int) -> int:
        mf = self.max_features
        if mf is None:            return n_feats
        if isinstance(mf, int):   return min(mf, n_feats)
        if isinstance(mf, float): return max(1, int(mf * n_feats))
        if mf == "sqrt":          return max(1, int(np.sqrt(n_feats)))
        if mf == "log2":          return max(1, int(np.log2(n_feats)))
        raise ValueError(f"max_features='{mf}' is not valid.")

    def _check_fitted(self) -> None:
        if not self.estimators_:
            raise RuntimeError("SLBTForest not fitted. Call fit() first.")

    # ================================================================
    #  REPR
    # ================================================================

    def __repr__(self) -> str:
        status = (f"{len(self.estimators_)} trees fitted"
                  if self.estimators_ else "not fitted")
        return (
            f"SLBTForest("
            f"n_estimators={self.n_estimators}, "
            f"homogeneity='{self.homogeneity}', "
            f"max_features={self.max_features!r}, "
            f"feature_sampling='{self.feature_sampling}', "
            f"bootstrap={self.bootstrap}, "
            f"[{status}])"
        )
