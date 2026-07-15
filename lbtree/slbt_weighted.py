"""
lbtree/slbt_weighted.py
=======================
SLBTWeighted — SLBT with per-sample weights (for AdaBoost).

Identical in structure to SLBT but every computation that depends on
observation counts is replaced by its weight-sum equivalent:

  - Gini impurity    → weighted Gini
  - GPI ranking      → weighted stratified GPI
  - Contingency mats → weighted stratified contingency
  - Leaf prediction  → argmax of per-class weight sums (= weighted majority)

The C backend functions (slba, gpi from libslbt) are completely agnostic
to weighting: they operate on float matrices and do not know whether
cells contain raw counts or weight sums.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base      import BaseLBTree, Node
from .reporting import TreeReporter
from ._utils.criteria import (
    _impurity_weighted,
    _distribution_weighted,
    _get_sizes_weighted_slbt,
    _gpi_stratified_weighted,
)
from ._utils.utils import _stratified_contingency_weighted
from ._tree.split  import score_slbt, _split, _splitS
from ._tree._homogeneity.base import get_homogeneity_strategy


class SLBTWeighted(BaseLBTree):
    """
    SLBT with per-sample weights — weak learner for SLBTAdaBoostForest.

    Parameters are identical to SLBT.  The only difference is that
    ``fit()`` accepts an optional ``sample_weight`` array which is
    propagated through every computation inside the tree.

    Parameters
    ----------
    homogeneity : {"none", "A", "B", "AB"}, default "none"
    min_ppi : float, default 0.001
    min_gpi : float, default 0.001
    min_impurity : float, default 0.001
    min_samples_split : int, default 2
    max_depth : int, default 5
    feats_viewed : int, default 10
    FAST : bool, default True
    """

    def __init__(
        self,
        homogeneity: str       = "none",
        min_ppi: float         = 0.001,
        min_gpi: float         = 0.001,
        min_impurity: float    = 0.001,
        min_samples_split: int = 2,
        max_depth: int         = 5,
        feats_viewed: int      = 10,
        FAST: bool             = True,
    ):
        super().__init__(
            min_ppi=min_ppi, min_gpi=min_gpi,
            min_impurity=min_impurity,
            min_samples_split=min_samples_split,
            max_depth=max_depth, feats_viewed=feats_viewed, FAST=FAST,
        )
        if homogeneity not in ("none", "A", "B", "AB"):
            raise ValueError(
                f"homogeneity='{homogeneity}' not supported. "
                "Choose from: 'none', 'A', 'B', 'AB'."
            )
        self.homogeneity = homogeneity

    def get_params(self, deep: bool = True) -> dict:
        p = super().get_params(deep)
        p["homogeneity"] = self.homogeneity
        p.pop("model", None)
        return p

    # ================================================================
    #  PUBLIC — fit / predict
    # ================================================================

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        x_s: np.ndarray | None  = None,
        sample_weight: np.ndarray | None = None,
    ) -> "SLBTWeighted":
        """
        Fit a weighted SLBT.

        Parameters
        ----------
        X             : pd.DataFrame — all-categorical predictors.
        y             : pd.Series    — categorical target.
        x_s           : np.ndarray, optional — stratum indicator.
                        If None, all observations share stratum 0 and
                        homogeneity is forced to "AB" (classic LBT).
        sample_weight : np.ndarray, optional — per-sample weights.
                        If None, uniform weights 1/n are used.

        Returns
        -------
        self
        """
        n = len(y)

        if sample_weight is None:
            sample_weight = np.ones(n, dtype=np.float64) / n
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            s = sample_weight.sum()
            if s > 0:
                sample_weight = sample_weight / s   # normalise to sum=1

        if x_s is None:
            x_s              = np.zeros(n, dtype=int)
            self.homogeneity = "AB"

        strategy = get_homogeneity_strategy(self.homogeneity)

        self.targhet_dist = [
            np.unique(y),
            _distribution_weighted(y, sample_weight),
        ]
        self.root_N   = n
        self.reporter = TreeReporter(homogeneity=self.homogeneity, decimals=4)

        W_root    = sample_weight.sum()          # 1.0 after normalisation
        root_imp  = _impurity_weighted(y, sample_weight)

        self.root = self._grow_tree(
            strategy, X, y, x_s, sample_weight,
            root_impurity=root_imp, W_root=W_root,
        )
        self._calculate_tree_partial_impurity_reduction()
        self._calculate_tree_partial_tau_reduction()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels for X."""
        if self.root is None:
            raise RuntimeError("Tree not fitted. Call fit() first.")
        return np.array([
            self._traverse_tree(X.iloc[i], self.root)
            for i in range(len(X))
        ])

    # ================================================================
    #  PRIVATE — tree growth
    # ================================================================

    def _grow_tree(
        self,
        strategy,
        X: pd.DataFrame,
        y: pd.Series,
        x_s: np.ndarray,
        w: np.ndarray,
        root_impurity: float  = 1.0,
        W_root: float         = 1.0,
        depth: int            = 0,
        pos: int              = 1,
        parent_cum_tau: float = 0.0,
    ) -> Node:

        X = self._drop_constant_columns(X)
        n_samples, n_feats, n_labels, impurity, distribution = (
            _get_sizes_weighted_slbt(X, y, w)
        )

        W_node = w.sum()
        impurity_decrease = (
            (root_impurity - impurity * W_node / W_root) / root_impurity
            if root_impurity > 0 else 0.0
        )

        cumulative_path_tau = parent_cum_tau

        # --- pre-split stopping criteria ---
        leaf = self._check_criteria_before(
            y, w, pos, impurity, distribution, depth,
            n_labels, n_samples, n_feats,
            impurity_decrease,
            parent_cum_tau=cumulative_path_tau,
        )
        if leaf is not None:
            return leaf

        # --- weighted stratified GPI ranking ---
        gpi_vals, gpi_order = _gpi_stratified_weighted(X, y, x_s, w)

        # --- best split search ---
        best_feature, best_treshold, best_pi, best_gpi, best_alpha, best_beta = (
            self._find_best_predictor(X, y, x_s, w, gpi_order, gpi_vals)
        )

        # --- post-split stopping criteria ---
        leaf = self._check_criteria_after(
            y, w, pos, impurity, distribution, depth,
            best_gpi, best_pi,
            impurity_decrease,
            parent_cum_tau=cumulative_path_tau,
        )
        if leaf is not None:
            return leaf

        tau_decrease  = best_pi * (1.0 - impurity_decrease)
        child_cum_tau = cumulative_path_tau + tau_decrease

        x_vals    = np.unique(X[best_feature])
        thresholds = strategy.get_treshold_values(best_treshold, x_vals, x_s)
        indexL, indexR = strategy.split(X[best_feature], x_s, thresholds)
        lift1, lift2   = strategy.compute_lift(best_beta, distribution)

        left  = self._grow_tree(
            strategy,
            X.loc[indexL], y[indexL], x_s[indexL], w[indexL],
            root_impurity, W_root, depth + 1, 2 * pos,
            parent_cum_tau=child_cum_tau,
        )
        right = self._grow_tree(
            strategy,
            X.loc[indexR], y[indexR], x_s[indexR], w[indexR],
            root_impurity, W_root, depth + 1, 2 * pos + 1,
            parent_cum_tau=child_cum_tau,
        )

        node = Node(
            gpi=best_gpi, pi=best_pi, position=pos,
            feature=best_feature, treshold=thresholds,
            left=left, right=right,
            impurity=impurity,
            impurity_decrease=impurity_decrease,
            tree_partial_impurity_reduction=0.0,
            tau_decrease=tau_decrease,
            cumulative_path_tau=cumulative_path_tau,
            tree_partial_tau_reduction=0.0,
            distribution=distribution, N=n_samples, labels=np.unique(y),
            LIFT_1=lift1, LIFT_2=lift2,
            GCR=None,
            strat_labels=np.unique(x_s),
        )
        if self.reporter is not None:
            self.reporter.add_node(node, is_leaf=False)
        return node

    # ================================================================
    #  PRIVATE — best predictor search
    # ================================================================

    def _find_best_predictor(self, X, y, x_s, w, gpi_order, gpi_vals):
        best = {
            "feature":   None,
            "threshold": None,
            "pi":        -np.inf,
            "gpi":       -np.inf,
            "alpha":     None,
            "beta":      None,
        }

        for rank, col in enumerate(gpi_order[: self.feats_viewed]):
            Fs_noN = _stratified_contingency_weighted(X[col], y, x_s, w, norm=False)
            Fs     = _stratified_contingency_weighted(X[col], y, x_s, w, norm=True)

            pi, S, alpha, beta = score_slbt(Fs_noN, Fs, self.homogeneity)

            if pi > best["pi"]:
                best["pi"]        = pi
                best["feature"]   = str(col)
                best["threshold"] = S
                best["alpha"]     = alpha
                best["beta"]      = beta
                best["gpi"]       = gpi_vals[rank]

            if (self.FAST
                    and rank < len(gpi_order) - 1
                    and best["pi"] > gpi_vals[rank + 1]):
                break

        return (
            best["feature"],
            best["threshold"],
            best["pi"],
            best["gpi"],
            best["alpha"],
            best["beta"],
        )

    # ================================================================
    #  PRIVATE — stopping criteria
    # ================================================================

    def _check_criteria_before(
        self, y, w, pos, impurity, distribution, depth,
        n_labels, n_samples, n_feats,
        impurity_decrease,
        parent_cum_tau=0.0,
    ):
        if (depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_samples_split
                or n_feats  == 0
                or impurity  < self.min_impurity):
            return self._make_leaf(
                y, w, pos, impurity, distribution,
                impurity_decrease, parent_cum_tau=parent_cum_tau,
            )
        return None

    def _check_criteria_after(
        self, y, w, pos, impurity, distribution, depth,
        best_gpi, best_pi,
        impurity_decrease,
        parent_cum_tau=0.0,
    ):
        if best_gpi < self.min_gpi or best_pi < self.min_ppi or best_pi < 1e-8:
            return self._make_leaf(
                y, w, pos, impurity, distribution,
                impurity_decrease, parent_cum_tau=parent_cum_tau,
            )
        return None

    def _make_leaf(
        self, y, w, pos, impurity, distribution,
        impurity_decrease, parent_cum_tau=0.0,
    ) -> Node:
        # Weighted majority: class with highest total weight
        y_arr   = np.asarray(y)
        w_arr   = np.asarray(w, dtype=np.float64)
        classes = np.unique(y_arr)
        class_weights = {c: w_arr[y_arr == c].sum() for c in classes}
        value   = max(class_weights, key=class_weights.get)

        leaf = Node(
            position=pos,
            value=value,
            impurity=impurity,
            impurity_decrease=impurity_decrease,
            tree_partial_impurity_reduction=0.0,
            tau_decrease=0.0,
            cumulative_path_tau=parent_cum_tau,
            tree_partial_tau_reduction=0.0,
            distribution=distribution,
            N=len(y),
            labels=classes,
            GCR=self._get_gcr(distribution, classes),
        )
        if self.reporter is not None:
            self.reporter.add_node(leaf, is_leaf=True)
        return leaf

    # ================================================================
    #  PRIVATE — utilities (shared with SLBT)
    # ================================================================

    def _traverse_tree(self, x: pd.Series, node: Node):
        if node._is_leaf_node():
            return node.value
        treshold = node.treshold
        if isinstance(treshold, np.ndarray) and treshold.ndim == 1:
            goes_left = x[node.feature] in treshold
        else:
            union = {v for arr in treshold for v in arr}
            goes_left = x[node.feature] in union
        if goes_left:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _drop_constant_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=X.columns[X.nunique(dropna=False) <= 1])

    def _get_gcr(self, distribution, labels):
        gcr = [0.0] * len(labels)
        for i, lbl in enumerate(labels):
            for j, g_lbl in enumerate(self.targhet_dist[0]):
                if g_lbl == lbl:
                    ref = self.targhet_dist[1][j]
                    gcr[i] = distribution[i] / ref if ref > 0 else 0.0
        return gcr

    def _collect_nodes(self, node: Node, nodes_list: list):
        if node is None:
            return
        nodes_list.append(node)
        if not node._is_leaf_node():
            self._collect_nodes(node.left,  nodes_list)
            self._collect_nodes(node.right, nodes_list)

    def _calculate_tree_partial_impurity_reduction(self):
        if self.root is None:
            return
        all_nodes = []
        self._collect_nodes(self.root, all_nodes)
        all_nodes.sort(key=lambda n: n.impurity_decrease)
        _bubble_sort_nodes(all_nodes)

        self.root.tree_partial_impurity_reduction = 0.0
        root_N                = self.root.N
        previous_part_imp_red = 0.0
        search                = True
        virtual_leaves     = [self.root.left, self.root.right]
        virtual_leaves_set = {self.root.left, self.root.right}

        for current in all_nodes[1:]:
            part_imp_red = sum(
                leaf.impurity_decrease * leaf.N / root_N
                for leaf in virtual_leaves
            )
            if part_imp_red - previous_part_imp_red < 0.01 and search:
                current.suggested_pruning = True
                search = False
            if current in virtual_leaves_set and not current._is_leaf_node():
                virtual_leaves.remove(current)
                virtual_leaves_set.discard(current)
                virtual_leaves.append(current.left)
                virtual_leaves_set.add(current.left)
                virtual_leaves.append(current.right)
                virtual_leaves_set.add(current.right)
                current.tree_partial_impurity_reduction = part_imp_red
                previous_part_imp_red = part_imp_red
            else:
                current.tree_partial_impurity_reduction = part_imp_red

    def _calculate_tree_partial_tau_reduction(self):
        if self.root is None:
            return
        all_nodes = []
        self._collect_nodes(self.root, all_nodes)
        all_nodes.sort(key=lambda n: (n.cumulative_path_tau or 0.0))
        _bubble_sort_tau_nodes(all_nodes)

        self.root.tree_partial_tau_reduction = 0.0
        part_tau           = 0.0
        virtual_leaves     = [self.root.left, self.root.right]
        virtual_leaves_set = {self.root.left, self.root.right}

        for current in all_nodes[1:]:
            if current in virtual_leaves_set and not current._is_leaf_node():
                virtual_leaves.remove(current)
                virtual_leaves_set.discard(current)
                virtual_leaves.append(current.left)
                virtual_leaves_set.add(current.left)
                virtual_leaves.append(current.right)
                virtual_leaves_set.add(current.right)
                part_tau += current.tau_decrease
                current.tree_partial_tau_reduction = part_tau
            else:
                current.tree_partial_tau_reduction = part_tau

    # ================================================================
    #  REPR
    # ================================================================

    def __repr__(self) -> str:
        status = "fitted" if self.root is not None else "not fitted"
        return (
            f"SLBTWeighted(homogeneity='{self.homogeneity}', "
            f"max_depth={self.max_depth}, "
            f"feats_viewed={self.feats_viewed}, "
            f"[{status}])"
        )


# ================================================================
#  Module-level sort helpers (mirrors SLBT private statics)
# ================================================================

def _bubble_sort_nodes(nodes):
    changed, max_iter, iteration = True, len(nodes) * 2, 0
    while changed and iteration < max_iter:
        changed   = False
        iteration += 1
        for i in range(len(nodes) - 1):
            if nodes[i].impurity_decrease > nodes[i + 1].impurity_decrease:
                nodes[i], nodes[i + 1] = nodes[i + 1], nodes[i]
                changed = True


def _bubble_sort_tau_nodes(nodes):
    changed, max_iter, iteration = True, len(nodes) * 2, 0
    while changed and iteration < max_iter:
        changed   = False
        iteration += 1
        for i in range(len(nodes) - 1):
            a = nodes[i].cumulative_path_tau     or 0.0
            b = nodes[i + 1].cumulative_path_tau or 0.0
            if a > b:
                nodes[i], nodes[i + 1] = nodes[i + 1], nodes[i]
                changed = True
