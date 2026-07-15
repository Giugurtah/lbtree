"""
lbtree/slbt.py
==============
SLBT — Simultaneous Latent Budget Tree for stratified categorical predictors.

Homogeneity modes
-----------------
"none" : per-stratum A and B (fully unconstrained)
"A"    : mixing coefficients A shared across strata
"B"    : latent budget profiles B shared across strata
"AB"   : both A and B shared (equivalent to non-stratified LBA)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base       import BaseLBTree, Node
from .reporting  import TreeReporter
from ._utils.criteria import _gpi_stratified, _get_sizes
from ._utils.utils    import _stratified_contingency
from ._tree.split     import score_slbt, _split, _splitS
from ._tree._homogeneity.base import get_homogeneity_strategy


class SLBT(BaseLBTree):
    """
    Simultaneous Latent Budget Tree for stratified categorical data.

    Parameters
    ----------
    homogeneity : {"none", "A", "B", "AB"}, default "none"
        Homogeneity constraint on the LBA parameters across strata.
    min_ppi : float, default 0.0
        Minimum PPI to accept a split.
    min_gpi : float, default 0.0
        Minimum GPI for a feature to be a candidate.
    min_impurity : float, default 0.0
        Nodes below this impurity become leaves.
    min_samples_split : int, default 1
        Minimum samples to attempt a split.
    max_depth : int, default 100
        Maximum tree depth.
    feats_viewed : int, default 10
        Top-GPI features evaluated per node.
    FAST : bool, default False
        Stop feature loop early when ``best_ppi > next_gpi``.

    Attributes (set after fit)
    --------------------------
    root : Node
    reporter : TreeReporter
    targhet_dist : list
    root_N : int
    """

    def __init__(
        self,
        homogeneity: str       = "none",
        min_ppi: float         = 0.0,
        min_gpi: float         = 0.0,
        min_impurity: float    = 0.0,
        min_samples_split: int = 1,
        max_depth: int         = 100,
        feats_viewed: int      = 10,
        FAST: bool             = False,
        node_max_features: int | None = None,
        rng                    = None,
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
        self.homogeneity       = homogeneity
        self.node_max_features = node_max_features
        self._rng              = rng

    def get_params(self, deep: bool = True) -> dict:
        p = super().get_params(deep)
        p["homogeneity"] = self.homogeneity
        p.pop("model", None)
        return p

    # ================================================================
    #  PUBLIC — fit / predict / prune
    # ================================================================

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        x_s: np.ndarray | None = None,
    ) -> "SLBT":
        """
        Fit the SLBT on stratified categorical data.

        Parameters
        ----------
        X   : pd.DataFrame — all-categorical predictor matrix
        y   : pd.Series    — categorical target
        x_s : np.ndarray, shape (n_samples,), optional
            Stratum indicator (integer per observation).
            If None, all observations belong to stratum 0 and the
            homogeneity is set to "AB" (equivalent to non-stratified LBA).

        Returns
        -------
        self
        """
        if x_s is None:
            x_s              = np.zeros(len(y), dtype=int)
            self.homogeneity = "AB"

        strategy = get_homogeneity_strategy(self.homogeneity)

        self.targhet_dist = [
            np.unique(y),
            np.unique(y, return_counts=True)[1] / len(y),
        ]
        self.root_N  = len(y)
        self.reporter = TreeReporter(homogeneity=self.homogeneity, decimals=4)

        self.root = self._grow_tree(strategy, X, y, x_s)
        self._calculate_tree_partial_impurity_reduction()
        self._calculate_tree_partial_tau_reduction()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels for X."""
        if self.root is None:
            raise RuntimeError("Tree not fitted. Call fit() first.")
        return np.array([self._traverse_tree(X.iloc[i], self.root)
                         for i in range(len(X))])

    def prune_after_vp(self, nodeID: int) -> None:
        """
        Prune the tree at the node identified by *nodeID*.

        All nodes with ``impurity_decrease`` ≤ that of *nodeID* are
        collapsed into leaves.
        """
        if self.root is None:
            return

        all_nodes = []
        self._collect_nodes(self.root, all_nodes)
        all_nodes.sort(key=lambda n: n.impurity_decrease)
        self._bubble_sort_nodes(all_nodes)

        virtual_leaves     = [self.root]
        virtual_leaves_set = {self.root}
        reached = False
        i = 0

        while not reached:
            current = all_nodes[i]
            if current.position != nodeID:
                i += 1
                if current in virtual_leaves_set and not current._is_leaf_node():
                    virtual_leaves.remove(current)
                    virtual_leaves_set.discard(current)
                    virtual_leaves.append(current.left)
                    virtual_leaves_set.add(current.left)
                    virtual_leaves.append(current.right)
                    virtual_leaves_set.add(current.right)
            else:
                reached = True

        for node in virtual_leaves:
            if not node._is_leaf_node():
                node.feature   = None
                node.treshold  = None
                node.left      = None
                node.right     = None
                node.gpi       = None
                node.pi        = None
                node.LIFT_1    = None
                node.LIFT_2    = None
                best_idx       = int(np.argmax(node.distribution))
                node.value     = node.labels[best_idx]
                node.GCR       = self._get_gcr(node.distribution, node.labels)

        self._rebuild_report()

    # ================================================================
    #  PRIVATE — tree growth
    # ================================================================

    def _grow_tree(
        self,
        strategy,
        X: pd.DataFrame,
        y: pd.Series,
        x_s: np.ndarray,
        root_impurity: float  = 1.0,
        root_N: int           = 1,
        depth: int            = 0,
        pos: int              = 1,
        parent_cum_tau: float = 0.0,
    ) -> Node:

        X = self._drop_constant_columns(X)
        n_samples, n_feats, n_labels, impurity, distribution = _get_sizes(X, y)

        if root_N == 1:
            root_N        = len(y)
            root_impurity = impurity
            impurity_decrease              = 0.0
            tree_partial_impurity_reduction = 0.0
        else:
            impurity_decrease = (
                (root_impurity - impurity * len(y) / root_N) / root_impurity
            )
            tree_partial_impurity_reduction = 0.0

        cumulative_path_tau = parent_cum_tau

        # --- pre-split stopping criteria ---
        leaf = self._check_criteria_before(
            y, pos, impurity, distribution, depth,
            n_labels, n_samples, n_feats,
            impurity_decrease, tree_partial_impurity_reduction,
            parent_cum_tau=cumulative_path_tau,
        )
        if leaf is not None:
            return leaf

        # --- per-node random feature subsampling (true RF mode) ---
        if self.node_max_features is not None and self._rng is not None:
            n = min(self.node_max_features, X.shape[1])
            sampled_cols = self._rng.choice(X.columns.tolist(), size=n, replace=False)
            X = X[sampled_cols]

        # --- stratified GPI ranking ---
        gpi_vals, gpi_order = _gpi_stratified(X, y, x_s)

        # --- best split search ---
        best_feature, best_treshold, best_pi, best_gpi, best_alpha, best_beta = (
            self._find_best_predictor(X, y, x_s, gpi_order, gpi_vals)
        )

        # --- post-split stopping criteria ---
        leaf = self._check_criteria_after(
            y, pos, impurity, distribution, depth,
            best_gpi, best_pi,
            impurity_decrease, tree_partial_impurity_reduction,
            parent_cum_tau=cumulative_path_tau,
        )
        if leaf is not None:
            return leaf

        tau_decrease  = best_pi * (1.0 - impurity_decrease)
        child_cum_tau = cumulative_path_tau + tau_decrease

        # --- threshold values and split ---
        x_vals    = np.unique(X[best_feature])
        thresholds = strategy.get_treshold_values(best_treshold, x_vals, x_s)

        indexL, indexR = strategy.split(X[best_feature], x_s, thresholds)

        # --- LIFT values ---
        lift1, lift2 = strategy.compute_lift(best_beta, distribution)

        left  = self._grow_tree(
            strategy, X.loc[indexL], y[indexL], x_s[indexL],
            root_impurity, root_N, depth + 1, 2 * pos,
            parent_cum_tau=child_cum_tau,
        )
        right = self._grow_tree(
            strategy, X.loc[indexR], y[indexR], x_s[indexR],
            root_impurity, root_N, depth + 1, 2 * pos + 1,
            parent_cum_tau=child_cum_tau,
        )

        node = Node(
            gpi=best_gpi, pi=best_pi, position=pos,
            feature=best_feature, treshold=thresholds,
            left=left, right=right,
            impurity=impurity,
            impurity_decrease=impurity_decrease,
            tree_partial_impurity_reduction=tree_partial_impurity_reduction,
            tau_decrease=tau_decrease,
            cumulative_path_tau=cumulative_path_tau,
            tree_partial_tau_reduction=0.0,
            distribution=distribution, N=len(y), labels=np.unique(y),
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

    def _find_best_predictor(self, X, y, x_s, gpi_order, gpi_vals):
        best = {
            "feature":   None,
            "threshold": None,
            "pi":        -np.inf,
            "gpi":       -np.inf,
            "alpha":     None,
            "beta":      None,
        }

        for rank, col in enumerate(gpi_order[: self.feats_viewed]):
            Fs_noN = _stratified_contingency(X[col], y, x_s, norm=False)
            Fs     = _stratified_contingency(X[col], y, x_s, norm=True)

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
        self, y, pos, impurity, distribution, depth,
        n_labels, n_samples, n_feats,
        impurity_decrease, tree_partial_impurity_reduction,
        parent_cum_tau=0.0,
    ):
        if (depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_samples_split
                or n_feats  == 0
                or impurity  < self.min_impurity):
            return self._make_leaf(
                y, pos, impurity, distribution,
                impurity_decrease, tree_partial_impurity_reduction,
                parent_cum_tau=parent_cum_tau,
            )
        return None

    def _check_criteria_after(
        self, y, pos, impurity, distribution, depth,
        best_gpi, best_pi,
        impurity_decrease, tree_partial_impurity_reduction,
        parent_cum_tau=0.0,
    ):
        if best_gpi < self.min_gpi or best_pi < self.min_ppi or best_pi < 1e-8:
            return self._make_leaf(
                y, pos, impurity, distribution,
                impurity_decrease, tree_partial_impurity_reduction,
                parent_cum_tau=parent_cum_tau,
            )
        return None

    def _make_leaf(
        self, y, pos, impurity, distribution,
        impurity_decrease, tree_partial_impurity_reduction,
        parent_cum_tau=0.0,
    ) -> Node:
        leaf = Node(
            position=pos,
            value=y.mode()[0],
            impurity=impurity,
            impurity_decrease=impurity_decrease,
            tree_partial_impurity_reduction=tree_partial_impurity_reduction,
            tau_decrease=0.0,
            cumulative_path_tau=parent_cum_tau,
            tree_partial_tau_reduction=0.0,
            distribution=distribution,
            N=len(y),
            labels=np.unique(y),
            GCR=self._get_gcr(distribution, np.unique(y)),
        )
        if self.reporter is not None:
            self.reporter.add_node(leaf, is_leaf=True)
        return leaf

    # ================================================================
    #  PRIVATE — utilities
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
                    gcr[i] = distribution[i] / self.targhet_dist[1][j]
        return gcr

    # ================================================================
    #  PRIVATE — pruning & impurity reduction
    # ================================================================

    def _prune_tree(self, node: Node):
        if node is None:
            return None, False
        if node._is_leaf_node():
            return node, False
        left,  cl = self._prune_tree(node.left)
        right, cr = self._prune_tree(node.right)
        node.left, node.right = left, right
        changed = False
        if (node.left  is not None and node.left._is_leaf_node()
                and node.right is not None and node.right._is_leaf_node()
                and node.left.value == node.right.value):
            node.value    = node.left.value
            node.gpi      = node.pi = node.feature = node.treshold = None
            node.LIFT_1   = None
            node.LIFT_2   = None
            node.GCR      = self._get_gcr(node.distribution, node.labels)
            node.left     = node.right = None
            changed       = True
        return node, cl or cr or changed

    def _calculate_tree_partial_impurity_reduction(self):
        if self.root is None:
            return
        all_nodes = []
        self._collect_nodes(self.root, all_nodes)
        all_nodes.sort(key=lambda n: n.impurity_decrease)
        self._bubble_sort_nodes(all_nodes)

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

    @staticmethod
    def _bubble_sort_nodes(nodes):
        changed, max_iter, iteration = True, len(nodes) * 2, 0
        while changed and iteration < max_iter:
            changed   = False
            iteration += 1
            for i in range(len(nodes) - 1):
                if nodes[i].impurity_decrease > nodes[i + 1].impurity_decrease:
                    nodes[i], nodes[i + 1] = nodes[i + 1], nodes[i]
                    changed = True

    def _calculate_tree_partial_tau_reduction(self):
        """
        Populate ``tree_partial_tau_reduction`` on every node.

        The tree is expanded split by split in ascending ``cumulative_path_tau``
        order (parent-before-child is guaranteed because a child always has
        strictly higher cumulative_path_tau than its parent).  At each step
        ``part_tau`` accumulates the ``tau_decrease`` of the node being
        expanded; that value is assigned to the expanding node.
        """
        if self.root is None:
            return
        all_nodes = []
        self._collect_nodes(self.root, all_nodes)
        all_nodes.sort(key=lambda n: (n.cumulative_path_tau or 0.0))
        self._bubble_sort_tau_nodes(all_nodes)

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

    @staticmethod
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

    def _collect_nodes(self, node: Node, nodes_list: list):
        if node is None:
            return
        nodes_list.append(node)
        if not node._is_leaf_node():
            self._collect_nodes(node.left,  nodes_list)
            self._collect_nodes(node.right, nodes_list)

    def _rebuild_report(self):
        if self.reporter is None:
            return
        new_reporter = TreeReporter(
            homogeneity=self.homogeneity,
            decimals=getattr(self.reporter, "decimals", 4),
        )

        def _traverse(n: Node):
            if n is None:
                return
            new_reporter.add_node(n, is_leaf=n._is_leaf_node())
            if not n._is_leaf_node():
                _traverse(n.left)
                _traverse(n.right)

        _traverse(self.root)
        self.reporter = new_reporter

    # ================================================================
    #  REPR
    # ================================================================

    def __repr__(self) -> str:
        status = "fitted" if self.root is not None else "not fitted"
        return (
            f"SLBT(homogeneity='{self.homogeneity}', "
            f"max_depth={self.max_depth}, "
            f"feats_viewed={self.feats_viewed}, "
            f"[{status}])"
        )
