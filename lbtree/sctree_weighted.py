"""
lbtree/sctree_weighted.py
=========================
SCTreeWeighted — Two-stage Decision Tree with sample weights (for AdaBoost).

Identical to SCTree but uses the weighted C functions to compute impurity,
GPI, and optimal splits, taking per-sample weights into account.

Supports the same three splitting models as SCTree:
"twoStage", "twoing", "twoClass".
For "twoClass" the numeric target is binarized locally at each node via
_binarize_y() before the weighted GPI ranking and split search.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base       import BaseLBTree, Node
from .reporting  import TreeReporter
from ._utils.criteria import _get_sizes, _variance, _y_stats
from ._utils.utils    import _contingency_matrix
from ._tree.split     import score_sctree_weighted, _split
from ._backend._lbtree import gpi_weighted as _c_gpi_weighted


class SCTreeWeighted(BaseLBTree):
    """
    Two-stage Decision Tree with sample weights (for AdaBoost).

    Identical to SCTree but ``fit()`` accepts a ``sample_weight`` array and
    uses the weighted C backend for GPI ranking and split scoring.

    Parameters
    ----------
    model : {"twoStage", "twoing"}, default "twoStage"
        Splitting algorithm.
    min_ppi : float, default 0.001
        Minimum PPI to accept a split.
    min_gpi : float, default 0.001
        Minimum GPI for a feature to be a candidate.
    min_impurity : float, default 0.001
        Nodes below this impurity become leaves.
    min_samples_split : int, default 2
        Minimum samples to attempt a split.
    max_depth : int, default 5
        Maximum tree depth.
    feats_viewed : int, default 10
        Top-GPI features evaluated per node.
    FAST : bool, default True
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
        model: str             = "twoStage",
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
        if model not in ("twoStage", "twoing", "twoClass"):
            raise ValueError(
                f"model='{model}' not supported. "
                "Choose 'twoStage', 'twoing', or 'twoClass'."
            )
        self.model = model

    def get_params(self, deep: bool = True) -> dict:
        p = super().get_params(deep)
        p["model"] = self.model
        p.pop("homogeneity", None)
        return p

    # ================================================================
    #  PUBLIC — fit / predict
    # ================================================================

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> "SCTreeWeighted":
        """
        Fit the tree on weighted data.

        Parameters
        ----------
        X : pd.DataFrame
            All-categorical predictor matrix.
        y : pd.Series
            Categorical target.
        sample_weight : np.ndarray, shape (n_samples,), optional
            Per-sample weights.  If None, all weights are set to 1.

        Returns
        -------
        self
        """
        if sample_weight is None:
            sample_weight = np.ones(len(y), dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if len(sample_weight) != len(y):
                raise ValueError("sample_weight length must equal len(y).")

        # For twoClass the global distribution is computed on the binarized
        # y (root-level) so that GCR labels match the 'low'/'high' categories.
        y_for_dist = self._binarize_y(y) if self.model == "twoClass" else y
        self.targhet_dist = [
            np.unique(y_for_dist),
            np.unique(y_for_dist, return_counts=True)[1] / len(y_for_dist),
        ]
        self.root_N  = len(y)
        self.reporter = TreeReporter(decimals=4)

        self.root = self._grow_tree(X, y, sample_weight)
        self._calculate_tree_partial_impurity_reduction()
        self._calculate_tree_partial_tau_reduction()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels for X."""
        if self.root is None:
            raise RuntimeError("Tree not fitted. Call fit() first.")
        return np.array([self._traverse_tree(X.iloc[i], self.root)
                         for i in range(len(X))])

    # ================================================================
    #  PRIVATE — tree growth
    # ================================================================

    def _grow_tree(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray,
        root_impurity: float  = 1.0,
        root_N: int           = 1,
        depth: int            = 0,
        pos: int              = 1,
        parent_cum_tau: float = 0.0,
    ) -> Node:

        X = self._drop_constant_columns(X)

        # --- twoClass: binarize numeric target locally at this node ---
        # y_original keeps the original numeric values to pass to child nodes,
        # so that each child can re-binarize on its own local distribution.
        y_original = y
        if self.model == "twoClass":
            y = self._binarize_y(y)

        n_samples, n_feats, n_labels, impurity, distribution = _get_sizes(X, y)

        # twoClass: override Gini impurity with variance of the original
        # numeric target (same logic as SCTree).
        # Also compute boxplot statistics for the gradient visualisation.
        if self.model == "twoClass":
            impurity     = _variance(y_original)
            node_y_stats = _y_stats(y_original)
        else:
            node_y_stats = None

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
            sample_weight=sample_weight,
            y_original=y_original if self.model == "twoClass" else None,
            parent_cum_tau=cumulative_path_tau,
        )
        if leaf is not None:
            return leaf

        # --- GPI ranking (weighted) ---
        gpi_vals, gpi_order = self._gpi_weighted(X, y, sample_weight)

        # --- best split search (weighted) ---
        best_feature, best_treshold, best_pi, best_gpi = self._find_best_predictor(
            X, y, sample_weight, gpi_order, gpi_vals
        )

        # --- post-split stopping criteria ---
        leaf = self._check_criteria_after(
            y, pos, impurity, distribution, depth,
            best_gpi, best_pi,
            impurity_decrease, tree_partial_impurity_reduction,
            sample_weight=sample_weight,
            y_original=y_original if self.model == "twoClass" else None,
            parent_cum_tau=cumulative_path_tau,
        )
        if leaf is not None:
            return leaf

        tau_decrease  = best_pi * (1.0 - impurity_decrease)
        child_cum_tau = cumulative_path_tau + tau_decrease

        # --- split ---
        threshold_labels = np.unique(X[best_feature])[np.array(best_treshold) > 0]
        indexL, indexR   = _split(X[best_feature], threshold_labels)

        # Pass y_original to children so each node re-binarizes on its own
        # local distribution (twoClass). For other models y_original == y.
        left  = self._grow_tree(
            X.loc[indexL], y_original[indexL], sample_weight[indexL],
            root_impurity, root_N, depth + 1, 2 * pos,
            parent_cum_tau=child_cum_tau,
        )
        right = self._grow_tree(
            X.loc[indexR], y_original[indexR], sample_weight[indexR],
            root_impurity, root_N, depth + 1, 2 * pos + 1,
            parent_cum_tau=child_cum_tau,
        )

        node = Node(
            gpi=best_gpi, pi=best_pi, position=pos,
            feature=best_feature, treshold=threshold_labels,
            left=left, right=right,
            impurity=impurity,
            impurity_decrease=impurity_decrease,
            tree_partial_impurity_reduction=tree_partial_impurity_reduction,
            tau_decrease=tau_decrease,
            cumulative_path_tau=cumulative_path_tau,
            tree_partial_tau_reduction=0.0,
            distribution=distribution, N=len(y), labels=np.unique(y),
            GCR=None,
            y_stats=node_y_stats,
        )
        if self.reporter is not None:
            self.reporter.add_node(node, is_leaf=False)
        return node

    # ================================================================
    #  PRIVATE — weighted GPI and best predictor search
    # ================================================================

    def _gpi_weighted(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray,
    ):
        """
        Compute weighted GPI for every column and return descending order.

        Returns
        -------
        gpi_vals  : tuple of floats, sorted descending
        gpi_order : tuple of column names, in the same order
        """
        vals  = []
        names = []
        for col in X.columns:
            F, W_row = self._contingency_matrix_weighted(X[col], y, sample_weight)
            I, J     = F.shape
            vals.append(_c_gpi_weighted(I, J, W_row, F.ravel().astype(np.float64)))
            names.append(col)

        vals, names = zip(*sorted(zip(vals, names), reverse=True))
        return vals, names

    def _find_best_predictor(self, X, y, sample_weight, gpi_order, gpi_vals):
        best = {"feature": None, "threshold": None, "pi": -np.inf, "gpi": -np.inf}

        n_to_view = self._resolve_feats_viewed(len(gpi_order))

        for rank, col in enumerate(gpi_order[:n_to_view]):
            F, W_row = self._contingency_matrix_weighted(X[col], y, sample_weight)
            pi, S    = score_sctree_weighted(F, W_row, self.model)

            if pi > best["pi"]:
                best["pi"]        = pi
                best["feature"]   = str(col)
                best["threshold"] = S
                best["gpi"]       = gpi_vals[rank]

            if (self.FAST
                    and rank < len(gpi_order) - 1
                    and best["pi"] > gpi_vals[rank + 1]):
                break

        return best["feature"], best["threshold"], best["pi"], best["gpi"]

    def _contingency_matrix_weighted(
        self,
        x: pd.Series,
        y: pd.Series,
        sample_weight: np.ndarray,
    ):
        """
        Build a weighted joint frequency matrix and row-weight vector.

        Returns
        -------
        F_weighted : np.ndarray, shape (I, J)
            Normalised joint probabilities weighted by ``sample_weight``.
        W_row : np.ndarray, shape (I,)
            Marginal weight for each predictor modality.
        """
        modalities_x = sorted(x.unique())
        modalities_y = sorted(y.unique())
        I = len(modalities_x)
        J = len(modalities_y)

        F_weighted = np.zeros((I, J), dtype=np.float64)
        for i, mod_x in enumerate(modalities_x):
            for j, mod_y in enumerate(modalities_y):
                mask = (x == mod_x) & (y == mod_y)
                F_weighted[i, j] = sample_weight[mask].sum()

        total = F_weighted.sum()
        if total > 0:
            F_weighted /= total

        W_row = F_weighted.sum(axis=1)
        return F_weighted, W_row

    # ================================================================
    #  PRIVATE — stopping criteria
    # ================================================================

    def _check_criteria_before(
        self, y, pos, impurity, distribution, depth,
        n_labels, n_samples, n_feats,
        impurity_decrease, tree_partial_impurity_reduction,
        sample_weight=None,
        y_original=None,
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
                sample_weight=sample_weight,
                y_original=y_original,
                parent_cum_tau=parent_cum_tau,
            )
        return None

    def _check_criteria_after(
        self, y, pos, impurity, distribution, depth,
        best_gpi, best_pi,
        impurity_decrease, tree_partial_impurity_reduction,
        sample_weight=None,
        y_original=None,
        parent_cum_tau=0.0,
    ):
        if best_gpi < self.min_gpi or best_pi < self.min_ppi or best_pi < 1e-8:
            return self._make_leaf(
                y, pos, impurity, distribution,
                impurity_decrease, tree_partial_impurity_reduction,
                sample_weight=sample_weight,
                y_original=y_original,
                parent_cum_tau=parent_cum_tau,
            )
        return None

    def _make_leaf(
        self, y, pos, impurity, distribution,
        impurity_decrease, tree_partial_impurity_reduction,
        sample_weight=None,
        y_original=None,
        parent_cum_tau=0.0,
    ) -> Node:
        # twoClass: use mean of original numeric target as leaf prediction.
        if y_original is not None:
            predicted_value = float(y_original.mean())
            leaf_y_stats    = _y_stats(y_original)
        else:
            # Use weighted majority class if sample_weight is provided,
            # otherwise fall back to unweighted mode.
            if sample_weight is not None and len(sample_weight) > 0:
                classes = np.unique(y)
                weights = np.array([sample_weight[y == c].sum() for c in classes])
                predicted_value = classes[np.argmax(weights)]
            else:
                predicted_value = y.mode()[0]
            leaf_y_stats    = None

        leaf = Node(
            position=pos,
            value=predicted_value,
            impurity=impurity,
            impurity_decrease=impurity_decrease,
            tree_partial_impurity_reduction=tree_partial_impurity_reduction,
            tau_decrease=0.0,
            cumulative_path_tau=parent_cum_tau,
            tree_partial_tau_reduction=0.0,
            distribution=distribution,
            N=len(y),
            labels=np.unique(y),
            GCR=None if y_original is not None else self._get_gcr(distribution, np.unique(y)),
            y_stats=leaf_y_stats,
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
        if x[node.feature] in node.treshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _drop_constant_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=X.columns[X.nunique(dropna=False) <= 1])

    def _binarize_y(self, y: pd.Series) -> pd.Series:
        """
        Binarize a numeric target into two categories ('low' / 'high').

        Called at every node when ``model == 'twoClass'`` before the weighted
        GPI ranking and split search.  The split threshold is determined by
        ``binarize_target_c`` in liblbtree (currently a stub returning the
        median of the local sorted unique values).

        The binarization is **local**: each node operates on its own subset
        of y, so the threshold can differ at every node.

        Parameters
        ----------
        y : pd.Series
            Numeric target values for the samples at the current node.

        Returns
        -------
        pd.Series
            Categorical series with values 'low' (y < threshold) or
            'high' (y >= threshold), same index as y.
        """
        from lbtree._backend._lbtree import binarize_target

        y_vals    = y.to_numpy(dtype=np.float64)
        y_sorted  = np.sort(np.unique(y_vals))
        threshold = binarize_target(len(y_sorted), y_sorted)

        return pd.Series(
            np.where(y_vals < threshold, "low", "high"),
            index=y.index,
            name=y.name,
        )

    def _get_gcr(self, distribution, labels):
        gcr = [0.0] * len(labels)
        for i, lbl in enumerate(labels):
            for j, g_lbl in enumerate(self.targhet_dist[0]):
                if g_lbl == lbl:
                    gcr[i] = distribution[i] / self.targhet_dist[1][j]
        return gcr

    # ================================================================
    #  PRIVATE — pruning & impurity reduction
    #  (reuse SCTree's implementations via inherited helpers)
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
        new_reporter = TreeReporter(decimals=getattr(self.reporter, "decimals", 4))

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
            f"SCTreeWeighted(model='{self.model}', "
            f"max_depth={self.max_depth}, "
            f"feats_viewed={self.feats_viewed}, "
            f"[{status}])"
        )
