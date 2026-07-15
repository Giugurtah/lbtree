"""
lbtree/base.py
==============
Shared base classes used by all estimators in lbtree.

Classes
-------
Node          — internal/leaf node of any lbtree decision tree
BaseLBTree    — common hyperparameter container and sklearn-compatible interface
"""

from __future__ import annotations


# ============================================================
#  Node
# ============================================================

class Node:
    """
    A single node in an LBTree decision tree.

    Internal nodes have ``value=None``; leaf nodes have ``value`` set to the
    predicted class label.  All other attributes are ``None`` when not
    applicable (e.g. leaf nodes do not have ``feature`` or ``treshold``).

    Parameters
    ----------
    gpi : float or None
        Global Predictability Index of the best splitting feature.
    pi : float or None
        Proportional Predictability Improvement (PPI) of the best split.
    position : int or None
        1-indexed position in the binary tree (root = 1, left child = 2*pos,
        right child = 2*pos+1).
    impurity : float or None
        Gini impurity at this node.
    impurity_decrease : float or None
        Normalised impurity decrease relative to the tree root.
    tree_partial_impurity_reduction : float or None
        Cumulative impurity reduction up to (and including) this node.
    tau_decrease : float or None
        Incremental tau (Δτ) of the split at this node.
        Δτ = PPI × (1 − impurity_decrease) = (I_t − I_L − I_R) / I_root.
        Zero for leaf nodes.
    cumulative_path_tau : float or None
        Sum of Δτ for all splits along the path from root to this node.
        Used as the y-axis in the tau visual-pruning layout.
        Zero at the root; increases monotonically toward leaves.
    tree_partial_tau_reduction : float or None
        Global cumulative tau of the tree when built up split by split
        (expansion ordered by cumulative_path_tau ascending).
        Analogous to tree_partial_impurity_reduction.
    suggested_pruning : bool or None
        True if this node was flagged as the suggested pruning point.
    feature : str or None
        Name of the splitting feature (internal nodes only).
    treshold : array-like or None
        Modality values assigned to the left child.
        For stratified splits (SLBT), this is a list of arrays, one per
        stratum.
    left : Node or None
        Left child node.
    right : Node or None
        Right child node.
    LIFT_1 : array-like or None
        LIFT values for the left child (SLBT only).
    LIFT_2 : array-like or None
        LIFT values for the right child (SLBT only).
    GCR : list or None
        Global Concentration Ratio per class.
    distribution : np.ndarray or None
        Relative class frequency vector at this node.
    N : int or None
        Number of samples at this node.
    labels : np.ndarray or None
        Unique class labels present at this node.
    strat_labels : np.ndarray or None
        Unique stratum labels (SLBT only).
    value : scalar or None
        Predicted class (leaf nodes only).
    """

    def __init__(
        self,
        gpi=None,
        pi=None,
        position=None,
        impurity=None,
        impurity_decrease=None,
        tree_partial_impurity_reduction=None,
        tau_decrease=None,
        cumulative_path_tau=None,
        tree_partial_tau_reduction=None,
        suggested_pruning=None,
        feature=None,
        treshold=None,
        left=None,
        right=None,
        LIFT_1=None,
        LIFT_2=None,
        GCR=None,
        distribution=None,
        N=None,
        labels=None,
        strat_labels=None,
        y_stats=None,
        *,
        value=None,
    ):
        self.gpi                             = gpi
        self.pi                              = pi
        self.position                        = position
        self.impurity                        = impurity
        self.impurity_decrease               = impurity_decrease
        self.tree_partial_impurity_reduction = tree_partial_impurity_reduction
        self.tau_decrease                    = tau_decrease
        self.cumulative_path_tau             = cumulative_path_tau
        self.tree_partial_tau_reduction      = tree_partial_tau_reduction
        self.suggested_pruning               = suggested_pruning
        self.feature                         = feature
        self.treshold                        = treshold
        self.left                            = left
        self.right                           = right
        self.LIFT_1                          = LIFT_1
        self.LIFT_2                          = LIFT_2
        self.GCR                             = GCR
        self.distribution                    = distribution
        self.N                               = N
        self.labels                          = labels
        self.strat_labels                    = strat_labels
        self.y_stats                         = y_stats   # dict with boxplot stats (twoClass only)
        self.value                           = value

    def _is_leaf_node(self) -> bool:
        """Return True if this node is a leaf (has a predicted value)."""
        return self.value is not None

    def __repr__(self) -> str:
        if self._is_leaf_node():
            return f"LeafNode(pos={self.position}, value={self.value!r}, N={self.N})"
        return (
            f"InternalNode(pos={self.position}, "
            f"feature={self.feature!r}, "
            f"gpi={self.gpi:.4f if self.gpi is not None else 'None'}, "
            f"pi={self.pi:.4f if self.pi is not None else 'None'}, "
            f"N={self.N})"
        )


# ============================================================
#  BaseLBTree
# ============================================================

class BaseLBTree:
    """
    Common hyperparameter container for all lbtree estimators.

    Subclasses (SCTree, SLBT, …) extend this class and implement
    ``fit()`` / ``predict()``.

    Parameters
    ----------
    min_ppi : float, default 0.0
        Minimum PPI required to accept a split.
    min_gpi : float, default 0.0
        Minimum GPI for a feature to be considered as a split candidate.
    min_impurity : float, default 0.0
        Nodes with impurity below this value become leaves.
    min_samples_split : int, default 1
        Minimum number of samples required to attempt a split.
    max_depth : int, default 100
        Maximum depth of the tree.
    feats_viewed : int | float, default 10
        Number of top-GPI features evaluated per node.
        - int   → exact count (capped at the number of available features).
        - float in (0.0, 1.0] → fraction of available features at each node,
          resolved to max(1, round(fraction * n_available)) at split time.
    FAST : bool, default False
        When True, the feature loop stops as soon as ``best_ppi > next_gpi``
        (the next candidate cannot improve the split).
    homogeneity : {"none", "A", "B", "AB"}, default "none"
        Homogeneity constraint for stratified splits (SLBT only).
        Ignored by SCTree.
    """

    def __init__(
        self,
        min_ppi: float          = 0.0,
        min_gpi: float          = 0.0,
        min_impurity: float     = 0.0,
        min_samples_split: int  = 1,
        max_depth: int          = 100,
        feats_viewed: int | float = 10,
        FAST: bool              = False,
        homogeneity: str        = "none",
    ):
        # Stopping criteria
        self.min_ppi           = min_ppi
        self.min_gpi           = min_gpi
        self.min_impurity      = min_impurity
        self.min_samples_split = min_samples_split
        self.max_depth         = max_depth

        # Search strategy
        if isinstance(feats_viewed, float) and not (0.0 < feats_viewed <= 1.0):
            raise ValueError(
                f"feats_viewed={feats_viewed!r}: when a float is provided it must "
                "be in (0.0, 1.0] to represent a fraction of available features."
            )
        if isinstance(feats_viewed, int) and feats_viewed < 1:
            raise ValueError(
                f"feats_viewed={feats_viewed!r}: integer value must be >= 1."
            )
        self.feats_viewed = feats_viewed
        self.FAST         = FAST

        # SLBT-specific
        self.homogeneity = homogeneity

        # Set after fit()
        self.root         = None
        self.reporter     = None
        self.targhet_dist = None
        self.root_N       = None
        self.depth        = None

    def _resolve_feats_viewed(self, n_available: int) -> int:
        """
        Resolve ``feats_viewed`` to a concrete integer for a node with
        ``n_available`` candidate features (after dropping constant columns).

        - int   → min(feats_viewed, n_available)
        - float → max(1, round(feats_viewed * n_available))
        """
        fv = self.feats_viewed
        if isinstance(fv, float):
            return max(1, round(fv * n_available))
        return min(fv, n_available)

    def get_params(self, deep: bool = True) -> dict:
        """scikit-learn compatible parameter getter."""
        return {
            "min_ppi":           self.min_ppi,
            "min_gpi":           self.min_gpi,
            "min_impurity":      self.min_impurity,
            "min_samples_split": self.min_samples_split,
            "max_depth":         self.max_depth,
            "feats_viewed":      self.feats_viewed,
            "FAST":              self.FAST,
            "homogeneity":       self.homogeneity,
        }

    def set_params(self, **params) -> "BaseLBTree":
        """scikit-learn compatible parameter setter."""
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter '{key}' for estimator {type(self).__name__}.")
            setattr(self, key, value)
        return self
