import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from lbtree import SCTree, Categorizer
from lbtree.plotting import plot_html

random_state = 42

# ── 1. Carica il dataset ──────────────────────────────────────────────────────
raw = load_wine()
df  = pd.DataFrame(raw.data, columns=raw.feature_names)
y   = pd.Series(raw.target.astype(str), name="class")   # target categorico

# ── 2. Rimuovi righe con NaN ──────────────────────────────────────────────────
mask = df.notna().all(axis=1)
df   = df[mask].reset_index(drop=True)
y    = y[mask].reset_index(drop=True)

# ── 3. Categorizza le variabili numeriche ─────────────────────────────────────
cat = Categorizer(method="elbow", k_max=5, label_style="interval")
X   = cat.fit_transform(df)

print("Bins trovati per feature:")
for col, k in cat.k_.items():
    print(f"  {col}: {k} bin")

# ── 4. Addestra SCTree twoStage ───────────────────────────────────────────────
clf = SCTree(
    model="twoStage",
    max_depth=10,
    min_ppi=0,
    feats_viewed=13,
)
clf.fit(X, y)

print("\nReport:")
print(clf.reporter.results[
    ["id", "node_type", "feature", "gpi", "pi",
     "tau_decrease", "cumulative_path_tau", "tree_partial_tau_reduction"]
].to_string(index=False))

# ── 5. Visual pruning — impurity decrease (classico) ─────────────────────────
plot_html(
    clf,
    output_file="wine_vp_impurity.html",
    title="Wine — Visual Pruning (Impurity)",
    visual_pruning=True,
    vp_metric="impurity",
)

# ── 6. Visual pruning — cumulative tau (nuovo) ────────────────────────────────
plot_html(
    clf,
    output_file="wine_vp_tau.html",
    title="Wine — Visual Pruning (τ cumulata)",
    visual_pruning=True,
    vp_metric="tau",
)