# lbtree — Latent Budget Tree

Decision trees for **categorical predictors** based on Latent Budget Analysis (LBA).

## Overview

`lbtree` implements a family of tree-based models that handle predictors with nominal (unordered) categories without requiring dummy encoding. Splits are found by maximising the *Predictability Index* (PPI/GPI) derived from LBA.

### Available models

| Class | Description |
|---|---|
| `SCTree` | Single decision tree (two-stage criterion) |
| `SCTreeForest` | Random forest of `SCTree` |
| `AdaBoostForest` | AdaBoost.M1 ensemble of `SCTree` |
| `SLBT` | Simultaneous Latent Budget Tree (stratified data) |
| `SLBTForest` | Random forest of `SLBT` |
| `SLBTAdaBoostForest` | AdaBoost ensemble of `SLBT` |
| `BINPI` | Boosted Incremental Non-Parametric Imputation |
| `Categorizer` | KMeans-based continuous → categorical discretizer |

## Requirements

- Python ≥ 3.10
- `numpy ≥ 1.23`, `pandas ≥ 1.5`
- A C compiler (`gcc` or `cc`) and `make` — required to compile the C backends at install time

## Installation

```bash
pip install git+https://github.com/<your-username>/lbtree.git
```

Or clone and install locally:

```bash
git clone https://github.com/<your-username>/lbtree.git
cd lbtree
pip install .
```

For development (editable install):

```bash
pip install -e .
```

> **Note:** `pip install` automatically compiles the C shared libraries via `make`. No manual build step is needed.

## Quick start

```python
from lbtree import SCTree

tree = SCTree(max_depth=3, criterion="twoStage")
tree.fit(X_train, y_train)          # X must contain categorical columns
predictions = tree.predict(X_test)
```

```python
from lbtree import Categorizer, SCTreeForest

cat = Categorizer(n_bins=5)
X_cat = cat.fit_transform(X_continuous)

forest = SCTreeForest(n_estimators=100, max_depth=4)
forest.fit(X_cat, y)
```

## License

MIT
