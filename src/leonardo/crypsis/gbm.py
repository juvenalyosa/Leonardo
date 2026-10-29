"""A small evaluator for gradient-boosted trees stored as plain arrays (exported from scikit-learn's
HistGradientBoostingClassifier by scripts/export_models.py), so the package needs neither scikit-learn nor
pickle at prediction time."""

import json

import numpy as np


class TreeEnsemble:
    def __init__(self, arrays):
        self.feature = arrays["feature"]
        self.threshold = arrays["threshold"]
        self.left = arrays["left"]
        self.right = arrays["right"]
        self.leaf = arrays["leaf"]
        self.value = arrays["value"]
        self.missing_left = arrays["missing_left"]
        self.offsets = arrays["offsets"]
        self.baseline = float(arrays["baseline"])
        self.n_features = int(arrays["n_features"])
        self.meta = json.loads(str(arrays["meta"])) if "meta" in arrays else {}
        self.extra = {k: arrays[k] for k in arrays if k.startswith("pca_")}

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls({k: z[k] for k in z.files})

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {X.shape}")
        raw = np.full(len(X), self.baseline)
        rows = np.arange(len(X))
        for t in range(len(self.offsets) - 1):
            base = self.offsets[t]
            node = np.full(len(X), base, dtype=np.int64)
            active = ~self.leaf[node]
            while active.any():
                i = node[active]
                x = X[rows[active], self.feature[i]]
                go_left = np.where(np.isnan(x), self.missing_left[i], x <= self.threshold[i])
                node[active] = base + np.where(go_left, self.left[i], self.right[i])
                active = ~self.leaf[node]
            raw += self.value[node]
        return raw

    def predict_proba(self, X):
        """Probability of the positive class (one column)."""
        return 1.0 / (1.0 + np.exp(-self.decision_function(X)))
