"""
AdaptAuth-ML · Modelo apilado (C2a + C2b + C3).

C2a: Isolation Forest -> s(x); reescalado s'(x) con s_ref = mediana de s en los legítimos de validación (Ec. 2).
C2b + C3: HistGradientBoosting sobre [x, s'(x)] -> R = P(ataque | x, s') (Ec. 3).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest


class ModeloApilado:
    def __init__(self, seed: int = 42, hgb_params: dict | None = None):
        self.seed = seed
        self.hgb_params = hgb_params or dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0)
        self.s_ref = None

    def _s(self, X):
        return -self.iso.score_samples(X)

    def score_if(self, X):
        """s'(x) reescalado (Ec. 2)."""
        s = self._s(X)
        return np.clip((s - self.s_ref) / (1.0 - self.s_ref), 0.0, 1.0)

    def fit(self, Xtr, ytr, Xval, yval):
        self.iso = IsolationForest(n_estimators=200, max_samples=256, random_state=self.seed, n_jobs=-1).fit(Xtr)
        self.s_ref = float(np.median(self._s(Xval)[np.asarray(yval) == 0]))
        self.hgb = HistGradientBoostingClassifier(random_state=self.seed, **self.hgb_params)
        self.hgb.fit(np.column_stack([Xtr, self.score_if(Xtr)]), ytr)
        return self

    def aumentar(self, X):
        return np.column_stack([X, self.score_if(X)])

    def predict_proba(self, X):
        return self.hgb.predict_proba(self.aumentar(X))
