"""
AdaptAuth-ML · Motor de detección (C2 + C3), versión 11.

C2/C3: HistGradientBoosting sobre el vector contextual x -> R = P(ataque | x) (Ec. 3).
Isolation Forest se evaluó como señal adicional (versión 10, modelo apilado) y se retiró del núcleo:
no mejora el F1 (ablación) ni el AUC-PR con pocas etiquetas (arranque en frío, run_levantamiento.py).
La clase conserva la interfaz del modelo apilado para que los scripts de requisitos no cambien.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class ModeloAdaptAuth:
    def __init__(self, seed: int = 42, hgb_params: dict | None = None):
        self.seed = seed
        self.hgb_params = hgb_params or dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0)
        self.s_ref = None

    def fit(self, Xtr, ytr, Xval=None, yval=None):
        self.hgb = HistGradientBoostingClassifier(random_state=self.seed, **self.hgb_params).fit(Xtr, ytr)
        return self

    def aumentar(self, X):
        return np.asarray(X)

    def predict_proba(self, X):
        return self.hgb.predict_proba(self.aumentar(X))
