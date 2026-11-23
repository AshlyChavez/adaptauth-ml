"""
AdaptAuth-ML · Segunda línea base con historial en CERT r4.2 (observación de la revisión del PI 15).

Compara AdaptAuth-ML v11 con una regresión logística entrenada con LAS MISMAS variables de historial del
empleado, los mismos 5 pliegues por empleado (semilla 42) y el mismo presupuesto de FPR ≤ 0.1 % fijado en
validación. Responde a la pregunta «¿el modelo solo gana porque las reglas son débiles?».

Salidas (results_cert_v11/):
  linea_base_logistica_cert.json   AUC-ROC, AUC-PR, recall/precisión con FPR ≤ 0.1 % y bootstrap por empleado
Uso:
    python run_cert_logistica.py --dir carpeta_con_logon_LDAP_y_answers
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import adaptauth as aa  # noqa: E402
import cert  # noqa: E402
from run_cert_v2 import F, K, N_BOOT_CERT, hgb  # noqa: E402

OUT = Path(__file__).parent / "results_cert_v11"


def logistica():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); args = ap.parse_args()
    cfg = aa.Config(cold_start_otp=False, label="anomalo")
    t0 = time.time()
    lg = cert.cargar_logon(cert.buscar(args.dir, "logon.csv")[0])
    ev, _ = cert.cargar_respuestas(args.dir, "4.2")
    ldap_dirs = [p for p in cert.buscar(args.dir, "LDAP") if Path(p).is_dir()]
    ld = cert.cargar_ldap(ldap_dirs[0]) if ldap_dirs else pd.DataFrame(columns=["user"])
    a = cert.construir_usuario_dia(lg, ev, lg["date"].min() + pd.DateOffset(months=6))
    a = cert.agregar_variables_v2(a).merge(ld, on="user", how="left")
    a["fold"] = cert.folds_por_empleado(a, K, cfg.seed).to_numpy()
    X = a[F].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    R = {m: np.zeros(len(a)) for m in ("adaptauth", "logistica")}
    pred = {m: np.zeros(len(a), dtype=int) for m in R}
    for k in range(K):
        kv = (k + 1) % K
        te, va = (a["fold"] == k).to_numpy(), (a["fold"] == kv).to_numpy()
        tr = ~(te | va)
        y_tr, y_va = a.loc[tr, "anomalo"].to_numpy(), a.loc[va, "anomalo"].to_numpy()
        for nombre, mod in (("adaptauth", hgb()), ("logistica", logistica())):
            Xtr = a.loc[tr, F].to_numpy() if nombre == "adaptauth" else X[tr].to_numpy()
            Xva = a.loc[va, F].to_numpy() if nombre == "adaptauth" else X[va].to_numpy()
            Xte = a.loc[te, F].to_numpy() if nombre == "adaptauth" else X[te].to_numpy()
            mod.fit(Xtr, y_tr)
            rv, rt = mod.predict_proba(Xva)[:, 1], mod.predict_proba(Xte)[:, 1]
            R[nombre][te] = rt
            pred[nombre][te] = (rt > float(np.quantile(rv[y_va == 0], 0.999))).astype(int)

    y = a["anomalo"].to_numpy()
    def met(m):
        p = pred[m]; tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
        return dict(auc_roc=float(roc_auc_score(y, R[m])), auc_pr=float(average_precision_score(y, R[m])),
                    recall_fpr01=tp / int(y.sum()), precision_fpr01=tp / max(tp + fp, 1), fpr=fp / int((y == 0).sum()), tp=tp, fp=fp)
    res = {m: met(m) for m in R}

    # bootstrap por empleado de las diferencias (AdaptAuth-ML − logística)
    u, inv = np.unique(a["user"].to_numpy(), return_inverse=True)
    rng = np.random.default_rng(42); d_auc, d_pr, d_rec = [], [], []
    for _ in range(N_BOOT_CERT):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)).astype(float)[inv]
        if (w * y).sum() == 0:
            continue
        d_auc.append(roc_auc_score(y, R["adaptauth"], sample_weight=w) - roc_auc_score(y, R["logistica"], sample_weight=w))
        d_pr.append(average_precision_score(y, R["adaptauth"], sample_weight=w) - average_precision_score(y, R["logistica"], sample_weight=w))
        P = (w * y).sum()
        d_rec.append(((w * y * pred["adaptauth"]).sum() - (w * y * pred["logistica"]).sum()) / P)
    ic = lambda v: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
    b = int(((pred["adaptauth"] == 1) & (pred["logistica"] == 0) & (y == 1)).sum())
    c = int(((pred["adaptauth"] == 0) & (pred["logistica"] == 1) & (y == 1)).sum())
    salida = dict(
        descripcion="Regresión logística (estandarización, class_weight balanced) con las mismas variables, pliegues y presupuesto FPR ≤ 0.1 %",
        variables=len(F), dias=int(len(a)), dias_maliciosos=int(y.sum()), resultados=res,
        diferencias_bootstrap_empleado=dict(replicas=len(d_auc), auc_roc=dict(valor=res["adaptauth"]["auc_roc"] - res["logistica"]["auc_roc"], ic95=ic(d_auc)),
                                            auc_pr=dict(valor=res["adaptauth"]["auc_pr"] - res["logistica"]["auc_pr"], ic95=ic(d_pr)),
                                            recall_fpr01=dict(valor=res["adaptauth"]["recall_fpr01"] - res["logistica"]["recall_fpr01"], ic95=ic(d_rec))),
        mcnemar_dias_maliciosos=dict(solo_adaptauth=b, solo_logistica=c, p_exacto=float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0),
        segundos=round(time.time() - t0),
    )
    (OUT / "linea_base_logistica_cert.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False))
    print(json.dumps(salida, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
