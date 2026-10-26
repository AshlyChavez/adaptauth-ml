"""
AdaptAuth-ML · Análisis para el levantamiento de observaciones de la semana 10 (SuspiciousLogin).

1. Arranque en frío (observación 1): se simula una PYME que todavía tiene pocas etiquetas.
   Del conjunto de entrenamiento (dic a mar) solo se conserva la etiqueta de una fracción de usuarios
   (0.5 % a 100 %), elegidos al azar con 5 semillas. Isolation Forest se ajusta con TODOS los eventos de
   entrenamiento, porque no necesita etiquetas; el clasificador solo ve los eventos etiquetados.
   Se comparan: Isolation Forest solo, supervisado sin IF y AdaptAuth-ML apilado. Métricas independientes
   del umbral (AUC-ROC y AUC-PR) en prueba (may y jun); con 0 % de etiquetas solo existe Isolation Forest.
   Nota: entrenar en un dataset y aplicar en el otro no es posible, porque SuspiciousLogin (evento de
   inicio de sesión, variables de red y ubicación) y CERT r4.2 (usuario-día, variables de jornada y equipo)
   no comparten el espacio de variables.

2. Recalibración de θ2 (observación 2): con AdaptAuth-ML v11 (supervisado, sin IF), θ1 se mantiene y θ2 se calibra en
   validación con cuatro restricciones: FPR de bloqueo ≤ 1 % (versión anterior), ≤ 0.5 %, ≤ 0.1 % y
   precisión de bloqueo ≥ 0.95. Se reporta la matriz 3×2 en prueba y el compromiso fricción-seguridad.

Uso:
    python run_levantamiento.py --data ruta/suspicious_logins_public_v1.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import adaptauth as aa  # noqa: E402
from run_suspiciouslogin import preparar  # noqa: E402

OUT = Path(__file__).parent / "results_levantamiento"
FRACCIONES = (0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 1.0)
SEMILLAS = (1, 2, 3, 4, 5)


def hgb(seed=42):
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=seed)


def particiones(path):
    d, X, _ = preparar(pd.read_csv(path))
    mes = d["month"].replace({12: 0})
    tr, va, te = (mes <= 3).to_numpy(), (mes == 4).to_numpy(), mes.isin([5, 6]).to_numpy()
    P = {}
    for k, m in (("tr", tr), ("va", va), ("te", te)):
        P[k] = dict(d=d[m].reset_index(drop=True), X=X[m].to_numpy(), y=d.loc[m, "anomalo"].to_numpy())
    return P


def score_if(P, seed=42):
    iso = IsolationForest(n_estimators=200, max_samples=256, random_state=seed, n_jobs=-1).fit(P["tr"]["X"])
    s = {k: -iso.score_samples(P[k]["X"]) for k in ("tr", "va", "te")}
    s_ref = float(np.median(s["va"][P["va"]["y"] == 0]))
    return {k: aa.rescale(v, s_ref) for k, v in s.items()}


def arranque_en_frio(P, sp):
    yt = P["te"]["y"]
    filas = [dict(fraccion=0.0, semilla=0, modelo="Isolation Forest solo", n_etiquetados=0, positivos=0,
                  auc_roc=roc_auc_score(yt, sp["te"]), auc_pr=average_precision_score(yt, sp["te"]))]
    usuarios = P["tr"]["d"]["actor_pseudo_id"].to_numpy()
    uniq = np.unique(usuarios)
    for f in FRACCIONES:
        for seed in (SEMILLAS if f < 1 else (1,)):
            rng = np.random.default_rng(seed)
            elegidos = rng.choice(uniq, size=max(1, int(round(f * len(uniq)))), replace=False)
            m = np.isin(usuarios, elegidos)
            Xl, yl = P["tr"]["X"][m], P["tr"]["y"][m]
            if yl.sum() == 0:
                continue
            sup = hgb(seed).fit(Xl, yl)
            api = hgb(seed).fit(np.column_stack([Xl, sp["tr"][m]]), yl)
            ps = sup.predict_proba(P["te"]["X"])[:, 1]
            pa = api.predict_proba(np.column_stack([P["te"]["X"], sp["te"]]))[:, 1]
            for nombre, r in (("AdaptAuth-ML v11 (sin IF)", ps), ("Apilado con IF (v10)", pa)):
                filas.append(dict(fraccion=f, semilla=seed, modelo=nombre, n_etiquetados=int(m.sum()),
                                  positivos=int(yl.sum()), auc_roc=roc_auc_score(yt, r),
                                  auc_pr=average_precision_score(yt, r)))
    return pd.DataFrame(filas)


def figura_frio(res, ruta):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    colores = {"AdaptAuth-ML v11 (sin IF)": "#1F5FAE", "Apilado con IF (v10)": "#6A7487"}
    for nombre, g in res[res.fraccion > 0].groupby("modelo"):
        s = g.groupby("fraccion")["auc_pr"].agg(["mean", "std"]).reset_index()
        ax.errorbar(s.fraccion * 100, s["mean"], yerr=s["std"].fillna(0), marker="o", capsize=3,
                    label=nombre, color=colores[nombre])
    base = res[res.fraccion == 0]["auc_pr"].iloc[0]
    ax.axhline(base, ls="--", color="#E06A2E", label=f"Isolation Forest solo, sin etiquetas ({base:.3f})")
    ax.set_xscale("log"); ax.set_xlabel("Usuarios de entrenamiento con etiqueta (%)")
    ax.set_ylabel("AUC-PR en prueba"); ax.grid(alpha=.3); ax.legend(frameon=False, fontsize=9)
    ax.set_xticks([0.5, 1, 2, 5, 10, 25, 100]); ax.set_xticklabels(["0.5", "1", "2", "5", "10", "25", "100"])
    fig.tight_layout(); fig.savefig(ruta); plt.close(fig)


def recalibrar_theta2(P, sp):
    sup = hgb().fit(P["tr"]["X"], P["tr"]["y"])
    Rv = sup.predict_proba(P["va"]["X"])[:, 1]
    Rt = sup.predict_proba(P["te"]["X"])[:, 1]
    cfg = aa.Config(cold_start_otp=False, label="anomalo")
    cal = aa.calibrate(P["va"]["d"].assign(c_i=0.0), Rv, cfg, (1.0,))
    t1, yv, yt = cal["theta1"], P["va"]["y"], P["te"]["y"]
    grid = np.array(cfg.theta_grid)
    reglas = {"FPR de bloqueo ≤ 1 % (v10)": ("fpr", 0.01),
              "FPR de bloqueo ≤ 0.5 %": ("fpr", 0.005),
              "FPR de bloqueo ≤ 0.1 % (v11)": ("fpr", 0.001),
              "Precisión de bloqueo ≥ 0.95": ("prec", 0.95)}
    filas = []
    for nombre, (tipo, lim) in reglas.items():
        t2 = 0.99
        for t in grid:
            if t <= t1:
                continue
            m = aa.binary_metrics(yv, (Rv > t).astype(int))
            ok = m["fpr"] <= lim if tipo == "fpr" else (m["precision"] >= lim and m["tp"] > 0)
            if ok:
                t2 = float(t)
                break
        dec = aa.decide(Rt, t1, t2, None, False)
        ev = aa.evaluate_decisions(yt, dec)
        cnt = lambda k, c: int(((dec == k) & (yt == c)).sum())  # noqa: E731
        filas.append(dict(regla_theta2=nombre, theta1=t1, theta2=t2,
                          acceso_legitimos=cnt(0, 0), acceso_anomalos=cnt(0, 1),
                          otp_legitimos=cnt(1, 0), otp_anomalos=cnt(1, 1),
                          bloqueo_legitimos=cnt(2, 0), bloqueo_anomalos=cnt(2, 1),
                          f1_theta1=ev["analisis_theta1"]["f1"], fpr_theta1=ev["analisis_theta1"]["fpr"],
                          precision_bloqueo=ev["analisis_theta2"]["precision"],
                          recall_bloqueo=ev["analisis_theta2"]["recall"],
                          f1_theta2=ev["analisis_theta2"]["f1"], fpr_bloqueo=ev["analisis_theta2"]["fpr"]))
    return pd.DataFrame(filas)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    P = particiones(args.data)
    sp = score_if(P)

    frio = arranque_en_frio(P, sp)
    frio.to_csv(OUT / "arranque_en_frio_detalle.csv", index=False)
    resumen = (frio.groupby(["fraccion", "modelo"])
               .agg(auc_roc=("auc_roc", "mean"), auc_pr=("auc_pr", "mean"), auc_pr_de=("auc_pr", "std"),
                    positivos=("positivos", "mean")).reset_index())
    piv = frio[frio.fraccion > 0].pivot_table(index=["fraccion", "semilla"], columns="modelo", values="auc_pr")
    dif = (piv["Apilado con IF (v10)"] - piv["AdaptAuth-ML v11 (sin IF)"]).groupby("fraccion")
    resumen_dif = pd.DataFrame({"dif_auc_pr_media": dif.mean(), "dif_min": dif.min(), "dif_max": dif.max(),
                                "semillas_a_favor_apilado": dif.apply(lambda s: int((s > 0).sum()))}).reset_index()
    resumen.to_csv(OUT / "arranque_en_frio_resumen.csv", index=False)
    resumen_dif.to_csv(OUT / "arranque_en_frio_diferencias.csv", index=False)
    figura_frio(frio, OUT / "arranque_en_frio_sl.png")
    print(resumen.round(4).to_string(index=False)); print(resumen_dif.round(4).to_string(index=False))

    t2 = recalibrar_theta2(P, sp)  # sp no se usa: el núcleo v11 no incluye IF
    t2.to_csv(OUT / "recalibracion_theta2_sl.csv", index=False)
    print(t2.round(4).to_string(index=False))
    json.dump({"arranque_en_frio": resumen.round(4).to_dict(orient="records"),
               "diferencias_apilado_menos_supervisado": resumen_dif.round(4).to_dict(orient="records"),
               "recalibracion_theta2": t2.round(4).to_dict(orient="records")},
              open(OUT / "levantamiento.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
