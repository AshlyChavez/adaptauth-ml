"""
AdaptAuth-ML · Análisis de resultados (PI 14) sobre los datos registrados en el PI 13 (SuspiciousLogin).

Lee results_validacion_pi13/ y reproduce, con la semilla de referencia (42), las predicciones de la corrida
congelada para estimar intervalos por bootstrap de usuarios. No introduce datos nuevos.

Salidas (results_analisis_pi14/):
  descriptiva_pi14.csv      media, mediana, DE y rango por indicador (5 semillas; línea base determinista)
  analisis_pi14.json        IC 95 % por usuario de F1, FPR y bloqueo; prueba unilateral F1 > 0.60;
                            McNemar con razón de discordancia (OR) e IC exacto; mejora absoluta y relativa
Uso:
    python run_analisis_pi14.py --data ruta/suspicious_logins_public_v1.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import adaptauth as aa  # noqa: E402
from modelo_adaptauth import ModeloAdaptAuth  # noqa: E402
from run_suspiciouslogin import preparar  # noqa: E402

BASE = Path(__file__).parent
OUT = BASE / "results_analisis_pi14"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    val = json.loads((BASE / "results_validacion_pi13" / "validacion_pi13_sl.json").read_text())
    sem = pd.read_csv(BASE / "results_validacion_pi13" / "cp01_semillas_sl.csv")

    # ---------- 1.2 Descriptiva ----------
    filas = []
    for ind in ("f1", "precision", "recall", "fpr", "fpr_bloqueo", "legitimos_bloqueados", "otp", "auc_pr"):
        v = sem[ind]
        filas.append(dict(indicador=ind, condicion="AdaptAuth-ML v11 (5 semillas)", n=len(v), media=v.mean(),
                          mediana=v.median(), de=v.std(ddof=1), minimo=v.min(), maximo=v.max()))
    for nombre, clave in (("Reglas, umbral original", "reglas_umbral_original"), ("Reglas calibradas a igual FPR", "reglas_calibradas_igual_fpr")):
        r = val["comparacion_igual_fpr"][clave]
        for ind in ("f1", "precision", "recall", "fpr"):
            filas.append(dict(indicador=ind, condicion=f"{nombre} (determinista)", n=1, media=r[ind], mediana=r[ind],
                              de=np.nan, minimo=r[ind], maximo=r[ind]))
    pd.DataFrame(filas).to_csv(OUT / "descriptiva_pi14.csv", index=False)

    # ---------- Predicciones de la corrida congelada (semilla 42) ----------
    cfg = aa.Config(cold_start_otp=False, label="anomalo")
    d, X, _ = preparar(pd.read_csv(args.data))
    mes = d["month"].replace({12: 0}).to_numpy()
    tr, va, te = mes <= 3, mes == 4, np.isin(mes, [5, 6])
    dv, dt = d[va].reset_index(drop=True), d[te].reset_index(drop=True)
    m = ModeloAdaptAuth(seed=42).fit(X[tr].to_numpy(), d[tr].anomalo.to_numpy())
    cal = aa.calibrate(dv.assign(c_i=0.0), m.predict_proba(X[va].to_numpy())[:, 1], cfg, (1.0,))
    dec = aa.decide(m.predict_proba(X[te].to_numpy())[:, 1], cal["theta1"], cal["theta2"], None, False)
    y = dt.anomalo.to_numpy(); pred = (dec >= 1).astype(int); blq = (dec == 2).astype(int)
    f1_obs = aa.binary_metrics(y, pred)["f1"]
    assert abs(f1_obs - val["comparacion_igual_fpr"]["adaptauth_v11"]["f1"]) < 1e-9, "no reproduce la corrida del PI 13"

    # ---------- Bootstrap por usuario del desempeño del modelo ----------
    u, inv = np.unique(dt["actor_pseudo_id"].to_numpy(), return_inverse=True)
    sm = lambda v: np.bincount(inv, weights=v.astype(float), minlength=len(u))
    tp, fp, pos, neg, bl = sm((pred == 1) & (y == 1)), sm((pred == 1) & (y == 0)), sm(y == 1), sm(y == 0), sm((blq == 1) & (y == 0))
    rng = np.random.default_rng(42); f1s, fprs, bls = [], [], []
    for _ in range(2000):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)).astype(float)
        TP, FP, P, N = w @ tp, w @ fp, w @ pos, w @ neg
        f1s.append(2 * TP / (2 * TP + FP + (P - TP))); fprs.append(FP / N); bls.append((w @ bl) / N)
    f1s, fprs, bls = map(np.array, (f1s, fprs, bls))
    ic = lambda v: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]

    # ---------- McNemar: razón de discordancia e IC exacto (Clopper-Pearson) ----------
    mc = val["mcnemar_sospechosos"]; b, c = mc["solo_adaptauth"], mc["solo_reglas"]
    ci = stats.binomtest(b, b + c).proportion_ci(confidence_level=0.95, method="exact")
    or_ic = [ci.low / (1 - ci.low), ci.high / (1 - ci.high)]

    comp = val["comparacion_igual_fpr"]; a, r = comp["adaptauth_v11"], comp["reglas_calibradas_igual_fpr"]
    salida = dict(
        hipotesis_he2_f1=dict(meta=0.60, f1=f1_obs, ic95_usuario=ic(f1s), p_unilateral_f1_menor_o_igual_meta=float((f1s <= 0.60).mean())),
        fpr=dict(valor=float(fp.sum() / neg.sum()), ic95_usuario=ic(fprs), meta=0.05),
        fpr_bloqueo=dict(valor=float(bl.sum() / neg.sum()), ic95_usuario=ic(bls), meta=0.001),
        mcnemar=dict(b_solo_adaptauth=b, c_solo_reglas=c, chi2_corregido=(abs(b - c) - 1) ** 2 / (b + c), p_exacto=mc["p_exacto"],
                     odds_ratio_discordantes=b / c, ic95_or=or_ic),
        mejora_frente_reglas_calibradas={k: dict(base=r[k], artefacto=a[k], absoluta=a[k] - r[k],
                                                 relativa_pct=100 * (a[k] - r[k]) / r[k] if r[k] else None)
                                         for k in ("recall", "f1", "precision", "fpr")},
        prueba_principal=val["prueba_principal_bootstrap_usuario"],
    )
    (OUT / "analisis_pi14.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False))
    print(pd.DataFrame(filas).round(4).to_string(index=False)); print(json.dumps(salida, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
