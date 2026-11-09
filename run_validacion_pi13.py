"""
AdaptAuth-ML · Ejecución de la validación (PI 13) sobre SuspiciousLogin, versión 11 congelada.

Atiende las observaciones del PI 11 + PI 12:
  O2  Línea base de reglas calibrada: el puntaje de reglas c_i (ponderación al estilo de Freeman et al., 2016)
      con umbral fijado en abril para el mismo FPR de validación que AdaptAuth-ML (θ1); se compara el recall
      a igual FPR. Los empates del puntaje discreto se rompen con un desempate aleatorio fijo (semilla 42).
  O3  Prueba principal única y preespecificada: bootstrap por usuario (2 000 réplicas) de la diferencia de
      recall a igual FPR; McNemar exacto sobre los accesos sospechosos como complemento.
  O6  Curva recall-FPR con umbrales calibrados en abril: recall con FPR de 1 %, 2 % y 5 %.
  CP01 Repetición con 5 semillas del clasificador (42, la de referencia, y 1 a 4): media y DE.
Los resultados de mayo y junio son NO CONFIRMATORIOS: ese periodo se usó en versiones anteriores (O1).

Uso:
    python run_validacion_pi13.py --data ruta/suspicious_logins_public_v1.csv
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import adaptauth as aa  # noqa: E402
from modelo_adaptauth import ModeloAdaptAuth  # noqa: E402
from run_suspiciouslogin import preparar  # noqa: E402

OUT = Path(__file__).parent / "results_validacion_pi13"
N_BOOT = 2000
SEMILLAS = (42, 1, 2, 3, 4)  # 42 es la semilla de referencia de la versión 11
FPR_OBJ = (0.005, 0.01, 0.02, 0.05)


def umbral_para_fpr(score_val, y_val, fpr):
    """Menor umbral cuyo FPR en validación no supera fpr."""
    return float(np.quantile(score_val[y_val == 0], 1 - fpr))


def conteos(y, pred):
    return dict(tp=int(((pred == 1) & (y == 1)).sum()), fn=int(((pred == 0) & (y == 1)).sum()),
                fp=int(((pred == 1) & (y == 0)).sum()), tn=int(((pred == 0) & (y == 0)).sum()))


def bootstrap_usuario(user, y, pred_a, pred_b, n_boot=N_BOOT, seed=42):
    """Bootstrap por conglomerado (usuario) de la diferencia B - A en recall, FPR y F1."""
    u, inv = np.unique(user, return_inverse=True)
    def suma(m):
        return np.bincount(inv, weights=m.astype(float), minlength=len(u))
    pos, neg = suma(y == 1), suma(y == 0)
    tpa, fpa = suma((pred_a == 1) & (y == 1)), suma((pred_a == 1) & (y == 0))
    tpb, fpb = suma((pred_b == 1) & (y == 1)), suma((pred_b == 1) & (y == 0))
    rng = np.random.default_rng(seed)
    out = {"recall": [], "fpr": [], "f1": []}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)).astype(float)
        P, N = w @ pos, w @ neg
        ra, rb = (w @ tpa) / P, (w @ tpb) / P
        fa, fb = (w @ fpa) / N, (w @ fpb) / N
        f1a = 2 * (w @ tpa) / (2 * (w @ tpa) + (w @ fpa) + (P - w @ tpa))
        f1b = 2 * (w @ tpb) / (2 * (w @ tpb) + (w @ fpb) + (P - w @ tpb))
        out["recall"].append(rb - ra); out["fpr"].append(fb - fa); out["f1"].append(f1b - f1a)
    res = {}
    for k, v in out.items():
        v = np.array(v)
        res[k] = dict(media=float(v.mean()), ic95=[float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))],
                      p_bilateral=float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))))
    res["usuarios"] = int(len(u)); res["usuarios_con_sospechosos"] = int((pos > 0).sum()); res["replicas"] = n_boot
    return res


def mcnemar(y, pred_a, pred_b):
    """McNemar exacto sobre los accesos sospechosos: B detecta y A no (b) frente a A detecta y B no (c)."""
    s = y == 1
    b = int(((pred_b == 1) & (pred_a == 0) & s).sum()); c = int(((pred_a == 1) & (pred_b == 0) & s).sum())
    p = float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c > 0 else 1.0
    return dict(solo_adaptauth=b, solo_reglas=c, ambos=int(((pred_a == 1) & (pred_b == 1) & s).sum()),
                ninguno=int(((pred_a == 0) & (pred_b == 0) & s).sum()), p_exacto=p)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); args = ap.parse_args()
    t0 = time.time()
    cfg = aa.Config(cold_start_otp=False, label="anomalo")
    OUT.mkdir(exist_ok=True); (OUT / "figuras").mkdir(exist_ok=True)
    d, X, _ = preparar(pd.read_csv(args.data))
    mes = d["month"].replace({12: 0}).to_numpy()
    tr, va, te = mes <= 3, mes == 4, np.isin(mes, [5, 6])
    dtr, dv, dt = (d[m].reset_index(drop=True) for m in (tr, va, te))
    Xtr, Xv, Xt = X[tr].to_numpy(), X[va].to_numpy(), X[te].to_numpy()
    yv, yt = dv.anomalo.to_numpy(), dt.anomalo.to_numpy()
    user = dt["actor_pseudo_id"].to_numpy()

    # ---------- CP01: AdaptAuth-ML con 5 semillas ----------
    filas, ref = [], None
    for s in SEMILLAS:
        m = ModeloAdaptAuth(seed=s).fit(Xtr, dtr.anomalo.to_numpy())
        pv, pt = m.predict_proba(Xv)[:, 1], m.predict_proba(Xt)[:, 1]
        cal = aa.calibrate(dv.assign(c_i=0.0), pv, cfg, (1.0,))
        dec = aa.decide(pt, cal["theta1"], cal["theta2"], None, False)
        r1 = aa.binary_metrics(yt, (dec >= 1).astype(int)); leg = yt == 0
        filas.append(dict(semilla=s, theta1=cal["theta1"], theta2=cal["theta2"], f1=r1["f1"], precision=r1["precision"],
                          recall=r1["recall"], fpr=r1["fpr"], fpr_bloqueo=float(((dec == 2) & leg).sum() / leg.sum()),
                          legitimos_bloqueados=int(((dec == 2) & leg).sum()), otp=int((dec == 1).sum()),
                          bloqueos=int((dec == 2).sum()), auc_roc=float(roc_auc_score(yt, pt)),
                          auc_pr=float(average_precision_score(yt, pt))))
        if s == 42:
            ref = dict(pv=pv, pt=pt, cal=cal, dec=dec)
    semillas = pd.DataFrame(filas); semillas.to_csv(OUT / "cp01_semillas_sl.csv", index=False)
    resumen_sem = semillas.drop(columns="semilla").agg(["mean", "std"]).T

    # ---------- O2: línea base de reglas calibrada a igual FPR ----------
    rng = np.random.default_rng(42)
    cv = dv["c_i"].to_numpy() + rng.uniform(0, 1e-6, len(dv)); ct = dt["c_i"].to_numpy() + rng.uniform(0, 1e-6, len(dt))
    pred_ada = (ref["dec"] >= 1).astype(int)
    fpr_val_ada = aa.binary_metrics(yv, (ref["pv"] > ref["cal"]["theta1"]).astype(int))["fpr"]
    thr_r = umbral_para_fpr(cv, yv, fpr_val_ada)
    pred_reg = (ct > thr_r).astype(int)
    pred_reg_fija = (aa.decide(dt["c_i"].to_numpy(), *[aa.calibrate(dv, np.zeros(len(dv)), cfg, (0.0,))[k] for k in ("theta1", "theta2")], None, False) >= 1).astype(int)
    comp = {}
    for nombre, pr in (("reglas_umbral_original", pred_reg_fija), ("reglas_calibradas_igual_fpr", pred_reg), ("adaptauth_v11", pred_ada)):
        m = aa.binary_metrics(yt, pr); comp[nombre] = dict(**{k: m[k] for k in ("f1", "precision", "recall", "fpr")}, **conteos(yt, pr))
    comp["fpr_validacion_objetivo"] = fpr_val_ada
    comp["auc_roc"] = dict(reglas=float(roc_auc_score(yt, dt["c_i"])), adaptauth=float(roc_auc_score(yt, ref["pt"])))
    comp["auc_pr"] = dict(reglas=float(average_precision_score(yt, dt["c_i"])), adaptauth=float(average_precision_score(yt, ref["pt"])))

    # ---------- O3: prueba principal (bootstrap por usuario) y McNemar ----------
    boot = bootstrap_usuario(user, yt, pred_reg, pred_ada)
    mc = mcnemar(yt, pred_reg, pred_ada)

    # ---------- O6: curva recall-FPR con umbrales calibrados en abril ----------
    curva = []
    for f in FPR_OBJ:
        ta, tr_ = umbral_para_fpr(ref["pv"], yv, f), umbral_para_fpr(cv, yv, f)
        ma, mr = aa.binary_metrics(yt, (ref["pt"] > ta).astype(int)), aa.binary_metrics(yt, (ct > tr_).astype(int))
        curva.append(dict(fpr_objetivo=f, umbral_adaptauth=ta, fpr_prueba_adaptauth=ma["fpr"], recall_adaptauth=ma["recall"],
                          precision_adaptauth=ma["precision"], fpr_prueba_reglas=mr["fpr"], recall_reglas=mr["recall"]))
    curva.append(dict(fpr_objetivo="θ1 (máximo F1)", umbral_adaptauth=ref["cal"]["theta1"], fpr_prueba_adaptauth=comp["adaptauth_v11"]["fpr"],
                      recall_adaptauth=comp["adaptauth_v11"]["recall"], precision_adaptauth=comp["adaptauth_v11"]["precision"],
                      fpr_prueba_reglas=comp["reglas_calibradas_igual_fpr"]["fpr"], recall_reglas=comp["reglas_calibradas_igual_fpr"]["recall"]))
    pd.DataFrame(curva).to_csv(OUT / "o6_recall_fpr_sl.csv", index=False)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fa, ra, _ = roc_curve(yt, ref["pt"]); fr, rr, _ = roc_curve(yt, dt["c_i"])
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(fa * 100, ra, color="#2a78d6", lw=2, label="AdaptAuth-ML v11")
    ax.plot(fr * 100, rr, color="#eb6834", lw=2, ls="--", label="Reglas estáticas (puntaje c_i)")
    for row in curva[:-1]:
        ax.scatter(row["fpr_prueba_adaptauth"] * 100, row["recall_adaptauth"], color="#2a78d6", zorder=3)
    ax.scatter(comp["adaptauth_v11"]["fpr"] * 100, comp["adaptauth_v11"]["recall"], marker="D", s=50, color="#1f1f1d", zorder=4, label="θ1 calibrado")
    ax.axvline(5, color="#9a9890", ls=":", lw=1); ax.text(5.1, 0.05, "Presupuesto FPR 5 %", fontsize=8, color="#6a7487")
    ax.set_xlim(0, 10); ax.set_ylim(0, 1); ax.set_xlabel("FPR en prueba (%)"); ax.set_ylabel("Recall")
    ax.grid(alpha=.3); ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.savefig(OUT / "figuras" / "recall_fpr_sl.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()
    import sklearn, scipy
    info = dict(commit=commit, fecha=time.strftime("%Y-%m-%d %H:%M"), python=platform.python_version(),
                sklearn=sklearn.__version__, pandas=pd.__version__, numpy=np.__version__, scipy=scipy.__version__,
                so=platform.platform(), cpu=platform.processor() or platform.machine(),
                eventos=dict(entrenamiento=int(tr.sum()), validacion=int(va.sum()), prueba=int(te.sum()),
                             sospechosos_prueba=int(yt.sum())),
                confirmatorio=False, nota="Mayo-junio se usó en versiones anteriores (v5, v6, v9, v10, v11, arranque en frío y ablación): resultados no confirmatorios.",
                segundos=round(time.time() - t0, 1))
    salida = dict(info=info, cp01_semillas=resumen_sem.to_dict(), comparacion_igual_fpr=comp,
                  prueba_principal_bootstrap_usuario=boot, mcnemar_sospechosos=mc, o6_curva=curva)
    (OUT / "validacion_pi13_sl.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False, default=str))
    pd.set_option("display.width", 200)
    print(json.dumps(info, indent=1, ensure_ascii=False)); print(resumen_sem.round(4))
    print(json.dumps(comp, indent=1)); print(json.dumps(boot, indent=1)); print(mc); print(pd.DataFrame(curva).round(4).to_string())


if __name__ == "__main__":
    main()
