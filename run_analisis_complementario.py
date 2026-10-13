"""
AdaptAuth-ML · Análisis complementarios solicitados en la revisión del parcial (SuspiciousLogin).

1. Ablación: supervisado sin IF frente a AdaptAuth-ML apilado (IF como variable del clasificador).
2. Prueba pareada (14 grupos mes x día) con Shapiro-Wilk reportado, y bootstrap por usuario
   como prueba complementaria que no asume independencia entre grupos.
3. Curva de la línea base: puntos de operación de las reglas estáticas para descartar mala calibración.
4. Importancia por grupo de variables del modelo apilado y desempeño según exista historial del usuario.
   (Latencias RNF-04/RNF-05 y RNF-12 se verifican en run_requisitos_sl.py.)
6. Tabla del límite inferior de la V de Aiken (Penfield y Giacobbi, 2004) para 5 y 7 jueces.

Uso:
    python run_analisis_complementario.py --data ruta/suspicious_logins_public_v1.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import adaptauth as aa  # noqa: E402
from run_suspiciouslogin import CATEGORICAS, GRUPOS, preparar, prueba_pareada_por_grupos  # noqa: E402

OUT = Path(__file__).parent / "results_complementarios"


def hgb(seed=42):
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=seed)


def evaluar(nombre, R_val, R_test, dv, dt, cfg):
    cal = aa.calibrate(dv.assign(c_i=0.0), R_val, cfg, (1.0,))
    dec = aa.decide(R_test, cal["theta1"], cal["theta2"], None, False)
    y = dt["anomalo"].to_numpy()
    ev = aa.evaluate_decisions(y, dec)
    return dict(modelo=nombre, auc_roc=float(roc_auc_score(y, R_test)), auc_pr=float(average_precision_score(y, R_test)),
                f1=ev["analisis_theta1"]["f1"], precision=ev["analisis_theta1"]["precision"],
                recall=ev["analisis_theta1"]["recall"], fpr=ev["analisis_theta1"]["fpr"],
                f1_theta2=ev["analisis_theta2"]["f1"], fpr_theta2=ev["analisis_theta2"]["fpr"],
                theta1=cal["theta1"], theta2=cal["theta2"]), dec


def bootstrap_usuarios(dt, dec_a, dec_b, reps=1000, seed=11):
    """IC 95 % de la diferencia de F1 (B - A) remuestreando usuarios completos (bootstrap por conglomerados).
    No asume independencia entre eventos ni entre grupos mes x día: el usuario es la unidad remuestreada."""
    y = dt["anomalo"].to_numpy()
    pa, pb = (dec_a >= 1).astype(int), (dec_b >= 1).astype(int)
    codigos, uniq = pd.factorize(dt["actor_pseudo_id"])
    orden = np.argsort(codigos, kind="stable")
    cortes = np.searchsorted(codigos[orden], np.arange(len(uniq) + 1))
    rng = np.random.default_rng(seed)
    difs = []
    for _ in range(reps):
        sel = rng.integers(0, len(uniq), size=len(uniq))
        ii = np.concatenate([orden[cortes[u]:cortes[u + 1]] for u in sel])
        difs.append(aa.binary_metrics(y[ii], pb[ii])["f1"] - aa.binary_metrics(y[ii], pa[ii])["f1"])
    difs = np.array(difs)
    return dict(unidad="usuario", n_usuarios=int(len(uniq)), reps=reps, dif_media=float(difs.mean()),
                ic95=[float(np.percentile(difs, 2.5)), float(np.percentile(difs, 97.5))],
                prop_dif_menor_igual_0=float((difs <= 0).mean()))


def puntos_operacion(R_val, R_test, yv, yt, fprs=(0.005, 0.01, 0.02, 0.05, 0.10, 0.20)):
    filas = []
    for f in fprs:
        thr = float(np.quantile(R_val[yv == 0], 1 - f))
        m = aa.binary_metrics(yt, (R_test > thr).astype(int))
        filas.append(dict(fpr_objetivo=f, umbral=thr, fpr_prueba=m["fpr"], recall=m["recall"],
                          precision=m["precision"], f1=m["f1"]))
    return pd.DataFrame(filas)


def aiken_li(V, n, k, z=1.96):
    """Límite inferior del IC score de Penfield y Giacobbi (2004); k = c - 1."""
    nk = n * k
    return (2 * nk * V + z ** 2 - z * np.sqrt(4 * nk * V * (1 - V) + z ** 2)) / (2 * (nk + z ** 2))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); args = ap.parse_args()
    cfg = aa.Config(cold_start_otp=False, label="anomalo")
    OUT.mkdir(exist_ok=True)
    raw = pd.read_csv(args.data)
    d, X, _ = preparar(raw)
    cols = list(X.columns)
    mes = d["month"].replace({12: 0})
    tr, va, te = (mes <= 3).to_numpy(), (mes == 4).to_numpy(), mes.isin([5, 6]).to_numpy()
    dtr, dv, dt = d[tr].reset_index(drop=True), d[va].reset_index(drop=True), d[te].reset_index(drop=True)
    Xtr, Xv, Xt = X[tr].to_numpy(), X[va].to_numpy(), X[te].to_numpy()
    ytr, yv, yt = dtr.anomalo.to_numpy(), dv.anomalo.to_numpy(), dt.anomalo.to_numpy()
    out = {}

    # ---------- 1. Ablación ----------
    filas, decs, Rv, Rt = [], {}, {}, {}
    r, decs["reglas"] = evaluar("Reglas estáticas", dv["c_i"].to_numpy(), dt["c_i"].to_numpy(), dv, dt, cfg)
    filas.append(r); Rv["reglas"], Rt["reglas"] = dv["c_i"].to_numpy(), dt["c_i"].to_numpy()
    iso = IsolationForest(n_estimators=200, max_samples=256, random_state=cfg.seed, n_jobs=-1).fit(Xtr)
    s_tr, s_v, s_t = (-iso.score_samples(M) for M in (Xtr, Xv, Xt))
    s_ref = float(np.median(s_v[yv == 0]))
    sp_tr, sp_v, sp_t = (aa.rescale(s, s_ref) for s in (s_tr, s_v, s_t))
    r, decs["iforest"] = evaluar("Isolation Forest", sp_v, sp_t, dv, dt, cfg); filas.append(r)
    Rv["iforest"], Rt["iforest"] = sp_v, sp_t
    sup = hgb().fit(Xtr, ytr)
    pv, pt = sup.predict_proba(Xv)[:, 1], sup.predict_proba(Xt)[:, 1]
    r, decs["supervisado"] = evaluar("Supervisado sin IF", pv, pt, dv, dt, cfg); filas.append(r)
    Rv["supervisado"], Rt["supervisado"] = pv, pt
    api = hgb().fit(np.column_stack([Xtr, sp_tr]), ytr)
    av, at = api.predict_proba(np.column_stack([Xv, sp_v]))[:, 1], api.predict_proba(np.column_stack([Xt, sp_t]))[:, 1]
    r, decs["apilado"] = evaluar("AdaptAuth-ML apilado (con IF)", av, at, dv, dt, cfg); filas.append(r)
    Rv["apilado"], Rt["apilado"] = av, at
    tabla = pd.DataFrame(filas)
    tabla.to_csv(OUT / "ablacion_sl.csv", index=False)
    out["ablacion"] = tabla.round(4).to_dict(orient="records")
    print(tabla.round(3).to_string(index=False))

    # ---------- 2. Pruebas pareadas + bootstrap por bloques ----------
    out["pareadas_14_grupos"] = {
        f"{b}_vs_{a}": {k: v for k, v in prueba_pareada_por_grupos(dt, decs[a], decs[b], ["month", "day_of_week"]).items()
                        if k not in ("f1_baseline", "f1_adaptauth")}
        for a, b in (("reglas", "apilado"), ("iforest", "apilado"), ("supervisado", "apilado"), ("reglas", "supervisado"))}
    out["bootstrap_usuarios"] = {f"{b}_vs_{a}": bootstrap_usuarios(dt, decs[a], decs[b])
                                for a, b in (("reglas", "apilado"), ("supervisado", "apilado"))}
    for k, v in out["pareadas_14_grupos"].items():
        print(f"[pareada] {k}: {v['prueba']} est={v['estadistico']:.2f} p={v['p_valor']:.3g} "
              f"Shapiro p={v['shapiro_p_diferencias']:.3f} dz={v['d_cohen_dz']}")
    for k, v in out["bootstrap_usuarios"].items():
        print(f"[bootstrap] {k}: ΔF1={v['dif_media']:.3f} IC95={v['ic95']} usuarios={v['n_usuarios']}")

    # ---------- 3. Curva de la línea base ----------
    po_reglas = puntos_operacion(Rv["reglas"], Rt["reglas"], yv, yt)
    po_api = puntos_operacion(Rv["apilado"], Rt["apilado"], yv, yt)
    po_reglas.to_csv(OUT / "puntos_operacion_reglas_sl.csv", index=False)
    po_api.to_csv(OUT / "puntos_operacion_apilado_sl.csv", index=False)
    niveles = pd.Series(dt["c_i"]).round(2).value_counts().sort_index()
    out["linea_base"] = dict(niveles_distintos_score=int(niveles.size),
                             puntos_operacion=po_reglas.round(4).to_dict(orient="records"),
                             recall_max_con_fpr_5=float(po_reglas.loc[po_reglas.fpr_objetivo == 0.05, "recall"].iloc[0]))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for k, nombre, color in (("reglas", "Reglas estáticas", "#9a9890"), ("iforest", "Isolation Forest", "#eb6834"),
                             ("supervisado", "Supervisado sin IF", "#1baf7a"), ("apilado", "AdaptAuth-ML apilado", "#2a78d6")):
        p, rc, _ = precision_recall_curve(yt, Rt[k])
        ax.plot(rc, p, color=color, lw=2, label=f"{nombre} (AUC-PR {average_precision_score(yt, Rt[k]):.3f})")
    ax.axhline(yt.mean(), color="#b5b4ab", ls="--", lw=1)
    ax.text(0.02, yt.mean() - 0.045, f"tasa base {yt.mean():.3f}", fontsize=8, color="#5f5e5a")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precisión"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(color="#e6e5df", lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.savefig(OUT / "curva_pr_modelos_sl.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # ---------- 4. Importancia por grupo (modelo apilado) y desempeño según historial ----------
    from run_suspiciouslogin import GRUPOS as G
    grupo_de = {c: g for g, cs in G.items() for c in cs}
    grupos_cols = {c: grupo_de.get(c, "autenticacion" if c.startswith("login_type") else "red_geo") for c in cols}
    grupos_cols["score_if"] = "score_if"
    Xt_aug = np.column_stack([Xt, sp_t]); nombres = cols + ["score_if"]
    base_ap = average_precision_score(yt, at)
    rng = np.random.default_rng(cfg.seed)
    imp = {}
    for g in sorted(set(grupos_cols.values())):
        ii = [i for i, c in enumerate(nombres) if grupos_cols[c] == g]
        caidas = []
        for _ in range(5):
            Xp = Xt_aug.copy(); perm = rng.permutation(len(Xp)); Xp[:, ii] = Xp[perm][:, ii]
            caidas.append(base_ap - average_precision_score(yt, api.predict_proba(Xp)[:, 1]))
        imp[g] = dict(media=float(np.mean(caidas)), de=float(np.std(caidas, ddof=1)))
    out["importancia_grupos_apilado"] = imp
    print("[importancia apilado]", {k: round(v["media"], 4) for k, v in imp.items()})
    hist = dt["hours_since_last_login"].to_numpy() > 0
    out["por_historial_apilado"] = {
        "con_historial": dict(n=int(hist.sum()), positivos=int(yt[hist].sum()),
                              **{k: aa.binary_metrics(yt[hist], (decs["apilado"][hist] >= 1).astype(int))[k] for k in ("f1", "precision", "recall")}),
        "primer_acceso": dict(n=int((~hist).sum()), positivos=int(yt[~hist].sum()),
                              **{k: aa.binary_metrics(yt[~hist], (decs["apilado"][~hist] >= 1).astype(int))[k] for k in ("f1", "precision", "recall")})}
    print("[historial]", out["por_historial_apilado"])

    # ---------- 6. V de Aiken ----------
    filas = []
    for n_j in (5, 7):
        k = 4
        for suma_desv in range(0, 4):  # puntos perdidos respecto del máximo (cada 4 en vez de 5 = 1 punto)
            V = (n_j * k - suma_desv) / (n_j * k)
            filas.append(dict(jueces=n_j, puntos_bajo_maximo=suma_desv, V=round(V, 3), LI_95=round(float(aiken_li(V, n_j, k)), 3)))
    va_tab = pd.DataFrame(filas)
    va_tab.to_csv(OUT / "aiken_limite_inferior.csv", index=False)
    out["aiken"] = va_tab.to_dict(orient="records")
    print(va_tab.to_string(index=False))

    aa.to_json(out, OUT / "analisis_complementario.json")
    print(f"[listo] {OUT}")


if __name__ == "__main__":
    main()
