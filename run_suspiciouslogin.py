"""
AdaptAuth-ML sobre SuspiciousLogin Dataset (Google Workspace, 73 528 inicios de sesión exitosos).

Fuente: https://github.com/salomaopena/SuspiciousLoginDataset (CC BY-NC 4.0)
Etiqueta: is_suspicious de Google Workspace (señal operativa de sospecha, no ataque confirmado).

Diseño:
  - C1: las variables ya calculadas por los autores, agrupadas en temporales, red/geo, historial
    del usuario y tipo de autenticación (el archivo público no trae IP, fecha exacta ni dispositivo).
  - Partición temporal por mes: entrenamiento dic-mar, validación abril, prueba mayo-junio
    (julio-agosto excluidos por vacaciones académicas, como hacen los autores).
  - Modelos: reglas estáticas (c_i), Isolation Forest (no supervisado), supervisado
    (HistGradientBoosting) e híbrido R = alpha * s'(IF) + (1 - alpha) * p(supervisado).
  - Calibración en validación: máx. F1 con FPR <= 5 % (θ1, OTP) y FPR de bloqueo <= 1 % (θ2).

Uso:
    python run_suspiciouslogin.py --data ruta/suspicious_logins_public_v1.csv
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent / "src"))
import adaptauth as aa  # noqa: E402
import figures  # noqa: E402

OUT = Path(__file__).parent / "results_sl"

GRUPOS = {
    "temporal": ["event_hour", "day_of_week", "is_weekend", "is_business_hours", "is_night_login"],
    "red_geo": ["ip_version", "new_ip", "distinct_ips_7d", "distinct_ips_30d", "is_new_country", "is_new_region",
                "is_new_city", "distinct_countries_cumulative", "distinct_regions_cumulative",
                "distinct_cities_cumulative", "country_changed", "continent_changed", "countries_seen_30d",
                "regions_seen_30d", "cities_seen_30d", "geo_jump", "multiple_country_logins_24h",
                "distinct_users_per_network_24h", "travel_distance_km", "travel_speed_kmh", "impossible_travel"],
    "historial": ["logins_24h", "logins_7d", "logins_30d", "avg_logins_per_day", "hours_since_last_login",
                  "days_since_first_login", "abnormal_login_hour", "historial_disponible"],
}
# login_challenge_method se EXCLUYE: es la verificación que Google aplicó porque ya marcó el acceso como
# sospechoso (p. ej. "sms" 100 %, "idv_preregistered_email" 89 % sospechosos). Es una consecuencia de la
# etiqueta, no un dato disponible antes de la decisión: usarla sería fuga de etiqueta.
CATEGORICAS = ["login_type", "continent", "ip_country"]
EXCLUIDAS_POR_FUGA = ["login_challenge_method", "risk_score", "risk_level"]

# Reglas estáticas (Alternativa 1), equivalentes a r1..r5 del PI7 con las variables disponibles
REGLAS = {"new_ip": 0.25, "is_new_country": 0.25, "impossible_travel": 0.30,
          "abnormal_login_hour": 0.15, "geo_cambio_continente": 0.05}


def preparar(d: pd.DataFrame):
    d = d.copy()
    d["historial_disponible"] = (d["hours_since_last_login"] > 0).astype(int)
    d["geo_cambio_continente"] = (d["geo_jump"] == 2).astype(int)
    top_paises = d["ip_country"].value_counts().head(15).index
    d["ip_country"] = d["ip_country"].where(d["ip_country"].isin(top_paises), "Otro")
    num = [c for g in GRUPOS.values() for c in g]
    X = pd.get_dummies(d[num + CATEGORICAS], columns=CATEGORICAS, dummy_na=True, dtype=float)
    X = X.fillna(0.0)
    d["anomalo"] = d["label"].astype(int)
    d["c_i"] = sum(w * d[r].astype(float) for r, w in REGLAS.items())
    d["arranque_frio"] = 1 - d["historial_disponible"]
    grupo_de = {c: g for g, cols in GRUPOS.items() for c in cols}
    grupos_X = {c: grupo_de.get(c, "autenticacion" if c.startswith("login_type") else "red_geo")
                for c in X.columns}
    return d, X, grupos_X


def f1_max(y, s):
    p, r, _ = precision_recall_curve(y, s)
    return float(np.max(np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)))


def evaluar(nombre, s_val, s_test, dv, dt, cfg, alpha_grid=(1.0,), c_val=None, c_test=None):
    v = dv.assign(c_i=0.0 if c_val is None else c_val)
    cal = aa.calibrate(v, s_val, cfg, alpha_grid)
    R = aa.risk_score(s_test, np.zeros(len(dt)) if c_test is None else c_test, cal["alpha"])
    R_val = aa.risk_score(s_val, v["c_i"].to_numpy(), cal["alpha"])
    dec = aa.decide(R, cal["theta1"], cal["theta2"], None, False)
    y = dt["anomalo"].to_numpy()
    res = dict(calibracion=cal, **aa.evaluate_decisions(y, dec),
               auc_roc=float(roc_auc_score(y, R)), auc_pr=float(average_precision_score(y, R)),
               f1_max_validacion=f1_max(dv["anomalo"].to_numpy(), R_val))
    for seg, mask in (("con_historial", dt["historial_disponible"].to_numpy() == 1),
                      ("primer_acceso", dt["historial_disponible"].to_numpy() == 0)):
        res[f"theta1_{seg}"] = aa.binary_metrics(y[mask], (dec[mask] >= 1).astype(int))
    m = res["analisis_theta1"]
    print(f"  {nombre:<17} AUC-ROC={res['auc_roc']:.3f} AUC-PR={res['auc_pr']:.3f} | θ1: F1={m['f1']:.3f} "
          f"P={m['precision']:.3f} R={m['recall']:.3f} FPR={m['fpr']:.3%} | θ2: F1={res['analisis_theta2']['f1']:.3f} "
          f"FPR={res['analisis_theta2']['fpr']:.3%} | α={cal['alpha']}"
          + ("  [FPR<=5% no alcanzable]" if cal.get("restriccion_fpr_incumplida") else ""))
    return res, dec, R, R_val


def prueba_pareada_por_grupos(dt, dec_a, dec_b, grupos):
    """Compara el F1 (θ1) de dos condiciones sobre los mismos grupos (mes x día de la semana)."""
    y = dt["anomalo"].to_numpy()
    f_a, f_b = [], []
    for _, idx in dt.groupby(grupos).indices.items():
        f_a.append(aa.binary_metrics(y[idx], (dec_a[idx] >= 1).astype(int))["f1"])
        f_b.append(aa.binary_metrics(y[idx], (dec_b[idx] >= 1).astype(int))["f1"])
    f_a, f_b = np.array(f_a), np.array(f_b)
    diff = f_b - f_a
    sh = float(stats.shapiro(diff).pvalue) if np.ptp(diff) > 0 else 1.0
    if sh > 0.05:
        t = stats.ttest_rel(f_b, f_a); nombre, est, p = "t de Student pareada", float(t.statistic), float(t.pvalue)
    else:
        w = stats.wilcoxon(f_b, f_a); nombre, est, p = "Wilcoxon de rangos con signo", float(w.statistic), float(w.pvalue)
    sd = diff.std(ddof=1)
    return dict(n_grupos=len(diff), f1_baseline=f_a.tolist(), f1_adaptauth=f_b.tolist(),
                media_baseline=float(f_a.mean()), media_adaptauth=float(f_b.mean()),
                de_baseline=float(f_a.std(ddof=1)), de_adaptauth=float(f_b.std(ddof=1)),
                shapiro_p_diferencias=sh, prueba=nombre, estadistico=est, p_valor=p,
                d_cohen_dz=float(diff.mean() / sd) if sd > 0 else None)


def puntos_operacion(R_val, R_test, yv, yt):
    filas = []
    for fpr in (0.005, 0.01, 0.02, 0.03, 0.05, 0.10):
        thr = float(np.quantile(R_val[yv == 0], 1 - fpr))
        m = aa.binary_metrics(yt, (R_test > thr).astype(int))
        filas.append(dict(fpr_objetivo=fpr, umbral=thr, fpr_prueba=m["fpr"], recall=m["recall"],
                          precision=m["precision"], f1=m["f1"]))
    return pd.DataFrame(filas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    cfg = aa.Config(cold_start_otp=False, label="anomalo")
    OUT.mkdir(exist_ok=True); (OUT / "figuras").mkdir(exist_ok=True)
    t0 = time.time()

    raw = pd.read_csv(args.data)
    d, X, grupos_X = preparar(raw)
    mes = d["month"].replace({12: 0})
    tr, va, te = (mes <= 3).to_numpy(), (mes == 4).to_numpy(), mes.isin([5, 6]).to_numpy()
    dtr, dv, dt = d[tr].reset_index(drop=True), d[va].reset_index(drop=True), d[te].reset_index(drop=True)
    Xtr, Xv, Xt = X[tr].to_numpy(), X[va].to_numpy(), X[te].to_numpy()
    print(f"[datos] {len(d):,} inicios | {d['actor_pseudo_id'].nunique():,} usuarios | {X.shape[1]} variables")
    print(f"        train dic-mar {tr.sum():,} ({dtr.anomalo.mean():.1%}) | val abr {va.sum():,} ({dv.anomalo.mean():.1%})"
          f" | prueba may-jun {te.sum():,} ({dt.anomalo.mean():.1%})")

    res, decs, Rs, Rvs = {}, {}, {}, {}
    print("[modelos] calibrados en validación, evaluados en prueba")
    res["reglas_estaticas"], decs["reglas_estaticas"], Rs["reglas_estaticas"], Rvs["reglas_estaticas"] = evaluar(
        "reglas_estaticas", np.zeros(len(dv)), np.zeros(len(dt)), dv, dt, cfg, (0.0,), dv["c_i"].to_numpy(), dt["c_i"].to_numpy())

    iso = IsolationForest(n_estimators=200, max_samples=256, random_state=cfg.seed, n_jobs=-1).fit(Xtr)
    s_v, s_t = -iso.score_samples(Xv), -iso.score_samples(Xt)
    s_ref = float(np.median(s_v[dv["anomalo"].to_numpy() == 0]))
    sp_v, sp_t = aa.rescale(s_v, s_ref), aa.rescale(s_t, s_ref)
    res["iforest"], decs["iforest"], Rs["iforest"], Rvs["iforest"] = evaluar("iforest", sp_v, sp_t, dv, dt, cfg)
    res["iforest"]["s_ref"] = s_ref

    hgb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                         random_state=cfg.seed).fit(Xtr, dtr["anomalo"].to_numpy())
    p_v, p_t = hgb.predict_proba(Xv)[:, 1], hgb.predict_proba(Xt)[:, 1]
    res["supervisado"], decs["supervisado"], Rs["supervisado"], Rvs["supervisado"] = evaluar(
        "supervisado", p_v, p_t, dv, dt, cfg)

    res["hibrido"], decs["hibrido"], Rs["hibrido"], Rvs["hibrido"] = evaluar(
        "hibrido", sp_v, sp_t, dv, dt, cfg, (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0), p_v, p_t)

    yv, yt = dv["anomalo"].to_numpy(), dt["anomalo"].to_numpy()
    tabla = pd.DataFrame([dict(
        modelo=k, auc_roc=r["auc_roc"], auc_pr=r["auc_pr"], f1_theta1=r["analisis_theta1"]["f1"],
        precision_theta1=r["analisis_theta1"]["precision"], recall_theta1=r["analisis_theta1"]["recall"],
        fpr_theta1=r["analisis_theta1"]["fpr"], f1_theta2=r["analisis_theta2"]["f1"], fpr_theta2=r["analisis_theta2"]["fpr"],
        mcc_theta1=r["analisis_theta1"]["mcc"], f1_con_historial=r["theta1_con_historial"]["f1"],
        f1_primer_acceso=r["theta1_primer_acceso"]["f1"], alpha=r["calibracion"]["alpha"],
        theta1=r["calibracion"]["theta1"], theta2=r["calibracion"]["theta2"]) for k, r in res.items()])
    tabla.to_csv(OUT / "tabla_comparativa_sl.csv", index=False)
    po = puntos_operacion(Rvs["hibrido"], Rs["hibrido"], yv, yt)
    po.to_csv(OUT / "puntos_operacion_sl.csv", index=False)
    aa.confusion_3x2(yt, decs["hibrido"]).to_csv(OUT / "matriz_3x2_hibrido_sl.csv")

    # Importancia por grupo de variables (permutando el grupo completo en el modelo supervisado)
    rng = np.random.default_rng(cfg.seed)
    base = average_precision_score(yt, p_t)
    imp_grupo = {}
    cols = list(X.columns)
    for g in sorted(set(grupos_X.values())):
        idx = [i for i, c in enumerate(cols) if grupos_X[c] == g]
        Xp = Xt.copy()
        perm = rng.permutation(len(Xp))
        Xp[:, idx] = Xp[perm][:, idx]
        imp_grupo[g] = float(base - average_precision_score(yt, hgb.predict_proba(Xp)[:, 1]))
    pi = permutation_importance(hgb, Xt, yt, scoring="average_precision", n_repeats=3, random_state=cfg.seed)
    imp_var = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False)

    lat_if = aa.inference_latency_ms(iso, pd.DataFrame(Xt, columns=cols), features=cols)
    lat_sup = aa.inference_latency_ms(hgb, pd.DataFrame(Xt, columns=cols), features=cols)
    resultados = dict(
        dataset="SuspiciousLogin (Google Workspace, 2 instituciones)", archivo=args.data,
        etiqueta="is_suspicious de Google Workspace (no es ataque confirmado)",
        variables_excluidas_por_fuga=EXCLUIDAS_POR_FUGA,
        particion="train dic-mar | val abr | prueba may-jun (jul-ago excluidos)",
        volumen=dict(total=len(d), usuarios=int(d["actor_pseudo_id"].nunique()), train=int(tr.sum()), val=int(va.sum()),
                     test=int(te.sum()), sospechosos_test=int(yt.sum()),
                     test_con_historial=int(dt["historial_disponible"].sum())),
        modelos=res, puntos_operacion_hibrido=po.round(4).to_dict(orient="records"),
        prueba_pareada_hibrido_vs_reglas=prueba_pareada_por_grupos(dt, decs["reglas_estaticas"], decs["hibrido"], ["month", "day_of_week"]),
        prueba_pareada_hibrido_vs_iforest=prueba_pareada_por_grupos(dt, decs["iforest"], decs["hibrido"], ["month", "day_of_week"]),
        importancia_por_grupo=imp_grupo, importancia_top15=imp_var.head(15).round(4).to_dict(),
        latencia_ms=dict(iforest=lat_if, supervisado=lat_sup, hibrido_p99_aprox=lat_if["p99_ms"] + lat_sup["p99_ms"]),
    )
    aa.to_json(resultados, OUT / "metricas_sl.json")

    c = res["hibrido"]["calibracion"]
    figures.risk_distribution(Rs["hibrido"], yt, c["theta1"], c["theta2"], OUT / "figuras" / "distribucion_riesgo_sl.png")
    figures.model_comparison(tabla, OUT / "figuras" / "comparacion_modelos_sl.png")
    figures.operating_points(po, OUT / "figuras" / "puntos_operacion_sl.png", titulo="AdaptAuth-ML híbrido")
    figures.shap_bars(pd.Series(imp_grupo).clip(lower=0), OUT / "figuras" / "importancia_grupos_sl.png",
                      xlabel="Caída de AUC-PR al permutar el grupo de variables")
    figures.paired_blocks(resultados["prueba_pareada_hibrido_vs_reglas"], OUT / "figuras" / "f1_por_grupo_sl.png",
                          etiquetas=("Reglas estáticas", "AdaptAuth-ML híbrido"))

    print(tabla[["modelo", "auc_roc", "auc_pr", "f1_theta1", "precision_theta1", "recall_theta1", "fpr_theta1",
                 "f1_con_historial", "f1_primer_acceso", "alpha"]].round(3).to_string(index=False))
    print("[puntos de operación, híbrido]"); print(po.round(3).to_string(index=False))
    for k in ("prueba_pareada_hibrido_vs_reglas", "prueba_pareada_hibrido_vs_iforest"):
        p = resultados[k]
        print(f"[{k}] {p['prueba']} ({p['n_grupos']} grupos): est={p['estadistico']:.2f} p={p['p_valor']:.2e} "
              f"dz={p['d_cohen_dz']} | M_A={p['media_baseline']:.3f} M_B={p['media_adaptauth']:.3f}")
    print("[importancia por grupo]", {k: round(v, 4) for k, v in imp_grupo.items()})
    print("[latencia p99 ms] IF", round(lat_if["p99_ms"], 2), "| supervisado", round(lat_sup["p99_ms"], 2))
    print(f"[listo] {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
