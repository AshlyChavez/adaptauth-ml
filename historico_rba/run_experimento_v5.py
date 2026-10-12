"""
Experimento v5 de AdaptAuth-ML: objetivo F1 >= 0.60 en el análisis θ1 (OTP o bloqueo).

Diagnóstico que lo motiva (v4, RBA real):
  - Con 7.6 % de ataques y FPR <= 5 %, F1 = 0.60 exige recall >= 0.5-0.7; v4 lograba 0.15.
  - Las variables de grano grueso (país, ASN) no distinguen legítimos y ataques que comparten la
    misma red. La etiqueta del RBA es por IP (Is Attack IP), así que se agregan señales por IP,
    por user agent y por fallos de la cuenta, todas causales y sin etiquetas.
  - Se quitan variables que crecen con el tiempo (deriva entre validación y prueba).

Modelos (calibrados en validación, evaluados en prueba):
    reglas_estaticas    baseline Alternativa 1 (c_i)
    iforest_v4          diseño v4 (continuidad)
    iforest_v5          Isolation Forest con FEATURES_V5 (no supervisado)
    supervisado_v5      HistGradientBoosting con FEATURES_V5 (aprende patrones de ataque conocidos)
    hibrido_v5          R = alpha * s'(IF v5) + (1 - alpha) * p(supervisado)   <- propuesta C3

Uso:
    python run_experimento_v5.py --data ruta/rba_muestra_usuarios_5pct.parquet
    python run_experimento_v5.py --synthetic
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import adaptauth as aa  # noqa: E402
import figures  # noqa: E402

OUT = Path(__file__).parent / "resultados" / "salida"
F_V4 = aa.FEATURES + aa.FEATURES_GLOBAL
F_V5 = aa.FEATURES_V5
FPR_OBJETIVOS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20)


def f1_max(y, score):
    p, r, _ = precision_recall_curve(y, score)
    f = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)
    return float(f.max())


def evaluar(nombre, s_val, s_test, val, test, cfg, alpha_grid=(1.0,), c_val=None, c_test=None):
    v = val.assign(c_i=0.0 if c_val is None else c_val)
    cal = aa.calibrate(v, s_val, cfg, alpha_grid)
    c = np.zeros(len(test)) if c_test is None else c_test
    R = aa.risk_score(s_test, c, cal["alpha"])
    R_val = aa.risk_score(s_val, v["c_i"].to_numpy(), cal["alpha"])
    dec = aa.decide(R, cal["theta1"], cal["theta2"], test["arranque_frio"].to_numpy(), cfg.cold_start_otp)
    y, yv = test[cfg.label].to_numpy(), val[cfg.label].to_numpy()
    res = dict(calibracion=cal, **aa.evaluate_decisions(y, dec),
               auc_roc=float(roc_auc_score(y, R)), auc_pr=float(average_precision_score(y, R)),
               f1_max_validacion=f1_max(yv, R_val),
               f1_max_oraculo_prueba=f1_max(y, R),   # solo diagnóstico: usa etiquetas de prueba
               score_medio_legitimos=float(R[y == 0].mean()))
    m = res["analisis_theta1"]
    print(f"  {nombre:<18} AUC-ROC={res['auc_roc']:.3f} AUC-PR={res['auc_pr']:.3f} | θ1: F1={m['f1']:.3f} "
          f"P={m['precision']:.3f} R={m['recall']:.3f} FPR={m['fpr']:.3f} | F1 máx val={res['f1_max_validacion']:.3f}"
          f" oráculo={res['f1_max_oraculo_prueba']:.3f}" + ("  [FPR<=5% no alcanzable]" if cal.get("restriccion_fpr_incumplida") else ""))
    return res, dec, R, R_val


def puntos_operacion(R_val, R_test, val, test, label):
    """Umbral fijado en validación para cada FPR objetivo; métricas reales en prueba."""
    yv, yt = val[label].to_numpy(), test[label].to_numpy()
    filas = []
    for fpr in FPR_OBJETIVOS:
        thr = float(np.quantile(R_val[yv == 0], 1 - fpr))
        m = aa.binary_metrics(yt, (R_test > thr).astype(int))
        filas.append(dict(fpr_objetivo=fpr, umbral=thr, fpr_prueba=m["fpr"], recall=m["recall"],
                          precision=m["precision"], f1=m["f1"]))
    return pd.DataFrame(filas)


def diagnostico_etiquetas(d, poblacion):
    """¿Qué tan 'aprendible' es la etiqueta Is Attack IP? (naturaleza por IP, mezcla con legítimos)."""
    from sklearn.metrics import roc_auc_score as auc
    g = d.groupby(aa.COL_IP)
    por_ip = pd.DataFrame({"ataque": g[aa.COL_ATTACK_IP].mean(), "usuarios": g[aa.COL_USER].nunique(),
                           "intentos": g.size(), "fallo": 1 - g[aa.COL_SUCCESS].mean()})
    atk_ip, leg_ip = por_ip[por_ip.ataque > 0], por_ip[por_ip.ataque == 0]
    ok = poblacion[poblacion["anomalo"] == 1]
    usuarios_con_ip_limpia = set(poblacion.loc[poblacion["anomalo"] == 0, aa.COL_USER])
    y = poblacion["anomalo"].to_numpy()
    return {
        "ips_con_etiqueta_mixta": round(float(((por_ip.ataque > 0) & (por_ip.ataque < 1)).mean()), 4),
        "ips_ataque": int(len(atk_ip)), "ips_legitimas": int(len(leg_ip)),
        "usuarios_por_ip_ataque_p50_p90_max": [float(atk_ip.usuarios.quantile(q)) for q in (.5, .9)] + [int(atk_ip.usuarios.max())],
        "usuarios_por_ip_legitima_p50_p90_max": [float(leg_ip.usuarios.quantile(q)) for q in (.5, .9)] + [int(leg_ip.usuarios.max())],
        "tasa_fallo_media_ip_ataque": round(float(atk_ip.fallo.mean()), 4),
        "tasa_fallo_media_ip_legitima": round(float(leg_ip.fallo.mean()), 4),
        "exitos_desde_ip_ataque_de_usuarios_que_tambien_entran_desde_ip_limpia":
            round(float(ok[aa.COL_USER].isin(usuarios_con_ip_limpia).mean()), 4),
        "auc_univariada_v5_poblacion": {c: round(float(auc(y, poblacion[c].astype(float))), 3) for c in F_V5},
        "nota": "AUC 0.5 = no separa; > 0.5 valores altos indican ataque; < 0.5 relación invertida",
    }


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--data")
    g.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no-shap", action="store_true")
    args = ap.parse_args()

    cfg = aa.Config(cold_start_otp=False)
    OUT.mkdir(exist_ok=True); (OUT / "figuras").mkdir(exist_ok=True)
    t0 = time.time()
    raw = aa.generate_synthetic_rba(seed=cfg.seed) if args.synthetic else aa.load_rba(args.data)
    d = aa.build_features(raw)
    contexto_completo = aa.apply_full_context(d)
    print("[contexto] IP/ASN/país/UA calculados con " + ("TODO el tráfico del dataset" if contexto_completo
          else "el tráfico de la muestra (más débil)"))
    pob = d[d[aa.COL_SUCCESS].astype(bool)].reset_index(drop=True)
    train, val, test = aa.temporal_split(pob, cfg.split)
    diag = diagnostico_etiquetas(d, pob)
    aa.to_json(diag, OUT / "diagnostico_etiquetas.json")
    print(f"[datos] {len(d):,} intentos | {len(pob):,} con contraseña correcta | prueba: {len(test):,} "
          f"({test['anomalo'].mean():.1%} ataques) | {time.time() - t0:.0f}s")
    print(f"[etiqueta] IPs mixtas={diag['ips_con_etiqueta_mixta']} | usuarios/IP ataque p50,p90={diag['usuarios_por_ip_ataque_p50_p90_max'][:2]}"
          f" vs legítima {diag['usuarios_por_ip_legitima_p50_p90_max'][:2]} | fallo IP ataque={diag['tasa_fallo_media_ip_ataque']}"
          f" vs legítima {diag['tasa_fallo_media_ip_legitima']}")

    yv_leg = val[cfg.label].to_numpy() == 0
    res, decs, Rs, Rvs = {}, {}, {}, {}
    print("[modelos]")
    res["reglas_estaticas"], decs["reglas_estaticas"], Rs["reglas_estaticas"], _ = evaluar(
        "reglas_estaticas", np.zeros(len(val)), np.zeros(len(test)), val, test, cfg, (0.0,),
        val["c_i"].to_numpy(), test["c_i"].to_numpy())

    ifs = {}
    for nombre, feats in (("iforest_v4", F_V4), ("iforest_v5", F_V5)):
        mdl = aa.train_iforest(train, cfg, feats)
        s_v, s_t = aa.anomaly_score(mdl, val, feats), aa.anomaly_score(mdl, test, feats)
        s_ref = float(np.median(s_v[yv_leg]))
        sp_v, sp_t = aa.rescale(s_v, s_ref), aa.rescale(s_t, s_ref)
        res[nombre], decs[nombre], Rs[nombre], Rvs[nombre] = evaluar(nombre, sp_v, sp_t, val, test, cfg)
        ifs[nombre] = (mdl, sp_v, sp_t)

    hgb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
                                         l2_regularization=1.0, early_stopping=True,
                                         validation_fraction=0.1, random_state=cfg.seed)
    hgb.fit(train[F_V5].to_numpy(), train[cfg.label].to_numpy())
    p_v = hgb.predict_proba(val[F_V5].to_numpy())[:, 1]
    p_t = hgb.predict_proba(test[F_V5].to_numpy())[:, 1]
    res["supervisado_v5"], decs["supervisado_v5"], Rs["supervisado_v5"], Rvs["supervisado_v5"] = evaluar(
        "supervisado_v5", p_v, p_t, val, test, cfg)

    # Híbrido (C3): alpha * s'(IF) + (1 - alpha) * p(supervisado); alpha calibrado en validación
    _, sp_v, sp_t = ifs["iforest_v5"]
    res["hibrido_v5"], decs["hibrido_v5"], Rs["hibrido_v5"], Rvs["hibrido_v5"] = evaluar(
        "hibrido_v5", sp_v, sp_t, val, test, cfg, (0.0, 0.1, 0.25, 0.5, 0.75, 1.0), p_v, p_t)

    tabla = pd.DataFrame([dict(
        modelo=k, auc_roc=r["auc_roc"], auc_pr=r["auc_pr"], f1_theta1=r["analisis_theta1"]["f1"],
        precision_theta1=r["analisis_theta1"]["precision"], recall_theta1=r["analisis_theta1"]["recall"],
        fpr_theta1=r["analisis_theta1"]["fpr"], f1_theta2=r["analisis_theta2"]["f1"],
        fpr_theta2=r["analisis_theta2"]["fpr"], mcc_theta1=r["analisis_theta1"]["mcc"],
        f1_max_validacion=r["f1_max_validacion"], f1_max_oraculo_prueba=r["f1_max_oraculo_prueba"],
        alpha=r["calibracion"]["alpha"], theta1=r["calibracion"]["theta1"], theta2=r["calibracion"]["theta2"],
        fpr_5pct_alcanzable=not r["calibracion"].get("restriccion_fpr_incumplida", False))
        for k, r in res.items()])
    tabla.to_csv(OUT / "tabla_comparativa_v5.csv", index=False)

    mejor = max(("iforest_v5", "supervisado_v5", "hibrido_v5"), key=lambda k: res[k]["f1_max_validacion"])
    po = puntos_operacion(Rvs[mejor], Rs[mejor], val, test, cfg.label)
    po.to_csv(OUT / "puntos_operacion_v5.csv", index=False)

    y = test[cfg.label].to_numpy()
    lat_if = aa.inference_latency_ms(ifs["iforest_v5"][0], test, features=F_V5)
    lat_sup = aa.inference_latency_ms(hgb, test, features=F_V5)
    resultados = dict(
        origen_datos="SINTETICO (solo prueba)" if args.synthetic else f"RBA real: {args.data}",
        poblacion="inicios con contraseña correcta",
        contexto_red="trafico completo" if contexto_completo else "solo muestra",
        volumen=dict(intentos=len(d), poblacion=len(pob), train=len(train), val=len(val), test=len(test),
                     ataques_test=int(y.sum()), ato_test=int(test[aa.COL_ATO].sum())),
        modelos=res, mejor_por_validacion=mejor,
        puntos_operacion=po.round(4).to_dict(orient="records"),
        prueba_pareada_hibrido_vs_reglas=aa.paired_block_test(test, decs["reglas_estaticas"], decs["hibrido_v5"], cfg.label, cfg.n_test_blocks),
        prueba_pareada_hibrido_vs_iforest_v4=aa.paired_block_test(test, decs["iforest_v4"], decs["hibrido_v5"], cfg.label, cfg.n_test_blocks),
        prueba_pareada_hibrido_vs_iforest_v5=aa.paired_block_test(test, decs["iforest_v5"], decs["hibrido_v5"], cfg.label, cfg.n_test_blocks),
        latencia_ms=dict(iforest_v5=lat_if, supervisado_v5=lat_sup,
                         hibrido_p99_aprox=lat_if["p99_ms"] + lat_sup["p99_ms"]),
    )
    aa.confusion_3x2(y, decs["hibrido_v5"]).to_csv(OUT / "matriz_3x2_hibrido_v5.csv")

    rng = np.random.default_rng(cfg.seed)
    sel = rng.choice(len(test), size=min(30000, len(test)), replace=False)
    pi = permutation_importance(hgb, test.iloc[sel][F_V5].to_numpy(), y[sel], scoring="average_precision",
                                n_repeats=3, random_state=cfg.seed)
    imp_sup = pd.Series(pi.importances_mean, index=F_V5).sort_values(ascending=False)
    resultados["importancia_supervisado_v5"] = imp_sup.round(4).to_dict()
    figures.shap_bars(imp_sup.clip(lower=0), OUT / "figuras" / "importancia_supervisado_v5.png",
                      xlabel="Caída de AUC-PR al permutar la variable")
    if not args.no_shap:
        import shap
        ex = shap.TreeExplainer(ifs["iforest_v5"][0])
        imp = []
        for b in np.array_split(np.arange(len(test)), 5):
            s = rng.choice(b, size=min(400, len(b)), replace=False)
            imp.append(np.abs(ex.shap_values(test.iloc[s][F_V5].to_numpy())).mean(axis=0))
        imp = pd.DataFrame(imp, columns=F_V5)
        top = set(imp.mean().sort_values(ascending=False).index[:3])
        coinc = sum(set(r.sort_values(ascending=False).index[:3]) == top for _, r in imp.iterrows())
        resultados["shap_iforest_v5"] = dict(top3=list(imp.mean().sort_values(ascending=False).index[:3]),
                                             bloques_que_coinciden=coinc, cumple=coinc >= 4)
        figures.shap_bars(imp.mean(), OUT / "figuras" / "shap_iforest_v5.png")

    c = res["hibrido_v5"]["calibracion"]
    figures.risk_distribution(Rs["hibrido_v5"], y, c["theta1"], c["theta2"], OUT / "figuras" / "distribucion_riesgo_v5.png")
    figures.paired_blocks(resultados["prueba_pareada_hibrido_vs_iforest_v4"], OUT / "figuras" / "f1_por_bloque_v5.png",
                          etiquetas=("AdaptAuth-ML v4", "AdaptAuth-ML v5 (híbrido)"))
    figures.model_comparison(tabla, OUT / "figuras" / "comparacion_modelos_v5.png")
    figures.operating_points(po, OUT / "figuras" / "puntos_operacion_v5.png", titulo=mejor)
    aa.to_json(resultados, OUT / "metricas_v5.json")
    print(tabla[["modelo", "auc_roc", "auc_pr", "f1_theta1", "recall_theta1", "fpr_theta1",
                 "f1_max_validacion", "alpha"]].round(3).to_string(index=False))
    print(f"[puntos de operación: {mejor}]")
    print(po.round(3).to_string(index=False))
    print(f"[listo] {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
