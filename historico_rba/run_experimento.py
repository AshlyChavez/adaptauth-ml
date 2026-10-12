"""
Experimento v4 de AdaptAuth-ML: corrección basada en el diagnóstico del RBA real.

Cambios respecto a v3 (run_baseline.py):
  - Población evaluada: inicios de sesión con contraseña correcta (punto donde actúa la
    autenticación basada en riesgo). Los intentos fallidos se usan solo como contexto.
  - Variables de contexto global sin etiquetas (FEATURES_GLOBAL): tipo de dispositivo y tasa
    previa de fallos / volumen del ASN y del país.
  - Regla P5 (OTP a todo primer acceso) desactivada por defecto: con mediana de 2 accesos por
    usuario generaba por sí sola la mayor parte del FPR. Se reporta también con P5 activa.
  - Comparación en una sola corrida:
        reglas_estaticas        baseline de la Alternativa 1 (solo c_i)
        iforest_usuario         diseño v3 (x_i en R^8)
        iforest_usuario_global  x_i + contexto global
        adaptauth_v4            alpha * s'(IF usuario+global) + (1 - alpha) * c_i   <- propuesta
        referencia_supervisada  HistGradientBoosting con etiquetas (cota superior, no es la propuesta)

Uso:
    python run_experimento.py --data ruta/rba_muestra_usuarios_5pct.parquet
    python run_experimento.py --synthetic
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import adaptauth as aa  # noqa: E402
import figures  # noqa: E402

OUT = Path(__file__).parent / "resultados" / "salida"
F_USR = aa.FEATURES
F_ALL = aa.FEATURES + aa.FEATURES_GLOBAL


def evaluar(nombre, s_val, s_test, val, test, cfg, alpha_grid=(1.0,), use_c=False):
    """Calibra en validación (máx. F1 con FPR <= 5 %) y evalúa en prueba."""
    v = val if use_c else val.assign(c_i=0.0)
    cal = aa.calibrate(v, s_val, cfg, alpha_grid)
    c = test["c_i"].to_numpy() if use_c else np.zeros(len(test))
    R = aa.risk_score(s_test, c, cal["alpha"])
    dec = aa.decide(R, cal["theta1"], cal["theta2"], test["arranque_frio"].to_numpy(), cfg.cold_start_otp)
    y = test[cfg.label].to_numpy()
    res = dict(calibracion=cal, **aa.evaluate_decisions(y, dec),
               auc_roc=float(roc_auc_score(y, R)), auc_pr=float(average_precision_score(y, R)),
               score_medio_legitimos=float(R[y == 0].mean()))
    print(f"  {nombre:<24} AUC-ROC={res['auc_roc']:.3f} AUC-PR={res['auc_pr']:.3f} | "
          f"θ1: F1={res['analisis_theta1']['f1']:.3f} FPR={res['analisis_theta1']['fpr']:.3f} | "
          f"θ2: F1={res['analisis_theta2']['f1']:.3f} FPR={res['analisis_theta2']['fpr']:.4f}"
          + ("  [FPR<=5% no alcanzable]" if cal.get("restriccion_fpr_incumplida") else ""))
    return res, dec, R


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--data")
    g.add_argument("--synthetic", action="store_true")
    ap.add_argument("--user-pct", type=float, default=None)
    ap.add_argument("--todos-los-intentos", action="store_true",
                    help="evaluar también intentos con contraseña incorrecta")
    ap.add_argument("--no-shap", action="store_true")
    args = ap.parse_args()

    cfg = aa.Config(cold_start_otp=False)
    OUT.mkdir(exist_ok=True); (OUT / "figuras").mkdir(exist_ok=True)
    t0 = time.time()
    raw = aa.generate_synthetic_rba(seed=cfg.seed) if args.synthetic else aa.load_rba(args.data, user_pct=args.user_pct)
    d = aa.build_features(raw)                      # el contexto se calcula con TODOS los intentos
    diag = aa.diagnose(raw, d)
    aa.to_json(diag, OUT / "diagnostico.json")
    poblacion = d if args.todos_los_intentos else d[d[aa.COL_SUCCESS].astype(bool)].reset_index(drop=True)
    train, val, test = aa.temporal_split(poblacion, cfg.split)
    print(f"[datos] {len(d):,} intentos | población evaluada {len(poblacion):,} "
          f"({'todos' if args.todos_los_intentos else 'contraseña correcta'}) | "
          f"anómalos en prueba {int(test['anomalo'].sum()):,} ({test['anomalo'].mean():.1%}) | {time.time()-t0:.0f}s")
    if val["anomalo"].sum() == 0 or test["anomalo"].sum() == 0:
        sys.exit("ERROR: validación o prueba sin eventos anómalos.")

    y_val = val[cfg.label].to_numpy() == 0
    res, decs, Rs = {}, {}, {}
    print("[modelos] calibrados en validación, evaluados en prueba:")

    # B0. Reglas estáticas (Alternativa 1 del PI6)
    res["reglas_estaticas"], decs["reglas_estaticas"], Rs["reglas_estaticas"] = evaluar(
        "reglas_estaticas", np.zeros(len(val)), np.zeros(len(test)), val, test, cfg, (0.0,), use_c=True)

    # M1 / M2. Isolation Forest con y sin contexto global
    modelos = {}
    for nombre, feats in (("iforest_usuario", F_USR), ("iforest_usuario_global", F_ALL)):
        m = aa.train_iforest(train, cfg, feats)
        s_v, s_t = aa.anomaly_score(m, val, feats), aa.anomaly_score(m, test, feats)
        s_ref = float(np.median(s_v[y_val]))
        sp_v, sp_t = aa.rescale(s_v, s_ref), aa.rescale(s_t, s_ref)
        res[nombre], decs[nombre], Rs[nombre] = evaluar(nombre, sp_v, sp_t, val, test, cfg)
        res[nombre]["s_ref"] = s_ref
        modelos[nombre] = (m, feats, sp_v, sp_t)

    # M3. AdaptAuth-ML v4 (propuesta): IF usuario+global combinado con c_i
    _, _, sp_v, sp_t = modelos["iforest_usuario_global"]
    res["adaptauth_v4"], decs["adaptauth_v4"], Rs["adaptauth_v4"] = evaluar(
        "adaptauth_v4", sp_v, sp_t, val, test, cfg, cfg.alpha_grid, use_c=True)
    cfg_p5 = aa.Config(cold_start_otp=True)
    res["adaptauth_v4_con_P5"], _, _ = evaluar(
        "adaptauth_v4_con_P5", sp_v, sp_t, val, test, cfg_p5, cfg.alpha_grid, use_c=True)

    # R1. Referencia supervisada (cota superior con las mismas variables)
    hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=cfg.seed)
    hgb.fit(train[F_ALL].to_numpy(), train[cfg.label].to_numpy())
    p_v = hgb.predict_proba(val[F_ALL].to_numpy())[:, 1]
    p_t = hgb.predict_proba(test[F_ALL].to_numpy())[:, 1]
    res["referencia_supervisada"], decs["referencia_supervisada"], Rs["referencia_supervisada"] = evaluar(
        "referencia_supervisada", p_v, p_t, val, test, cfg)

    # Tabla comparativa
    filas = []
    for k, r in res.items():
        filas.append(dict(modelo=k, auc_roc=r["auc_roc"], auc_pr=r["auc_pr"],
                          f1_theta1=r["analisis_theta1"]["f1"], fpr_theta1=r["analisis_theta1"]["fpr"],
                          recall_theta1=r["analisis_theta1"]["recall"], precision_theta1=r["analisis_theta1"]["precision"],
                          f1_theta2=r["analisis_theta2"]["f1"], fpr_theta2=r["analisis_theta2"]["fpr"],
                          mcc_theta1=r["analisis_theta1"]["mcc"], alpha=r["calibracion"]["alpha"],
                          theta1=r["calibracion"]["theta1"], theta2=r["calibracion"]["theta2"],
                          fpr_5pct_alcanzable=not r["calibracion"].get("restriccion_fpr_incumplida", False)))
    tabla = pd.DataFrame(filas)
    tabla.to_csv(OUT / "tabla_comparativa.csv", index=False)

    y = test[cfg.label].to_numpy()
    resultados = dict(
        origen_datos="SINTETICO (solo prueba)" if args.synthetic else f"RBA real: {args.data}",
        poblacion="todos los intentos" if args.todos_los_intentos else "inicios con contraseña correcta",
        volumen=dict(intentos_totales=len(d), poblacion=len(poblacion), train=len(train), val=len(val),
                     test=len(test), anomalos_test=int(y.sum()), ato_test=int(test[aa.COL_ATO].sum()),
                     primer_acceso_test=int(test["arranque_frio"].sum())),
        modelos=res,
        prueba_pareada_vs_reglas=aa.paired_block_test(test, decs["reglas_estaticas"], decs["adaptauth_v4"],
                                                      cfg.label, cfg.n_test_blocks),
        prueba_pareada_vs_v3=aa.paired_block_test(test, decs["iforest_usuario"], decs["adaptauth_v4"],
                                                  cfg.label, cfg.n_test_blocks),
        latencia_inferencia_iforest=aa.inference_latency_ms(modelos["iforest_usuario_global"][0], test,
                                                            features=F_ALL),
    )
    aa.confusion_3x2(y, decs["adaptauth_v4"]).to_csv(OUT / "matriz_3x2_adaptauth_v4.csv")

    if not args.no_shap:
        import shap
        m, feats = modelos["iforest_usuario_global"][:2]
        ex = shap.TreeExplainer(m)
        rng = np.random.default_rng(cfg.seed)
        imp = []
        for b in np.array_split(np.arange(len(test)), 5):
            sel = rng.choice(b, size=min(400, len(b)), replace=False)
            imp.append(np.abs(ex.shap_values(test.iloc[sel][feats].to_numpy())).mean(axis=0))
        imp = pd.DataFrame(imp, columns=feats)
        top_ref = set(imp.mean().sort_values(ascending=False).index[:3])
        coinc = sum(set(r.sort_values(ascending=False).index[:3]) == top_ref for _, r in imp.iterrows())
        resultados["shap_iforest_usuario_global"] = dict(
            top3=list(imp.mean().sort_values(ascending=False).index[:3]), bloques_que_coinciden=coinc,
            cumple=coinc >= 4, media_abs=imp.mean().round(4).to_dict())
        figures.shap_bars(imp.mean(), OUT / "figuras" / "shap_importancia.png")
        # importancia en la referencia supervisada (qué variables separan de verdad los ataques)
        from sklearn.inspection import permutation_importance
        sel = rng.choice(len(test), size=min(20000, len(test)), replace=False)
        pi = permutation_importance(hgb, test.iloc[sel][F_ALL].to_numpy(), y[sel], scoring="average_precision",
                                    n_repeats=3, random_state=cfg.seed)
        resultados["importancia_referencia_supervisada"] = dict(
            sorted(zip(F_ALL, np.round(pi.importances_mean, 4)), key=lambda kv: -kv[1]))

    c = res["adaptauth_v4"]["calibracion"]
    figures.risk_distribution(Rs["adaptauth_v4"], y, c["theta1"], c["theta2"], OUT / "figuras" / "distribucion_riesgo.png")
    figures.paired_blocks(resultados["prueba_pareada_vs_reglas"], OUT / "figuras" / "f1_por_bloque.png")
    figures.model_comparison(tabla, OUT / "figuras" / "comparacion_modelos.png")
    aa.to_json(resultados, OUT / "metricas_v4.json")
    print(tabla[["modelo", "auc_roc", "auc_pr", "f1_theta1", "fpr_theta1", "fpr_5pct_alcanzable"]].round(3).to_string(index=False))
    print(f"[listo] {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
