"""
Ejecuta el experimento base de la Semana 8 de AdaptAuth-ML.

Uso:
    python run_baseline.py --data data/rba-dataset.csv [--nrows 2000000]   # datos reales RBA
    python run_baseline.py --synthetic                                     # datos de prueba

Salidas en results/: metricas.json, tablas .csv y figuras .png.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import adaptauth as aa  # noqa: E402
import figures  # noqa: E402

OUT = Path(__file__).parent / "resultados" / "salida"


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--data", help="ruta al CSV real de RBA")
    g.add_argument("--synthetic", action="store_true", help="usar datos sintéticos de prueba")
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--user-pct", type=float, default=None,
                    help="muestreo por usuario sobre todo el archivo (p. ej. 5 = 5 %% de usuarios)")
    ap.add_argument("--no-shap", action="store_true")
    args = ap.parse_args()

    cfg = aa.Config()
    OUT.mkdir(exist_ok=True); (OUT / "figuras").mkdir(exist_ok=True)
    origen = ("SINTETICO (solo prueba del pipeline)" if args.synthetic else
              f"RBA real: {args.data} | " + (f"muestreo por usuario {args.user_pct} %" if args.user_pct
                                            else f"primeras {args.nrows} filas"))

    t0 = time.time()
    raw = aa.generate_synthetic_rba(seed=cfg.seed) if args.synthetic else aa.load_rba(args.data, args.nrows, args.user_pct)
    d = aa.build_features(raw)
    diag = aa.diagnose(raw, d)
    aa.to_json(diag, OUT / "diagnostico.json")
    print("[EDA] ordenado por fecha:", diag["archivo_ordenado_por_fecha"], "| rango:", diag["rango_fechas"],
          "| usuarios:", f"{diag['usuarios']:,}", "| anomalos:", diag["tasa_anomalos"])
    train, val, test = aa.temporal_split(d, cfg.split)
    print(f"[C1] {len(d):,} eventos | train {len(train):,} | val {len(val):,} | test {len(test):,}"
          f" | {time.time() - t0:.1f}s")
    for name, part in (("train", train), ("val", val), ("test", test)):
        print(f"     {name}: anomalos={int(part['anomalo'].sum()):,} | ATO={int(part[aa.COL_ATO].sum()):,}"
              f" | primeros accesos={int(part['arranque_frio'].sum()):,}")
    if val["anomalo"].sum() == 0 or test["anomalo"].sum() == 0:
        sys.exit("ERROR: validación o prueba no tienen eventos anómalos; aumenta --nrows.")

    # C2: Isolation Forest (no supervisado: no ve etiquetas)
    model = aa.train_iforest(train, cfg)
    s_val, s_test = aa.anomaly_score(model, val), aa.anomaly_score(model, test)

    # Ecuación 1b: s_ref = mediana de s en accesos legítimos de VALIDACIÓN
    s_ref = float(np.median(s_val[val[cfg.label].to_numpy() == 0]))
    sp_val, sp_test = aa.rescale(s_val, s_ref), aa.rescale(s_test, s_ref)

    # Calibración solo en validación
    cal_adapt = aa.calibrate(val, sp_val, cfg, cfg.alpha_grid)          # AdaptAuth-ML
    cal_base = aa.calibrate(val, sp_val, cfg, (0.0,))                   # baseline: solo reglas
    print(f"[C3/C4] s_ref={s_ref:.3f} | AdaptAuth {cal_adapt} | Baseline {cal_base}")

    y = test[cfg.label].to_numpy(); y_ato = test[aa.COL_ATO].to_numpy()
    cold = test["arranque_frio"].to_numpy(); c = test["c_i"].to_numpy()
    R_adapt = aa.risk_score(sp_test, c, cal_adapt["alpha"])
    R_base = aa.risk_score(sp_test, c, 0.0)
    dec_adapt = aa.decide(R_adapt, cal_adapt["theta1"], cal_adapt["theta2"], cold, cfg.cold_start_otp)
    dec_base = aa.decide(R_base, cal_base["theta1"], cal_base["theta2"], cold, cfg.cold_start_otp)
    # Isolation Forest solo (sin reglas ni arranque en frío), como referencia técnica
    cal_if = aa.calibrate(val.assign(c_i=0.0, arranque_frio=0), sp_val,
                          aa.Config(cold_start_otp=False), (1.0,))
    dec_if = aa.decide(sp_test, cal_if["theta1"], cal_if["theta2"], None, False)

    legit = y == 0
    resultados = {
        "origen_datos": origen,
        "advertencia": ("Datos sintéticos: estas cifras validan que el pipeline funciona, "
                        "NO son resultados de la tesis.") if args.synthetic else None,
        "config": cfg,
        "volumen": dict(total=len(d), train=len(train), val=len(val), test=len(test),
                        anomalos_test=int(y.sum()), ato_test=int(y_ato.sum()),
                        arranque_frio_test=int(cold.sum())),
        "s_ref": s_ref,
        "calibracion": dict(adaptauth=cal_adapt, baseline_reglas=cal_base, iforest_solo=cal_if),
        "prueba": {
            "adaptauth": aa.evaluate_decisions(y, dec_adapt),
            "baseline_reglas": aa.evaluate_decisions(y, dec_base),
            "iforest_solo": aa.evaluate_decisions(y, dec_if),
            # Misma evaluación excluyendo primeros accesos (regla P5 de arranque en frío)
            "adaptauth_sin_arranque_frio": aa.evaluate_decisions(y[cold == 0], dec_adapt[cold == 0]),
            "adaptauth_recall_ato":aa.binary_metrics(y_ato, (dec_adapt >= 1).astype(int))["recall"],
            "auc_roc_s": float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y, s_test)),
            "auc_pr_s": float(__import__("sklearn.metrics", fromlist=["average_precision_score"]).average_precision_score(y, s_test)),
            "auc_roc_R": float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y, R_adapt)),
            "score_medio_legitimos_R": float(R_adapt[legit].mean()),       # CA-C3-02
            "R_fuera_de_rango": int(((R_adapt < 0) | (R_adapt > 1)).sum()),  # CA-C3-01
        },
        "prueba_pareada": aa.paired_block_test(test, dec_base, dec_adapt, cfg.label, cfg.n_test_blocks),
        "latencia_inferencia": aa.inference_latency_ms(model, test),
    }

    aa.confusion_3x2(y, dec_adapt).to_csv(OUT / "matriz_3x2_adaptauth.csv")
    aa.confusion_3x2(y, dec_base).to_csv(OUT / "matriz_3x2_baseline.csv")
    pd.Series(dec_adapt).map(aa.DECISION_NAMES).value_counts().to_csv(OUT / "distribucion_decisiones.csv")

    # OE1: importancia de variables con SHAP por bloque temporal (estabilidad del top-3)
    if not args.no_shap:
        import shap
        explainer = shap.TreeExplainer(model)
        blocks = np.array_split(np.arange(len(test)), 5)
        rng = np.random.default_rng(cfg.seed)
        imp = []
        for b in blocks:
            sel = rng.choice(b, size=min(400, len(b)), replace=False)
            sv = explainer.shap_values(test.iloc[sel][aa.FEATURES].to_numpy())
            imp.append(np.abs(sv).mean(axis=0))
        imp = pd.DataFrame(imp, columns=aa.FEATURES, index=[f"bloque_{i+1}" for i in range(5)])
        imp.to_csv(OUT / "shap_por_bloque.csv")
        top3 = [tuple(r.sort_values(ascending=False).index[:3]) for _, r in imp.iterrows()]
        ref = set(imp.mean().sort_values(ascending=False).index[:3])
        coincid = sum(set(t) == ref for t in top3)
        resultados["shap"] = dict(top3_global=list(imp.mean().sort_values(ascending=False).index[:3]),
                                  bloques_que_coinciden=coincid, criterio="top-3 coincide en >= 4 de 5 bloques",
                                  cumple=coincid >= 4, media_abs=imp.mean().to_dict())
        figures.shap_bars(imp.mean(), OUT / "figuras" / "shap_importancia.png")

    figures.risk_distribution(R_adapt, y, cal_adapt["theta1"], cal_adapt["theta2"],
                              OUT / "figuras" / "distribucion_riesgo.png")
    figures.paired_blocks(resultados["prueba_pareada"], OUT / "figuras" / "f1_por_bloque.png")

    aa.to_json(resultados, OUT / "metricas.json")
    m = resultados["prueba"]["adaptauth"]
    print(f"[Prueba] AdaptAuth theta1: F1={m['analisis_theta1']['f1']:.3f} "
          f"FPR={m['analisis_theta1']['fpr']:.3f} | theta2: F1={m['analisis_theta2']['f1']:.3f} "
          f"FPR={m['analisis_theta2']['fpr']:.4f}")
    print(f"[Listo] {time.time() - t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
