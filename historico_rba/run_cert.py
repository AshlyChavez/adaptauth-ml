"""
AdaptAuth-ML sobre CERT Insider Threat r4.2: componente de historial del empleado (usuario-día).

Uso:
    python run_cert.py --dir carpeta_con_logon_LDAP_y_answers
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import adaptauth as aa  # noqa: E402
import cert  # noqa: E402
import figures  # noqa: E402

OUT = Path(__file__).parent / "resultados" / "cert_v1"
F = cert.FEATURES_CERT


def evaluar(nombre, s_val, s_test, dv, dt, cfg, alpha_grid=(1.0,), c_val=None, c_test=None):
    v = dv.assign(c_i=0.0 if c_val is None else c_val)
    cal = aa.calibrate(v, s_val, cfg, alpha_grid)
    R = aa.risk_score(s_test, np.zeros(len(dt)) if c_test is None else c_test, cal["alpha"])
    R_val = aa.risk_score(s_val, v["c_i"].to_numpy(), cal["alpha"])
    dec = aa.decide(R, cal["theta1"], cal["theta2"], None, False)
    y = dt["anomalo"].to_numpy()
    res = dict(calibracion=cal, **aa.evaluate_decisions(y, dec),
               auc_roc=float(roc_auc_score(y, R)) if 0 < y.sum() < len(y) else None,
               auc_pr=float(average_precision_score(y, R)) if y.sum() > 0 else None,
               recall_dias_cualquier_actividad_maliciosa=aa.binary_metrics(dt["anomalo_cualquier"].to_numpy(),
                                                                          (dec >= 1).astype(int))["recall"])
    m = res["analisis_theta1"]
    print(f"  {nombre:<17} AUC-ROC={res['auc_roc'] or 0:.3f} AUC-PR={res['auc_pr'] or 0:.3f} | θ1: F1={m['f1']:.3f} "
          f"P={m['precision']:.3f} R={m['recall']:.3f} FPR={m['fpr']:.3%} | α={cal['alpha']}"
          + ("  [FPR<=5% no alcanzable]" if cal.get("restriccion_fpr_incumplida") else ""))
    return res, dec, R, R_val


def prueba_pareada(dt, dec_a, dec_b, n=10):
    """F1 (θ1) por grupos de empleados de prueba (mismos grupos en ambas condiciones)."""
    y = dt["anomalo"].to_numpy()
    usuarios = np.array(sorted(dt["user"].unique()))
    rng = np.random.default_rng(7); rng.shuffle(usuarios)
    grupo = dt["user"].map({u: i % n for i, u in enumerate(usuarios)}).to_numpy()
    fa, fb = [], []
    for k in range(n):
        idx = grupo == k
        if y[idx].sum() == 0:
            continue
        fa.append(aa.binary_metrics(y[idx], (dec_a[idx] >= 1).astype(int))["f1"])
        fb.append(aa.binary_metrics(y[idx], (dec_b[idx] >= 1).astype(int))["f1"])
    fa, fb = np.array(fa), np.array(fb)
    if len(fa) < 3:
        return dict(nota="muy pocos grupos con casos maliciosos para una prueba pareada", n_grupos=len(fa))
    d = fb - fa
    sh = float(stats.shapiro(d).pvalue) if np.ptp(d) > 0 else 1.0
    t = stats.ttest_rel(fb, fa) if sh > 0.05 else stats.wilcoxon(fb, fa)
    return dict(n_grupos=len(fa), prueba="t de Student pareada" if sh > 0.05 else "Wilcoxon de rangos con signo",
                media_baseline=float(fa.mean()), media_adaptauth=float(fb.mean()), shapiro_p=sh,
                estadistico=float(t.statistic), p_valor=float(t.pvalue),
                d_cohen_dz=float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else None,
                f1_baseline=fa.tolist(), f1_adaptauth=fb.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="carpeta con logon.csv, LDAP/ y answers/")
    args = ap.parse_args()
    cfg = aa.Config(cold_start_otp=False, label="anomalo")
    OUT.mkdir(exist_ok=True); (OUT / "figuras").mkdir(exist_ok=True)
    t0 = time.time()

    logon_path = cert.buscar(args.dir, "logon.csv")[0]
    lg = cert.cargar_logon(logon_path)
    ev, ins = cert.cargar_respuestas(args.dir, "4.2")
    ldap_dirs = [p for p in cert.buscar(args.dir, "LDAP") if Path(p).is_dir()]
    ld = cert.cargar_ldap(ldap_dirs[0]) if ldap_dirs else pd.DataFrame(columns=["user"])
    corte = lg["date"].min() + pd.DateOffset(months=6)
    a = cert.construir_usuario_dia(lg, ev, corte)
    a = a.merge(ld, on="user", how="left")
    print(f"[CERT r4.2] {len(lg):,} eventos de sesión | {lg['user'].nunique():,} empleados | {len(ins)} insiders | "
          f"{len(ev):,} eventos maliciosos ({(ev['tipo'] == 'logon').sum():,} de inicio de sesión)")
    print(f"[usuario-día] {len(a):,} filas | días con inicio de sesión malicioso: {int(a['anomalo'].sum())} | "
          f"días con cualquier actividad maliciosa: {int(a['anomalo_cualquier'].sum())} | {time.time() - t0:.0f}s")
    if a["anomalo"].sum() < 10:
        print("AVISO: muy pocos días con inicio de sesión malicioso; revisa el formato de answers.")

    tr, va, te = cert.particion_por_usuario(a, cfg.seed)
    print(f"[partición por empleado] train {tr['user'].nunique()} ({int(tr.anomalo.sum())} días mal.) | "
          f"val {va['user'].nunique()} ({int(va.anomalo.sum())}) | prueba {te['user'].nunique()} ({int(te.anomalo.sum())})")

    res, decs, Rs, Rvs = {}, {}, {}, {}
    res["reglas_estaticas"], decs["reglas_estaticas"], Rs["reglas_estaticas"], Rvs["reglas_estaticas"] = evaluar(
        "reglas_estaticas", np.zeros(len(va)), np.zeros(len(te)), va, te, cfg, (0.0,), va["c_i"].to_numpy(), te["c_i"].to_numpy())

    iso = IsolationForest(n_estimators=200, max_samples=256, random_state=cfg.seed, n_jobs=-1).fit(tr[F].to_numpy())
    s_v, s_t = -iso.score_samples(va[F].to_numpy()), -iso.score_samples(te[F].to_numpy())
    s_ref = float(np.median(s_v[va["anomalo"].to_numpy() == 0]))
    sp_v, sp_t = aa.rescale(s_v, s_ref), aa.rescale(s_t, s_ref)
    res["iforest"], decs["iforest"], Rs["iforest"], Rvs["iforest"] = evaluar("iforest", sp_v, sp_t, va, te, cfg)

    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                         random_state=cfg.seed).fit(tr[F].to_numpy(), tr["anomalo"].to_numpy())
    p_v, p_t = hgb.predict_proba(va[F].to_numpy())[:, 1], hgb.predict_proba(te[F].to_numpy())[:, 1]
    res["supervisado"], decs["supervisado"], Rs["supervisado"], Rvs["supervisado"] = evaluar("supervisado", p_v, p_t, va, te, cfg)
    res["hibrido"], decs["hibrido"], Rs["hibrido"], Rvs["hibrido"] = evaluar(
        "hibrido", sp_v, sp_t, va, te, cfg, (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0), p_v, p_t)

    tabla = pd.DataFrame([dict(modelo=k, auc_roc=r["auc_roc"], auc_pr=r["auc_pr"], f1_theta1=r["analisis_theta1"]["f1"],
                               precision_theta1=r["analisis_theta1"]["precision"], recall_theta1=r["analisis_theta1"]["recall"],
                               fpr_theta1=r["analisis_theta1"]["fpr"], f1_theta2=r["analisis_theta2"]["f1"],
                               fpr_theta2=r["analisis_theta2"]["fpr"], alpha=r["calibracion"]["alpha"],
                               theta1=r["calibracion"]["theta1"], theta2=r["calibracion"]["theta2"]) for k, r in res.items()])
    tabla.to_csv(OUT / "tabla_comparativa_cert.csv", index=False)

    # Evaluación por unidad funcional ("PYME simulada": unidades de decenas a cientos de empleados)
    por_unidad = []
    col_u = "functional_unit" if "functional_unit" in te.columns else None
    if col_u:
        for u, idx in te.groupby(col_u).indices.items():
            y = te["anomalo"].to_numpy()[idx]
            m = aa.binary_metrics(y, (decs["hibrido"][idx] >= 1).astype(int))
            por_unidad.append(dict(unidad=u, empleados=int(te.iloc[idx]["user"].nunique()), dias=len(idx),
                                   dias_maliciosos=int(y.sum()), f1=m["f1"], recall=m["recall"], fpr=m["fpr"]))
    por_unidad = pd.DataFrame(por_unidad)
    if len(por_unidad):
        por_unidad.to_csv(OUT / "resultados_por_unidad_cert.csv", index=False)

    imp = {}
    rng = np.random.default_rng(cfg.seed)
    yt = te["anomalo"].to_numpy()
    if yt.sum() > 0:
        base = average_precision_score(yt, p_t)
        for c in F:
            Xp = te[F].to_numpy().copy(); j = F.index(c); Xp[:, j] = rng.permutation(Xp[:, j])
            imp[c] = round(float(base - average_precision_score(yt, hgb.predict_proba(Xp)[:, 1])), 4)
    resultados = dict(
        dataset="CERT Insider Threat r4.2 (CMU)", unidad="usuario-día (solo inicios/cierres de sesión)",
        etiqueta="día con al menos un inicio de sesión listado como malicioso en answers",
        particion="por empleado, estratificada (70/15/15)",
        volumen=dict(eventos=len(lg), empleados=int(lg["user"].nunique()), insiders=int(len(ins)),
                     usuario_dias=len(a), dias_maliciosos=int(a["anomalo"].sum()),
                     dias_cualquier_malicioso=int(a["anomalo_cualquier"].sum())),
        modelos=res, por_unidad=por_unidad.round(4).to_dict(orient="records"),
        prueba_pareada_hibrido_vs_reglas=prueba_pareada(te, decs["reglas_estaticas"], decs["hibrido"]),
        prueba_pareada_hibrido_vs_iforest=prueba_pareada(te, decs["iforest"], decs["hibrido"]),
        importancia_supervisado=dict(sorted(imp.items(), key=lambda kv: -kv[1])),
        latencia_ms=dict(iforest=aa.inference_latency_ms(iso, te, features=F),
                         supervisado=aa.inference_latency_ms(hgb, te, features=F)),
    )
    aa.to_json(resultados, OUT / "metricas_cert.json")
    figures.model_comparison(tabla.fillna(0), OUT / "figuras" / "comparacion_modelos_cert.png")
    if imp:
        figures.shap_bars(pd.Series(imp).clip(lower=0), OUT / "figuras" / "importancia_cert.png",
                          xlabel="Caída de AUC-PR al permutar la variable")
    print(tabla.round(3).to_string(index=False))
    if len(por_unidad):
        print("[por unidad funcional]"); print(por_unidad.round(3).to_string(index=False))
    for k in ("prueba_pareada_hibrido_vs_reglas", "prueba_pareada_hibrido_vs_iforest"):
        print(f"[{k}]", {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in resultados[k].items()
                         if kk not in ("f1_baseline", "f1_adaptauth")})
    print(f"[listo] {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
