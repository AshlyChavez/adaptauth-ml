"""
AdaptAuth-ML sobre CERT r4.2, versión 2.

Problema de la v1: solo 25 días maliciosos en prueba y 15 en validación, así que el umbral y el F1
eran muy inestables (cada caso movía el recall 4 puntos) y el umbral de Isolation Forest quedó tan alto
que no detectó nada pese a su AUC de 0.94.

Cambios:
  - Validación cruzada de 5 folds POR EMPLEADO (estratificada): cada fold es prueba una vez, otro fold es
    validación y los 3 restantes entrenamiento. Así se evalúan los 115 días maliciosos, no solo 25.
  - Más variables de hábito del empleado (jornada, desbalance de sesiones, desviaciones respecto de su
    propio historial, proporción previa de días fuera de horario / en PC ajena).
  - Híbrido apilado: el score de Isolation Forest entra como variable del modelo supervisado, para
    aprovechar que IF ordena bien (AUC 0.94) aunque por sí solo no fije un buen umbral.

Versión 11 (levantamiento de observaciones, semana 10):
  - AdaptAuth-ML = clasificador supervisado sin Isolation Forest (IF se retiró del núcleo: ablación y
    arranque en frío sin aporte). El apilado con IF se conserva solo como comparación.
  - θ2 de bloqueo calibrado con FPR ≤ 0.1 % (Config.max_fpr_theta2).
  - --frio: arranque en frío por empleado (fracción de empleados de entrenamiento con etiqueta).
  - Resultados en results_cert_v11/.

Uso:
    python run_cert_v2.py --dir carpeta_con_logon_LDAP_y_answers [--frio]
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

sys.path.insert(0, str(Path(__file__).parent / "src"))
import adaptauth as aa  # noqa: E402
import cert  # noqa: E402
import figures  # noqa: E402

OUT = Path(__file__).parent / "results_cert_v11"
MEJOR = "supervisado"   # AdaptAuth-ML v11, definido antes de ver la prueba
FRIO_FRACCIONES = (0.2, 0.5)
FRIO_SEMILLAS = (1, 2)
F = cert.FEATURES_CERT_V2
K = 5
MODELOS = ("reglas_estaticas", "iforest", "supervisado", "hibrido", "hibrido_apilado")
FPR_OBJ = (0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01)


def hgb():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, class_weight="balanced", random_state=42)


def ajustar_fold(tr, va, te, cfg):
    """Entrena en tr, calibra en va y devuelve scores y decisiones para te (y va) de cada modelo."""
    out = {}
    iso = IsolationForest(n_estimators=200, max_samples=256, random_state=cfg.seed, n_jobs=-1).fit(tr[F].to_numpy())
    s = {n: -iso.score_samples(d[F].to_numpy()) for n, d in (("tr", tr), ("va", va), ("te", te))}
    s_ref = float(np.median(s["va"][va["anomalo"].to_numpy() == 0]))
    sp = {n: aa.rescale(v, s_ref) for n, v in s.items()}

    m_sup = hgb().fit(tr[F].to_numpy(), tr["anomalo"].to_numpy())
    p = {n: m_sup.predict_proba(d[F].to_numpy())[:, 1] for n, d in (("va", va), ("te", te))}

    Fa = F + ["score_if"]
    tr2, va2, te2 = tr.assign(score_if=sp["tr"]), va.assign(score_if=sp["va"]), te.assign(score_if=sp["te"])
    m_ap = hgb().fit(tr2[Fa].to_numpy(), tr2["anomalo"].to_numpy())
    q = {n: m_ap.predict_proba(d[Fa].to_numpy())[:, 1] for n, d in (("va", va2), ("te", te2))}

    defs = {
        "reglas_estaticas": (np.zeros(len(va)), np.zeros(len(te)), (0.0,), va["c_i"].to_numpy(), te["c_i"].to_numpy()),
        "iforest": (sp["va"], sp["te"], (1.0,), None, None),
        "supervisado": (p["va"], p["te"], (1.0,), None, None),
        "hibrido": (sp["va"], sp["te"], (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0), p["va"], p["te"]),
        "hibrido_apilado": (q["va"], q["te"], (1.0,), None, None),
    }
    for nombre, (sv, st, ag, cv, ct) in defs.items():
        v = va.assign(c_i=0.0 if cv is None else cv)
        cal = aa.calibrate(v, sv, cfg, ag)
        R = aa.risk_score(st, np.zeros(len(te)) if ct is None else ct, cal["alpha"])
        Rv = aa.risk_score(sv, v["c_i"].to_numpy(), cal["alpha"])
        dec = aa.decide(R, cal["theta1"], cal["theta2"], None, False)
        out[nombre] = dict(R=R, Rv=Rv, dec=dec, cal=cal)
    return out, m_sup, sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--frio", action="store_true")
    args = ap.parse_args()
    cfg = aa.Config(cold_start_otp=False, label="anomalo")
    OUT.mkdir(exist_ok=True); (OUT / "figuras").mkdir(exist_ok=True)
    t0 = time.time()

    lg = cert.cargar_logon(cert.buscar(args.dir, "logon.csv")[0])
    ev, ins = cert.cargar_respuestas(args.dir, "4.2")
    ldap_dirs = [p for p in cert.buscar(args.dir, "LDAP") if Path(p).is_dir()]
    ld = cert.cargar_ldap(ldap_dirs[0]) if ldap_dirs else pd.DataFrame(columns=["user"])
    a = cert.construir_usuario_dia(lg, ev, lg["date"].min() + pd.DateOffset(months=6))
    a = cert.agregar_variables_v2(a).merge(ld, on="user", how="left")
    a["fold"] = cert.folds_por_empleado(a, K, cfg.seed).to_numpy()
    print(f"[CERT r4.2 v2] {len(a):,} días-empleado | {int(a.anomalo.sum())} días maliciosos | {len(F)} variables | "
          f"días maliciosos por fold: {a.groupby('fold').anomalo.sum().astype(int).tolist()} | {time.time()-t0:.0f}s")

    pooled = {m: dict(R=np.zeros(len(a)), dec=np.zeros(len(a), dtype=int)) for m in MODELOS}
    por_fold = {m: [] for m in MODELOS}
    puntos = {f: dict(tp=0, fp=0, fn=0, tn=0) for f in FPR_OBJ}
    imp_acum = []
    frio = []
    for k in range(K):
        kv = (k + 1) % K
        te_m, va_m = a["fold"] == k, a["fold"] == kv
        tr_m = ~(te_m | va_m)
        tr, va, te = a[tr_m].reset_index(drop=True), a[va_m].reset_index(drop=True), a[te_m].reset_index(drop=True)
        out, m_sup, sp = ajustar_fold(tr, va, te, cfg)
        idx = np.where(te_m.to_numpy())[0]
        y = te["anomalo"].to_numpy()
        for mname, o in out.items():
            pooled[mname]["R"][idx] = o["R"]; pooled[mname]["dec"][idx] = o["dec"]
            por_fold[mname].append(aa.binary_metrics(y, (o["dec"] >= 1).astype(int))["f1"])
        # puntos de operación de AdaptAuth-ML v11: umbral por FPR objetivo en validación
        o = out[MEJOR]; yv = va["anomalo"].to_numpy()
        for f in FPR_OBJ:
            thr = float(np.quantile(o["Rv"][yv == 0], 1 - f))
            m = aa.binary_metrics(y, (o["R"] > thr).astype(int))
            for c in ("tp", "fp", "fn", "tn"):
                puntos[f][c] += m[c]
        if y.sum() > 0:
            rng = np.random.default_rng(k)
            Fa = list(F)
            X = te[Fa].to_numpy()
            base = average_precision_score(y, m_sup.predict_proba(X)[:, 1])
            fila = {}
            for j, c in enumerate(Fa):
                Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
                fila[c] = base - average_precision_score(y, m_sup.predict_proba(Xp)[:, 1])
            imp_acum.append(fila)
        if args.frio:
            Fa = F + ["score_if"]
            emps = tr["user"].unique()
            for fr in FRIO_FRACCIONES:
                for sd in FRIO_SEMILLAS:
                    sel = np.random.default_rng(100 * k + sd).choice(emps, size=max(1, int(fr * len(emps))), replace=False)
                    mk = tr["user"].isin(sel).to_numpy()
                    if tr["anomalo"].to_numpy()[mk].sum() == 0 or y.sum() == 0:
                        continue
                    ms = hgb().fit(tr[F].to_numpy()[mk], tr["anomalo"].to_numpy()[mk])
                    ma = hgb().fit(tr.assign(score_if=sp["tr"])[Fa].to_numpy()[mk], tr["anomalo"].to_numpy()[mk])
                    frio.append(dict(fold=k, fraccion=fr, semilla=sd,
                                     auc_pr_adaptauth=float(average_precision_score(y, ms.predict_proba(te[F].to_numpy())[:, 1])),
                                     auc_pr_apilado=float(average_precision_score(y, ma.predict_proba(te.assign(score_if=sp["te"])[Fa].to_numpy())[:, 1])),
                                     auc_pr_iforest=float(average_precision_score(y, sp["te"]))))
        print(f"  fold {k + 1}/{K}: prueba {te['user'].nunique()} empleados, {int(y.sum())} días maliciosos | "
              + " | ".join(f"{m}: F1={por_fold[m][-1]:.2f}" for m in MODELOS))

    y = a["anomalo"].to_numpy()
    res, filas = {}, []
    for mname in MODELOS:
        ev_ = aa.evaluate_decisions(y, pooled[mname]["dec"])
        r = dict(**ev_, auc_roc=float(roc_auc_score(y, pooled[mname]["R"])),
                 auc_pr=float(average_precision_score(y, pooled[mname]["R"])),
                 f1_por_fold=por_fold[mname], f1_media_folds=float(np.mean(por_fold[mname])),
                 f1_de_folds=float(np.std(por_fold[mname], ddof=1)),
                 recall_dias_cualquier_actividad_maliciosa=aa.binary_metrics(a["anomalo_cualquier"].to_numpy(),
                                                                            (pooled[mname]["dec"] >= 1).astype(int))["recall"])
        res[mname] = r
        m1 = r["analisis_theta1"]
        filas.append(dict(modelo=mname, auc_roc=r["auc_roc"], auc_pr=r["auc_pr"], f1_theta1=m1["f1"],
                          precision_theta1=m1["precision"], recall_theta1=m1["recall"], fpr_theta1=m1["fpr"],
                          tp=m1["tp"], fp=m1["fp"], f1_theta2=r["analisis_theta2"]["f1"],
                          f1_media_folds=r["f1_media_folds"], f1_de_folds=r["f1_de_folds"]))
    tabla = pd.DataFrame(filas)
    tabla.to_csv(OUT / "tabla_comparativa_cert_v2.csv", index=False)

    po = []
    for f, c in puntos.items():
        P = c["tp"] / max(c["tp"] + c["fp"], 1); Rr = c["tp"] / max(c["tp"] + c["fn"], 1)
        po.append(dict(fpr_objetivo=f, fpr_prueba=c["fp"] / max(c["fp"] + c["tn"], 1), recall=Rr, precision=P,
                       f1=2 * P * Rr / max(P + Rr, 1e-12), alertas_falsas=c["fp"], dias_detectados=c["tp"]))
    po = pd.DataFrame(po); po.to_csv(OUT / "puntos_operacion_cert_v2.csv", index=False)

    mejor = MEJOR
    pu = []
    if "functional_unit" in a.columns:
        for u, idx in a.groupby("functional_unit").indices.items():
            yy = y[idx]
            m = aa.binary_metrics(yy, (pooled[mejor]["dec"][idx] >= 1).astype(int))
            pu.append(dict(unidad=u, empleados=int(a.iloc[idx]["user"].nunique()), dias=len(idx),
                           dias_maliciosos=int(yy.sum()), f1=m["f1"], precision=m["precision"], recall=m["recall"], fpr=m["fpr"]))
    pu = pd.DataFrame(pu)
    if len(pu):
        pu.to_csv(OUT / "por_unidad_cert_v2.csv", index=False)

    def pareada(ma, mb, ngr=10):
        usuarios = np.array(sorted(a["user"].unique())); rng = np.random.default_rng(7); rng.shuffle(usuarios)
        g = a["user"].map({u: i % ngr for i, u in enumerate(usuarios)}).to_numpy()
        fa, fb = [], []
        for k in range(ngr):
            i = g == k
            if y[i].sum() == 0:
                continue
            fa.append(aa.binary_metrics(y[i], (pooled[ma]["dec"][i] >= 1).astype(int))["f1"])
            fb.append(aa.binary_metrics(y[i], (pooled[mb]["dec"][i] >= 1).astype(int))["f1"])
        fa, fb = np.array(fa), np.array(fb); d = fb - fa
        sh = float(stats.shapiro(d).pvalue) if np.ptp(d) > 0 else 1.0
        t = stats.ttest_rel(fb, fa) if sh > 0.05 else stats.wilcoxon(fb, fa)
        return dict(n_grupos=len(d), prueba="t de Student pareada" if sh > 0.05 else "Wilcoxon de rangos con signo",
                    media_A=float(fa.mean()), media_B=float(fb.mean()), shapiro_p=sh, estadistico=float(t.statistic),
                    p_valor=float(t.pvalue), d_cohen_dz=float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else None)

    imp = pd.DataFrame(imp_acum).mean().sort_values(ascending=False) if imp_acum else pd.Series(dtype=float)
    resultados = dict(
        dataset="CERT r4.2", unidad="usuario-día (solo sesiones)", evaluacion=f"{K} folds por empleado (estratificados)",
        volumen=dict(dias=len(a), empleados=int(a.user.nunique()), dias_maliciosos=int(y.sum())),
        modelos=res, mejor=mejor, puntos_operacion_adaptauth=po.round(5).to_dict(orient="records"),
        por_unidad=pu.round(4).to_dict(orient="records") if len(pu) else [],
        pareada_mejor_vs_reglas=pareada("reglas_estaticas", mejor),
        pareada_mejor_vs_iforest=pareada("iforest", mejor),
        pareada_apilado_vs_adaptauth=pareada(mejor, "hibrido_apilado"),
        importancia_adaptauth=imp.round(4).to_dict(),
        arranque_en_frio=frio,
    )
    aa.to_json(resultados, OUT / "metricas_cert_v2.json")
    figures.model_comparison(tabla, OUT / "figuras" / "comparacion_modelos_cert_v2.png")
    if len(imp):
        figures.shap_bars(imp.clip(lower=0), OUT / "figuras" / "importancia_cert_v2.png",
                          xlabel="Caída de AUC-PR al permutar la variable")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=160)
    ax.plot(po["fpr_prueba"] * 100, po["recall"], marker="o", color="#1F5FAE", lw=2)
    for _, r in po.iterrows():
        ax.annotate(f"{r.recall:.2f}", (r.fpr_prueba * 100, r.recall), textcoords="offset points", xytext=(4, -12), fontsize=8)
    ax.axhline(0.70, ls="--", color="#E06A2E", lw=1); ax.axvline(0.10, ls="--", color="#E06A2E", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("Falsas alarmas en prueba (FPR, %)"); ax.set_ylabel("Recall de días maliciosos")
    ax.set_title("CERT r4.2 · AdaptAuth-ML v11: detección según el presupuesto de alertas", fontsize=10)
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(OUT / "figuras" / "puntos_operacion_cert_v11.png"); plt.close(fig)
    if frio:
        fz = pd.DataFrame(frio).groupby("fraccion")[["auc_pr_adaptauth", "auc_pr_apilado", "auc_pr_iforest"]].mean()
        fz.to_csv(OUT / "arranque_en_frio_cert.csv"); print("[arranque en frío CERT, AUC-PR medio]"); print(fz.round(4).to_string())
    print(tabla.round(4).to_string(index=False))
    print("[puntos de operación, AdaptAuth-ML v11]"); print(po.round(5).to_string(index=False))
    if len(pu):
        print(f"[por unidad funcional, {mejor}]"); print(pu.round(3).to_string(index=False))
    for k in ("pareada_mejor_vs_reglas", "pareada_mejor_vs_iforest", "pareada_apilado_vs_adaptauth"):
        if resultados[k]:
            print(f"[{k}]", {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in resultados[k].items()})
    print("[importancia top 8]", dict(list(imp.round(4).items())[:8]))
    print(f"[listo] {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
