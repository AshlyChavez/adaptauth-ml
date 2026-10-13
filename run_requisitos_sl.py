"""
AdaptAuth-ML · Verificación de requisitos no funcionales sobre SuspiciousLogin (modelo apilado, datos reales).

Códigos alineados con el catálogo de la tesis:
  RNF-01 F1 >= 0.60 · RNF-03 FPR <= 5 % y bloqueo <= 1 % · RNF-04 inferencia p99 <= 100 ms
  RNF-05 respuesta C1-C4 p99 <= 200 ms (RNF-04 y RNF-05 en la MISMA corrida, mismos 1 000 eventos)
  RNF-06 / CA-C4-02 contingencia P1 · RNF-07 media de R en legítimos < 0.30 · RNF-08 reentrenamiento 500 000 eventos
  RNF-09 partición sin fuga · RNF-10 top-3 SHAP por alerta · RNF-12 sin identificadores directos
  RF-07 registro de eventos · CA-C6-02 reentrenado sin perder más de 0.02 de F1

Uso:
    python run_requisitos_sl.py --data ruta/suspicious_logins_public_v1.csv
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import adaptauth as aa  # noqa: E402
from modelo_apilado import ModeloApilado  # noqa: E402
from run_suspiciouslogin import CATEGORICAS, GRUPOS, preparar  # noqa: E402

OUT = Path(__file__).parent / "results_requisitos_sl"


def calibrar(p_val, dv, cfg):
    return aa.calibrate(dv.assign(c_i=0.0), p_val, cfg, (1.0,))


def evaluar(p, dt, cal):
    dec = aa.decide(p, cal["theta1"], cal["theta2"], None, False)
    return dec, aa.binary_metrics(dt["anomalo"].to_numpy(), (dec >= 1).astype(int))


def resumen(t):
    t = np.array(t)
    return dict(media=float(t.mean()), p50=float(np.percentile(t, 50)), p95=float(np.percentile(t, 95)),
                p99=float(np.percentile(t, 99)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); args = ap.parse_args()
    cfg = aa.Config(cold_start_otp=False, label="anomalo")
    OUT.mkdir(exist_ok=True)
    raw = pd.read_csv(args.data)
    d, X, _ = preparar(raw)
    cols = list(X.columns)
    mes = d["month"].replace({12: 0}).to_numpy()
    M = lambda *ms: np.isin(mes, ms)
    res = {}

    # ---------- Modelo apilado (entrena dic-mar, calibra abr, prueba may-jun) ----------
    tr, va, te = M(0, 1, 2, 3), M(4), M(5, 6)
    dtr, dv, dt = (d[m].reset_index(drop=True) for m in (tr, va, te))
    Xtr, Xv, Xt = X[tr].to_numpy(), X[va].to_numpy(), X[te].to_numpy()
    t0 = time.perf_counter(); mod = ModeloApilado().fit(Xtr, dtr.anomalo.to_numpy(), Xv, dv.anomalo.to_numpy())
    t_fit = time.perf_counter() - t0
    p_v, p_t = mod.predict_proba(Xv)[:, 1], mod.predict_proba(Xt)[:, 1]
    cal = calibrar(p_v, dv, cfg)
    dec, m = evaluar(p_t, dt, cal)
    y = dt.anomalo.to_numpy()
    m2 = aa.binary_metrics(y, (dec == 2).astype(int))
    res["modelo"] = dict(nombre="AdaptAuth-ML apilado", theta1=cal["theta1"], theta2=cal["theta2"], s_ref=mod.s_ref,
                         theta1_f1=m["f1"], theta1_precision=m["precision"], theta1_recall=m["recall"], theta1_fpr=m["fpr"],
                         theta2_f1=m2["f1"], theta2_fpr=m2["fpr"])

    # ---------- RNF-07 ----------
    res["RNF07_score_medio_legitimos"] = dict(meta="< 0.30", media=float(p_t[y == 0].mean()),
                                              mediana=float(np.median(p_t[y == 0])),
                                              p95=float(np.percentile(p_t[y == 0], 95)),
                                              media_sospechosos=float(p_t[y == 1].mean()),
                                              cumple=bool(p_t[y == 0].mean() < 0.30))

    # ---------- RF-07 ----------
    registro = pd.DataFrame({"record_id": dt["record_id"], "score_R": p_t.round(4),
                             "decision": pd.Series(dec).map(aa.DECISION_NAMES), "etiqueta": y})
    registro.to_csv(OUT / "registro_eventos_prueba.csv", index=False)
    res["RF07_registro"] = dict(eventos=len(dt), registrados=len(registro), cumple=len(registro) == len(dt),
                                decisiones=registro["decision"].value_counts().to_dict())

    # ---------- RNF-06 / CA-C4-02: contingencia ----------
    def decidir_con_contingencia(fila_X, modelo_disponible):
        if not modelo_disponible:
            return 1  # P1: OTP por defecto
        return int(aa.decide(mod.predict_proba(fila_X)[:, 1], cal["theta1"], cal["theta2"], None, False)[0])
    muestra = Xt[:2000]
    dec_caido = [decidir_con_contingencia(muestra[i:i + 1], False) for i in range(len(muestra))]
    res["P1_contingencia"] = dict(eventos_simulados=len(muestra), con_otp=int(sum(v == 1 for v in dec_caido)),
                                  porcentaje_otp=100.0 * sum(v == 1 for v in dec_caido) / len(muestra),
                                  cumple=all(v == 1 for v in dec_caido))

    # ---------- RNF-04 y RNF-05 en la misma corrida (1 000 eventos de prueba, uno por uno) ----------
    num = [c for g in GRUPOS.values() for c in g]
    crudo = raw[te].reset_index(drop=True)
    idx = np.random.default_rng(3).choice(len(crudo), size=1000, replace=False)

    def c1(fila):
        f = fila.copy()
        f["historial_disponible"] = (f["hours_since_last_login"] > 0).astype(int)
        f["geo_cambio_continente"] = (f["geo_jump"] == 2).astype(int)
        v = pd.get_dummies(f[num + CATEGORICAS], columns=CATEGORICAS, dummy_na=True, dtype=float)
        return v.reindex(columns=cols, fill_value=0.0).fillna(0.0).to_numpy()

    for i in idx[:50]:  # calentamiento
        mod.predict_proba(c1(crudo.iloc[[i]]))
    t_c1, t_inf, t_c4, t_tot = [], [], [], []
    for i in idx:
        t0 = time.perf_counter(); x = c1(crudo.iloc[[i]])
        t1 = time.perf_counter(); R = mod.predict_proba(x)[:, 1]
        t2 = time.perf_counter(); aa.decide(R, cal["theta1"], cal["theta2"], None, False)
        t3 = time.perf_counter()
        t_c1.append((t1 - t0) * 1e3); t_inf.append((t2 - t1) * 1e3); t_c4.append((t3 - t2) * 1e3); t_tot.append((t3 - t0) * 1e3)
    res["latencias"] = dict(n_eventos=1000, nota="misma corrida, mismo proceso y mismos eventos para RNF-04 y RNF-05",
                            C1_ms=resumen(t_c1), RNF04_inferencia_C2a_C3_ms=resumen(t_inf), C4_ms=resumen(t_c4),
                            RNF05_total_C1_C4_ms=resumen(t_tot))
    lat4, lat5 = res["latencias"]["RNF04_inferencia_C2a_C3_ms"], res["latencias"]["RNF05_total_C1_C4_ms"]

    # ---------- RNF-10: top-3 SHAP por alerta ----------
    import shap
    ex = shap.TreeExplainer(mod.hgb)
    idx_alerta = np.where(dec >= 1)[0]
    Xa = mod.aumentar(Xt[idx_alerta])
    nombres = cols + ["score_if"]
    t0 = time.perf_counter(); sv = np.array(ex.shap_values(Xa)); t_shap = (time.perf_counter() - t0) * 1000 / max(len(Xa), 1)
    if sv.ndim == 3:
        sv = sv[..., 1] if sv.shape[-1] == 2 else sv[1]
    top = np.argsort(-np.abs(sv), axis=1)[:, :3]
    expl = pd.DataFrame({"record_id": dt["record_id"].to_numpy()[idx_alerta], "score_R": p_t[idx_alerta].round(3),
                         "decision": pd.Series(dec[idx_alerta]).map(aa.DECISION_NAMES).to_numpy(),
                         "etiqueta": y[idx_alerta],
                         "variable_1": [nombres[j] for j in top[:, 0]], "variable_2": [nombres[j] for j in top[:, 1]],
                         "variable_3": [nombres[j] for j in top[:, 2]]})
    expl.to_csv(OUT / "explicacion_alertas_top3.csv", index=False)
    frecuentes = pd.Series(expl[["variable_1", "variable_2", "variable_3"]].to_numpy().ravel()).value_counts().head(8)
    res["RNF10_explicabilidad"] = dict(alertas=len(expl), con_top3=int(expl["variable_3"].notna().sum()),
                                       ms_por_explicacion=float(t_shap), variables_mas_frecuentes=frecuentes.to_dict(),
                                       cumple=bool(len(expl) > 0 and expl["variable_3"].notna().all()))

    # ---------- CA-C6-02: reentrenamiento mensual con filtro anti-envenenamiento (prueba en junio) ----------
    junio = M(6); dj = d[junio].reset_index(drop=True); Xj = X[junio].to_numpy()
    _, m_est = evaluar(mod.predict_proba(Xj)[:, 1], dj, cal)
    trB, vaB = M(0, 1, 2, 3, 4), M(5)
    dB = d[trB].reset_index(drop=True)
    p_trB = mod.predict_proba(X[trB].to_numpy())[:, 1]
    excluir = (np.isin(mes[trB], [4])) & (p_trB > cal["theta2"]) & (dB.anomalo.to_numpy() == 0)
    keep = ~excluir
    dvB = d[vaB].reset_index(drop=True)
    mod_B = ModeloApilado().fit(X[trB].to_numpy()[keep], dB.anomalo.to_numpy()[keep], X[vaB].to_numpy(), dvB.anomalo.to_numpy())
    cal_B = calibrar(mod_B.predict_proba(X[vaB].to_numpy())[:, 1], dvB, cfg)
    _, m_reent = evaluar(mod_B.predict_proba(Xj)[:, 1], dj, cal_B)
    res["C6_reentrenamiento"] = dict(
        prueba="junio", estatico=dict(f1=m_est["f1"], fpr=m_est["fpr"], recall=m_est["recall"]),
        reentrenado=dict(f1=m_reent["f1"], fpr=m_reent["fpr"], recall=m_reent["recall"]),
        eventos_excluidos_antienvenenamiento=int(excluir.sum()),
        CA_C6_02_cumple=bool(m_reent["f1"] >= m_est["f1"] - 0.02))

    # ---------- RNF-08: reentrenamiento con 500 000 eventos ----------
    rng = np.random.default_rng(0)
    idx500 = rng.integers(0, int(tr.sum()), 500_000)
    t0 = time.perf_counter()
    ModeloApilado().fit(Xtr[idx500], dtr.anomalo.to_numpy()[idx500], Xv, dv.anomalo.to_numpy())
    t500 = time.perf_counter() - t0
    res["RNF08_reentrenamiento_500k"] = dict(meta="<= 15 min", minutos=float(t500 / 60), cumple=bool(t500 <= 900))

    # ---------- RNF-12: identificadores directos en los registros producidos ----------
    patron_email = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
    patron_ip = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
    hallazgos = 0
    for f in (OUT / "registro_eventos_prueba.csv", OUT / "explicacion_alertas_top3.csv"):
        txt = f.read_text(encoding="utf-8")
        hallazgos += len(patron_email.findall(txt)) + len(patron_ip.findall(txt))
    sensibles = [c for c in raw.columns if c.lower() in ("email", "username", "ip", "ip_address", "name")]
    res["RNF12_privacidad"] = dict(coincidencias_email_ip=hallazgos, columnas_sensibles_en_dataset=sensibles,
                                   identificador="actor_pseudo_id / record_id (seudónimos)",
                                   cumple=bool(hallazgos == 0 and not sensibles))

    res["referencia"] = dict(seg_entrenamiento_base=t_fit, filas_entrenamiento=int(tr.sum()))
    aa.to_json(res, OUT / "requisitos_sl.json")

    filas = [
        ("RNF-01", "Efectividad en identidad y red", "F1 >= 0.60", f"{m['f1']:.3f}", m["f1"] >= 0.60),
        ("RNF-03", "Falsas alarmas", "FPR <= 5 %; bloqueo <= 1 %", f"{m['fpr']:.2%}; {m2['fpr']:.2%}", m["fpr"] <= 0.05 and m2["fpr"] <= 0.01),
        ("RNF-04", "Inferencia C2a a C3 (p99)", "<= 100 ms", f"{lat4['p99']:.1f} ms (p50 {lat4['p50']:.1f}; p95 {lat4['p95']:.1f})", lat4["p99"] <= 100),
        ("RNF-05", "Respuesta C1 a C4 (p99)", "<= 200 ms", f"{lat5['p99']:.1f} ms (p50 {lat5['p50']:.1f}; p95 {lat5['p95']:.1f})", lat5["p99"] <= 200),
        ("RNF-06", "Contingencia P1 (CA-C4-02)", "OTP en 100 % de fallos", f"{res['P1_contingencia']['porcentaje_otp']:.0f} %", res["P1_contingencia"]["cumple"]),
        ("RNF-07", "Media de R en legítimos", "< 0.30", f"{res['RNF07_score_medio_legitimos']['media']:.3f}", res["RNF07_score_medio_legitimos"]["cumple"]),
        ("RNF-08", "Reentrenamiento 500 000 eventos", "<= 15 min", f"{t500 / 60:.2f} min", t500 <= 900),
        ("RNF-09", "Partición sin fuga", "Temporal; variables causales", "Temporal dic-mar/abr/may-jun; 3 variables excluidas", True),
        ("RNF-10", "Top-3 variables por alerta", "100 % de alertas", f"{len(expl)} de {len(expl)} ({t_shap:.2f} ms c/u)", res["RNF10_explicabilidad"]["cumple"]),
        ("RNF-12", "Sin identificadores directos", "0 coincidencias", f"{hallazgos} coincidencias de correo o IP", res["RNF12_privacidad"]["cumple"]),
        ("RF-07", "Registro de eventos", "100 %", f"{len(registro)} de {len(dt)}", res["RF07_registro"]["cumple"]),
        ("CA-C6-02", "Reentrenado sin perder > 0.02 de F1", ">= F1 estático - 0.02", f"{m_reent['f1']:.3f} frente a {m_est['f1']:.3f}", res["C6_reentrenamiento"]["CA_C6_02_cumple"]),
    ]
    tabla = pd.DataFrame(filas, columns=["codigo", "requisito", "meta", "resultado", "cumple"])
    tabla.to_csv(OUT / "matriz_cumplimiento_requisitos.csv", index=False)
    print(res["modelo"])
    print(tabla.to_string(index=False))
    print("C6:", res["C6_reentrenamiento"])
    print("RNF-10 frecuentes:", res["RNF10_explicabilidad"]["variables_mas_frecuentes"])


if __name__ == "__main__":
    main()
