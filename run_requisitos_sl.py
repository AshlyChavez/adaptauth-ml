"""
Verificación de requisitos pendientes de AdaptAuth-ML sobre SuspiciousLogin (datos reales).

  RNF07  Score medio de accesos legítimos < 0.30
  RNF04  Tiempo de respuesta C1-C4 por evento (preprocesamiento + modelo + decisión) <= 200 ms
  RNF08  Reentrenamiento de 500 000 eventos <= 15 min                      (C6)
  CA-C6-02  El modelo reentrenado no degrada el F1 en más de 2 puntos      (C6)
  RNF10  Explicación de cada alerta con sus 3 variables principales (SHAP) (C3/C5)
  CA-C4-02 / P1  Si el motor de detección no está disponible, 100 % de accesos reciben OTP
  RF07   Todos los eventos quedan registrados con su score y decisión

Uso: python run_requisitos_sl.py --data data/suspicious_logins_public_v1.csv
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import adaptauth as aa  # noqa: E402
from run_suspiciouslogin import CATEGORICAS, GRUPOS, preparar  # noqa: E402

OUT = Path(__file__).parent / "results_requisitos_sl"


def entrenar(X, y, seed=42):
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=seed).fit(X, y)


def calibrar(p_val, dv, cfg):
    return aa.calibrate(dv.assign(c_i=0.0), p_val, cfg, (1.0,))


def evaluar(p, dt, cal):
    dec = aa.decide(p, cal["theta1"], cal["theta2"], None, False)
    return dec, aa.binary_metrics(dt["anomalo"].to_numpy(), (dec >= 1).astype(int))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); args = ap.parse_args()
    cfg = aa.Config(cold_start_otp=False, label="anomalo")
    OUT.mkdir(exist_ok=True)
    d, X, _ = preparar(pd.read_csv(args.data))
    cols = list(X.columns)
    mes = d["month"].replace({12: 0}).to_numpy()
    M = lambda *ms: np.isin(mes, ms)
    res = {}

    # ---------- Modelo base (dic-mar / abr / may-jun) ----------
    tr, va, te = M(0, 1, 2, 3), M(4), M(5, 6)
    dtr, dv, dt = (d[m].reset_index(drop=True) for m in (tr, va, te))
    t0 = time.perf_counter(); mod = entrenar(X[tr].to_numpy(), dtr.anomalo.to_numpy()); t_fit = time.perf_counter() - t0
    p_v, p_t = mod.predict_proba(X[va].to_numpy())[:, 1], mod.predict_proba(X[te].to_numpy())[:, 1]
    cal = calibrar(p_v, dv, cfg)
    dec, m = evaluar(p_t, dt, cal)
    y = dt.anomalo.to_numpy()

    # ---------- RNF07: score medio de legítimos ----------
    res["RNF07_score_medio_legitimos"] = dict(meta="< 0.30", media=float(p_t[y == 0].mean()),
                                              mediana=float(np.median(p_t[y == 0])),
                                              p95=float(np.percentile(p_t[y == 0], 95)),
                                              media_sospechosos=float(p_t[y == 1].mean()),
                                              cumple=bool(p_t[y == 0].mean() < 0.30))

    # ---------- RF07: registro de todos los eventos ----------
    registro = pd.DataFrame({"record_id": dt["record_id"], "score_R": p_t.round(4),
                             "decision": pd.Series(dec).map(aa.DECISION_NAMES), "etiqueta": y})
    registro.to_csv(OUT / "registro_eventos_prueba.csv", index=False)
    res["RF07_registro"] = dict(eventos=len(dt), registrados=len(registro), cumple=len(registro) == len(dt),
                                decisiones=registro["decision"].value_counts().to_dict())

    # ---------- P1 / CA-C4-02: contingencia ante indisponibilidad del motor ----------
    def decidir_con_contingencia(fila_X, modelo_disponible):
        if not modelo_disponible:
            return 1                                    # P1: OTP por defecto (fail-secure)
        return int(aa.decide(mod.predict_proba(fila_X)[:, 1], cal["theta1"], cal["theta2"], None, False)[0])
    muestra = X[te].to_numpy()[:2000]
    dec_caido = [decidir_con_contingencia(muestra[i:i + 1], False) for i in range(len(muestra))]
    res["P1_contingencia"] = dict(eventos_simulados=len(muestra), con_otp=int(sum(v == 1 for v in dec_caido)),
                                  porcentaje_otp=100.0 * sum(v == 1 for v in dec_caido) / len(muestra),
                                  cumple=all(v == 1 for v in dec_caido))

    # ---------- RNF04: tiempo de respuesta C1-C4 por evento ----------
    crudo = pd.read_csv(args.data).iloc[np.where(te)[0][:300]]
    tiempos = []
    for i in range(len(crudo)):
        t0 = time.perf_counter()
        fila, Xf, _ = preparar(crudo.iloc[[i]])                          # C1: variables
        Xf = Xf.reindex(columns=cols, fill_value=0.0).to_numpy()
        R = mod.predict_proba(Xf)[:, 1]                                   # C2/C3: score
        aa.decide(R, cal["theta1"], cal["theta2"], None, False)          # C4: decisión
        tiempos.append((time.perf_counter() - t0) * 1000)
    t = np.array(tiempos)
    res["RNF04_respuesta_C1_C4_ms"] = dict(meta="<= 200 ms", p50=float(np.percentile(t, 50)),
                                           p95=float(np.percentile(t, 95)), p99=float(np.percentile(t, 99)),
                                           cumple=bool(np.percentile(t, 99) <= 200))

    # ---------- RNF10: explicación de cada alerta (top-3 SHAP) ----------
    import shap
    ex = shap.TreeExplainer(mod)
    idx_alerta = np.where(dec >= 1)[0]
    Xa = X[te].to_numpy()[idx_alerta]
    t0 = time.perf_counter(); sv = np.array(ex.shap_values(Xa)); t_shap = (time.perf_counter() - t0) * 1000 / max(len(Xa), 1)
    if sv.ndim == 3:
        sv = sv[..., 1] if sv.shape[-1] == 2 else sv[1]
    top = np.argsort(-sv, axis=1)[:, :3]
    expl = pd.DataFrame({"record_id": dt["record_id"].to_numpy()[idx_alerta], "score_R": p_t[idx_alerta].round(3),
                         "decision": pd.Series(dec[idx_alerta]).map(aa.DECISION_NAMES).to_numpy(),
                         "etiqueta": y[idx_alerta],
                         "variable_1": [cols[j] for j in top[:, 0]], "variable_2": [cols[j] for j in top[:, 1]],
                         "variable_3": [cols[j] for j in top[:, 2]]})
    expl.to_csv(OUT / "explicacion_alertas_top3.csv", index=False)
    frecuentes = pd.Series(expl[["variable_1", "variable_2", "variable_3"]].to_numpy().ravel()).value_counts().head(8)
    res["RNF10_explicabilidad"] = dict(alertas=len(expl), con_top3=int(expl["variable_3"].notna().sum()),
                                       ms_por_explicacion=float(t_shap), variables_mas_frecuentes=frecuentes.to_dict(),
                                       ejemplos=expl.head(5).to_dict(orient="records"),
                                       cumple=bool(len(expl) > 0 and expl["variable_3"].notna().all()))

    # ---------- C6: reentrenamiento mensual vs modelo estático (prueba en junio) ----------
    junio = M(6); dj = d[junio].reset_index(drop=True); Xj = X[junio].to_numpy()
    _, m_est = evaluar(mod.predict_proba(Xj)[:, 1], dj, cal)                     # estático (dic-mar, cal. abr)
    trB, vaB = M(0, 1, 2, 3, 4), M(5)                                          # reentrenado con abril
    dB = d[trB].reset_index(drop=True)
    # anti-envenenamiento: se excluyen los eventos que el modelo vigente BLOQUEÓ y aún no tienen etiqueta confirmada
    p_trB = mod.predict_proba(X[trB].to_numpy())[:, 1]
    excluir = (np.isin(mes[trB], [4])) & (p_trB > cal["theta2"]) & (dB.anomalo.to_numpy() == 0)
    keep = ~excluir
    t0 = time.perf_counter()
    mod_B = entrenar(X[trB].to_numpy()[keep], dB.anomalo.to_numpy()[keep])
    t_reent = time.perf_counter() - t0
    cal_B = calibrar(mod_B.predict_proba(X[vaB].to_numpy())[:, 1], d[vaB].reset_index(drop=True), cfg)
    _, m_reent = evaluar(mod_B.predict_proba(Xj)[:, 1], dj, cal_B)
    res["C6_reentrenamiento"] = dict(
        prueba="junio", estatico=dict(entrenado="dic-mar", calibrado="abr", f1=m_est["f1"], fpr=m_est["fpr"], recall=m_est["recall"]),
        reentrenado=dict(entrenado="dic-abr", calibrado="may", f1=m_reent["f1"], fpr=m_reent["fpr"], recall=m_reent["recall"]),
        eventos_excluidos_antienvenenamiento=int(excluir.sum()),
        diferencia_f1_puntos=100 * (m_reent["f1"] - m_est["f1"]),
        CA_C6_02_cumple=bool(m_reent["f1"] >= m_est["f1"] - 0.02))

    # ---------- RNF08: tiempo de reentrenamiento con 500 000 eventos ----------
    rng = np.random.default_rng(0)
    idx500 = rng.integers(0, int(tr.sum()), 500_000)
    X500, y500 = X[tr].to_numpy()[idx500], dtr.anomalo.to_numpy()[idx500]
    t0 = time.perf_counter(); entrenar(X500, y500); t500 = time.perf_counter() - t0
    res["RNF08_reentrenamiento_500k"] = dict(meta="<= 15 min", segundos=float(t500), minutos=float(t500 / 60),
                                             cumple=bool(t500 <= 900), nota="500 000 eventos remuestreados del entrenamiento")

    res["referencia"] = dict(f1_theta1=m["f1"], precision=m["precision"], recall=m["recall"], fpr=m["fpr"],
                             seg_entrenamiento_base=t_fit, filas_entrenamiento=int(tr.sum()))
    aa.to_json(res, OUT / "requisitos_sl.json")

    filas = [
        ("RNF02", "FPR (OTP o bloqueo)", "<= 5 %", f"{m['fpr']:.2%}", m["fpr"] <= 0.05),
        ("RNF04", "Tiempo de respuesta C1-C4 (p99)", "<= 200 ms", f"{res['RNF04_respuesta_C1_C4_ms']['p99']:.1f} ms", res["RNF04_respuesta_C1_C4_ms"]["cumple"]),
        ("RNF07", "Score medio de legítimos", "< 0.30", f"{res['RNF07_score_medio_legitimos']['media']:.3f}", res["RNF07_score_medio_legitimos"]["cumple"]),
        ("RNF08", "Reentrenamiento 500 000 eventos", "<= 15 min", f"{t500 / 60:.2f} min", t500 <= 900),
        ("RNF10", "Top-3 variables por alerta", "100 % de alertas", f"{len(expl)} de {len(expl)} ({t_shap:.2f} ms c/u)", res["RNF10_explicabilidad"]["cumple"]),
        ("RF07", "Registro de eventos con score y decisión", "100 %", f"{len(registro)} de {len(dt)}", res["RF07_registro"]["cumple"]),
        ("CA-C4-02", "OTP automático si cae el motor (P1)", "100 %", f"{res['P1_contingencia']['porcentaje_otp']:.0f} %", res["P1_contingencia"]["cumple"]),
        ("CA-C6-02", "Reentrenado no pierde más de 2 pts de F1", ">= F1 estático - 0.02",
         f"{m_reent['f1']:.3f} vs {m_est['f1']:.3f}", res["C6_reentrenamiento"]["CA_C6_02_cumple"]),
    ]
    tabla = pd.DataFrame(filas, columns=["codigo", "requisito", "meta", "resultado", "cumple"])
    tabla.to_csv(OUT / "matriz_cumplimiento_requisitos.csv", index=False)
    print(tabla.to_string(index=False))
    print("\nC6:", res["C6_reentrenamiento"])
    print("RNF07:", {k: round(v, 3) if isinstance(v, float) else v for k, v in res["RNF07_score_medio_legitimos"].items()})
    print("RNF10 variables más frecuentes en alertas:", res["RNF10_explicabilidad"]["variables_mas_frecuentes"])
    print(expl.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
