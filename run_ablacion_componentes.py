"""
AdaptAuth-ML · Ablación de componentes del modelo integral sobre SuspiciousLogin (versión 11).

Mide el valor de cada componente de decisión manteniendo el mismo clasificador:
  A. Estructura de la decisión: tres niveles (θ1, θ2) frente a esquemas de dos niveles y a OTP universal.
  B. Contingencia P1: política ante fallo del motor (OTP frente a dejar pasar o bloquear).
  C. Reentrenamiento protegido: excluir los bloqueos no confirmados frente a etiquetarlos como legítimos.

Uso:
    python run_ablacion_componentes.py --data ruta/suspicious_logins_public_v1.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import adaptauth as aa  # noqa: E402
from modelo_adaptauth import ModeloAdaptAuth  # noqa: E402
from run_suspiciouslogin import preparar  # noqa: E402

OUT = Path(__file__).parent / "results_ablacion_componentes"


def calibrar(p_val, dv, cfg):
    return aa.calibrate(dv.assign(c_i=0.0), p_val, cfg, (1.0,))


def resumen_decision(nombre, dec, y):
    """dec: 0 acceso directo, 1 OTP, 2 bloqueo."""
    leg, sos = y == 0, y == 1
    return dict(
        esquema=nombre,
        legitimos_otp=int(((dec == 1) & leg).sum()),
        legitimos_bloqueados=int(((dec == 2) & leg).sum()),
        legitimos_con_friccion=int(((dec >= 1) & leg).sum()),
        sospechosos_sin_verificacion=int(((dec == 0) & sos).sum()),
        sospechosos_otp=int(((dec == 1) & sos).sum()),
        sospechosos_bloqueados=int(((dec == 2) & sos).sum()),
        pct_legitimos_bloqueados=100 * float(((dec == 2) & leg).sum() / leg.sum()),
        pct_sospechosos_detenidos=100 * float(((dec >= 1) & sos).sum() / sos.sum()),
    )


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); args = ap.parse_args()
    cfg = aa.Config(cold_start_otp=False, label="anomalo")
    OUT.mkdir(exist_ok=True)
    d, X, _ = preparar(pd.read_csv(args.data))
    mes = d["month"].replace({12: 0}).to_numpy()
    M = lambda *ms: np.isin(mes, ms)
    tr, va, te = M(0, 1, 2, 3), M(4), M(5, 6)
    dtr, dv, dt = (d[m].reset_index(drop=True) for m in (tr, va, te))
    mod = ModeloAdaptAuth().fit(X[tr].to_numpy(), dtr.anomalo.to_numpy(), X[va].to_numpy(), dv.anomalo.to_numpy())
    p_v, p_t = mod.predict_proba(X[va].to_numpy())[:, 1], mod.predict_proba(X[te].to_numpy())[:, 1]
    cal = calibrar(p_v, dv, cfg)
    t1, t2 = cal["theta1"], cal["theta2"]
    y = dt.anomalo.to_numpy()

    # ---------- A. Estructura de la decisión ----------
    esquemas = [
        ("Tres niveles: acceso, OTP y bloqueo (v11)", np.where(p_t > t2, 2, np.where(p_t > t1, 1, 0))),
        ("Dos niveles: acceso y bloqueo con θ1", np.where(p_t > t1, 2, 0)),
        ("Dos niveles: acceso y bloqueo con θ2", np.where(p_t > t2, 2, 0)),
        ("Dos niveles: acceso y OTP con θ1 (sin bloqueo)", np.where(p_t > t1, 1, 0)),
        ("OTP universal (MFA estática)", np.ones_like(y)),
    ]
    A = pd.DataFrame([resumen_decision(n, dec, y) for n, dec in esquemas])
    A.to_csv(OUT / "ablacion_estructura_decision.csv", index=False)

    # ---------- B. Contingencia ante fallo del motor (todo el periodo de prueba sin motor) ----------
    politicas = [
        ("P1: OTP ante fallo (v11)", np.ones_like(y)),
        ("Sin contingencia: dejar pasar (fail-open)", np.zeros_like(y)),
        ("Sin contingencia: bloquear (fail-closed)", np.full_like(y, 2)),
    ]
    B = pd.DataFrame([resumen_decision(n, dec, y) for n, dec in politicas])
    B.to_csv(OUT / "ablacion_contingencia.csv", index=False)

    # ---------- C. Reentrenamiento protegido (reentrena con dic-abr, calibra en may, prueba en jun) ----------
    junio = M(6); dj = d[junio].reset_index(drop=True); Xj = X[junio].to_numpy(); yj = dj.anomalo.to_numpy()
    trB, vaB = M(0, 1, 2, 3, 4), M(5)
    XB, yB = X[trB].to_numpy(), d[trB].anomalo.to_numpy().copy()
    dvB = d[vaB].reset_index(drop=True)
    pB = mod.predict_proba(XB)[:, 1]
    # En operación, los eventos de abril que el modelo bloqueó no tienen confirmación: el usuario no entró.
    no_confirmados = np.isin(mes[trB], [4]) & (pB > t2)

    def reentrenar(Xr, yr):
        m = ModeloAdaptAuth().fit(Xr, yr, X[vaB].to_numpy(), dvB.anomalo.to_numpy())
        c = calibrar(m.predict_proba(X[vaB].to_numpy())[:, 1], dvB, cfg)
        dec = aa.decide(m.predict_proba(Xj)[:, 1], c["theta1"], c["theta2"], None, False)
        r = aa.binary_metrics(yj, (dec >= 1).astype(int))
        return dict(f1=r["f1"], precision=r["precision"], recall=r["recall"], fpr=r["fpr"])

    filas = []
    todos_bloqueados = pB > t2  # escenario acumulado: ningún bloqueo de la ventana fue confirmado
    est = aa.decide(mod.predict_proba(Xj)[:, 1], t1, t2, None, False)
    r = aa.binary_metrics(yj, (est >= 1).astype(int))
    filas.append(dict(variante="Modelo sin reentrenar (referencia)", eventos_entrenamiento=int(tr.sum()),
                      f1=r["f1"], precision=r["precision"], recall=r["recall"], fpr=r["fpr"]))
    y_ingenuo = yB.copy(); y_ingenuo[no_confirmados] = 0
    filas.append(dict(variante="Sin protección: bloqueos no confirmados etiquetados como legítimos",
                      eventos_entrenamiento=int(len(yB)), **reentrenar(XB, y_ingenuo)))
    keep = ~no_confirmados
    filas.append(dict(variante="Con protección (v11): bloqueos no confirmados excluidos",
                      eventos_entrenamiento=int(keep.sum()), **reentrenar(XB[keep], yB[keep])))
    y_ing2 = yB.copy(); y_ing2[todos_bloqueados] = 0
    filas.append(dict(variante="Acumulado sin protección: ningún bloqueo de dic-abr confirmado, etiquetados como legítimos",
                      eventos_entrenamiento=int(len(yB)), **reentrenar(XB, y_ing2)))
    filas.append(dict(variante="Acumulado con protección: ningún bloqueo de dic-abr confirmado, excluidos",
                      eventos_entrenamiento=int((~todos_bloqueados).sum()), **reentrenar(XB[~todos_bloqueados], yB[~todos_bloqueados])))
    filas.append(dict(variante="Etiquetas completas (cota superior, no disponible en operación)",
                      eventos_entrenamiento=int(len(yB)), **reentrenar(XB, yB)))
    C = pd.DataFrame(filas)
    C.to_csv(OUT / "ablacion_reentrenamiento.csv", index=False)

    info = dict(theta1=t1, theta2=t2, eventos_prueba=int(len(y)), sospechosos_prueba=int(y.sum()),
                bloqueos_abril_no_confirmados=int(no_confirmados.sum()),
                de_ellos_sospechosos=int(yB[no_confirmados].sum()),
                bloqueos_ventana_completa=int(todos_bloqueados.sum()), de_ellos_sospechosos_ventana=int(yB[todos_bloqueados].sum()))
    aa.to_json(dict(info=info, estructura=A.to_dict(orient="records"), contingencia=B.to_dict(orient="records"),
                    reentrenamiento=C.to_dict(orient="records")), OUT / "ablacion_componentes.json")
    pd.set_option("display.width", 220)
    print(info); print(A.to_string(index=False)); print(B.to_string(index=False)); print(C.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
