"""
CERT Insider Threat r4.2 (CMU, CC BY 4.0) para el componente de historial del usuario de AdaptAuth-ML.

Unidad de análisis: usuario-día (cada día de cada empleado con actividad de inicio de sesión).
Solo se usan los registros de inicio/cierre de sesión (logon.csv) y la estructura de la empresa (LDAP):
no se usan USB, web ni correo, para que el modelo siga siendo de AUTENTICACIÓN y no de amenaza interna.

Etiqueta principal: el usuario-día contiene al menos un inicio de sesión listado como malicioso en las
respuestas oficiales (answers). Etiqueta secundaria: cualquier evento malicioso ese día.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

FMT = "%m/%d/%Y %H:%M:%S"
FEATURES_CERT = ["n_logons", "n_eventos", "primera_hora", "ultima_hora", "n_fuera_horario", "frac_fuera_horario",
                 "n_pcs", "n_pc_nuevo", "n_pc_de_otro", "fin_de_semana", "z_primera_hora", "z_ultima_hora",
                 "z_n_logons", "z_n_pcs", "dias_desde_ultimo", "log_dias_historial"]
REGLAS_CERT = {"r1_pc_de_otro": 0.30, "r2_pc_nuevo": 0.25, "r3_fuera_horario": 0.25,
               "r4_fin_de_semana": 0.10, "r5_inicio_atipico": 0.10}


def buscar(base: str, patron: str) -> list[str]:
    return sorted(glob.glob(os.path.join(base, "**", patron), recursive=True))


def cargar_logon(path: str) -> pd.DataFrame:
    lg = pd.read_csv(path)
    lg.columns = [c.strip().lower() for c in lg.columns]
    lg["date"] = pd.to_datetime(lg["date"], format=FMT, errors="coerce")
    return lg.dropna(subset=["date"])


def cargar_ldap(carpeta: str) -> pd.DataFrame:
    """Último registro conocido de cada empleado (unidad funcional, departamento, cargo)."""
    archivos = buscar(carpeta, "*.csv")
    if not archivos:
        return pd.DataFrame(columns=["user"])
    partes = []
    for f in archivos:
        t = pd.read_csv(f)
        t.columns = [c.strip().lower() for c in t.columns]
        t["_mes"] = os.path.basename(f)
        partes.append(t)
    ld = pd.concat(partes, ignore_index=True).sort_values("_mes")
    ld = ld.rename(columns={"user_id": "user"}).drop_duplicates("user", keep="last")
    cols = [c for c in ("user", "role", "business_unit", "functional_unit", "department", "team", "supervisor")
            if c in ld.columns]
    return ld[cols]


def cargar_respuestas(carpeta: str, release: str = "4.2"):
    """
    Lee answers/insiders.csv y los archivos de detalle del release. Cada línea de detalle empieza con el
    tipo de evento (logon, device, http, email, file) seguido de id, fecha, usuario y PC.
    Devuelve (eventos maliciosos, tabla de insiders).
    """
    ins_path = buscar(carpeta, "insiders.csv")
    if not ins_path:
        raise FileNotFoundError("No se encontró insiders.csv dentro de las respuestas (answers).")
    ins = pd.read_csv(ins_path[0])
    ins.columns = [c.strip().lower() for c in ins.columns]
    ins = ins[ins["dataset"].astype(str).str.replace("r", "", regex=False).str.strip() == release]
    filas = []
    for det in ins["details"].astype(str):
        rutas = buscar(carpeta, det)
        if not rutas:
            continue
        with open(rutas[0], encoding="utf-8", errors="ignore") as fh:
            for linea in fh:
                p = linea.rstrip("\n").split(",", 5)
                if len(p) >= 5:
                    filas.append(p[:5])
    ev = pd.DataFrame(filas, columns=["tipo", "id", "date", "user", "pc"])
    ev["tipo"] = ev["tipo"].str.strip().str.lower()
    ev["date"] = pd.to_datetime(ev["date"], format=FMT, errors="coerce")
    ev["dia"] = ev["date"].dt.normalize()
    return ev, ins


def construir_usuario_dia(lg: pd.DataFrame, ev: pd.DataFrame, fecha_corte_pc: pd.Timestamp) -> pd.DataFrame:
    """Agrega los inicios de sesión por usuario-día con variables causales (solo días previos del usuario)."""
    lg = lg.sort_values("date", kind="mergesort").reset_index(drop=True)
    lg["dia"] = lg["date"].dt.normalize()
    lg["hora"] = lg["date"].dt.hour + lg["date"].dt.minute / 60.0
    es_logon = (lg["activity"].astype(str).str.lower() == "logon")
    lg["es_logon"] = es_logon.astype(int)
    lg["fuera_horario"] = (((lg["hora"] < 7) | (lg["hora"] >= 19)) & es_logon).astype(int)
    lg["pc_nuevo"] = ((~lg.duplicated(["user", "pc"])) & es_logon).astype(int)
    # PC "de otro": su usuario principal (en los primeros meses) es otra persona
    base = lg[(lg["date"] < fecha_corte_pc) & es_logon]
    dueno = (base.groupby(["pc", "user"]).size().reset_index(name="n").sort_values("n")
             .drop_duplicates("pc", keep="last").set_index("pc")["user"])
    lg["pc_dueno"] = lg["pc"].map(dueno)
    lg["pc_de_otro"] = (lg["pc_dueno"].notna() & (lg["pc_dueno"] != lg["user"]) & es_logon).astype(int)
    ids_mal = set(ev.loc[ev["tipo"] == "logon", "id"])
    lg["malicioso"] = lg["id"].isin(ids_mal).astype(int)

    a = lg.groupby(["user", "dia"]).agg(
        n_logons=("es_logon", "sum"), n_eventos=("id", "size"), primera_hora=("hora", "min"),
        ultima_hora=("hora", "max"), n_fuera_horario=("fuera_horario", "sum"), n_pcs=("pc", "nunique"),
        n_pc_nuevo=("pc_nuevo", "sum"), n_pc_de_otro=("pc_de_otro", "sum"), anomalo=("malicioso", "max"),
    ).reset_index().sort_values(["user", "dia"], kind="mergesort").reset_index(drop=True)

    g = a.groupby("user")
    a["dias_historial"] = g.cumcount()
    n = a["dias_historial"].replace(0, np.nan)
    for col in ("primera_hora", "ultima_hora", "n_logons", "n_pcs"):
        x = a[col].astype(float)
        s1 = x.groupby(a["user"]).cumsum() - x
        s2 = (x ** 2).groupby(a["user"]).cumsum() - x ** 2
        media = s1 / n
        sd = np.sqrt((s2 / n - media ** 2).clip(lower=0))
        a[f"z_{col}"] = ((x - media) / (sd + 0.5)).fillna(0.0)
    a["dias_desde_ultimo"] = g["dia"].diff().dt.days.fillna(30).clip(upper=60)
    a["fin_de_semana"] = (a["dia"].dt.dayofweek >= 5).astype(int)
    a["frac_fuera_horario"] = a["n_fuera_horario"] / a["n_logons"].clip(lower=1)
    a["log_dias_historial"] = np.log1p(a["dias_historial"])
    a["historial_disponible"] = (a["dias_historial"] >= 5).astype(int)
    a["arranque_frio"] = 1 - a["historial_disponible"]

    dias_mal = set(zip(ev["user"], ev["dia"]))
    a["anomalo_cualquier"] = [int(k in dias_mal) for k in zip(a["user"], a["dia"])]

    a["r1_pc_de_otro"] = (a["n_pc_de_otro"] > 0).astype(int)
    a["r2_pc_nuevo"] = ((a["n_pc_nuevo"] > 0) & (a["dias_historial"] >= 5)).astype(int)
    a["r3_fuera_horario"] = (a["n_fuera_horario"] > 0).astype(int)
    a["r4_fin_de_semana"] = a["fin_de_semana"]
    a["r5_inicio_atipico"] = (a["z_primera_hora"].abs() > 2).astype(int)
    a["c_i"] = sum(w * a[r] for r, w in REGLAS_CERT.items())
    return a


def particion_por_usuario(a: pd.DataFrame, seed: int = 42, prop=(0.70, 0.15, 0.15)):
    """
    Divide por EMPLEADO (todos sus días en una sola partición), estratificando por si alguna vez tuvo un
    día malicioso. Con tan pocos insiders, una partición temporal dejaría validación o prueba sin casos.
    """
    rng = np.random.default_rng(seed)
    pos = a.groupby("user")["anomalo"].max()
    grupos = {}
    for valor in (0, 1):
        us = pos[pos == valor].index.to_numpy().copy()
        rng.shuffle(us)
        n1, n2 = int(len(us) * prop[0]), int(len(us) * (prop[0] + prop[1]))
        for u in us[:n1]: grupos[u] = "train"
        for u in us[n1:n2]: grupos[u] = "val"
        for u in us[n2:]: grupos[u] = "test"
    part = a["user"].map(grupos)
    return (a[part == "train"].reset_index(drop=True), a[part == "val"].reset_index(drop=True),
            a[part == "test"].reset_index(drop=True))


def generar_cert_sintetico(base: str, n_users: int = 120, dias: int = 200, seed: int = 0):
    """Archivos con el FORMATO de CERT r4.2 (solo para probar el pipeline). No son datos reales."""
    import tarfile
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(base, "r4.2", "LDAP"), exist_ok=True)
    users = [f"USR{i:04d}" for i in range(n_users)]
    pcs = {u: f"PC-{1000 + i}" for i, u in enumerate(users)}
    filas, k = [], 0
    inicio = pd.Timestamp("2010-01-04")
    for d in range(dias):
        dia = inicio + pd.Timedelta(days=d)
        if dia.dayofweek >= 5 and rng.random() < 0.9:
            continue
        for u in users:
            if rng.random() < 0.1:
                continue
            h = rng.normal(8.5, 0.6)
            for t, act in ((h, "Logon"), (h + rng.normal(8.5, 0.7), "Logoff")):
                k += 1
                filas.append((f"{{ID-{k:08d}}}", (dia + pd.Timedelta(hours=float(t))).strftime(FMT), u, pcs[u], act))
    malos, det = [], {}
    for j, u in enumerate(users[:6]):
        victima = users[(j + 10) % n_users]
        for d in rng.choice(np.arange(60, dias), size=3, replace=False):
            dia = inicio + pd.Timedelta(days=int(d))
            k += 1
            pc = pcs[victima] if j % 2 else pcs[u]
            usuario = victima if j % 2 else u
            linea = (f"{{ID-{k:08d}}}", (dia + pd.Timedelta(hours=float(rng.uniform(20, 23)))).strftime(FMT), usuario, pc, "Logon")
            filas.append(linea)
            det.setdefault(u, []).append("logon," + ",".join(linea))
    lg = pd.DataFrame(filas, columns=["id", "date", "user", "pc", "activity"])
    lg["_t"] = pd.to_datetime(lg["date"], format=FMT)
    lg.sort_values("_t").drop(columns="_t").to_csv(os.path.join(base, "r4.2", "logon.csv"), index=False)
    ld = pd.DataFrame({"employee_name": users, "user_id": users, "email": [u + "@x.com" for u in users],
                       "role": "Employee", "business_unit": 1,
                       "functional_unit": [f"Unidad {i % 4}" for i in range(n_users)],
                       "department": [f"Depto {i % 8}" for i in range(n_users)], "team": 1, "supervisor": "Jefe"})
    for mes in ("2010-01", "2010-02"):
        ld.to_csv(os.path.join(base, "r4.2", "LDAP", f"{mes}.csv"), index=False)
    os.makedirs(os.path.join(base, "answers", "r4.2-1"), exist_ok=True)
    ins = []
    for u, lineas in det.items():
        nombre = f"r4.2-1-{u}.csv"
        with open(os.path.join(base, "answers", "r4.2-1", nombre), "w") as fh:
            fh.write("\n".join(lineas) + "\n")
        ins.append(dict(dataset=4.2, scenario=1, details=nombre, user=u, start="", end=""))
    pd.DataFrame(ins).to_csv(os.path.join(base, "answers", "insiders.csv"), index=False)
    with tarfile.open(os.path.join(base, "r4.2.tar.bz2"), "w:bz2") as t:
        t.add(os.path.join(base, "r4.2"), arcname="r4.2")
    with tarfile.open(os.path.join(base, "answers.tar.bz2"), "w:bz2") as t:
        t.add(os.path.join(base, "answers"), arcname="answers")


# ----------------------------------------------------------------------------------------------
# v2: más contexto del hábito de cada empleado (todo causal: solo días previos del mismo empleado)
# ----------------------------------------------------------------------------------------------
FEATURES_CERT_V2 = FEATURES_CERT + [
    "jornada_horas", "desbalance_logon_logoff", "frac_pc_de_otro", "dia_semana",
    "z_n_fuera_horario", "z_n_pc_de_otro", "z_n_pc_nuevo", "z_jornada_horas",
    "frac_prev_dias_fuera_horario", "frac_prev_dias_pc_de_otro", "frac_prev_dias_fin_de_semana",
]


def _z_causal(a: pd.DataFrame, col: str) -> pd.Series:
    x = a[col].astype(float)
    n = a.groupby("user").cumcount().replace(0, np.nan)
    s1 = x.groupby(a["user"]).cumsum() - x
    s2 = (x ** 2).groupby(a["user"]).cumsum() - x ** 2
    media = s1 / n
    sd = np.sqrt((s2 / n - media ** 2).clip(lower=0))
    return ((x - media) / (sd + 0.5)).fillna(0.0)


def agregar_variables_v2(a: pd.DataFrame) -> pd.DataFrame:
    a = a.sort_values(["user", "dia"], kind="mergesort").reset_index(drop=True)
    a["jornada_horas"] = a["ultima_hora"] - a["primera_hora"]
    a["desbalance_logon_logoff"] = a["n_logons"] - (a["n_eventos"] - a["n_logons"])
    a["frac_pc_de_otro"] = a["n_pc_de_otro"] / a["n_logons"].clip(lower=1)
    a["dia_semana"] = a["dia"].dt.dayofweek
    for col in ("n_fuera_horario", "n_pc_de_otro", "n_pc_nuevo", "jornada_horas"):
        a[f"z_{col}"] = _z_causal(a, col)
    n = a.groupby("user").cumcount().replace(0, np.nan)
    for col, nombre in (("n_fuera_horario", "frac_prev_dias_fuera_horario"),
                        ("n_pc_de_otro", "frac_prev_dias_pc_de_otro"),
                        ("fin_de_semana", "frac_prev_dias_fin_de_semana")):
        flag = (a[col] > 0).astype(float)
        a[nombre] = ((flag.groupby(a["user"]).cumsum() - flag) / n).fillna(0.0)
    return a


def folds_por_empleado(a: pd.DataFrame, k: int = 5, seed: int = 42) -> pd.Series:
    """Asigna cada empleado a un fold (0..k-1), estratificando por si tuvo días maliciosos."""
    rng = np.random.default_rng(seed)
    pos = a.groupby("user")["anomalo"].max()
    asign = {}
    for valor in (1, 0):
        us = pos[pos == valor].index.to_numpy().copy()
        rng.shuffle(us)
        for i, u in enumerate(us):
            asign[u] = i % k
    return a["user"].map(asign)
