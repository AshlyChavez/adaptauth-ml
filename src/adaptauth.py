"""
AdaptAuth-ML - pipeline del baseline S8 (Isolation Forest + reglas contextuales).

Implementa el diseño corregido del PI7 (Fase 0):
  - x_i en R^8 (hora cíclica, desviación horaria, historial, frecuencias de país/ASN/dispositivo, RTT)
  - c_i = 0.25 r1 + 0.25 r2 + 0.30 r3 + 0.15 r4 + 0.05 r5
  - s(x) de Isolation Forest (ecuación 1) y reescalado s'(x) (ecuación 1b)
  - R_i = alpha * s'(x_i) + (1 - alpha) * c_i (ecuación 2)
  - Política: acceso directo / OTP / bloqueo, con regla de arranque en frío (P5)
  - Partición temporal 70/15/15 y calibración de alpha, theta1, theta2 solo en validación

Los nombres de columnas siguen el dataset RBA (Wiefling et al., 2022), de modo que el
mismo código corre sobre el archivo real o sobre los datos sintéticos de prueba.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

# --------------------------------------------------------------------------------------
# Columnas del dataset RBA
# --------------------------------------------------------------------------------------
COL_TS = "Login Timestamp"
COL_USER = "User ID"
COL_RTT = "Round-Trip Time [ms]"
COL_IP = "IP Address"
COL_COUNTRY = "Country"
COL_ASN = "ASN"
COL_BROWSER = "Browser Name and Version"
COL_OS = "OS Name and Version"
COL_DEVICE = "Device Type"
COL_SUCCESS = "Login Successful"
COL_ATTACK_IP = "Is Attack IP"          # SOLO etiqueta de evaluación, nunca entrada
COL_ATO = "Is Account Takeover"         # SOLO etiqueta de evaluación, nunca entrada

LABEL_COLUMNS = (COL_ATTACK_IP, COL_ATO)

FEATURES = [
    "hora_sin", "hora_cos", "desv_horaria", "log_n_prev",
    "frec_pais", "frec_asn", "frec_dispositivo", "log_rtt",
]
# Contexto global (v4): no usa etiquetas, solo el pasado de TODA la población
FEATURES_GLOBAL = ["es_movil", "es_desktop", "tasa_fallo_asn", "tasa_fallo_pais", "log_vol_asn"]
# v5: se quitan variables que crecen con el tiempo (log_vol_asn, log_rtt casi sin datos) y se agregan
# señales por IP, por user agent y por fallos de la cuenta
FEATURES_V5 = ["hora_sin", "hora_cos", "desv_horaria", "log_n_prev", "frec_pais", "frec_asn",
               "frec_dispositivo", "es_movil", "es_desktop", "tasa_fallo_asn", "tasa_fallo_pais",
               "tasa_fallo_ua", "ip_tasa_fallo", "ip_log_intentos_prev", "ip_log_usuarios_prev",
               "ip_log_horas_desde_prev", "ip_nueva_para_usuario", "usr_tasa_fallo",
               "usr_fallos_recientes", "usr_log_horas_desde_prev"]
RULES = ["r1_dispositivo_nuevo", "r2_pais_poco_habitual", "r3_viaje_imposible",
         "r4_horario_atipico", "r5_asn_nuevo"]
RULE_WEIGHTS = {"r1_dispositivo_nuevo": 0.25, "r2_pais_poco_habitual": 0.25,
                "r3_viaje_imposible": 0.30, "r4_horario_atipico": 0.15, "r5_asn_nuevo": 0.05}

DECISION_NAMES = {0: "acceso_directo", 1: "otp", 2: "bloqueo"}


@dataclass
class Config:
    seed: int = 42
    n_estimators: int = 100          # t
    max_samples: int = 256           # n
    split: tuple = (0.70, 0.15, 0.15)
    alpha_grid: tuple = (0.3, 0.5, 0.7, 0.85, 1.0)
    theta_grid: tuple = tuple(np.round(np.arange(0.02, 0.951, 0.01), 2))
    max_fpr_theta1: float = 0.05     # RNF02 / CA-C2-02
    max_fpr_theta2: float = 0.01     # bloqueo: tolerancia más estricta
    cold_start_otp: bool = True      # regla P5
    n_test_blocks: int = 10          # bloques temporales para la prueba pareada
    label: str = "anomalo"           # columna de verdad usada para calibrar


# --------------------------------------------------------------------------------------
# 1. Datos
# --------------------------------------------------------------------------------------
def _open_csv(path: str):
    if str(path).endswith(".zip"):          # se lee directo del .zip, sin descomprimir ~9 GB
        import zipfile
        zf = zipfile.ZipFile(path)
        member = next(n for n in zf.namelist() if n.endswith(".csv"))
        return zf.open(member)
    return path


def load_rba(path: str, nrows: int | None = None, user_pct: float | None = None,
             chunksize: int = 2_000_000) -> pd.DataFrame:
    """
    Carga el CSV real de RBA (rba-dataset.csv o el .zip de Zenodo).
      - nrows: primeras N filas (ventana de tiempo corta: muchos usuarios sin historial).
      - user_pct: muestreo POR USUARIO sobre todo el archivo (p. ej. 5 = 5 % de usuarios con
        TODOS sus accesos). Conserva historiales completos; es la opción recomendada.
    """
    usecols = [COL_TS, COL_USER, COL_RTT, COL_IP, COL_COUNTRY, COL_ASN, COL_BROWSER,
               COL_OS, COL_DEVICE, COL_SUCCESS, COL_ATTACK_IP, COL_ATO]
    if str(path).endswith(".parquet"):      # muestra ya filtrada y guardada (lectura en segundos)
        df = pd.read_parquet(path)
        df[COL_TS] = pd.to_datetime(df[COL_TS])
        return df
    if user_pct:
        parts = []
        for ch in pd.read_csv(_open_csv(path), usecols=usecols, chunksize=chunksize):
            h = pd.util.hash_pandas_object(ch[COL_USER], index=False).to_numpy() % 10000
            parts.append(ch[h < user_pct * 100])
        df = pd.concat(parts, ignore_index=True)
    else:
        df = pd.read_csv(_open_csv(path), usecols=usecols, nrows=nrows)
    df[COL_TS] = pd.to_datetime(df[COL_TS])
    for col in (COL_SUCCESS, COL_ATTACK_IP, COL_ATO):
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1"])
    return df


CTX_COLS = ["tasa_fallo_asn", "tasa_fallo_pais", "tasa_fallo_ua", "ip_tasa_fallo",
            "ip_log_intentos_prev", "ip_log_usuarios_prev", "ip_log_horas_desde_prev"]


def build_full_context_sample(path: str, user_pct: float, chunksize: int = 2_000_000,
                              m: float = 20.0, verbose: bool = True) -> pd.DataFrame:
    """
    Recorre TODO el archivo en orden temporal y calcula el contexto de red (IP, ASN, país,
    user agent) con el tráfico completo, como lo vería el sistema en producción. Devuelve solo
    los accesos del user_pct % de usuarios, con columnas ctx_* listas para reemplazar las
    calculadas dentro de la muestra. Todo es causal: cada evento solo ve eventos anteriores.
    """
    usecols = [COL_TS, COL_USER, COL_RTT, COL_IP, COL_COUNTRY, COL_ASN, COL_BROWSER,
               COL_OS, COL_DEVICE, COL_SUCCESS, COL_ATTACK_IP, COL_ATO]
    empty = lambda: pd.Series(dtype="float64")
    state = {k: (empty(), empty()) for k in ("asn", "pais", "ua", "ip")}
    ip_users, ip_last = empty(), empty()
    seen = np.empty(0, dtype=np.uint64)
    glob_n = glob_f = 0.0
    t_ref, parts, fila0 = None, [], 0
    t0 = time.time()
    for i, ch in enumerate(pd.read_csv(_open_csv(path), usecols=usecols, chunksize=chunksize)):
        ch = ch.reset_index(drop=True)
        ch["_fila"] = np.arange(fila0, fila0 + len(ch)); fila0 += len(ch)
        ts = pd.to_datetime(ch[COL_TS])
        t_ref = ts.min() if t_ref is None else t_ref
        horas = pd.Series((ts - t_ref).dt.total_seconds().to_numpy() / 3600.0)
        ok = ch[COL_SUCCESS]
        ok = ok.astype(str).str.lower().isin(["true", "1"]) if ok.dtype == object else ok.astype(bool)
        fail = pd.Series((~ok.to_numpy()).astype(int))
        keys = {"asn": ch[COL_ASN].astype(str), "pais": ch[COL_COUNTRY].astype(str),
                "ua": ch[COL_BROWSER].astype(str) + "|" + ch[COL_OS].astype(str) + "|" + ch[COL_DEVICE].astype(str),
                "ip": ch[COL_IP].astype(str)}
        f_cum = fail.cumsum().to_numpy() - fail.to_numpy()
        p0 = (glob_f + f_cum + 1.0) / (glob_n + np.arange(len(ch)) + 4.0)
        prior = {}
        for k, key in keys.items():
            cnt, fl = state[k]
            n_k = key.map(cnt).fillna(0).to_numpy() + fail.groupby(key).cumcount().to_numpy()
            f_k = key.map(fl).fillna(0).to_numpy() + (fail.groupby(key).cumsum() - fail).to_numpy()
            prior[k] = (n_k, f_k)
            state[k] = (cnt.add(key.value_counts(), fill_value=0), fl.add(fail.groupby(key).sum(), fill_value=0))
        out = {}
        for k, name in (("asn", "tasa_fallo_asn"), ("pais", "tasa_fallo_pais"), ("ua", "tasa_fallo_ua")):
            n_k, f_k = prior[k]
            out[name] = (f_k + m * p0) / (n_k + m)
        n_ip, f_ip = prior["ip"]
        out["ip_tasa_fallo"] = (f_ip + 5.0 * out["tasa_fallo_asn"]) / (n_ip + 5.0)
        out["ip_log_intentos_prev"] = np.log1p(n_ip)
        h = pd.util.hash_pandas_object(ch[[COL_IP, COL_USER]].astype(str), index=False).to_numpy()
        first = pd.Series(((~pd.Series(h).duplicated()).to_numpy() & ~np.isin(h, seen)).astype(int))
        users_prev = keys["ip"].map(ip_users).fillna(0).to_numpy() + (first.groupby(keys["ip"]).cumsum() - first).to_numpy()
        out["ip_log_usuarios_prev"] = np.log1p(users_prev)
        ip_users = ip_users.add(first.groupby(keys["ip"]).sum(), fill_value=0)
        seen = np.union1d(seen, h[first.to_numpy() == 1])
        prev = horas.groupby(keys["ip"]).shift(1).fillna(keys["ip"].map(ip_last))
        out["ip_log_horas_desde_prev"] = np.log1p((horas - prev).fillna(24 * 365).clip(lower=0)).to_numpy()
        last = horas.groupby(keys["ip"]).max()
        ip_last = pd.concat([ip_last, last]); ip_last = ip_last[~ip_last.index.duplicated(keep="last")]
        glob_n += len(ch); glob_f += float(fail.sum())
        keep = (pd.util.hash_pandas_object(ch[COL_USER], index=False).to_numpy() % 10000) < user_pct * 100
        sub = ch.loc[keep].copy()
        for name, arr in out.items():
            sub["ctx_" + name] = np.asarray(arr)[keep]
        parts.append(sub)
        if verbose:
            print(f"  bloque {i + 1}: {fila0:,} intentos leídos | IPs vistas {len(ip_users):,} | {time.time() - t0:.0f}s", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df[COL_TS] = pd.to_datetime(df[COL_TS])
    for col in (COL_SUCCESS, COL_ATTACK_IP, COL_ATO):
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1"])
    return df


def apply_full_context(d: pd.DataFrame) -> bool:
    """Si la muestra trae columnas ctx_* (tráfico completo), reemplaza las calculadas en la muestra."""
    if not all(("ctx_" + c) in d.columns for c in CTX_COLS):
        return False
    for c in CTX_COLS:
        d[c] = d["ctx_" + c].to_numpy()
    return True


def generate_synthetic_rba(n_users: int = 2000, days: int = 365, seed: int = 42) -> pd.DataFrame:
    """
    Datos SINTÉTICOS con la misma estructura de columnas que RBA. Sirven solo para
    probar el pipeline de punta a punta; sus métricas NO son resultados de la tesis.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2020-02-03")
    countries = np.array(["NO", "SE", "DE", "GB", "US", "FR", "ES", "PL", "BR", "IN"])
    home_p = np.array([0.55, 0.08, 0.07, 0.06, 0.06, 0.05, 0.04, 0.04, 0.03, 0.02])
    asn_pool = {c: [10000 + 100 * i + k for k in range(6)] for i, c in enumerate(countries)}
    attacker_countries = np.array(["US", "RU", "CN", "VN", "BR", "NL", "UA", "ID"])
    attacker_asns = [64000 + k for k in range(60)]       # proveedores de hosting
    browsers = ["Chrome 88", "Firefox 85", "Safari 14", "Edge 88", "Chrome Mobile 88",
                "Samsung Internet 13", "Opera 74"]
    oses = ["Windows 10", "Mac OS X 10.15", "iOS 14.4", "Android 10", "Linux", "Android 11"]
    devices = ["desktop", "mobile", "tablet"]
    base_rtt = {c: v for c, v in zip(countries, [30, 45, 60, 65, 140, 70, 90, 70, 220, 260])}

    rows = []
    for uid in range(n_users):
        home = rng.choice(countries, p=home_p)
        user_asns = rng.choice(asn_pool[home], size=rng.integers(1, 3), replace=False)
        user_ips = [f"10.{uid % 250}.{k}.{rng.integers(255)}" for k in range(rng.integers(1, 4))]
        n_dev = rng.integers(1, 4)
        user_devs = [(rng.choice(browsers), rng.choice(oses), rng.choice(devices, p=[.5, .42, .08]))
                     for _ in range(n_dev)]
        pref_hour = np.clip(rng.normal(13, 3), 6, 22)
        n_events = int(np.clip(rng.lognormal(4.2, 0.9), 2, 900))
        times = np.sort(rng.uniform(0, days, n_events))
        for t in times:
            day = np.floor(t)
            hour = (rng.normal(pref_hour, 2.5)) % 24
            ts = start + pd.Timedelta(days=float(day)) + pd.Timedelta(hours=float(hour))
            ctry, asn = home, rng.choice(user_asns)
            if rng.random() < 0.02:                       # viaje legítimo
                ctry = rng.choice(countries)
                asn = rng.choice(asn_pool[ctry])
            elif rng.random() < 0.08:                     # red móvil / otra red del país
                asn = rng.choice(asn_pool[home])
            dev = user_devs[rng.integers(len(user_devs))]
            if rng.random() < 0.03:                       # dispositivo nuevo legítimo
                dev = (rng.choice(browsers), rng.choice(oses), rng.choice(devices))
                user_devs.append(dev)
            rtt = rng.lognormal(np.log(base_rtt[ctry]), 0.35)
            ip = (f"100.64.{rng.integers(40)}.{rng.integers(255)}" if rng.random() < 0.15   # NAT móvil compartida
                  else user_ips[rng.integers(len(user_ips))])
            rows.append((ts, uid, rtt, ctry, asn, dev[0], dev[1], dev[2],
                         rng.random() > 0.04, 0, 0, ip))

    n_legit = len(rows)
    # Relleno de credenciales (credential stuffing) desde IPs de ataque
    n_attack = int(n_legit * 0.08)
    for _ in range(n_attack):
        uid = int(rng.integers(n_users))
        ts = start + pd.Timedelta(days=float(rng.uniform(0, days)))
        ctry = rng.choice(attacker_countries)
        rtt = rng.lognormal(np.log(180), 0.6)
        rows.append((ts, uid, rtt, ctry, rng.choice(attacker_asns), rng.choice(browsers),
                     rng.choice(oses), "desktop", rng.random() < 0.02, 1, 0,
                     f"203.0.{rng.integers(150)}.1"))
    # Toma de cuenta (account takeover): acceso exitoso del atacante
    n_ato = int(n_legit * 0.002)
    for _ in range(n_ato):
        uid = int(rng.integers(n_users))
        ts = start + pd.Timedelta(days=float(rng.uniform(30, days)))
        ctry = rng.choice(attacker_countries)
        rtt = rng.lognormal(np.log(180), 0.6)
        rows.append((ts, uid, rtt, ctry, rng.choice(attacker_asns), rng.choice(browsers),
                     rng.choice(oses), rng.choice(devices), True, 1, 1, f"203.0.{rng.integers(150)}.1"))

    df = pd.DataFrame(rows, columns=[COL_TS, COL_USER, COL_RTT, COL_COUNTRY, COL_ASN,
                                     COL_BROWSER, COL_OS, COL_DEVICE, COL_SUCCESS,
                                     COL_ATTACK_IP, COL_ATO, COL_IP])
    return df


# --------------------------------------------------------------------------------------
# 2. C1 - Ingesta y feature engineering (solo usa el pasado de cada usuario)
# --------------------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye x_i (R^8), las reglas r1..r5 y c_i. El historial de cada usuario se forma
    SOLO con sus inicios de sesión exitosos anteriores al evento, de modo que no hay
    fuga temporal ni se usan las etiquetas de ataque como entrada.
    """
    d = df.sort_values(COL_TS, kind="mergesort").reset_index(drop=True).copy()
    d["anomalo"] = ((d[COL_ATTACK_IP] == 1) | (d[COL_ATO] == 1)).astype(int)
    d["dispositivo"] = (d[COL_BROWSER].astype(str) + "|" + d[COL_OS].astype(str) + "|"
                        + d[COL_DEVICE].astype(str))
    succ = d[COL_SUCCESS].astype(bool).astype(int)
    d["_succ"] = succ

    hour = d[COL_TS].dt.hour + d[COL_TS].dt.minute / 60.0
    ang = 2 * np.pi * hour / 24.0
    d["hora_sin"], d["hora_cos"] = np.sin(ang), np.cos(ang)

    g = d.groupby(COL_USER)
    n_prev = g["_succ"].cumsum() - succ                   # accesos exitosos previos
    d["n_prev"] = n_prev
    d["log_n_prev"] = np.log1p(n_prev)

    def prior_share(key: str) -> pd.Series:
        same = d.groupby([COL_USER, key])["_succ"].cumsum() - succ
        return np.where(n_prev > 0, same / n_prev.clip(lower=1), 0.0)

    d["frec_pais"] = prior_share(COL_COUNTRY)
    d["frec_asn"] = prior_share(COL_ASN)
    d["frec_dispositivo"] = prior_share("dispositivo")

    # Hora habitual: media circular de los accesos exitosos previos
    s_cum = (d["hora_sin"] * succ).groupby(d[COL_USER]).cumsum() - d["hora_sin"] * succ
    c_cum = (d["hora_cos"] * succ).groupby(d[COL_USER]).cumsum() - d["hora_cos"] * succ
    mean_ang = np.arctan2(s_cum, c_cum)
    diff = np.abs(np.angle(np.exp(1j * (ang - mean_ang))))     # 0..pi
    desv_h = diff * 24 / (2 * np.pi)                            # 0..12 h
    d["desv_horas"] = np.where(n_prev > 0, desv_h, np.nan)
    d["desv_horaria"] = np.where(n_prev > 0, desv_h / 12.0, 0.5)

    d["log_rtt"] = np.log1p(d[COL_RTT].astype(float).fillna(d[COL_RTT].median()))

    # Último acceso exitoso previo (para viaje imposible)
    d["_ctry_s"] = d[COL_COUNTRY].where(succ == 1)
    d["_ts_s"] = d[COL_TS].where(succ == 1)
    # Vectorizado (sin lambda por usuario) para soportar millones de filas
    d["prev_pais"] = d.groupby(COL_USER)["_ctry_s"].shift(1).groupby(d[COL_USER]).ffill()
    d["prev_ts"] = d.groupby(COL_USER)["_ts_s"].shift(1).groupby(d[COL_USER]).ffill()
    horas_desde_prev = (d[COL_TS] - d["prev_ts"]).dt.total_seconds() / 3600.0

    hist = n_prev >= 1
    d["r1_dispositivo_nuevo"] = (hist & (d["frec_dispositivo"] == 0)).astype(int)
    d["r2_pais_poco_habitual"] = (hist & (d["frec_pais"] < 0.2)).astype(int)
    d["r3_viaje_imposible"] = (d["prev_pais"].notna() & (d["prev_pais"] != d[COL_COUNTRY])
                               & (horas_desde_prev < 2)).astype(int)
    d["r4_horario_atipico"] = ((n_prev >= 5) & (d["desv_horas"] > 4)).astype(int)
    d["r5_asn_nuevo"] = (hist & (d["frec_asn"] == 0)).astype(int)
    d["c_i"] = sum(RULE_WEIGHTS[r] * d[r] for r in RULES)
    d["arranque_frio"] = (n_prev == 0).astype(int)

    # ---- Contexto global (v4), causal: cada evento solo ve eventos anteriores ----
    dev = d[COL_DEVICE].astype(str).str.lower()
    d["es_movil"] = (dev == "mobile").astype(int)
    d["es_desktop"] = (dev == "desktop").astype(int)
    fail = 1 - succ
    g_n = np.arange(len(d))
    g_f = fail.cumsum().to_numpy() - fail.to_numpy()
    p0 = (g_f + 1.0) / (g_n + 4.0)                      # tasa global de fallo previa (suavizada)
    m = 20.0                                            # fuerza del suavizado bayesiano
    for key, name in ((COL_ASN, "tasa_fallo_asn"), (COL_COUNTRY, "tasa_fallo_pais")):
        n_k = d.groupby(key).cumcount().to_numpy()
        f_k = fail.groupby(d[key]).cumsum().to_numpy() - fail.to_numpy()
        d[name] = (f_k + m * p0) / (n_k + m)
    d["log_vol_asn"] = np.log1p(d.groupby(COL_ASN).cumcount())

    # ---- v5: IP, user agent y fallos de la cuenta (causales, sin etiquetas) ----
    horas = (d[COL_TS] - d[COL_TS].min()).dt.total_seconds() / 3600.0
    ip = d[COL_IP].astype(str)
    ip_n = d.groupby(ip).cumcount()
    ip_f = fail.groupby(ip).cumsum() - fail
    d["ip_tasa_fallo"] = (ip_f + 5.0 * d["tasa_fallo_asn"]) / (ip_n + 5.0)   # previa jerárquica: ASN
    d["ip_log_intentos_prev"] = np.log1p(ip_n)
    first_pair = (~d.duplicated([COL_IP, COL_USER])).astype(int)
    d["ip_log_usuarios_prev"] = np.log1p(first_pair.groupby(ip).cumsum() - first_pair)
    d["ip_log_horas_desde_prev"] = np.log1p(horas.groupby(ip).diff().fillna(24 * 365).clip(lower=0))
    pair_ok_prev = d.groupby([COL_USER, COL_IP])["_succ"].cumsum() - succ
    d["ip_nueva_para_usuario"] = (pair_ok_prev == 0).astype(int)
    ua = d["dispositivo"]
    ua_n = d.groupby(ua).cumcount()
    ua_f = fail.groupby(ua).cumsum() - fail
    d["tasa_fallo_ua"] = (ua_f + m * p0) / (ua_n + m)
    u_n = d.groupby(COL_USER).cumcount()
    u_f = fail.groupby(d[COL_USER]).cumsum() - fail                     # fallos previos de la cuenta
    d["usr_tasa_fallo"] = (u_f + 3.0 * p0) / (u_n + 3.0)
    d["usr_fallos_recientes"] = u_f - u_f.groupby(d[COL_USER]).shift(5).fillna(0)   # en sus 5 intentos previos
    d["usr_log_horas_desde_prev"] = np.log1p(horas.groupby(d[COL_USER]).diff().fillna(24 * 365).clip(lower=0))

    return d.drop(columns=["_succ", "_ctry_s", "_ts_s"])


def diagnose(raw: pd.DataFrame, d: pd.DataFrame) -> dict:
    """
    Análisis exploratorio orientado a la detección: ¿qué variables separan de verdad los
    accesos anómalos de los legítimos? (Etapa 1 del procedimiento metodológico).
    """
    from sklearn.metrics import roc_auc_score
    y = d["anomalo"].to_numpy()
    lab = np.where(y == 1, "anomalo", "legitimo")
    per_user = d.groupby(COL_USER).size()

    def auc(col, mask=None):
        m = np.ones(len(d), bool) if mask is None else mask
        yy, xx = y[m], d.loc[m, col].astype(float).fillna(0).to_numpy()
        return round(float(roc_auc_score(yy, xx)), 3) if 0 < yy.sum() < len(yy) else None

    hist = (d["n_prev"] >= 1).to_numpy()
    cols = FEATURES + FEATURES_GLOBAL + RULES + ["c_i", COL_RTT]

    def top_share(col, k=5):
        out = {}
        for name in ("legitimo", "anomalo"):
            s = d.loc[lab == name, col].astype(str).value_counts(normalize=True).head(k)
            out[name] = {i: round(float(v), 3) for i, v in s.items()}
        return out

    legit_asn = set(d.loc[y == 0, COL_ASN])
    legit_users = set(d.loc[(y == 0) & d[COL_SUCCESS].astype(bool), COL_USER])
    atk = d[y == 1]
    return {
        "archivo_ordenado_por_fecha": bool(raw[COL_TS].is_monotonic_increasing),
        "rango_fechas": [str(d[COL_TS].min()), str(d[COL_TS].max())],
        "eventos": int(len(d)), "usuarios": int(per_user.size),
        "eventos_por_usuario_p50_p90_max": [int(per_user.quantile(.5)), int(per_user.quantile(.9)),
                                            int(per_user.max())],
        "tasa_anomalos": round(float(y.mean()), 4), "n_ato": int(d[COL_ATO].sum()),
        "exito_login": {k: round(float(d.loc[lab == k, COL_SUCCESS].astype(bool).mean()), 4)
                        for k in ("legitimo", "anomalo")},
        "primer_acceso": {k: round(float((d.loc[lab == k, "n_prev"] == 0).mean()), 4)
                          for k in ("legitimo", "anomalo")},
        "anomalos_sobre_usuarios_con_historial_legitimo": round(float(atk[COL_USER].isin(legit_users).mean()), 4),
        "anomalos_con_asn_visto_en_legitimos": round(float(atk[COL_ASN].isin(legit_asn).mean()), 4),
        "pais_top5": top_share(COL_COUNTRY), "tipo_dispositivo_top5": top_share(COL_DEVICE),
        "asn_top5": top_share(COL_ASN),
        "auc_univariada_todos": {c: auc(c) for c in cols},
        "auc_univariada_con_historial": {c: auc(c, hist) for c in cols},
        "tasa_reglas": {r: {k: round(float(d.loc[lab == k, r].mean()), 4) for k in ("legitimo", "anomalo")}
                        for r in RULES},
        "nota_auc": "0.5 = no separa; > 0.5 = valores altos indican anomalía; < 0.5 = relación invertida",
    }


def temporal_split(d: pd.DataFrame, split=(0.70, 0.15, 0.15)):
    n = len(d)
    a, b = int(n * split[0]), int(n * (split[0] + split[1]))
    return d.iloc[:a].copy(), d.iloc[a:b].copy(), d.iloc[b:].copy()


# --------------------------------------------------------------------------------------
# 3. C2 / C3 / C4 - detección, consolidación y decisión
# --------------------------------------------------------------------------------------
def train_iforest(train: pd.DataFrame, cfg: Config, features=None) -> IsolationForest:
    features = features or FEATURES
    model = IsolationForest(n_estimators=cfg.n_estimators, max_samples=cfg.max_samples,
                            contamination="auto", random_state=cfg.seed, n_jobs=-1)
    model.fit(train[features].to_numpy())
    return model


def anomaly_score(model: IsolationForest, d: pd.DataFrame, features=None) -> np.ndarray:
    """Ecuación 1: s(x, n) = 2^(-E[h(x)]/c(n)). En scikit-learn, score_samples = -s(x, n)."""
    return -model.score_samples(d[features or FEATURES].to_numpy())


def rescale(s: np.ndarray, s_ref: float) -> np.ndarray:
    """Ecuación 1b: s'(x) = max(0, (s - s_ref) / (1 - s_ref))."""
    return np.clip((s - s_ref) / (1.0 - s_ref), 0.0, 1.0)


def risk_score(s_prime: np.ndarray, c: np.ndarray, alpha: float) -> np.ndarray:
    """Ecuación 2: R_i = alpha * s'(x_i) + (1 - alpha) * c_i."""
    return alpha * s_prime + (1.0 - alpha) * c


def decide(R: np.ndarray, theta1: float, theta2: float, cold: np.ndarray | None = None,
           cold_start_otp: bool = True) -> np.ndarray:
    """Reglas P2, P3 y P5. 0 = acceso directo, 1 = OTP, 2 = bloqueo."""
    dec = np.where(R > theta2, 2, np.where(R > theta1, 1, 0))
    if cold_start_otp and cold is not None:
        dec = np.where((cold == 1) & (dec == 0), 1, dec)
    return dec


# --------------------------------------------------------------------------------------
# 4. Métricas (dos análisis binarios sobre la decisión de tres niveles)
# --------------------------------------------------------------------------------------
def binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1,
                fpr=fpr, fnr=fnr, mcc=mcc)


def evaluate_decisions(y: np.ndarray, dec: np.ndarray) -> dict:
    """Análisis theta1: positivo = OTP o bloqueo. Análisis theta2: positivo = bloqueo."""
    return {"analisis_theta1": binary_metrics(y, (dec >= 1).astype(int)),
            "analisis_theta2": binary_metrics(y, (dec == 2).astype(int))}


def confusion_3x2(y: np.ndarray, dec: np.ndarray) -> pd.DataFrame:
    tab = pd.crosstab(pd.Series(dec).map(DECISION_NAMES).rename("decision"),
                      pd.Series(y).map({0: "legitimo", 1: "anomalo"}).rename("etiqueta_real"))
    return tab.reindex(index=list(DECISION_NAMES.values()), columns=["legitimo", "anomalo"],
                       fill_value=0)


# --------------------------------------------------------------------------------------
# 5. Calibración en validación (el conjunto de prueba no se usa aquí)
# --------------------------------------------------------------------------------------
def calibrate(val: pd.DataFrame, s_prime: np.ndarray, cfg: Config, alpha_grid) -> dict:
    """
    Búsqueda en malla: maximiza F1 del análisis theta1 sujeto a FPR <= max_fpr_theta1.
    theta2 es el menor umbral > theta1 cuyo FPR de bloqueo <= max_fpr_theta2.
    """
    y = val[cfg.label].to_numpy()
    cold = val["arranque_frio"].to_numpy()
    c = val["c_i"].to_numpy()
    best, fallback = None, None
    for alpha in alpha_grid:
        R = risk_score(s_prime, c, alpha)
        for t1 in cfg.theta_grid:
            dec = decide(R, t1, 1.01, cold, cfg.cold_start_otp)
            m = binary_metrics(y, (dec >= 1).astype(int))
            cand = dict(alpha=float(alpha), theta1=float(t1), f1_val=m["f1"], fpr_val=m["fpr"])
            if m["fpr"] <= cfg.max_fpr_theta1 and (best is None or m["f1"] > best["f1_val"]):
                best = cand
            # respaldo: menor FPR posible y, a igualdad, mayor F1
            if fallback is None or (m["fpr"], -m["f1"]) < (fallback["fpr_val"], -fallback["f1_val"]):
                fallback = cand
    if best is None:   # ninguna combinación cumple FPR <= umbral: se declara y se usa el respaldo
        best = dict(fallback, restriccion_fpr_incumplida=True)
    R = risk_score(s_prime, c, best["alpha"])
    best["theta2"] = 0.95
    for t2 in cfg.theta_grid:
        if t2 <= best["theta1"]:
            continue
        m = binary_metrics(y, (R > t2).astype(int))
        if m["fpr"] <= cfg.max_fpr_theta2:
            best["theta2"] = float(t2)
            break
    return best


# --------------------------------------------------------------------------------------
# 6. Prueba pareada por bloques temporales (baseline vs AdaptAuth-ML)
# --------------------------------------------------------------------------------------
def paired_block_test(test: pd.DataFrame, dec_a: np.ndarray, dec_b: np.ndarray,
                      y_col: str, n_blocks: int) -> dict:
    """
    Divide el conjunto de prueba en n bloques temporales consecutivos y compara el F1
    (análisis theta1) de ambas condiciones sobre los MISMOS bloques: muestras pareadas.
    """
    y = test[y_col].to_numpy()
    idx = np.array_split(np.arange(len(test)), n_blocks)
    f1_a = np.array([binary_metrics(y[i], (dec_a[i] >= 1).astype(int))["f1"] for i in idx])
    f1_b = np.array([binary_metrics(y[i], (dec_b[i] >= 1).astype(int))["f1"] for i in idx])
    diff = f1_b - f1_a
    shapiro_p = float(stats.shapiro(diff).pvalue) if np.ptp(diff) > 0 else 1.0
    if shapiro_p > 0.05:
        t = stats.ttest_rel(f1_b, f1_a)
        test_name, stat, p = "t de Student pareada", float(t.statistic), float(t.pvalue)
    else:
        w = stats.wilcoxon(f1_b, f1_a)
        test_name, stat, p = "Wilcoxon de rangos con signo", float(w.statistic), float(w.pvalue)
    sd = diff.std(ddof=1)
    dz = float(diff.mean() / sd) if sd > 0 else float("inf")
    return dict(n_bloques=n_blocks, f1_baseline=f1_a.tolist(), f1_adaptauth=f1_b.tolist(),
                media_baseline=float(f1_a.mean()), media_adaptauth=float(f1_b.mean()),
                de_baseline=float(f1_a.std(ddof=1)), de_adaptauth=float(f1_b.std(ddof=1)),
                shapiro_p_diferencias=shapiro_p, prueba=test_name, estadistico=stat, p_valor=p,
                d_cohen_dz=dz)


# --------------------------------------------------------------------------------------
# 7. Latencia (RNF03 / CA-C2-03)
# --------------------------------------------------------------------------------------
def inference_latency_ms(model, d: pd.DataFrame, n: int = 300, seed: int = 0, features=None) -> dict:
    rng = np.random.default_rng(seed)
    X = d[features or FEATURES].to_numpy()
    score = model.score_samples if hasattr(model, "score_samples") else model.predict_proba
    times = []
    for i in rng.integers(0, len(X), n):
        t0 = time.perf_counter()
        score(X[i:i + 1])
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return dict(p50_ms=float(np.percentile(t, 50)), p95_ms=float(np.percentile(t, 95)),
                p99_ms=float(np.percentile(t, 99)), media_ms=float(t.mean()))


def to_json(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, Config):
            return asdict(o)
        raise TypeError(type(o))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=conv)
