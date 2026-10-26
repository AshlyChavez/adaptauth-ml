# AdaptAuth-ML

Modelo de autenticación adaptativa basado en aprendizaje automático para la detección de accesos anómalos en PYMES del sector de servicios empresariales.

Repositorio de la tesis *Modelo de autenticación adaptativa basado en Machine Learning para la detección de accesos anómalos en PYMES del sector de servicios empresariales en Lima, 2026*. Universidad Nacional Mayor de San Marcos, Facultad de Ingeniería de Sistemas e Informática, Tesis II.

- **Autora:** Ashly Herminia Jasmín Chávez Marón
- **Asesor:** Ing. Daniel Alfonso Quinto Pazce
- **Método:** Design Science Research con evaluación cuasi-experimental anidada (línea base frente a AdaptAuth-ML)
- **Tipo de artefacto:** modelo computacional / pipeline (aporte conceptual, Grupo A)
- **Versión:** 11 (Isolation Forest evaluado y retirado del núcleo; θ2 con FPR de bloqueo ≤ 0.1 %)

## Descripción del modelo

Cuando un usuario ya validó su contraseña, AdaptAuth-ML hace tres cosas:
- construye un vector de variables contextuales con información previa al evento;
- estima un score de riesgo R ∈ [0, 1];
- asigna una de tres respuestas: **acceso directo**, **verificación OTP** o **bloqueo**.

| Componente | Función | Implementación |
|---|---|---|
| C1. Ingesta y variables | Variables causales de hora, red, ubicación, dispositivo e historial | `src/adaptauth.py`, `src/cert.py` |
| C2. Motor de detección | Clasificador supervisado sobre el vector de variables | HistGradientBoostingClassifier (`src/modelo_adaptauth.py`) |
| C3. Score de riesgo | R = P(ataque \| x) | `predict_proba` del clasificador |
| C4. Decisión adaptativa | Umbrales θ1 y θ2 calibrados en validación; reglas P1 a P3 | `calibrate`, `decide` en `src/adaptauth.py` |
| C5. Monitoreo y explicación | Registro de decisiones y top-3 variables por alerta | SHAP, `run_requisitos_sl.py` |
| C6. Reentrenamiento | Reentrenamiento mensual sin bloqueos no confirmados | `run_requisitos_sl.py` (prueba CA-C6-02) |

**Ecuaciones:**

- **Score de riesgo:** R = P(ataque | x)
- **Decisión:** acceso directo si R ≤ θ1, OTP si θ1 < R ≤ θ2 y bloqueo si R > θ2.

**Calibración:**
- θ1 maximiza F1 con FPR ≤ 5 %.
- θ2 es el menor umbral mayor que θ1 con FPR de bloqueo ≤ 0.1 %.
- Ambos umbrales se calibran solo con el conjunto de validación.

**Isolation Forest.** Se evaluó como señal adicional del clasificador (modelo apilado de la versión 10, `src/modelo_apilado.py`). No mejoró el F1 ni el AUC-PR, ni siquiera con pocas etiquetas (arranque en frío), por lo que se retiró del núcleo y se conserva solo como línea base y ablación.

## Estructura

```
adaptauth-ml/
├── src/
│   ├── adaptauth.py            # núcleo: reescalado, score, calibración, decisión, métricas, pruebas pareadas
│   ├── cert.py                 # carga y variables de CERT r4.2 (unidad usuario-día)
│   ├── modelo_adaptauth.py     # AdaptAuth-ML v11: HistGradientBoosting
│   ├── modelo_apilado.py       # ablación v10: Isolation Forest + HistGradientBoosting apilado
│   └── figures.py              # figuras de resultados
├── run_suspiciouslogin.py      # experimento principal en SuspiciousLogin
├── run_requisitos_sl.py        # verificación de requisitos no funcionales
├── run_analisis_complementario.py  # ablación con IF, pruebas pareadas, bootstrap por usuario, V de Aiken
├── run_levantamiento.py        # arranque en frío y recalibración de θ2
├── run_cert_v2.py              # CERT r4.2 con validación cruzada por empleado (--frio: arranque en frío)
├── notebooks/                  # cuadernos de Google Colab
│   └── historico/              # versiones previas (dataset RBA)
├── results_sl/                 # resultados reales de SuspiciousLogin
├── results_requisitos_sl/      # matriz de cumplimiento, registro de eventos y explicaciones
├── results_cert_v11/           # resultados de CERT r4.2, versión 11
├── results_cert_v2/            # resultados de CERT r4.2, versión 10 (histórico)
├── results_complementarios/    # ablación, pruebas pareadas, bootstrap, curvas PR y tabla de V de Aiken
├── results_levantamiento/      # arranque en frío y recalibración de θ2 en SuspiciousLogin
├── historico_rba/              # iteraciones v1 a v4 con el dataset RBA y CERT v1 (bitácora)
├── docs/figuras/               # figuras del informe de avance
└── data/                       # instrucciones para obtener los datos (los datos no se versionan)
```

## Datos

Los conjuntos de datos no se incluyen en el repositorio. En `data/README.md` está la forma de obtenerlos. SuspiciousLogin se usa en su versión 1 (commit `68cf94b` del repositorio fuente); el autor publicó una versión 2 con otras columnas el 22-10-2026.

| Dataset | Uso en el modelo | Licencia |
|---|---|---|
| SuspiciousLogin (Google Workspace, 73 528 inicios de sesión, 36 691 usuarios) | Señales de identidad, red y ubicación | CC BY-NC 4.0 |
| CERT Insider Threat r4.2 (CMU, 1 000 empleados, 330 452 días-empleado) | Historial diario del empleado | CC BY 4.0 |

## Ejecución

```bash
pip install -r requirements.txt

# SuspiciousLogin: experimento principal y verificación de requisitos
python run_suspiciouslogin.py --data data/suspicious_logins_public_v1.csv
python run_requisitos_sl.py   --data data/suspicious_logins_public_v1.csv
python run_analisis_complementario.py --data data/suspicious_logins_public_v1.csv
python run_levantamiento.py --data data/suspicious_logins_public_v1.csv

# CERT r4.2: carpeta con logon.csv, LDAP/ y answers/
python run_cert_v2.py --dir data/CERT_r4.2 --frio
```

**Demostración en Colab:** `notebooks/00_AdaptAuthML_Demo.ipynb` descarga este repositorio y el dataset, ejecuta los experimentos (pasos 1 a 7) y copia los resultados de CERT a Google Drive ([abrir en Colab](https://colab.research.google.com/github/AshlyChavez/adaptauth-ml/blob/main/notebooks/00_AdaptAuthML_Demo.ipynb)).

En Google Colab, los cuadernos de `notebooks/` montan Google Drive, leen los datos desde la carpeta `AdaptAuth-ML` y guardan los resultados en la misma carpeta.

## Resultados principales

**SuspiciousLogin** (prueba: mayo y junio, n = 20 852, 7.6 % sospechosos)

| Modelo | AUC-ROC | AUC-PR | F1 | Precisión | Recall | FPR |
|---|---|---|---|---|---|---|
| Reglas estáticas | 0.549 | 0.090 | 0.009 | 0.250 | 0.004 | 0.11 % |
| Isolation Forest | 0.630 | 0.109 | 0.044 | 0.051 | 0.038 | 5.80 % |
| Apilado con IF (v10, ablación) | 0.881 | 0.640 | 0.620 | 0.814 | 0.501 | 0.93 % |
| **AdaptAuth-ML (v11)** | **0.859** | **0.634** | **0.628** | **0.851** | **0.498** | **0.71 %** |

- **Prueba t pareada** en 14 grupos: frente a las reglas, t(13) = 33.87, p < 0.001, dz = 9.05.
- **Bootstrap por usuario** (1 000 réplicas): diferencia de F1 frente a las reglas de 0.620 (IC 95 %: 0.568 a 0.665).
- **Ablación:** el apilado con IF frente a AdaptAuth-ML, t(13) = −2.10, p = 0.056; bootstrap −0.008 (IC 95 %: −0.014 a −0.002).
- **Bloqueo (θ2):** 9 legítimos bloqueados de 19 275 (FPR de bloqueo 0.05 %); con la regla anterior (≤ 1 %) eran 85.
- **Arranque en frío:** con 0.5 % a 100 % de usuarios etiquetados, la diferencia de AUC-PR del apilado está entre −0.007 y +0.007 (`results_levantamiento/`).

**CERT r4.2** (5 grupos de empleados, 115 días maliciosos)

| Modelo | AUC-ROC | AUC-PR | F1 | Recall | FPR |
|---|---|---|---|---|---|
| Reglas estáticas | 0.899 | 0.003 | 0.006 | 0.391 | 4.20 % |
| Isolation Forest | 0.944 | 0.007 | 0.008 | 0.287 | 2.46 % |
| Apilado con IF (v10, ablación) | 0.947 | 0.256 | 0.378 | 0.617 | 0.06 % |
| **AdaptAuth-ML (v11)** | **0.963** | **0.268** | **0.365** | **0.626** | **0.06 %** |

- **Prueba t pareada** en 10 grupos, frente a las reglas: t(9) = 6.74, p < 0.001, dz = 2.13.
- **Ablación:** apilado con IF frente a AdaptAuth-ML, t(9) = 0.85, p = 0.416.
- **Presupuesto de alertas:** con el umbral calibrado en validación para FPR ≤ 0.1 %, detecta 81 de 115 días maliciosos (recall 0.704) con FPR de 0.0999 %.
- **Arranque en frío:** con 20 % y 50 % de empleados etiquetados, AUC-PR de 0.113 y 0.215 (apilado: 0.104 y 0.224; Isolation Forest solo: 0.008).

**Requisitos no funcionales** (`results_requisitos_sl/matriz_cumplimiento_requisitos.csv`)

| Requisito | Resultado |
|---|---|
| RNF-04 inferencia (p50 / p95 / p99) | 2.7 / 4.2 / 5.2 ms |
| RNF-05 respuesta C1 a C4 (p50 / p95 / p99), misma corrida | 9.7 / 14.4 / 15.6 ms |
| RNF-07 score medio de legítimos | 0.024 |
| RNF-08 reentrenamiento de 500 000 eventos | 0.63 min |
| RNF-10 alertas explicadas | 922 de 922 |
| RNF-12 identificadores directos en registros | 0 |
| RNF-06 contingencia OTP | 100 % |
| CA-C6-02 F1 reentrenado frente al estático | 0.661 frente a 0.663 |

## Prevención de fuga de información

- Todas las variables se calculan con eventos anteriores al instante evaluado.
- **SuspiciousLogin:** se excluyen `login_challenge_method`, `risk_score` y `risk_level`, porque son consecuencia de la evaluación de riesgo de la plataforma.
- **Particiones:**
  - SuspiciousLogin usa una partición temporal: entrenamiento de diciembre a marzo, validación en abril y prueba en mayo y junio.
  - CERT r4.2 usa validación cruzada por empleado, de modo que ningún empleado aparece en entrenamiento y prueba a la vez.

## Limitaciones

- La etiqueta de SuspiciousLogin es la señal de sospecha de Google Workspace, no un ataque confirmado.
- CERT r4.2 es un conjunto sintético con escenarios de ataque guionados.
- Ningún dataset proviene de PYMES peruanas.

## Licencia

El código se publica bajo la licencia MIT. Los datasets conservan las licencias de sus autores.
