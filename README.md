# AdaptAuth-ML

Modelo de autenticación adaptativa basado en aprendizaje automático para la detección de accesos anómalos en PYMES del sector de servicios empresariales.

Repositorio de la tesis *Modelo de autenticación adaptativa basado en Machine Learning para la detección de accesos anómalos en PYMES del sector de servicios empresariales en Lima, 2026*. Universidad Nacional Mayor de San Marcos, Facultad de Ingeniería de Sistemas e Informática, Tesis II.

- **Autora:** Ashly Herminia Jasmín Chávez Marón
- **Asesor:** Ing. Daniel Alfonso Quinto Pazce
- **Método:** Design Science Research con evaluación cuasi-experimental anidada (línea base frente a AdaptAuth-ML)
- **Tipo de artefacto:** modelo computacional / pipeline (aporte conceptual, Grupo A)
- **Versión:** 12 (v11 + 8 variables de comportamiento por usuario y decisión escalonada que prioriza el OTP)

## Descripción del modelo

Cuando un usuario ya validó su contraseña, AdaptAuth-ML hace tres cosas:
- construye un vector de variables contextuales con información previa al evento;
- estima un score de riesgo R ∈ [0, 1];
- asigna una de tres respuestas: **acceso directo**, **verificación OTP** o **bloqueo**.

| Componente | Función | Implementación |
|---|---|---|
| C1. Ingesta y variables | Variables de hora, red, ubicación, dispositivo e historial, más 8 variables que comparan el acceso con los accesos previos del mismo usuario; todas calculadas solo con eventos previos (sin fuga temporal) | `src/adaptauth.py`, `src/cert.py` |
| C2. Motor de detección | Clasificador supervisado sobre el vector de variables | HistGradientBoostingClassifier (`src/modelo_adaptauth.py`) |
| C3. Score de riesgo | R = P(ataque \| x) | `predict_proba` del clasificador |
| C4. Decisión adaptativa | Umbrales θ1 y θ2 calibrados en validación; decisión escalonada; reglas P1 a P3 | `calibrate`, `decide_escalonado` en `src/adaptauth.py` |
| C5. Monitoreo y explicación | Registro de decisiones y top-3 variables por alerta | SHAP, `run_requisitos_sl.py` |
| C6. Reentrenamiento | Reentrenamiento mensual sin bloqueos no confirmados | `run_requisitos_sl.py` (prueba CA-C6-02) |

**Ecuaciones:**

- **Score de riesgo:** R = P(ataque | x)
- **Decisión (v12):** acceso directo si R ≤ θ1; OTP si R > θ1; bloqueo si el OTP falla o si R > θ2 con viaje imposible.

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
│   ├── modelo_adaptauth.py     # AdaptAuth-ML: HistGradientBoosting
│   ├── modelo_apilado.py       # ablación v10: Isolation Forest + HistGradientBoosting apilado
│   └── figures.py              # figuras de resultados
├── run_suspiciouslogin.py      # experimento principal en SuspiciousLogin
├── run_requisitos_sl.py        # verificación de requisitos no funcionales
├── run_analisis_complementario.py  # ablación con IF, pruebas pareadas, bootstrap por usuario, V de Aiken
├── run_levantamiento.py        # arranque en frío y recalibración de θ2
├── run_ablacion_componentes.py # ablación del modelo integral: niveles de decisión, contingencia, reentrenamiento protegido
├── run_cert_v2.py              # CERT r4.2 con validación cruzada por empleado (--frio: arranque en frío)
├── notebooks/                  # cuadernos de Google Colab
│   └── historico/              # versiones previas (dataset RBA)
├── results_sl/                 # resultados reales de SuspiciousLogin
├── results_requisitos_sl/      # matriz de cumplimiento, registro de eventos y explicaciones
├── results_cert_v11/           # resultados de CERT r4.2, versión 11
├── results_cert_v2/            # resultados de CERT r4.2, versión 10 (histórico)
├── results_complementarios/    # ablación, pruebas pareadas, bootstrap, curvas PR y tabla de V de Aiken
├── results_levantamiento/      # arranque en frío y recalibración de θ2 en SuspiciousLogin
├── results_ablacion_componentes/  # ablación de componentes del modelo integral
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
python run_ablacion_componentes.py --data data/suspicious_logins_public_v1.csv
python run_validacion_pi13.py --data data/suspicious_logins_public_v1.csv

# CERT r4.2: carpeta con logon.csv, LDAP/ y answers/
python run_cert_v2.py --dir data/CERT_r4.2 --frio
```

**Demostración en Colab:** `notebooks/00_AdaptAuthML_Demo.ipynb` descarga este repositorio y el dataset, ejecuta los experimentos (pasos 1 a 7) y copia los resultados de CERT a Google Drive ([abrir en Colab](https://colab.research.google.com/github/AshlyChavez/adaptauth-ml/blob/main/notebooks/00_AdaptAuthML_Demo.ipynb)).

En Google Colab, los cuadernos de `notebooks/` montan Google Drive, leen los datos desde la carpeta `AdaptAuth-ML` y guardan los resultados en la misma carpeta.

## Informe de la versión 12

### 1. Objetivo

Evaluar si incorporar variables de comportamiento por usuario eleva el F1 de AdaptAuth-ML sobre la meta de 0.60 con significancia estadística, y si una política de decisión escalonada reduce los bloqueos sin pérdida de detección.

### 2. Método

**Datos.** SuspiciousLogin v1, partición temporal: entrenamiento de diciembre a marzo (35 969 eventos), validación en abril (12 094) y prueba en mayo y junio (20 852; 1 577 sospechosos; 12 765 usuarios).

**Variables incorporadas.** Ocho variables calculadas solo con los accesos anteriores del mismo usuario (`variables_usuario` en `run_suspiciouslogin.py`):

| Variable | Definición |
|---|---|
| `u_accesos_previos` | log(1 + número de accesos previos) |
| `u_frac_mismo_pais` | Fracción de accesos previos desde el mismo país |
| `u_frac_mismo_tipo` | Fracción de accesos previos con el mismo tipo de inicio |
| `u_pais_nuevo` | 1 si el país no aparece en el historial del usuario |
| `u_tipo_nuevo` | 1 si el tipo de inicio no aparece en el historial del usuario |
| `u_desv_hora` | 1 − cos(hora actual − hora media circular previa) |
| `u_ratio_gap` | Horas desde el último acceso / intervalo medio previo, acotado a [0, 50] |
| `u_noche_inusual` | Acceso nocturno × (1 − fracción previa de accesos nocturnos) |

**Protocolo de selección.** Se compararon tres variantes solo con el mes de validación, por AUC-PR; la variante elegida se evaluó una única vez en la prueba.

| Variante | AUC-PR en validación |
|---|---|
| A. Variables de la v11 | 0.589 |
| **B. v11 + variables por usuario (elegida)** | **0.682** |
| C. B + rareza del contexto (país × tipo, país × franja horaria) | 0.682 |

C no superó a B, por lo que se eligió la variante más simple.

**Clasificador y calibración.** Sin cambios respecto de la v11: HistGradientBoosting (semilla 42; 400 iteraciones; tasa 0.05; 31 hojas; L2 = 1.0). θ1 maximiza el F1 con FPR ≤ 5 % y θ2 cumple FPR de bloqueo ≤ 0.1 %, ambos calibrados en abril.

**Política de decisión.** Todo acceso con R > θ1 recibe OTP, incluido el riesgo alto. El bloqueo directo requiere R > θ2 junto con viaje imposible; en operación también se bloquea cuando el OTP falla (`decide_escalonado` en `src/adaptauth.py`).

**Análisis estadístico.** Bootstrap por conglomerados de usuario (2 000 réplicas), McNemar exacto sobre los sospechosos, cinco semillas y prueba t pareada en 14 grupos.

### 3. Resultados

**Tabla 1.** Desempeño en la prueba (mayo y junio, n = 20 852).

| Indicador | v11 | v12 |
|---|---|---|
| AUC-ROC | 0.859 | 0.886 |
| AUC-PR | 0.634 | 0.688 |
| F1 | 0.628 | **0.667** |
| IC 95 % del F1 | 0.581 a 0.678 | **0.620 a 0.712** |
| p (F1 ≤ 0.60) | 0.115 | **0.004** |
| Precisión | 0.851 | 0.879 |
| Recall | 0.498 | 0.537 |
| FPR | 0.71 % | 0.61 % |
| θ1 / θ2 | 0.21 / 0.56 | 0.22 / 0.95 |

**Tabla 2.** Decisiones de la v12 en la prueba.

| Decisión | Legítimos | Sospechosos |
|---|---|---|
| Acceso directo | 19 158 | 730 |
| OTP | 117 | 847 |
| Bloqueo | 0 | 0 |

**Tabla 3.** Comparación con las reglas calibradas al mismo FPR de validación.

| Indicador | Reglas | v12 | Diferencia (IC 95 %) |
|---|---|---|---|
| Recall | 0.021 | 0.537 | +0.517 (0.461 a 0.574) |
| FPR | 1.29 % | 0.61 % | −0.67 pp (−0.94 a −0.39) |
| F1 | 0.036 | 0.667 | +0.631 (0.585 a 0.678) |

- **McNemar exacto:** 830 sospechosos detectados solo por la v12 frente a 16 solo por las reglas (p < 0.001).
- **Cinco semillas:** F1 de 0.666 ± 0.007, AUC-PR de 0.690 ± 0.004 y 0 legítimos bloqueados.
- **Prueba t pareada (14 grupos):** frente a las reglas, t(13) = 50.54, p < 0.001, dz = 13.51.
- **Importancia por permutación (pérdida de AUC-PR):** red y geolocalización 0.559; comportamiento por usuario 0.084; autenticación 0.059; temporal 0.007.
- **Requisitos no funcionales:** inferencia p99 de 3.6 ms y respuesta completa p99 de 9.7 ms, con consulta del historial del usuario incluida.

### 4. Conclusión

La v12 alcanza un F1 de 0.667 con límite inferior del IC 95 % de 0.620, de modo que supera la meta de 0.60 con significancia (p = 0.004). La política escalonada mantiene la detección (53.7 % de sospechosos detenidos) y elimina los bloqueos directos: los 847 sospechosos detenidos pasan por OTP.

### 5. Reproducción

```bash
python run_suspiciouslogin.py --data data/suspicious_logins_public_v1.csv
python run_validacion_pi13.py --data data/suspicious_logins_public_v1.csv
python run_analisis_pi14.py --data data/suspicious_logins_public_v1.csv
python run_ablacion_componentes.py --data data/suspicious_logins_public_v1.csv
python run_requisitos_sl.py --data data/suspicious_logins_public_v1.csv
```

Las ablaciones históricas (`run_levantamiento.py`, `run_analisis_complementario.py`) se ejecutan con las variables de la v11 (`preparar(..., usuario=False)`).

## Resultados principales

**SuspiciousLogin** (prueba: mayo y junio, n = 20 852, 7.6 % sospechosos)

| Modelo | AUC-ROC | AUC-PR | F1 | Precisión | Recall | FPR |
|---|---|---|---|---|---|---|
| Reglas estáticas | 0.549 | 0.090 | 0.009 | 0.250 | 0.004 | 0.11 % |
| Isolation Forest | 0.630 | 0.109 | 0.044 | 0.051 | 0.038 | 5.80 % |
| Apilado con IF (v10, ablación) | 0.881 | 0.640 | 0.620 | 0.814 | 0.501 | 0.93 % |
| AdaptAuth-ML (v11) | 0.859 | 0.634 | 0.628 | 0.851 | 0.498 | 0.71 % |
| **AdaptAuth-ML (v12)** | **0.886** | **0.688** | **0.667** | **0.879** | **0.537** | **0.61 %** |

- **Meta del F1 (HE2):** F1 = 0.667, IC 95 % por bootstrap de usuarios de 0.620 a 0.712; p (F1 ≤ 0.60) = 0.004.
- **Prueba t pareada** en 14 grupos: v12 frente a las reglas, t(13) = 50.54, p < 0.001, dz = 13.51.
- **Ablación (v11):** el apilado con IF frente a AdaptAuth-ML v11, t(13) = −2.10, p = 0.056; bootstrap −0.008 (IC 95 %: −0.014 a −0.002).
- **Decisión escalonada (v12):** 964 accesos con OTP (847 sospechosos y 117 legítimos) y 0 bloqueos directos; se detiene el 53.7 % de los sospechosos, igual que con bloqueo directo sobre θ2.
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
| RNF-04 inferencia (p50 / p95 / p99) | 1.4 / 2.2 / 3.6 ms |
| RNF-05 respuesta C1 a C4 (p50 / p95 / p99), misma corrida | 5.0 / 9.0 / 9.7 ms |
| RNF-07 score medio de legítimos | 0.024 |
| RNF-08 reentrenamiento de 500 000 eventos (prueba de carga: remuestreo con reposición del entrenamiento) | 0.60 min |
| RNF-10 alertas explicadas | 964 de 964 |
| RNF-12 identificadores directos en registros | 0 |
| RNF-06 contingencia OTP | 100 % |
| CA-C6-02 F1 reentrenado frente al estático (un ciclo simulado; prueba solo en junio) | 0.701 frente a 0.682 |

Las latencias se midieron en la misma corrida, con 1 000 eventos procesados uno por uno: RNF-04 mide solo la inferencia (C2 y C3) y RNF-05 la respuesta completa de C1 a C4, que incluye construir el vector de variables y consultar el historial del usuario.

**Ablación de componentes** (`results_ablacion_componentes/`, prueba de SuspiciousLogin, mismo clasificador)

| Esquema de decisión | Legítimos bloqueados | Legítimos con OTP | Sospechosos sin verificación |
|---|---|---|---|
| Escalonado con OTP priorizado (v12) | 0 | 117 | 730 |
| Tres niveles con bloqueo directo sobre θ2 (v11) | 0 | 117 | 730 |
| Dos niveles: acceso o bloqueo con θ1 | 117 | 0 | 730 |
| Dos niveles: acceso o bloqueo con θ2 | 0 | 0 | 886 |
| OTP universal | 0 | 19 275 | 0 |

- **Contingencia:** ante una caída del motor, sin la regla P1 hay que elegir entre dejar pasar a los 1 577 sospechosos o bloquear a los 19 275 legítimos.
- **Reentrenamiento protegido:** en un ciclo mejora el F1 (0.679 frente a 0.667 sin protección); con bloqueos sin confirmar acumulados limita la caída (0.443 frente a 0.144), pero los bloqueos deben confirmarse para conservar la señal de ataque.
- **CERT r4.2:** la unidad es el día-empleado y las variables usan la actividad del día completo; ese escenario demuestra detección diaria con historial, no la decisión en el inicio de sesión.

## Validación (PI 13)

La corrida de validación usa el commit congelado `e67f5a7` (v12; la v11 quedó en `5f6f3b1`), las versiones exactas de `requirements-validacion.txt` y el cuaderno `notebooks/03_Validacion_PI13.ipynb`. Los resultados de mayo y junio de SuspiciousLogin son **no confirmatorios**, porque ese periodo se usó en versiones anteriores.

| Comparación en SuspiciousLogin (prueba, 20 852 eventos) | Reglas con umbral para el mismo FPR de validación | AdaptAuth-ML v12 |
|---|---|---|
| Recall | 0.021 | 0.537 |
| FPR | 1.29 % | 0.61 % |
| F1 | 0.036 | 0.667 |

- **Prueba principal:** bootstrap por usuario (2 000 réplicas, 12 765 usuarios) de la diferencia de recall: +0.517 (IC 95 %: 0.461 a 0.574). McNemar exacto sobre los 1 577 sospechosos: 830 detectados solo por AdaptAuth-ML frente a 16 solo por las reglas (p < 0.001).
- **Cinco semillas:** F1 0.666 ± 0.007; AUC-PR 0.690 ± 0.004; legítimos bloqueados 0.
- **Curva recall-FPR** (`results_validacion_pi13/o6_recall_fpr_sl.csv`): con el umbral para FPR ≤ 5 % el recall sube a 0.637, pero la precisión baja a 0.54.
- **CERT r4.2:** `run_cert_v2.py` calcula además la diferencia de AUC y de recall con FPR ≤ 0.1 % frente a las reglas, con IC 95 % por bootstrap de empleados (`results_cert_v11/validacion_pi13_cert.json`).

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

## Uso de inteligencia artificial

La autora utilizó herramientas de inteligencia artificial generativa como apoyo para orientar el análisis de los modelos, revisar la precisión de los cálculos y del código, e identificar puntos de mejora. Las decisiones de diseño, la ejecución de los experimentos, la verificación de los resultados y la interpretación son responsabilidad de la autora.

## Licencia

El código se publica bajo la licencia MIT. Los datasets conservan las licencias de sus autores.
