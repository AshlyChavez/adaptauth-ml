# AdaptAuth-ML

Modelo de autenticación adaptativa basado en aprendizaje automático para la detección de accesos anómalos en PYMES del sector de servicios empresariales.

Repositorio de la tesis *Desarrollo de un Sistema Inteligente de Autenticación Adaptativa Basado en Machine Learning para la Detección de Accesos Anómalos en PYMES en el Sector de Servicios Empresariales en Lima, 2026*. Universidad Nacional Mayor de San Marcos, Facultad de Ingeniería de Sistemas e Informática, Tesis II.

- **Autora:** Ashly Herminia Jasmín Chávez Marón
- **Asesor:** Ing. Daniel Alfonso Quinto Pazce
- **Método:** Design Science Research con evaluación cuasi-experimental anidada (línea base frente a AdaptAuth-ML)
- **Tipo de artefacto:** modelo computacional / pipeline (aporte conceptual, Grupo A)

## Descripción del modelo

Cuando un usuario ya validó su contraseña, AdaptAuth-ML hace tres cosas:
- construye un vector de variables contextuales con información previa al evento;
- estima un score de riesgo R ∈ [0, 1];
- asigna una de tres respuestas: **acceso directo**, **verificación OTP** o **bloqueo**.

| Componente | Función | Implementación |
|---|---|---|
| C1. Ingesta y variables | Variables causales de hora, red, ubicación, dispositivo e historial | `src/adaptauth.py`, `src/cert.py` |
| C2a. Motor no supervisado | Score de anomalía reescalado s′(x) | Isolation Forest (200 árboles, 256 muestras) |
| C2b + C3. Motor supervisado y consolidación | R = P(ataque \| x, s′) por apilamiento | HistGradientBoostingClassifier |
| C4. Decisión adaptativa | Umbrales θ1 y θ2 calibrados en validación; reglas P1 a P3 | `calibrate`, `decide` en `src/adaptauth.py` |
| C5. Monitoreo y explicación | Registro de decisiones y top-3 variables por alerta | SHAP, `run_requisitos_sl.py` |
| C6. Reentrenamiento | Reentrenamiento mensual sin bloqueos no confirmados | `run_requisitos_sl.py` (prueba CA-C6-02) |

**Ecuaciones:**

- **Score de Isolation Forest:** s(x) = 2^(−E[h(x)] / c(n))
- **Reescalado:** s′(x) = max(0, (s − s_ref) / (1 − s_ref)), donde s_ref es la mediana de s en los accesos legítimos de validación.
- **Score consolidado:** R = P(ataque | x, s′)
- **Decisión:** acceso directo si R ≤ θ1, OTP si θ1 < R ≤ θ2 y bloqueo si R > θ2.

**Calibración:**
- θ1 maximiza F1 con FPR ≤ 5 %.
- θ2 se fija con FPR de bloqueo ≤ 1 %.
- Ambos umbrales se calibran solo con el conjunto de validación.

## Estructura

```
adaptauth-ml/
├── src/
│   ├── adaptauth.py            # núcleo: reescalado, score, calibración, decisión, métricas, pruebas pareadas
│   ├── cert.py                 # carga y variables de CERT r4.2 (unidad usuario-día)
│   └── figures.py              # figuras de resultados
├── run_suspiciouslogin.py      # experimento principal en SuspiciousLogin
├── run_requisitos_sl.py        # verificación de requisitos no funcionales
├── run_cert_v2.py              # experimento en CERT r4.2 con validación cruzada por empleado
├── notebooks/                  # cuadernos de Google Colab
│   └── historico/              # versiones previas (dataset RBA)
├── results_sl/                 # resultados reales de SuspiciousLogin
├── results_requisitos_sl/      # matriz de cumplimiento, registro de eventos y explicaciones
├── results_cert_v2/            # resultados reales de CERT r4.2
├── historico_rba/              # iteraciones v1 a v4 con el dataset RBA y CERT v1 (bitácora)
├── docs/figuras/               # figuras del informe de avance
└── data/                       # instrucciones para obtener los datos (los datos no se versionan)
```

## Datos

Los conjuntos de datos no se incluyen en el repositorio. En `data/README.md` está la forma de obtenerlos.

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

# CERT r4.2: carpeta con logon.csv, LDAP/ y answers/
python run_cert_v2.py --dir data/CERT_r4.2
```

**Demostración en Colab:** `notebooks/00_AdaptAuthML_Demo.ipynb` descarga este repositorio y el dataset, ejecuta los tres experimentos y muestra los resultados ([abrir en Colab](https://colab.research.google.com/github/AshlyChavez/adaptauth-ml/blob/main/notebooks/00_AdaptAuthML_Demo.ipynb)).

En Google Colab, los cuadernos de `notebooks/` montan Google Drive, leen los datos desde la carpeta `AdaptAuth-ML` y guardan los resultados en la misma carpeta.

## Resultados principales

**SuspiciousLogin** (prueba: mayo y junio, n = 20 852, 7.6 % sospechosos)

| Modelo | AUC-ROC | AUC-PR | F1 | Precisión | Recall | FPR |
|---|---|---|---|---|---|---|
| Reglas estáticas | 0.549 | 0.090 | 0.009 | 0.250 | 0.004 | 0.11 % |
| Isolation Forest | 0.630 | 0.109 | 0.044 | 0.051 | 0.038 | 5.80 % |
| **AdaptAuth-ML** | **0.859** | **0.634** | **0.628** | **0.851** | **0.498** | **0.71 %** |

Prueba t pareada en 14 grupos: frente a las reglas, t(13) = 33.87 con p < 0.001.

**CERT r4.2** (5 grupos de empleados, 115 días maliciosos)

| Modelo | AUC-ROC | AUC-PR | F1 | Recall | FPR |
|---|---|---|---|---|---|
| Reglas estáticas | 0.899 | 0.003 | 0.006 | 0.391 | 4.20 % |
| Isolation Forest | 0.944 | 0.007 | 0.008 | 0.287 | 2.46 % |
| Supervisado sin IF | 0.963 | 0.268 | 0.366 | 0.626 | 0.06 % |
| **AdaptAuth-ML (apilado)** | **0.947** | **0.256** | **0.378** | **0.617** | **0.06 %** |

- **Prueba t pareada** en 10 grupos, frente a las reglas: t(9) = 7.27, p < 0.001, dz = 2.30.
- **Presupuesto de alertas:** con FPR de 0.11 %, el modelo detecta el 70 % de los días maliciosos.

**Requisitos no funcionales** (`results_requisitos_sl/matriz_cumplimiento_requisitos.csv`)

| Requisito | Resultado |
|---|---|
| Tiempo de respuesta p99 | 19.3 ms |
| Score medio de legítimos | 0.024 |
| Reentrenamiento de 500 000 eventos | 0.66 min |
| Alertas explicadas | 922 de 922 |
| Contingencia OTP | 100 % |
| F1 del modelo reentrenado frente al estático | 0.667 frente a 0.663 |

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
