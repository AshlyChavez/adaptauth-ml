# Histórico: iteraciones con el dataset RBA

Esta carpeta conserva las primeras iteraciones de construcción como evidencia del proceso de diseño (bitácora de intervención). Los scripts usan el dataset RBA (Wiefling et al., 2022), que se descartó por no representar el contexto de una PYME.

| Versión | Script | Resultado principal | Decisión |
|---|---|---|---|
| v1 | `run_baseline.py` | F1 0.14; FPR 32 % | Muestreo por usuario |
| v3 | `run_baseline.py` (muestra del 5 % de usuarios) | F1 0.28; FPR 26 %; mediana de 2 accesos por usuario | Accesos exitosos y contexto global |
| v4 | `run_experimento.py` | AUC 0.80; F1 0.17; FPR 5.1 % | La señal está en la red y no en el usuario: cambio de dataset |
| v5 | `run_experimento_v5.py` | Prueba del núcleo híbrido sobre RBA | Adoptar el núcleo híbrido |
| CERT v1 | `run_cert.py` | 25 positivos en prueba con partición temporal | Validación cruzada por empleado (`run_cert_v2.py`) |

Las métricas de cada versión están en `resultados/`.
