# Anexo D. Ficha de registro de experimentos

Cada corrida genera un JSON con estos campos (por ejemplo, `results_validacion_pi13/validacion_pi13_sl.json`):

| Campo | Contenido |
|---|---|
| Identificación | Versión del modelo, commit, fecha y hora, script |
| Datos | Dataset, versión o commit, número de eventos y proporción de positivos |
| Partición | Temporal o por empleado, meses o grupos, semilla |
| Configuración | Hiperparámetros y reglas de calibración de θ1 y θ2 |
| Umbrales | θ1 y θ2 calibrados en validación |
| Métricas | F1, precisión, recall, FPR, FPR de bloqueo, AUC-ROC, AUC-PR, matriz 3×2 |
| Pruebas | Prueba, estadístico o IC, p |
| Entorno | Python, librerías y sistema operativo |
| Archivos | CSV, JSON y figuras generados |
