# Anexo B. Cuestionario de juicio de expertos

**Datos del juez:** nombre y apellidos; profesión y grado académico; años de experiencia en seguridad de la información o gestión de TI; relación con PYMES (sí / no); fecha.

**Instrucciones.** Después del recorrido del modelo con casos de los datasets (acceso legítimo, OTP y bloqueo), marque su grado de acuerdo con cada afirmación: 1 = totalmente en desacuerdo, 2 = en desacuerdo, 3 = ni de acuerdo ni en desacuerdo, 4 = de acuerdo y 5 = totalmente de acuerdo. El cuestionario mide la pertinencia y la utilidad percibida; la eficacia del modelo se demuestra con métricas.

| Código | Criterio | Ítem | Valoración |
|---|---|---|---|
| U1 | Utilidad | La respuesta en tres niveles (acceso directo, OTP y bloqueo) es pertinente para el control de acceso de una PYME de servicios empresariales. | 1 2 3 4 5 |
| U2 | Utilidad | El equilibrio entre seguridad y fricción obtenido (0.71 % de falsas alarmas y 0.05 % de usuarios legítimos bloqueados) es aceptable para la operación de una PYME. | 1 2 3 4 5 |
| U3 | Utilidad | Que la propia organización calibre los umbrales según la fricción que tolera aporta valor frente a un motor de riesgo cerrado del proveedor. | 1 2 3 4 5 |
| U4 | Utilidad | El modelo es aplicable a sistemas que no pasan por el motor de riesgo de Google Workspace, como un ERP, una VPN o un directorio local. | 1 2 3 4 5 |
| U5 | Utilidad | La carga de revisar los bloqueos y las alertas es asumible por el personal de TI de una PYME. | 1 2 3 4 5 |
| C1 | Calidad | La arquitectura de seis componentes es coherente y cubre el flujo completo de una autenticación adaptativa. | 1 2 3 4 5 |
| C2 | Calidad | Las reglas de decisión (umbrales θ1 y θ2 y contingencia ante fallo) están definidas con claridad. | 1 2 3 4 5 |
| C3 | Calidad | La explicación de cada alerta con sus tres variables más influyentes es comprensible y útil para el responsable de TI. | 1 2 3 4 5 |
| C4 | Calidad | El registro de decisiones permite auditar por qué se permitió, verificó o bloqueó cada acceso. | 1 2 3 4 5 |
| C5 | Calidad | El tratamiento de los datos (seudonimización y variables calculadas solo con eventos previos) es adecuado para la protección de datos personales. | 1 2 3 4 5 |
| E1 | Pertinencia de la evaluación | Las métricas empleadas (F1, FPR, AUC-PR y recall con presupuesto de alertas) son pertinentes para juzgar el modelo. | 1 2 3 4 5 |
| E2 | Pertinencia de la evaluación | La comparación con la línea base de reglas calibrada y las pruebas estadísticas son pertinentes para sustentar la mejora. | 1 2 3 4 5 |
| E3 | Pertinencia de la evaluación | Los umbrales calibrados son adecuados para el escenario de una PYME. | 1 2 3 4 5 |
| E4 | Pertinencia de la evaluación | La contingencia ante fallos y el reentrenamiento protegido son pertinentes para la operación del modelo. | 1 2 3 4 5 |
| E5 | Pertinencia de la evaluación | Las limitaciones declaradas (etiqueta del motor de Google, datos sintéticos de CERT r4.2 y ausencia de datos de una PYME) están adecuadamente reconocidas. | 1 2 3 4 5 |

U3, U4, C3 y C4 recogen la valoración (percepción) de las propiedades de diseño que no se miden con los datos. Regla de decisión con el número real de jueces: un ítem es válido si V de Aiken ≥ 0.80; el IC 95 % (método score de Penfield y Giacobbi, 2004) se reporta como información de precisión.

Observaciones del juez: ____________________ Firma: ____________
