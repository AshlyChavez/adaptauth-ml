# Anexo C. Guía de walk-through para los expertos

Objetivo: que cada experto conozca el funcionamiento de AdaptAuth-ML con casos de los datasets públicos antes de responder el cuestionario (Anexo B). Duración aproximada: 25 minutos; el mismo orden con todos los jueces.

| Paso | Contenido | Material y caso que se muestra |
|---|---|---|
| 1 | Problema y propósito (3 min) | Acceso con contraseña válida robada; limitación de las reglas fijas. |
| 2 | Arquitectura (4 min) | Figura de los seis componentes C1 a C6 y flujo de una decisión. |
| 3 | Acceso legítimo (3 min) | Evento 14982: R = 0.0002, acceso directo. |
| 4 | Caso dudoso (3 min) | Evento 28380: R = 0.219, se solicita OTP. |
| 5 | Bloqueo con explicación (3 min) | Evento 28447: R = 0.726, bloqueo; tres variables más influyentes. |
| 6 | Contingencia y reentrenamiento (3 min) | Fallo simulado del motor (OTP en el 100 % de los casos); exclusión de bloqueos no confirmados. |
| 7 | Resultados y limitaciones (4 min) | Comparación con la línea base calibrada, resultados en CERT r4.2 y limitaciones. |
| 8 | Preguntas y cuestionario | Entrega del Anexo B; el expositor no interviene durante el llenado. |

Se registra un acta con fecha, duración, preguntas de cada juez y aclaraciones dadas.
