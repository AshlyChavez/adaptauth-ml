# Datos

Los datasets no se versionan en el repositorio. Para reproducir los experimentos se descargan y se colocan en esta carpeta.

## SuspiciousLogin

- **Fuente:** https://github.com/salomaopena/SuspiciousLoginDataset
- **Archivo:** `suspicious_logins_public_v1.csv`, que se coloca en `data/`.
- **Contenido:** 73 528 inicios de sesión exitosos en Google Workspace de dos instituciones, en un periodo de ocho meses.
- **Etiqueta:** `is_suspicious`.
- **Licencia:** CC BY-NC 4.0.

## CERT Insider Threat Test Dataset r4.2

- **Fuente:** Lindauer, B. (2020). *Insider Threat Test Dataset*. Carnegie Mellon University. https://doi.org/10.1184/R1/12841247.v1
- **Archivos necesarios:**
  - `logon.csv`;
  - la carpeta `LDAP/`;
  - la carpeta `answers/` (respuestas oficiales de r4.2).
- **Ubicación:** se colocan en `data/CERT_r4.2/`.
- **Licencia:** CC BY 4.0.

## Google Colab

Los cuadernos esperan los datos en `MyDrive/AdaptAuth-ML/`. Si el archivo ya está en Drive, no se vuelve a descargar.
