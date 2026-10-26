# Proyecto I — Pipelines de calidad de datos

[![CI](https://github.com/gca21/dsia-26-27-gonzalo-carrasco/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/gca21/dsia-26-27-gonzalo-carrasco/actions/workflows/ci.yml)

Dos aplicaciones independientes para cargar, validar y analizar datos: un pipeline de ventas y otro del catálogo de satélites SATCAT de Space-Track. Ambas usan pandas, exponen una CLI y tienen pruebas pytest. El workflow común está en [`../.github/workflows/ci.yml`](../.github/workflows/ci.yml).

## Requisitos e instalación

Se requiere Python 3.12, igual que en CI.

```bash
cd proyecto_i
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Ejecutar los pipelines

Desde `proyecto_i/`. Cada CLI escribe el CSV válido en la ruta indicada y un resumen JSON (`calidad_datos.json` o `quality_data.json`) en el mismo directorio.

### Parte A: ventas

```bash
python -m ventas_app.cli \
  --input data/ventas.csv \
  --output data/salidas/ventas_validas.csv
```

Se validan las unidades y el precio unitario (ambos deben ser numéricos y mayores que cero). Las métricas impresas son el importe por región, los tres productos con más importe y los clientes recurrentes.

### Parte B: satélites

El repo incluye una muestra para poder ejecutar el pipeline sin credenciales ni acceso a internet:

```bash
python -m satellites.cli \
  --input data/satellites/muestra_satellites.csv \
  --output data/salidas/satellites_validos.csv
```

La carga conserva las columnas SATCAT utilizadas por la aplicación. La validación comprueba, entre otras reglas, identificadores NORAD positivos y únicos, nombre y tipo de objeto conocidos, fechas de lanzamiento/caída coherentes y rangos físicos de órbita. Se derivan el año de lanzamiento y la clase orbital. Las métricas son el crecimiento anual acumulado, la concentración por país y la distribución por clase orbital.

#### Origen y condiciones de los datos satelitales

- **Fuente:** [Space-Track SATCAT](https://www.space-track.org/basicspacedata/query/class/satcat/CURRENT/Y/orderby/NORAD_CAT_ID%20asc/format/csv), catálogo de objetos espaciales actuales. La consulta del script `satellites/downloader.py` requiere iniciar sesión en Space-Trac.
- **Muestra del repositorio:** [`data/satellites/muestra_satellites.csv`](data/satellites/muestra_satellites.csv), 2.246 registros más la cabecera.
- **Fecha:** 11-10-2026, fecha de extracción de la muestra presente en este repositorio.
- **Licencia/condiciones:** el CSV no incluye una licencia abierta propia. El acceso y uso están sujetos a los [términos y condiciones de Space-Track](https://www.space-track.org/).
- **Descarga opcional:** configura `SPACETRACK_USER` y `SPACETRACK_PASS` en un `.env` y ejecuta `python -m satellites.downloader`. El script descarga el catálogo actual, por lo que sus resultados pueden cambiar con el tiempo.

## Pruebas y cobertura

Desde `proyecto_i/`:

```bash
python -m pytest -q
python -m pytest -q -m "not integration"
python -m pytest --cov=ventas_app --cov=satellites \
  --cov-report=term-missing --cov-fail-under=60
```

Las pruebas unitarias usan fixtures en memoria o `tmp_path`; no necesitan red ni rutas de la máquina. Hay pruebas de `loader`, `validator`, `metrics` y `cli` para ambas aplicaciones. La prueba marcada `integration` carga el CSV versionado de ventas y ejecuta la validación sobre los datos reales del proyecto.

## Estado frente al checkpoint del enunciado

El `data/ventas.csv` actualmente versionado contiene **150 filas: 140 válidas y 10 inválidas**, y la prueba de integración comprueba ese checkpoint.

## CI

El workflow [`../.github/workflows/ci.yml`](../.github/workflows/ci.yml) se ejecuta en `push` y `pull_request` a `main`; configura Python 3.12, instala `requirements.txt` y exige pytest-cov con cobertura mínima del 60 % para `ventas_app` y `satellites`. El badge superior refleja el estado de la rama `main` de [GitHub Actions](https://github.com/gca21/dsia-26-27-gonzalo-carrasco/actions/workflows/ci.yml).

## Estructura

```text
proyecto_i/
├── ventas_app/                 # Parte A: loader, validator, metrics y cli
├── satellites/                 # Parte B: loader, validator, metrics, cli y downloader
├── tests/
│   ├── ventas/
│   └── satellites/
├── data/
│   ├── ventas.csv
│   └── satellites/
├── requirements.txt
└── pytest.ini
```

## Uso de IA

Durante el desarrollo de este proyecto se han usado modelos de inteligencia artificial como apoyo, recayendo siempre la implementación final y el entendimiento de la misma sobre el alumno. Las principales tareas en las que la IA ha sido de apoyo son las siguientes:

- Documentación del proyecto
- Debugging
- Explicación de funciones y métodos proporcionados por las librerías utilizadas como pandas o pytest
- Limpieza del código