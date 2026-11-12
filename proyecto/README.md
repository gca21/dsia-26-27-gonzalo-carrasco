# Proyecto II — Clasificación y explicación de objetos espaciales

Pipeline reproducible para validar el catálogo SATCAT de Space-Track, entrenar un clasificador scikit-learn y predecir el tipo de objetos cuyo catálogo no lo especifica. Para las predicciones más seguras, un cliente opcional de Hugging Face puede redactar una explicación en español. El modo `mock` permite ejecutar las pruebas y el pipeline sin llamar a un servicio de IA.

Este repositorio es la base del Proyecto III: el mismo núcleo de datos, entrenamiento, inferencia y cliente de IA se podrá ampliar con una API o aplicación, despliegue y monitorización.

## Dominio, datos y motivación

El dominio es la clasificación de objetos espaciales en `PAYLOAD`, `ROCKET BODY` o `DEBRIS`. El modelo se entrena con objetos de tipo conocido y se aplica a los registros `UNKNOWN`, `TBA` u otros tipos fuera de esas tres clases. La motivación es automatizar una primera clasificación de los registros incompletos del catálogo y facilitar su revisión con explicaciones legibles.

- **Origen:** catálogo SATCAT de [Space-Track](https://www.space-track.org/basicspacedata/query/class/satcat/CURRENT/Y/orderby/NORAD_CAT_ID%20asc/format/csv). La descarga del catálogo actual requiere cuenta y credenciales de Space-Track.
- **Muestra incluida:** [`data/satellites/muestra_satellites.csv`](data/satellites/muestra_satellites.csv), una muestra pequeña para explorar el formato. El entrenamiento requiere al menos 10.000 filas válidas.
- **Licencia y condiciones:** El acceso y el uso se rigen por los [términos de Space-Track](https://www.space-track.org/).

## Instalación y configuración

Se requiere Python 3.12 y las dependencias de [`requirements.txt`](requirements.txt).

```bash
cd proyecto
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
cp .env.example .env
```

El pipeline descarga los datos por defecto. Para ello, rellena en `.env` las credenciales `SPACETRACK_USER` y `SPACETRACK_PASS`. La IA usa `AI_PROVIDER=mock` por defecto y no necesita token. La plantilla configura `HF_MODEL=openai/gpt-oss-120b` y `HF_INFERENCE_PROVIDER=groq` como opción para el cliente HF. Para activar llamadas reales, cambia `AI_PROVIDER=hf` y proporciona en `HF_TOKEN` una credencial válida para el proveedor. Consulta [AI_USAGE.md](AI_USAGE.md) para más detalles. Las pruebas siguen usando mock o clientes simulados y no realizan llamadas reales.

## Ejecutar el pipeline

Desde este directorio:

```bash
python -m satellites.cli \
  --input data/satellites/satellites.csv \
  --output-dir output/run
```

La ejecución descarga el SATCAT actual, carga y valida las filas, comprueba el umbral de calidad, entrena y evalúa el clasificador, y predice los registros no clasificados. El modelo explica hasta tres predicciones con mayor probabilidad usando el cliente configurado. Para procesar un CSV ya descargado, añade `--no-download`. La ruta de entrada debe existir y contener datos suficientes para entrenar.

Opciones disponibles:

- `--max-error-rate`: proporción máxima de filas inválidas (por defecto `0.1`).
- `--min-f1`: F1 mínimo para superar el control del modelo (por defecto `0.5`).
- `--explain-top`: máximo de predicciones a explicar (por defecto `3`, `0` desactiva las explicaciones).
- `--no-download`: utiliza el CSV local en vez de descargarlo.

El directorio de salida contiene `clean_satellites.csv`, `invalid_satellites.csv`, `model.joblib`, `metadata.json`, `predictions.csv`, `metrics.json` y `run.log`. El modelo se vuelve a entrenar en cada ejecución y se guarda junto con sus metadatos, puede regenerarse con el comando anterior y una fuente de datos válida.

## Modelo scikit-learn

`SatellitesTrainer` entrena un `RandomForestClassifier` de 100 árboles con ponderación balanceada. Se descartan del entrenamiento las clases distintas de `PAYLOAD`, `ROCKET BODY` y `DEBRIS`. Las variables incluyen datos orbitales, fecha y lugar de lanzamiento, clase orbital y tamaño radar. El preprocesamiento (imputación, escalado y codificación one-hot) está integrado en el `Pipeline` de scikit-learn para guardar y reutilizar el mismo proceso en inferencia.

La evaluación utiliza una partición agrupada y estratificada por evento de lanzamiento (`LAUNCH` + `SITE`) para reducir la fuga de información entre fragmentos de un mismo evento. Se informan accuracy, F1, métricas por clase, matriz de confusión y una línea base `DummyClassifier`. El pipeline detiene la ejecución si no se supera el umbral de F1.

Como referencia, la ejecución registrada en `output/metrics.json` el 12-11-2026 procesó 71.305 filas, validó 69.196 (tasa de error del 2,96 %) y obtuvo accuracy `0.9063` y F1 `0.8579`, frente a `0.2213` de F1 para la línea base. Predijo 478 objetos y generó tres explicaciones. Estas métricas dependen de la versión y del contenido del catálogo y no son una garantía para ejecuciones futuras.

## Hugging Face y explicación

El borde de IA está definido en [`satellites/ai/client.py`](satellites/ai/client.py): `MockClient` y `HFClient` comparten el contrato `complete(prompt) -> str`. El cliente real usa `huggingface_hub.InferenceClient` y la tarea `chat_completion`, con `openai/gpt-oss-120b` enrutado a `groq` según `.env.example`. `AI_PROVIDER=mock` sigue siendo el valor inicial: esa configuración no activa por sí sola el servicio real. El prompt de [`satellites/ai/explain.py`](satellites/ai/explain.py) solicita una explicación breve a partir de los datos orbitales y las probabilidades del clasificador.

La IA generativa no decide la clase ni modifica la predicción, esa decisión pertenece al modelo local de scikit-learn. Las explicaciones son informativas y no se validan automáticamente. En la ejecución del 12-11-2026, [`output/run.log`](output/run.log) registra tres respuestas HTTP 200 del endpoint de chat de Groq y [`output/predictions.csv`](output/predictions.csv) conserva las tres explicaciones. Las pruebas sustituyen el cliente de red por dobles o usan `MockClient`, no necesitan un token ni hacen llamadas a proveedores.

## Pruebas y automatización

```bash
python -m pytest -q
python -m pytest -q -m integration
python -m pytest --cov=satellites --cov-report=term-missing
```

La suite incluye pruebas unitarias de carga, validación, construcción de características, entrenamiento, carga e inferencia del modelo, cliente de IA y CLI, además de pruebas de integración del pipeline con datos y clientes controlados.

En local, las pruebas de integración del pipeline usan `mock` y las pruebas del cliente HF simulan `InferenceClient`; no se necesita red ni credenciales de IA.

## Estructura

```text
proyecto/
├── satellites/             # Carga, validación, características, train, predict y pipeline
│   ├── ai/                 # Cliente mock/HF y prompt de explicación
│   └── scripts/            # Ejecución programada y comprobación de métricas
├── tests/satellites/       # Pruebas unitarias y de integración
├── data/satellites/        # Muestra y datos locales
├── output/                 # Artefactos, métricas y logs de ejecuciones
├── .env.example            # Plantilla de configuración, sin secretos
├── requirements.txt
└── AI_USAGE.md             # Registro de uso de IA
```

## Continuidad hacia el Proyecto III

En el Proyecto III se podrá añadir una API o aplicación sobre las funciones existentes de entrenamiento, predicción y explicación, definir un contrato de análisis, y completar CI, despliegue y monitorización. El dominio y el pipeline se mantienen como base, sin rehacer el núcleo de datos.

## Uso de IA

El uso de IA en el producto y como asistente de desarrollo, las comprobaciones realizadas y las limitaciones conocidas se documentan en [AI_USAGE.md](AI_USAGE.md).
