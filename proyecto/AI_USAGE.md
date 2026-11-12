# Uso de inteligencia artificial

Este documento distingue la IA integrada en el pipeline de la IA utilizada como ayuda durante el desarrollo. La ejecución registrada en `output/run.log` y `output/predictions.csv` aporta evidencia de llamadas reales al proveedor Groq a través de Hugging Face.

## 1. IA integrada en el pipeline

| Aspecto | Estado |
|---|---|
| Borde de cliente | `ModelClient`, con implementaciones `MockClient` y `HFClient` en [`satellites/ai/client.py`](satellites/ai/client.py) |
| Proveedor real previsto | Hugging Face Inference Providers, mediante `huggingface_hub.InferenceClient` |
| Tarea | `chat_completion` |
| Modelo del Hub | `openai/gpt-oss-120b` (configurado en `.env.example`) |
| Proveedor de inferencia | Groq, mediante `HF_INFERENCE_PROVIDER=groq` en `.env.example` |
| Estado por defecto | `AI_PROVIDER=mock`. HF/Groq no se activa hasta cambiar a `AI_PROVIDER=hf` y proporcionar `HF_TOKEN` |
| Uso | Redactar una explicación breve de predicciones locales sobre objetos que el catálogo no clasifica |
| Evidencia de uso real | Ejecución del 12-11-2026: tres solicitudes `POST` a Groq con respuesta HTTP 200. Se guardaron tres explicaciones en `output/predictions.csv` |

### Papel y límites

El clasificador `RandomForestClassifier` de scikit-learn determina siempre el tipo de objeto y sus probabilidades. El modelo generativo solo redacta una explicación para las predicciones más seguras (tres por defecto); no reemplaza ni modifica la clasificación.

El prompt de [`satellites/ai/explain.py`](satellites/ai/explain.py) solicita dos o tres frases y prohíbe inventar datos o cambiar la predicción. Incluye la clase predicha, las probabilidades y, cuando existen, periodo, inclinación, apogeo, perigeo, clase orbital, año de lanzamiento, país, tamaño radar y fecha de reentrada. No incluye el nombre del satélite ni su identificador NORAD. Las explicaciones se guardan en `predictions.csv`.

### Configuración

Las variables se leen del entorno o de `.env` (plantilla en [`.env.example`](.env.example)):

```dotenv
AI_PROVIDER=mock
HF_TOKEN=
HF_MODEL=openai/gpt-oss-120b
HF_INFERENCE_PROVIDER=groq
```

`mock` es el valor predeterminado: no usa la red y devuelve una respuesta fija. La configuración para el modo real es `openai/gpt-oss-120b` a través de Groq, usando el cliente de Hugging Face. Para repetir la llamada real hay que cambiar `AI_PROVIDER=hf` y proporcionar en `HF_TOKEN` una credencial válida para el proveedor configurado.

Si la llamada falla o devuelve texto vacío, el pipeline registra un warning y omite la explicación. Si falta el token o el identificador de modelo al inicializar el modo HF, el CLI informa el error de configuración y termina con código `2`.

### Pruebas, acceso y coste

- Las pruebas de integración del pipeline usan `MockClient`, las pruebas del cliente HF sustituyen `InferenceClient` por implementaciones falsas. Por tanto, las pruebas no envían datos a Hugging Face ni a otro proveedor.
- La ejecución real del 12-11-2026 queda reflejada en `output/run.log`: se resolvió el modelo `openai/gpt-oss-120b` en Hugging Face y se enviaron tres solicitudes a `https://api.groq.com/openai/v1/chat/completions`, todas con respuesta HTTP 200. `output/predictions.csv` contiene las tres explicaciones generadas, y `output/metrics.json` informa `explained: 3`.
- `.env.example` mantiene `AI_PROVIDER=mock` y `HF_TOKEN` vacío para que la configuración por defecto y las pruebas no hagan llamadas de red. La ejecución real se hizo con una configuración local que no se versiona. El coste y las condiciones de uso dependen de la cuenta y el plan de Groq.

### Datos y privacidad

En la ejecución real se enviaron al proveedor el prompt con la clase predicha, sus probabilidades y los campos orbitales seleccionados descritos arriba. No se incluyeron el nombre del satélite ni su identificador NORAD. No son datos personales conocidos por este proyecto. El token se usa para autenticar la solicitud y no forma parte del prompt. El catálogo está sujeto a los términos de Space-Track.

## 2. IA como asistente de desarrollo

Se ha usado inteligencia artificial generativa como apoyo en tareas de diseño y depuración de código, así como en la redacción de documentación, recayendo siempre la implementación final y el entendimiento de la misma en el alumno.
