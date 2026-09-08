1. Diferencia entre Python global y `.venv`.
   - El .venv tiene un intérprete de python enlazado, además cuenta con su propio pip, site-packages y scripts de activación. El entorno debe ser activado para que la shell priorice ese python sobre el global.
2. ¿Para qué sirve `python -m pip` frente a llamar solo a `pip`?
   - Sirve para especificar en que versión de python instalar los paquetes. El primer comando especifica instalar los paquetes en ese python, mientras que el segundo comandos los instala en el primer python que encuentra.
3. Explica working tree / staging / commit.  
   - El working tree es el directorio de trabajo. Cuando se realizan cambios a los archivos pasan a la zona de staging que es un punto intermedio para elegir qué se va a añadir al commit. El commit es un conjunto de cambios seleccionados que pasan a formar parte del historial del proyecto.
4. ¿Clone y pull son lo mismo? ¿Por qué?
   - Clone sirve para traer los archivos de un repositorio remoto que no se tiene en local, mientras que pull sirve para actualizar el repositorio local ya inicializado, trayendo los cambios más recientes del repositorio remoto.
5. Lista cuatro cosas que no se suben a GitHub y justifica una.
   - El entorno virtual: Es un directorio pesado además de redundante si se incluyen las dependencias correctamente en el repositorio.
   - El .env: Ya que contiene información sensible como API Keys.
   - Datasets: En entornos reales suelen ser muy pesados y hay maneras más efectivas y eficientes de gestionarlos que incluyendolos al repositorio directamente.
   - Información confidencial
6. Reescribe a buen estilo: `update`, `fix final`, `cambios varios`.
   - Los mensajes de los commits deben ser precisos y claros. Unas alternativas para estos mensajes podrían ser: `Update function X to make Y` y `Fix bug Z` para los dos primeros, mientras que el último habría que dividirlo en varios commits siempre que fuera posible, siguiendo el formato de los anteriores.
7. ¿Qué haces si el IDE no importa `pandas` pero la terminal sí?
   - Probablemente se deba a que el IDE está utilizando un intérprete diferente para el proyecto que se esta desarrollando. Hay que asegurarse que el entorno está activado y que se selecciona en el IDE el intérprete que tiene `pandas` instalado. Por ejemplo en VSCode sería usando `Ctrl+Shift+P` y seleccionando el intérprete correspondiente.
8. ¿Qué haces si subiste `.env` por error?
   - Lo más urgente sería revocar las claves que hubiera contenidas en el archivo y eliminarlas del propio archivo. A continuación habría que limpiar el historial y por último comunicar la incidencia al resto de contribuyentes del repositorio.