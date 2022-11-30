# Logging para sistemas ML

Mantenga registros de los eventos importantes en nuestra aplicación.

## Introducción

Logging es el proceso de rastrear y registrar eventos clave que ocurren en nuestras aplicaciones con el propósito de inspección, depuración, etc.
Son mucho más poderosos que las sentencias `print` porque nos permiten enviar información específica a ubicaciones específicas con formato personalizado, interfaces compartidas, etc.
Esto hace que el logueo sea un defensor clave para poder obtener información detallada de los procesos internos de nuestra aplicación.

## Componentes

Hay algunos conceptos generales a tener en cuenta:

* `Logger`: emite los mensajes de log de nuestra aplicación.
* `Handler`: envía registros de log a una ubicación específica.
* `Formatter`: da formato y estilo a los registros.

## Niveles

Antes de crear nuestro logger, veamos cómo se ven los mensajes de log usando la configuración básica.

import logging
import sys

```python
# Crear logger super basico
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Niveles de logeo (de menor a mayor prioridad)
logging.debug("Se utiliza para depurar su código.")
logging.info("Mensajes informativos de tu código.")
logging.warning("Todo funciona, pero hay algo a tener en cuenta.")
logging.error("Ha habido un error con el proceso.")
logging.critical("Hay algo terriblemente mal y el proceso puede terminar.")
```

Definimos nuestro logger usando `basicConfig` para emitir mensajes de log a stdout, pero también podríamos haber escrito en cualquier otro flujo o incluso en un archivo.
También definimos nuestro log para que sea sensible a los mensajes a partir del nivel DEBUG, que es el nivel más bajo.
Si hubiéramos hecho el nivel `ERROR`, solo se mostrarían los mensajes de registro `ERROR` y `CRÍTICO`.

## Configuración

Primero estableceremos la ubicación de nuestros logs en el script `config.py`:

```python
# config/config.py
# Directorios
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")
STORES_DIR = Path(BASE_DIR, "stores")
LOGS_DIR = Path(BASE_DIR, "logs")

# Stores
MODEL_REGISTRY = Path(STORES_DIR, "model")

# Crear carpetas
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
```

A continuación, configuraremos el logger para nuestra aplicación:

```python
# config/config.py
import logging
import sys

# Logger
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {
        "handlers": ["console", "info", "error"],
        "level": logging.INFO,
        "propagate": True,
    },
}
```

Elegimos usar un diccionario para configurar nuestro logger, pero hay otras formas, como script de Python, archivo de configuración, etc:

<details>
<summary>Script de Python</summary>

```python
import logging
from rich.logging import RichHandler

# Obtener logger root
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Crear handlers
console_handler = RichHandler(markup=True)
console_handler.setLevel(logging.DEBUG)
info_handler = logging.handlers.RotatingFileHandler(
    filename=Path(LOGS_DIR, "info.log"),
    maxBytes=10485760,  # 1 MB
    backupCount=10,
)
info_handler.setLevel(logging.INFO)
error_handler = logging.handlers.RotatingFileHandler(
    filename=Path(LOGS_DIR, "error.log"),
    maxBytes=10485760,  # 1 MB
    backupCount=10,
)
error_handler.setLevel(logging.ERROR)

# Crear formatters
minimal_formatter = logging.Formatter(fmt="%(message)s")
detailed_formatter = logging.Formatter(
    fmt="%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
)

# Enganchar todo
console_handler.setFormatter(fmt=minimal_formatter)
info_handler.setFormatter(fmt=detailed_formatter)
error_handler.setFormatter(fmt=detailed_formatter)
logger.addHandler(hdlr=console_handler)
logger.addHandler(hdlr=info_handler)
logger.addHandler(hdlr=error_handler)
```

</details>

<details>
<summary>Archivo de configuración</summary>

1. Coloque esto dentro de un archivo `logging.config`:

    ```python
    [formatters]
    keys=minimal,detailed

    [formatter_minimal]
    format=%(message)s

    [formatter_detailed]
    format=
        %(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]
        %(message)s

    [handlers]
    keys=console,info,error

    [handler_console]
    class=StreamHandler
    level=DEBUG
    formatter=minimal
    args=(sys.stdout,)

    [handler_info]
    class=handlers.RotatingFileHandler
    level=INFO
    formatter=detailed
    backupCount=10
    maxBytes=10485760
    args=("logs/info.log",)

    [handler_error]
    class=handlers.RotatingFileHandler
    level=ERROR
    formatter=detailed
    backupCount=10
    maxBytes=10485760
    args=("logs/error.log",)

    [loggers]
    keys=root

    [logger_root]
    level=INFO
    handlers=console,info,error
    ```

2. Coloque esto dentro de su script de Python:

    ```python
    import logging
    import logging.config
    from rich.logging import RichHandler

    # Usar el archivo de configuración para inicializar el logger
    logging.config.fileConfig(Path(CONFIG_DIR, "logging.config"))
    logger = logging.getLogger()
    logger.handlers[0] = RichHandler(markup=True)  # setear rich handler
    ```

</details>

Podemos cargar nuestro diccionario de configuración del log así:

```python
# config/config.py
from rich.logging import RichHandler

logging.config.dictConfig(logging_config)
logger = logging.getLogger()
logger.handlers[0] = RichHandler(markup=True)  # pretty formatting

# Mensajes de muestra (tenga en cuenta que ahora usamos `logger` configurado)
logging.debug("Se utiliza para depurar su código.")
logging.info("Mensajes informativos de tu código.")
logging.warning("Todo funciona, pero hay algo a tener en cuenta.")
logging.error("Ha habido un error con el proceso.")
logging.critical("Hay algo terriblemente mal y el proceso puede terminar.")
```

Necesitaremos instalar la librería `RichHandler` y agregarla a `requirements.txt`:

```bash
pip install rich==12.4.4
```

```bash
# Agregar a requirements.txt
rich==12.4.4
```

## Aplicación

En nuestro proyecto, podemos reemplazar todas nuestras sentencias print en sentencias de logging:

```python
print("✅ Datos guardados!")
```

se convierte en:

```python
from config.config import logger

logger.info("✅ Datos guardados!")
```

* qué: registre todos los detalles necesarios que desea que surjan de nuestra aplicación que serán útiles durante el desarrollo y luego para la inspección retrospectiva.
* donde: una mejor práctica es no saturar nuestras funciones modulares con sentencias de log.
  En su lugar, deberíamos registrar mensajes fuera de funciones pequeñas y dentro de flujos de trabajo más grandes.
  Por ejemplo, no hay mensajes de log dentro de ninguno de nuestros scripts, excepto los archivos `main.py` y `train.py`.
