# APIs para la publicación de modelos

Diseño e implementación de una API para servir modelos de machine learning.

## Introducción

Existen varias limitaciones para servir nuestros modelos con una CLI:

* los usuarios necesitan acceso a la terminal, código base, entorno virtual, etc.
* las salidas CLI en el terminal no son exportables.

Para abordar estos problemas, vamos a desarrollar una interfaz de programación de aplicaciones (API) que permitirá que cualquier persona interactúe con nuestra aplicación con una simple solicitud.
Es posible que el usuario final no interactúe directamente con nuestra API, pero puede usar componentes de UI/UX que envíen solicitudes a nuestra API.

## Servicio

Las API permiten que diferentes aplicaciones se comuniquen entre sí en tiempo real. Pero primero debemos decidir si lo haremos en lotes o en tiempo real.

### Servicio por lotes

Podemos hacer predicciones por lotes en un conjunto finito de entradas que luego se escriben en una base de datos para una inferencia de baja latencia.

### Servicio en tiempo real

También podemos ofrecer predicciones en vivo, generalmente a través de un request a una API con los datos de entrada adecuados.

## Request

Los usuarios interactuarán con nuestra API en forma de un request. Los diferentes componentes de un request son:

### URI (Uniform Resource Identifier)

Es un identificador para un recurso específico.
Ej.: `https://localhost:8000/models/{modelId}/?filter=passed#details`

### Método

Es la operación a ejecutar sobre el recurso específico definido por la URI.
Los cuatro metodos a continuación son los más populares, que a menudo se denominan CRUD porque le permiten crear, leer, actualizar y eliminar.

* GET: obtener un recurso.
* POST: crear o actualizar un recurso.
* PUT/PATCH: crear o actualizar un recurso.
* DELETE: eliminar un recurso.

La principal diferencia entre `POST` y `PUT` es que `PUT` es idempotente, lo que significa que puede llamar al método repetidamente y producirá el mismo estado cada vez.
Podemos usar cURL para ejecutar nuestras llamadas API.

```curl
curl -X GET "http://localhost:8000/models"
```

### Headers

Contienen información sobre un determinado evento y generalmente se encuentran tanto en el request del cliente como en la respuesta del servidor.
Puede variar desde qué tipo de formato enviarán y recibirán, información de autenticación y almacenamiento en caché, etc.

```curl
curl -X GET "http://localhost:8000/" \          # method and URI
    -H  "accept: application/json"  \           # client accepts JSON
    -H  "Content-Type: application/json" \      # client sends JSON
```

### Body

Contiene información que puede ser necesaria para que se procese la solicitud. Por lo general, es un objeto JSON enviado durante los métodos de solicitud POST, PUT/PATCH, DELETE.

```curl
curl -X POST "http://localhost:8000/models" \   # method and URI
    -H  "accept: application/json" \            # client accepts JSON
    -H  "Content-Type: application/json" \      # client sends JSON
    -d "{'name': 'RoBERTa', ...}"               # request body
```

## Response

La respuesta es el resultado de la solicitud que enviamos. También incluye encabezados y un cuerpo que debe incluir el código de estado HTTP adecuado, así como mensajes explícitos, datos, etc.

```json
{
  "message": "OK",
  "method": "GET",
  "status-code": 200,
  "url": "http://localhost:8000/",
  "data": {}
}
```

## Mejores prácticas

Al diseñar nuestra API, hay algunas mejores prácticas a seguir:

* Las rutas de URI, los mensajes, etc. deben ser lo más explícitos posible.
* Use sustantivos, en lugar de verbos, para nombrar los recursos. (✅ GET /users not ❌ GET /get_users).
* Sustantivos en plural (✅ GET /users/{userId} no ❌ GET /user/{userID}).
* Utilice guiones en URI para recursos y parámetros de ruta, pero use guiones bajos para parámetros de consulta (GET /nlp-models/?find_desc=bert).
* Devolver mensajes HTTP e informativos apropiados al usuario.

## Aplicación

Vamos a definir nuestra API en un directorio de aplicaciones separado. Dentro de nuestro directorio de aplicaciones, crearemos los siguientes scripts:

```bash
mkdir app
cd app
touch api.py gunicorn.py schemas.py
cd ../
```

* api.py: el script principal que incluirá la inicialización y los endpoints de nuestra API.
* gunicorn.py: script para definir las configuraciones de trabajo de la API.
* schemas.py: definiciones para los diferentes objetos que usaremos en nuestros endpoints de recursos.

## FastAPI

Vamos a utilizar FastAPI como nuestro framework para construir nuestro servicio de API.
Las ventajas de FastAPI incluyen:

* desarrollo en python
* de alto rendimiento
* validación de datos a través de pydantic
* documentación generada automáticamente
* inyección de dependencia
* seguridad a través de OAuth2

```bash
pip install fastapi==0.78.0
```

```bash
# Agregar a requirements.txt
fastapi==0.78.0
```

### Inicialización

El primer paso es inicializar nuestra API en nuestro script `api.py` definiendo metadatos como el título, la descripción y la versión:

```python
# app/api.py
from fastapi import FastAPI

# Definir aplicación
app = FastAPI(
    title="CoE MLOps Template",
    description="Clasificación de proyectos de machine learning.",
    version="0.1",
)
```

Nuestro primer endpoint será uno simple en el que queremos mostrar que todo funciona según lo previsto. La ruta para el endpoint será `/` y será una solicitud `GET`.

```python
# app/api.py
from http import HTTPStatus
from typing import Dict

@app.get("/")
def _index() -> Dict:
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response
```

### Lanzamiento

Usamos Uvicorn, un servidor ASGI rápido que puede ejecutar código asíncrono en un solo proceso para iniciar nuestra aplicación.

```bash
pip install uvicorn==0.17.6
```

```bash
# Agregar a requirements.txt
uvicorn==0.17.6
```

Podemos lanzar nuestra aplicación con el siguiente comando:

```bash
uvicorn app.api:app \               # ubicación de la aplicación
    --host 0.0.0.0 \                # localhost
    --port 8000 \                   # puerto 8000
    --reload \                      # recargar cada vez que actualizamos
    --reload-dir coe_template \     # solo recargar en las actualizaciones del directorio `coe_template`
    --reload-dir app                # y el directorio `app`
```

Si queremos administrar múltiples trabajadores de uvicorn para habilitar el paralelismo en nuestra aplicación, podemos usar Gunicorn junto con Uvicorn.
Podemos iniciar todos los trabajadores con el siguiente comando:

```bash
gunicorn -c config/gunicorn.py -k uvicorn.workers.UvicornWorker app.api:app
```

También agregaremos estos dos comandos a nuestro archivo README.md:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload --reload-dir coe_template --reload-dir app  # dev
gunicorn -c app/gunicorn.py -k uvicorn.workers.UvicornWorker app.api:app  # prod
```

### Requests

Podemos enviar nuestra solicitud GET usando varios métodos diferentes:

* Visitar el endpoint en un navegador en [<http://localhost:8000/>](http://localhost:8000/)
* cURL: `curl -X GET http://localhost:8000/`
* Acceder a los endpoints a través del código:

    ```python
    import json
    import requests

    response = requests.get("http://localhost:8000/")
    print (json.loads(response.text))
    ```

* Usar herramientas externas como Postman

### Decoradores

Usemos decoradores para agregar automáticamente metadatos relevantes a nuestras respuestas

```python
# app/api.py
from datetime import datetime
from functools import wraps

from fastapi import FastAPI, Request


def construct_response(f):
    """Construir una respuesta JSON para un endpoint."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs) -> Dict:
        results = f(request, *args, **kwargs)
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }
        if "data" in results:
            response["data"] = results["data"]
        return response

    return wrap
```

Estamos pasando una instancia de Request para que podamos acceder a información como el método request y la URL.
Por lo tanto, nuestras funciones endpoint también deben tener este objeto Request como argumento de entrada.
Una vez que recibimos los resultados de nuestra función f, podemos agregar los detalles adicionales y devolver una respuesta más informativa.
Para usar este decorador, solo tenemos que envolver nuestras funciones.

```python
@app.get("/")
@construct_response
def _index(request: Request) -> Dict:
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response
```

También hay algunos decoradores incorporados que debemos tener en cuenta, como el decorador de eventos (@app.on_event()) que podemos usar para iniciar y cerrar nuestra aplicación.

```python
from pathlib import Path

from config import config
from config.config import logger
from coe_template import main


@app.on_event("startup")
def load_artifacts():
    global artifacts
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = main.load_artifacts(run_id=run_id)
    logger.info("Listo para la inferencia!")
```

### Documentación

Cuando definimos un endpoint, FastAPI genera automáticamente cierta documentación (se adhiere a los estándares de OpenAPI) en función de sus entradas, escritura, salidas, etc.
Podemos acceder a la interfaz de usuario de Swagger para nuestra documentación yendo al endpoint /docs.
Tenga en cuenta que nuestro endpoint está organizado en secciones en la interfaz de usuario. Podemos usar etiquetas al definir nuestros endpoints en el script:

```python
@app.get("/", tags=["General"])
@construct_response
def _index(request: Request) -> Dict:
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response
```

## Recursos

Al diseñar los recursos para nuestra API, debemos pensar en las siguientes preguntas:

* [USUARIOS]: ¿Quiénes son los usuarios finales? Esto definirá qué recursos deben exponerse.
  * desarrolladores que quieran interactuar con la API.
  * equipo del producto que quiere probar e inspeccionar el modelo y su rendimiento.
  * servicio backend que quiere clasificar los proyectos entrantes.
* [ACCIONES]: ¿Qué acciones quieren poder realizar nuestros usuarios?
  * predicción para un conjunto dado de entradas
  * inspección de performance
  * inspección de argumentos de entrenamiento

### Parámetros de consulta

Podemos pasar un filtro de parámetro de consulta opcional para indicar el subconjunto de rendimiento que nos interesa.

```python
@app.get("/performance", tags=["Performance"])
@construct_response
def _performance(request: Request, filter: str = None) -> Dict:
    """Obtener las métricas de performance."""
    performance = artifacts["performance"]
    data = {"performance":performance.get(filter, performance)}
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": data,
    }
    return response
```

Podemos incluir este parámetro en nuestra solicitud `GET` así:

```curl
curl -X "GET" \
  "http://localhost:8000/performance?filter=overall" \
  -H "accept: application/json"
```

### Parámetros de path

En el siguiente endpoint obtenemos los argumentos utilizados para entrenar el modelo. Esta vez, usamos un parámetro de ruta args, que es un campo obligatorio en el URI.

```python
@app.get("/args/{arg}", tags=["Arguments"])
@construct_response
def _arg(request: Request, arg: str) -> Dict:
    """Obtener el valor de un parámetro específico utilizado para la ejecución."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            arg: vars(artifacts["args"]).get(arg, ""),
        },
    }
    return response
```

Podemos realizar nuestra solicitud GET de esta manera:

```curl
curl -X "GET" \
  "http://localhost:8000/args/learning_rate" \
  -H "accept: application/json"
```

También podemos crear un endpoint para producir todos los argumentos que se usaron:

```python
@app.get("/args", tags=["Arguments"])
@construct_response
def _args(request: Request) -> Dict:
    """Obtener todos los argumentos utilizados para la ejecución."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "args": vars(artifacts["args"]),
        },
    }
    return response
```

Podemos realizar nuestra solicitud `GET` de esta manera

```curl
curl -X "GET" \
"http://localhost:8000/args" \
-H "accept: application/json"
```

### Schemas

Ahora es el momento de definir nuestro endpoint para la predicción.
Necesitamos consumir las entradas que queremos clasificar y, por lo tanto, debemos definir el esquema que debe seguirse al definir esas entradas.

```python
# app/schemas.py
from typing import List
from fastapi import Query
from pydantic import BaseModel


class Text(BaseModel):
    text: str = Query(None, min_length=1)


class PredictPayload(BaseModel):
    texts: List[Text]
```

Aquí estamos definiendo un objeto `PredictPayload` como una lista de objetos de texto llamados `texts`.
Cada objeto `Text` es una cadena cuyo valor predeterminado es `None` y debe tener una longitud mínima de 1 carácter.
Ahora podemos usar este payload en nuestro endpoint:

```python
from app.schemas import PredictPayload
from coe_template import predict

@app.post("/predict", tags=["Prediction"])
@construct_response
def _predict(request: Request, payload: PredictPayload) -> Dict:
    """Predecir etiquetas para una lista de textos."""
    texts = [item.text for item in payload.texts]
    predictions = predict.predict(texts=texts, artifacts=artifacts)
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"predictions": predictions},
    }
    return response
```

Necesitamos adherirnos al esquema `PredictPayload` cuando queremos usar nuestro `/predict`:

```curl
curl -X 'POST' 'http://localhost:8000/predict' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "texts": [
        {"text": "Transfer learning with transformers for text classification."},
        {"text": "Generative adversarial networks for image generation."}
      ]
    }'
```

### Validación

**Built-in**
Estamos usando el objeto BaseModel de pydantic aquí porque ofrece una validación integrada para todos nuestros esquemas.
En nuestro caso, si una instancia de texto tiene menos de 1 carácter, nuestro servicio devolverá el código y mensaje de error correspondientes.

**Custom**
También podemos agregar una validación personalizada en una entidad específica usando el decorador @validator, como lo hacemos para asegurarnos de que la lista de textos no esté vacía.

```python
from pydantic import BaseModel, validator


class PredictPayload(BaseModel):
    texts: List[Text]

    @validator("texts")
    def list_must_not_be_empty(cls, value):
        if not len(value):
            raise ValueError("La lista de textos a clasificar no puede estar vacía.")
        return value
```

### Extras

Por último, podemos agregar un objeto `schema_extra` en una clase `Config` para representar cómo debería verse un `PredictPayload` de ejemplo.
Cuando hacemos esto, aparece automáticamente en la documentación de nuestro endpoint.

```python
class PredictPayload(BaseModel):
    texts: List[Text]

    @validator("texts")
    def list_must_not_be_empty(cls, value):
        if not len(value):
            raise ValueError("La lista de textos a clasificar no puede estar vacía.")
        return value


    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    {"text": "Transfer learning with transformers for text classification."},
                    {"text": "Generative adversarial networks in both PyTorch and TensorFlow."},
                ]
            }
        }
```

## Servidor de modelos

Además de envolver nuestros modelos como microservicios separados y escalables, también podemos tener un servidor de modelos especialmente diseñado para alojar nuestros modelos.
Los servidores de modelos proporcionan un registro con una capa de API para inspeccionar, actualizar, servir, revertir, etc. múltiples versiones de modelos.
También ofrecen escalado automático para satisfacer las necesidades de rendimiento y latencia.
Las opciones populares incluyen [BentoML](https://www.bentoml.com/), [MLFlow](https://docs.databricks.com/applications/mlflow/model-serving.html), [TorchServe](https://pytorch.org/serve/),
[RedisAI](https://oss.redislabs.com/redisai/), [Nvidia Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server), etc.
Los servidores de modelo están experimentando una gran adopción por su capacidad para estandarizar la implementación del modelo y los procesos de servicio en todo el equipo,
lo que permite actualizaciones, validación e integración sin inconvenientes.
