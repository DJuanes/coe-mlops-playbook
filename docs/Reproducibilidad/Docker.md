# Contenedorización

Empaquetar nuestra aplicación en contenedores reproducibles y escalables.

## Introducción

El último paso para lograr la reproducibilidad es desplegar nuestro código versionado y los artefactos en un entorno reproducible.
Esto va mucho más allá del entorno virtual que configuramos para nuestras aplicaciones Python porque hay especificaciones a nivel de sistema (sistema operativo, paquetes implícitos necesarios, etc.)
que no estamos capturando.
Queremos ser capaces de encapsular todos los requisitos que necesitamos para que no haya dependencias externas que impidan a otra persona reproducir nuestra aplicación exacta.

## Docker

En soluciones para la reproducibilidad a nivel de sistema el motor de contenedores Docker es el más popular por varias ventajas clave:

* reproducibilidad vía Dockerfile con instrucciones explícitas para desplegar nuestra aplicación en un sistema específico.
* aislamiento a través de los contenedores para no afectar a otras aplicaciones que también puedan ejecutarse en el mismo sistema operativo subyacente.
* y muchas más ventajas como el tamaño, la velocidad, Docker Hub, etc.

Vamos a utilizar Docker para desplegar nuestra aplicación localmente de forma aislada, reproducible y escalable.
Una vez que hagamos esto, cualquier máquina con el motor Docker instalado podrá reproducir nuestro trabajo.

## Arquitectura

El motor de contenedores Docker se encarga de poner en marcha los contenedores configurados, que contienen nuestra aplicación y sus dependencias (binarios, librerías, etc.).
El motor de contenedores es muy eficiente, ya que no necesita crear un sistema operativo distinto para cada aplicación en contenedor.
Esto también significa que nuestros contenedores pueden compartir los recursos del sistema a través del motor Docker.

## Instalación

Una vez [instalado](https://docs.docker.com/get-docker/), podemos iniciar el Docker Desktop que nos permitirá crear y desplegar nuestras aplicaciones en contenedores.

```bash
docker --version
```

## Imágenes

El primer paso es construir una imagen Docker que tenga la aplicación y todas sus dependencias especificadas.
Podemos crear esta imagen utilizando un Dockerfile que describe un conjunto de instrucciones.
Estas instrucciones esencialmente construyen capas de imagen de sólo lectura en la parte superior de cada uno para construir nuestra imagen completa.
Echemos un vistazo al Dockerfile de nuestra aplicación y a las capas de imagen que crea.

## Dockerfile

Empezaremos creando un Dockerfile:

```bash
touch Dockerfile
```

La primera línea que escribiremos en nuestro Dockerfile especifica la imagen base de la que queremos extraer.
Queremos usar la imagen base para ejecutar aplicaciones basadas en Python versión 3.9 con la variante slim.
Esta variante slim con paquetes mínimos satisface nuestros requisitos y mantiene el tamaño de la capa de la imagen bajo.

```dockerfile
# Base image
FROM python:3.9-slim
```

A continuación vamos a instalar las dependencias de nuestra aplicación. En primer lugar, vamos a copiar los archivos necesarios de nuestro sistema de archivos local.
Una vez que tenemos nuestros archivos, podemos instalar los paquetes necesarios para instalar las dependencias de nuestra aplicación utilizando el comando RUN.
Una vez que hayamos terminado de usar los paquetes, podemos eliminarlos para mantener el tamaño de nuestra capa de imagen al mínimo.

```dockerfile
# Instalar dependencias
WORKDIR /mlops
COPY setup.py setup.py
COPY requirements.txt requirements.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install -e . --no-cache-dir \
    && python3 -m pip install protobuf==3.20.1 --no-cache-dir \
    && apt-get purge -y --auto-remove gcc build-essential
```

A continuación, estamos listos para copiar los archivos necesarios para ejecutar nuestra aplicación.

```dockerfile
# Copy
COPY coe_template coe_template
COPY app app
COPY data data
COPY config config
COPY stores stores

# Extraer activos de S3
RUN dvc init --no-scm
RUN dvc remote add -d storage stores/blob
RUN dvc pull
```

Como nuestra aplicación (API) requiere que el puerto 8000 esté abierto, necesitamos especificar en nuestro Dockerfile que lo exponga.

```dockerfile
# Exponer puertos
EXPOSE 8000
```

El último paso es especificar el ejecutable que se ejecutará cuando se construya un contenedor a partir de nuestra imagen.
Para nuestra aplicación, queremos lanzar nuestra API con gunicorn.

```dockerfile
# Iniciar aplicación
ENTRYPOINT ["gunicorn", "-c", "app/gunicorn.py", "-k", "uvicorn.workers.UvicornWorker", "app.api:app"]
```

## Compilar imágenes

Una vez que hemos terminado de componer el Dockerfile, estamos listos para construir nuestra imagen utilizando el comando build que nos permite añadir una etiqueta
y especificar la ubicación del Dockerfile a utilizar.

```bash
docker build -t coe_template:latest -f Dockerfile .
```

Podemos inspeccionar todas las imágenes construidas y sus atributos:

```bash
docker images
```

También podemos eliminar alguna o todas las imágenes en función de su identificación única.

```bash
docker rmi <IMAGE_ID>              # eliminar una imagen
docker rmi $(docker images -a -q)  # eliminar todas las imágenes
```

## Ejecutar contenedores

Una vez que hemos construido nuestra imagen, estamos listos para ejecutar un contenedor usando esa imagen con el comando run que nos permite especificar la imagen, el reenvío de puertos, etc.

```bash
docker run -p 8000:8000 --name coe_template coe_template:latest
```

Una vez que tenemos nuestro contenedor en funcionamiento, podemos utilizar la API gracias al puerto que estamos compartiendo (8000):

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "texts": [
    {
      "text": "Transfer learning with transformers for text classification."
    }
  ]
}'
```

Podemos inspeccionar todos los contenedores así:

```bash
docker ps     # contenedores en funcionamiento
docker ps -a  # contenedores detenidos
```

También podemos detener y eliminar cualquiera o todos los contenedores en función de sus identificadores únicos:

```bash
docker stop <CONTAINER_ID>      # detener un contenedor en ejecución
docker rm <CONTAINER_ID>        # eliminar un contenedor
docker stop $(docker ps -a -q)  # detener todos los contenedores
docker rm $(docker ps -a -q)    # eliminar todos los contenedores
```

## Depurar

En el caso de que nos encontremos con errores mientras construimos nuestras capas de imagen, una forma muy fácil de depurar el problema
es ejecutar el contenedor con las capas de imagen que se han construido hasta el momento.
Podemos hacer esto incluyendo sólo los comandos que se han ejecutado con éxito hasta ahora en el Dockerfile.
Y luego tenemos que reconstruir la imagen (ya que alteramos el Dockerfile) y ejecutar el contenedor:

```bash
docker build -t coe_template:latest -f Dockerfile .
docker run -p 8000:8000 -it coe_template /bin/bash
```

Una vez que tenemos nuestro contenedor funcionando, podemos utilizar nuestra aplicación como lo haríamos en nuestra máquina local, pero ahora es reproducible en cualquier sistema operativo.

## Producción

Este `Dockerfile` es comúnmente el artefacto final para desplegar y escalar sus servicios, con algunos cambios:

* los activos de datos se extraerían de una ubicación de almacenamiento remota.
* los artefactos del modelo se cargarían desde un registro de modelos remoto.
* el código se cargaría desde un repositorio remoto.

Todos estos cambios implicarían el uso de las credenciales adecuadas (a través de secretos encriptados e incluso pueden desplegarse automáticamente a través de flujos de trabajo CI/CD).
Pero, por supuesto, hay responsabilidades posteriores como el monitoreo.
