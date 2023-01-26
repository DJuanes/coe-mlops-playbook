# Makefiles

Una herramienta de automatización que organiza comandos para los procesos de nuestra aplicación.

## Introducción

Para ayudar a organizar todo, vamos a utilizar un Makefile, que es una herramienta de automatización que organiza nuestros comandos.
Comenzaremos creando este archivo en el directorio raíz de nuestro proyecto.

```bash
touch Makefile
```

En la parte superior de nuestro Makefile, debemos especificar el entorno de shell en el que queremos que se ejecuten todos nuestros comandos:

```bash
# Makefile
SHELL = /bin/bash
```

## Componentes

Dentro de nuestro Makefile, crearemos una lista de reglas. Estas reglas tienen un `target` que a veces puede tener requisitos previos que deben cumplirse
y en la siguiente línea una tabulación seguida de una receta que especifica cómo crear el target.
Por ejemplo, si quisiéramos crear una regla para el estilo, agregaríamos lo siguiente:

```makefile
# Styling
.PHONY: style
style:
    black coe_template
    flake8 coe_template
    isort coe_template
```

## Targets

Podemos ejecutar cualquiera de las reglas escribiendo `make <target>` en la terminal:

```bash
# Make a target
$ make style
```

Si está usando Windows, primero debe instalar `make` en Windows. Siga los pasos en este [link](https://linuxhint.com/run-makefile-windows/).

## PHONY

Los Makefiles se usan comúnmente como accesos directos de comandos, lo que puede generar confusión cuando un target de Makefile y un archivo comparten el mismo nombre.
A veces, nombraremos nuestros targets y querremos que se ejecuten, ya sea que exista como un archivo real o no.
En estos escenarios, queremos definir un target PHONY en nuestro archivo MAKE agregando esta línea arriba del target:

```makefile
.PHONY: <target_name>
```

La mayoría de las reglas en nuestro Makefile requerirán el target PHONY porque queremos que se ejecuten incluso si hay un archivo que comparte el nombre del target.

## Prerrequisitos

Antes de hacer un target, podemos adjuntarles requisitos previos. Estos pueden ser targets de archivos que deben existir o comandos target PHONY que deben ejecutarse antes de crear este target.
Por ejemplo, estableceremos el target de estilo como un requisito previo para el target de limpieza para que todos los archivos tengan el formato adecuado antes de limpiarlos.

```makefile
# Cleaning
.PHONY: clean
clean: style
    find . -type f -name "*.DS_Store" -ls -delete
    find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
    find . | grep -E ".pytest_cache" | xargs rm -rf
    find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
    find . | grep -E ".trash" | xargs rm -rf
    rm -f .coverage
```

## Variables

También podemos establecer y usar variables dentro de nuestro Makefile para organizar todas nuestras reglas.

* Podemos establecer las variables directamente dentro del Makefile. Si la variable no está definida en el Makefile, por defecto sería cualquier variable de entorno con el mismo nombre.

    ```makefile
    # Set variable
    MESSAGE := "hello world"

    # Use variable
    greeting:
        @echo ${MESSAGE}
    ```

* Podemos usar variables pasadas al ejecutar la regla de esta manera:

    ```bash
    make greeting MESSAGE="hi"
    ```

## Shells

Cada línea en una receta para una regla se ejecutará en una sub-shell separado. Sin embargo, para ciertas recetas, como activar un entorno virtual y cargar paquetes,
queremos ejecutar todos los pasos en un solo shell. Para hacer esto, podemos agregar el target especial .ONESHELL encima de cualquier target.

```makefile
# Environment
.ONESHELL:
venv:
    python3 -m venv venv
    source venv/bin/activate && \
    python -m pip install --upgrade pip setuptools wheel && \
    pip install -e
```

## Ayuda

Lo último que agregaremos a nuestro Makefile es un target de ayuda en la parte superior. Esta regla proporcionará un mensaje informativo para las capacidades de este Makefile:

```makefile
.PHONY: help
help:
    @echo "venv    : crea un virtual environment."
    @echo "style   : ejecuta el formato de estilo."
    @echo "clean   : limpia todos los archivos innecesarios."
```

```bash
make help
```
