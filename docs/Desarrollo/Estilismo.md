# Estilo y formato de código

Convenciones de estilo y formato para mantener la coherencia de nuestro código.

## Introducción

Cuando escribimos un fragmento de código, casi nunca es la última vez que lo vemos o la última vez que se edita.
Así que necesitamos explicar lo que está pasando (a través de la documentación) y hacer que sea fácil de leer.
Una de las maneras más fáciles de hacer que el código sea más legible es seguir convenciones de estilo y formato consistentes.
Hay muchas opciones cuando se trata de convenciones de estilo de Python a las que adherirse, pero la mayoría se basan en convenciones de PEP8.
Los aspectos más importantes son:

* consistencia: todos siguen los mismos estándares.
* automatización: el formateo debería ser en gran medida sin esfuerzo después de la configuración inicial.

## Herramientas

Usaremos una combinación muy popular de convenciones de estilo y formato.

* Black: un reformateador que se adhiere a PEP8.
* isort: ordena y da formato a las sentencias import.
* flake8: un linter de código con convenciones estilísticas que se adhieren a PEP8.

Instale los paquetes necesarios:

```bash
python -m pip install black==22.3.0 flake8==3.9.2 isort==5.10.1
```

Dado que estos paquetes de estilo no son parte integral de las operaciones principales de aprendizaje automático,
creemos una lista separada en nuestro `setup.py`:

```bash
# setup.py
style_packages = [
    "black==22.3.0",
    "flake8==3.9.2",
    "isort==5.10.1"
]

# Definir nuestro paquete
setup(
    ...
    extras_require={
        "dev": docs_packages + style_packages,
        "docs": docs_packages,
    },
)
```

## Configuración

Antes de que podamos usar correctamente estas herramientas, tendremos que configurarlas.

### Black

Para configurar Black es más eficiente hacerlo a través de un archivo.
Así que crearemos un `pyproject.toml` en la raíz de nuestro directorio de proyectos con lo siguiente:

```bash
touch pyproject.toml
```

```ini
# Black formatting
[tool.black]
line-length = 100
include = '\.pyi?$'
exclude = '''
/(
      .eggs         # excluir algunos directorios comunes en el
    | .git          # raíz del proyecto
    | .hg
    | .mypy_cache
    | .tox
    | venv
    | _build
    | buck-out
    | build
    | dist
  )/
'''
```

### isort

A continuación, vamos a configurar isort en nuestro archivo `pyproject.toml`:

```ini
# isort
[tool.isort]
profile = "black"
line_length = 100
multi_line_output = 3
include_trailing_comma = true
virtual_env = "venv"
```

### flake8

Por último, configuraremos flake8, pero esta vez necesitamos crear un archivo `.flake8` separado para definir sus configuraciones:

```bash
touch .flake8
```

```ini
[flake8]
exclude = venv
ignore = W503, E226
max-line-length = 100

# W503: Line break occurred before binary operator
# E226: Missing white space around arithmetic operator
```

Además de definir opciones de configuración aquí, que se aplican globalmente,
también podemos optar por ignorar específicamente ciertas convenciones línea por línea.
Aquí hay un ejemplo de cómo utilizamos esto:

```python
# coe_template/config.py
import pretty_errors  # NOQA: F401 (imported but unused)
```

## Uso

Para usar estas herramientas que hemos configurado, tenemos que ejecutarlas desde el directorio del proyecto:

```bash
black coe_template
flake8 coe_template
isort coe_template
```

En la sección de makefile, aprenderemos cómo combinar todos estos comandos en uno.
Y en la sección pre-commit, aprenderemos cómo ejecutar automáticamente este formato siempre que hagamos cambios en nuestro código.
