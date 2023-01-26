# Pre-commit

Usando los hooks pre-commit de git para asegurar las comprobaciones antes del commit.

## Introducción

Antes de realizar un commit en nuestro repositorio local, hay muchos elementos en nuestra lista de tareas, que van desde el estilo, el formato, las pruebas, etc.
Y es muy fácil olvidarse de algunos de estos pasos.
Para ayudarnos a gestionar todos estos pasos importantes, podemos utilizar hooks de pre-commit, que se activarán automáticamente cuando intentemos realizar un commit.

## Instalación

Utilizaremos el framework [Pre-commit](https://pre-commit.com/) para ayudarnos a realizar automáticamente comprobaciones importantes mediante hooks cuando hagamos un commit.

```bash
# Instalar pre-commit
python -m pip install pre-commit==2.19.0
pre-commit install
```

Y agregaremos esto a nuestro script `setup.py` en lugar de nuestro archivo `requirements.txt` porque no es fundamental para las operaciones de aprendizaje automático.

```python
# setup.py
setup(
    ...
    extras_require={
        "dev": docs_packages + style_packages + test_packages + ["pre-commit==2.19.0"],
        "docs": docs_packages,
        "test": test_packages,
    },
)
```

## Configuración

Definimos nuestros hooks de pre-commit a través de un archivo de configuración `.pre-commit-config.yaml`.
Podemos crear nuestra configuración yaml desde cero o utilizar la CLI de pre-commit para crear una configuración de ejemplo a la que podemos añadir.

```bash
# Simple config
pre-commit sample-config > .pre-commit-config.yaml
cat .pre-commit-config.yaml
```

```yaml
# See https://pre-commit.com for more information
# See https://pre-commit.com/hooks.html for more hooks
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v3.2.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files
```

## Hooks

A la hora de crear y utilizar hooks, tenemos varias opciones para elegir:

### Integrado (Built-in)

Dentro de la configuración de ejemplo, podemos ver que pre-commit ha añadido algunos hooks por defecto de su repositorio.
Especifica la ubicación del repositorio, la versión, así como los identificadores específicos de los hooks a utilizar.

```yaml
# .pre-commit-config.yaml
...
-   id: check-added-large-files
    args: ['--maxkb=1000']
    exclude: "notebooks"
...
```

### Personalizado

También hay muchos hooks personalizados y populares de terceros entre los que podemos elegir.
Por ejemplo, si queremos aplicar comprobaciones de formato con Black, podemos aprovechar el hook de precommit de Black.

```yaml
# .pre-commit-config.yaml
...
-   repo: https://github.com/psf/black
    rev: 22.6.0
    hooks:
    -   id: black
        args: []
        files: .
...
```

### Local

También podemos crear nuestros propios hooks locales sin configurar un .pre-commit-hooks.yaml separado.
Aquí estamos definiendo dos hooks pre-commit, test-non-training y clean, para ejecutar algunos comandos que hemos definido en nuestro Makefile.
Del mismo modo, podemos ejecutar cualquier comando de entrada con argumentos para crear hooks muy rápidamente.

```yaml
# .pre-commit-config.yaml
...
- repo: local
  hooks:
    - id: test
      name: test
      entry: make
      args: ["test"]
      language: system
      pass_filenames: false
    - id: clean
      name: clean
      entry: make
      args: ["clean"]
      language: system
      pass_filenames: false
```

Creemos nuestro archivo de configuación de pre-commits:

```bash
touch .pre-commit-config.yaml
```

<details>
<summary>Ver nuestro .pre-commit-config.yaml</summary>

```yaml
# See https://pre-commit.com for more information
# See https://pre-commit.com/hooks.html for more hooks
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.3.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
        exclude: "config/run_id.txt"
    -   id: check-yaml
        exclude: "mkdocs.yml"
    -   id: check-added-large-files
        args: ['--maxkb=1000']
        exclude: "notebooks"
    -   id: check-ast
    -   id: check-json
    -   id: check-merge-conflict
    -   id: detect-private-key
-   repo: https://github.com/psf/black
    rev: 22.6.0
    hooks:
    -   id: black
        args: []
        files: .
-   repo: https://github.com/pycqa/flake8
    rev: 3.9.2
    hooks:
    -   id: flake8
-   repo: https://github.com/PyCQA/isort
    rev: 5.10.1
    hooks:
    -   id: isort
        args: []
        files: .
-   repo: https://github.com/asottile/pyupgrade  # update python syntax
    rev: v2.37.3
    hooks:
    -   id: pyupgrade
        args: [--py36-plus]
- repo: local
  hooks:
    - id: test
      name: test
      entry: make
      args: ["test"]
      language: system
      pass_filenames: false
    - id: clean
      name: clean
      entry: make
      args: ["clean"]
      language: system
      pass_filenames: false
```

</details>

Si está usando Windows tiene que realizar un paso adicional, sino dára error Great Expectations al ejecutar por consola, por los problemas con los caracteres especiales.
Dentro del virtual enviroment debe ir a la librería `great_expectations`, luego a la carpeta `cli`.
Aquí debe abrir `checkpoint.py` y reemplazar la línea 305 por esto:

```python
cli_message(str(status_line.encode('utf-8')))
```

## Commit

Nuestros hooks de pre-commit se ejecutarán automáticamente cuando intentemos hacer un commit. Podremos ver si cada hook pasó o falló y hacer cualquier cambio.
Si alguno de los hooks ha fallado, tendremos que arreglar el archivo correspondiente o, en muchos casos, el reformateo se producirá automáticamente.
En el caso de que alguno de los hooks haya fallado, tenemos que añadir y comitear de nuevo para asegurarnos de que todos los hooks son aprobados.

```git
git add .
git commit -m <MENSAJE>
```

## Ejecutar

Aunque los hooks de precommit están pensados para ejecutarse antes de un commit, podemos activar manualmente todos o algunos hooks en todos o en un conjunto de archivos.

```bash
# Run
pre-commit run --all-files  # ejecutar todos los hooks en todos los archivos
pre-commit run <HOOK_ID> --all-files # ejecutar un hook en todos los archivos
pre-commit run --files <PATH_TO_FILE>  # ejecutar todos los hooks en un archivo
pre-commit run <HOOK_ID> --files <PATH_TO_FILE> # ejecutar un hook en un archivo
```

## Skip

No es recomendable omitir la ejecución de ninguno de los hooks porque están ahí por una razón. Pero para algunos commits muy urgentes y que salvan el mundo, podemos usar el flag no-verify.

```bash
# Commit without hooks
git commit -m <MENSAJE> --no-verify
```

## Actualizar

En nuestro archivo de configuración `.pre-commit-config.yaml`, hemos tenido que especificar las versiones de cada uno de los repositorios para poder utilizar sus últimos hooks.
Pre-commit tiene un comando CLI de auto-actualización que actualizará estas versiones a medida que estén disponibles.

```bash
# Autoupdate
pre-commit autoupdate
```

También podemos añadir este comando a nuestro `Makefile` para que se ejecute cuando se cree un entorno de desarrollo para que todo esté actualizado.

```makefile
# Makefile
.ONESHELL:
venv:
    python3 -m venv venv
    source venv/bin/activate && \
    python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -e ".[dev]" && \
    pre-commit install && \
    pre-commit autoupdate
```
