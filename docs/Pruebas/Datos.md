# Testing de Datos

Hasta ahora, hemos utilizado pruebas unitarias y de integración para probar las funciones que interactúan con nuestros datos, pero no hemos probado la validez de los datos en sí.
Vamos a utilizar la librería [great expectations](https://github.com/great-expectations/great_expectations) para probar cómo se espera que sean nuestros datos.
Es una librería que nos permite crear expectativas sobre el aspecto que deberían tener nuestros datos de forma estandarizada.
También proporciona módulos para conectarse sin problemas con fuentes de datos de backend como sistemas de archivos locales, cloud, bases de datos, etc.

```bash
pip install great-expectations==0.15.15
```

Y agregaremos esto a nuestro script `setup.py`:

```python
# setup.py
test_packages = ["pytest==7.1.2", "pytest-cov==2.10.1", "great-expectations==0.15.15"]
```

Primero cargaremos los datos sobre los que queremos aplicar nuestras expectativas.

```python
import great_expectations as ge
import json
import pandas as pd
from urllib.request import urlopen

# Cargar proyectos etiquetados
projects = pd.read_csv("https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/projects.csv")
tags = pd.read_csv("https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/tags.csv")
df = ge.dataset.PandasDataset(pd.merge(projects, tags, on="id"))
print (f"{len(df)} projects")
df.head(5)
```

## Expectativas

A la hora de crear expectativas sobre el aspecto que deben tener nuestros datos, debemos pensar en todo nuestro conjunto de datos y en todos los features (columnas) que contiene.

```python
# Presencia de features específicos.
df.expect_table_columns_to_match_ordered_list(
    column_list=["id", "created_on", "title", "description", "tag"]
)

# Combinaciones únicas de features (¡detecta fugas de datos!)
df.expect_compound_columns_to_be_unique(column_list=["title", "description"])

# Valores faltantes
df.expect_column_values_to_not_be_null(column="tag")

# Valores únicos
df.expect_column_values_to_be_unique(column="id")

# Adherencia al tipo
df.expect_column_values_to_be_of_type(column="title", type_="str")

# Lista (categórica) / rango (continuo) de valores permitidos
tags = ["computer-vision", "graph-learning", "reinforcement-learning",
        "natural-language-processing", "mlops", "time-series"]
df.expect_column_values_to_be_in_set(column="tag", value_set=tags)
```

Cada una de estas expectativas creará una salida con detalles sobre el éxito o el fracaso, los valores esperados y observados, las expectativas planteadas, etc.
Éstas son sólo algunas de las diferentes expectativas que podemos crear. Aquí hay otras expectativas populares:

* relaciones de valores de features con otros valores de features → `expect_column_pair_values_a_to_be_greater_than_b`
* recuento de filas (exacto o en un rango) de muestras → `expect_table_row_count_to_be_between`
* estadísticas de valores (media, std, mediana, max, min, suma, etc.) → `expect_column_mean_to_be_between`

## Organización

A la hora de organizar las expectativas, se recomienda empezar por las de nivel de tabla y luego pasar a las columnas de features individuales.
**Expectativas de la tabla**:

```python
# columnas
df.expect_table_columns_to_match_ordered_list(
    column_list=["id", "created_on", "title", "description", "tag"])

# data leak
df.expect_compound_columns_to_be_unique(column_list=["title", "description"])
```

**Expectativas de columnas**:

```python
# id
df.expect_column_values_to_be_unique(column="id")

# created_on
df.expect_column_values_to_not_be_null(column="created_on")
df.expect_column_values_to_match_strftime_format(
    column="created_on", strftime_format="%Y-%m-%d %H:%M:%S")

# title
df.expect_column_values_to_not_be_null(column="title")
df.expect_column_values_to_be_of_type(column="title", type_="str")

# description
df.expect_column_values_to_not_be_null(column="description")
df.expect_column_values_to_be_of_type(column="description", type_="str")

# tag
df.expect_column_values_to_not_be_null(column="tag")
df.expect_column_values_to_be_of_type(column="tag", type_="str")
```

Podemos agrupar todas las expectativas para crear un objeto Expectation Suite que podemos utilizar para validar cualquier módulo Dataset.

```python
# Expectation suite
expectation_suite = df.get_expectation_suite(discard_failed_expectations=False)
print(df.validate(expectation_suite=expectation_suite, only_return_failures=True))
```

## Proyectos

Hasta ahora hemos trabajado con la librería Great Expectations a nivel de script / notebook, pero podemos organizar aún más nuestras expectativas creando un Proyecto.

```bash
cd tests
great_expectations init
```

Esto creará un directorio tests/great_expectations con la siguiente estructura:

```bash
tests/great_expectations/
├── checkpoints/
├── expectations/
├── plugins/
├── uncommitted/
├── .gitignore
└── great_expectations.yml
```

**Data source**
El primer paso es establecer nuestro datasource:

```bash
great_expectations datasource new
```

```bash
What data would you like Great Expectations to connect to?
    1. Files on a filesystem (for processing with Pandas or Spark) 👈
    2. Relational database (SQL)
```

```bash
What are you processing your files with?
1. Pandas 👈
2. PySpark
```

```bash
Enter the path of the root directory where the data files are stored: ../data
```

Ejecutamos las celdas en el notebook generado y cambiamos el `datasource_name` por `local_data`.
Después de ejecutar las celdas, podemos cerrar el notebook y podemos ver el Datasource que fue añadido a `great_expectations.yml`.

**Suites**
Cree expectativas de forma manual, interactiva o automática y guárdelas como suites (un conjunto de expectativas para un activo de datos concreto).

```bash
great_expectations suite new
```

```bash
How would you like to create your Expectation Suite?
    1. Manually, without interacting with a sample batch of data (default)
    2. Interactively, with a sample batch of data 👈
    3. Automatically, using a profiler
```

```bash
Which data asset (accessible by data connector "default_inferred_data_connector_name") would you like to use?
    1. labeled_projects.csv
    2. projects.csv 👈
    3. tags.csv
```

```bash
Name the new Expectation Suite [projects.csv.warning]: projects
```

Esto abrirá un notebook interactivo donde podemos añadir expectativas.
Copie y pegue las expectativas a continuación y ejecute todas las celdas.

<details>
<summary>Expectativas para projects.csv</summary>

Expectativas de tabla

```bash
# Presencia de features
validator.expect_table_columns_to_match_ordered_list(
    column_list=["id", "created_on", "title", "description"])
validator.expect_compound_columns_to_be_unique(column_list=["title", "description"])  # data leak
```

Expectativas de columna

```bash
# id
validator.expect_column_values_to_be_unique(column="id")

# create_on
validator.expect_column_values_to_not_be_null(column="created_on")
validator.expect_column_values_to_match_strftime_format(
    column="created_on", strftime_format="%Y-%m-%d %H:%M:%S")

# title
validator.expect_column_values_to_not_be_null(column="title")
validator.expect_column_values_to_be_of_type(column="title", type_="str")

# description
validator.expect_column_values_to_not_be_null(column="description")
validator.expect_column_values_to_be_of_type(column="description", type_="str")
```

</details>

<details>
<summary>Expectativas para tags.csv</summary>

Expectativas de tabla

```bash
# Presencia de features
validator.expect_table_columns_to_match_ordered_list(column_list=["id", "tag"])
```

Expectativas de columna

```bash
# id
validator.expect_column_values_to_be_unique(column="id")

# tag
validator.expect_column_values_to_not_be_null(column="tag")
validator.expect_column_values_to_be_of_type(column="tag", type_="str")
```

</details>

<details>
<summary>Expectativas para labeled_projects.csv</summary>

Expectativas de tabla

```bash
# Presencia de features
validator.expect_table_columns_to_match_ordered_list(
    column_list=["id", "created_on", "title", "description", "tag"])
validator.expect_compound_columns_to_be_unique(column_list=["title", "description"])  # data leak
```

Expectativas de columna

```bash
# id
validator.expect_column_values_to_be_unique(column="id")

# create_on
validator.expect_column_values_to_not_be_null(column="created_on")
validator.expect_column_values_to_match_strftime_format(
    column="created_on", strftime_format="%Y-%m-%d %H:%M:%S")

# title
validator.expect_column_values_to_not_be_null(column="title")
validator.expect_column_values_to_be_of_type(column="title", type_="str")

# description
validator.expect_column_values_to_not_be_null(column="description")
validator.expect_column_values_to_be_of_type(column="description", type_="str")

# tag
validator.expect_column_values_to_not_be_null(column="tag")
validator.expect_column_values_to_be_of_type(column="tag", type_="str")
```

</details>

Todas estas expectativas se guardarán en `great_expectations/expectations`.
También podemos enumerar las suites con:

```bash
great_expectations suite list
```

Para editar una suite, podemos ejecutar el siguiente comando:

```bash
great_expectations suite edit <SUITE_NAME>
```

**Checkpoints**
Crea checkpoints donde se aplica una suite de expectativas a un activo de datos específico.
Esta es una gran manera de aplicar programáticamente los puntos de control en nuestras fuentes de datos existentes y nuevas.
Para nuestro ejemplo, sería:

```bash
cd tests
great_expectations checkpoint new projects
great_expectations checkpoint new tags
great_expectations checkpoint new labeled_projects
```

Cada llamada de creación de checkpoint lanzará un notebook en el que podemos definir a qué suites aplicar este checkpoint.
Tenemos que cambiar las líneas para `data_asset_name` (sobre qué activo de datos ejecutar el checkpoint suite) y `expectation_suite_name` (nombre de la suite a utilizar).
Luego podemos ejecutarlo:

```bash
great_expectations checkpoint run projects
great_expectations checkpoint run tags
great_expectations checkpoint run labeled_projects
```

## Documentación

Cuando creamos expectativas utilizando la aplicación CLI, Great Expectations genera automáticamente documentación para nuestras pruebas.
También almacena información sobre las ejecuciones de validación y sus resultados.
Podemos lanzar la documentación de los datos generados con el siguiente comando: `great_expectations docs build`
Por defecto, Great Expectations almacena nuestras expectativas, resultados y métricas localmente, pero para producción, querremos configurar almacenes de metadatos remotos.

## Producción

La ventaja de utilizar una biblioteca como great expectations, frente a las sentencias assert aisladas, es que podemos:

* reducir los esfuerzos redundantes para crear pruebas en todas las modalidades de datos
* crear automáticamente puntos de control de las pruebas para ejecutarlas a medida que crece nuestro conjunto de datos
* generar automáticamente documentación sobre las expectativas e informar sobre las ejecuciones
* conectar fácilmente con fuentes de datos backend como sistemas de archivos locales, cloud, bases de datos, etc.
