# Testing de Código

## Aplicación

Comenzaremos creando un directorio de pruebas separado con un subdirectorio de código para probar nuestros scripts. Mas adelante crearemos subdirectorios para probar los datos y los modelos.

```bash
tests/
└── code/
│   ├── test_data.py
│   ├── test_evaluate.py
│   ├── test_main.py
│   ├── test_predict.py
│   └── test_utils.py
```

```bash
mkdir tests
cd tests
mkdir code
cd code
touch test_data.py test_evaluate.py test_main.py test_predict.py test_utils.py
cd ../../
```

Vamos a utilizar pytest como nuestro framework de pruebas por sus poderosas características incorporadas como parametrización, fixtures, marcadores y más.

```bash
pip install pytest==7.1.2
```

Dado que este paquete no es parte integral de las operaciones principales de machine learning, creemos una lista separada en nuestro `setup.py` y la agregamos a `extras_require`:

```bash
# setup.py
test_packages = ["pytest==7.1.2"]

# Definir nuestro paquete
setup(
    ...
    extras_require={
        "dev": docs_packages + style_packages + test_packages,
        "docs": docs_packages,
        "test": test_packages,
    },
)
```

De esta manera tendremos mayor flexibilidad, podemos ejecutar sólo las pruebas, o todo los pasos necesarios para el entorno de desarrollo.

## Configuración

Pytest espera que las pruebas se organicen bajo un directorio tests por defecto.
Una vez en el directorio, pytest busca scripts de python que empiecen por `tests_*.py` pero podemos configurarlo para que lea también cualquier otro patrón de archivos.

## Assertions

Veamos cómo es una prueba de ejemplo y sus resultados. Supongamos que tenemos una función simple que determina si una fruta es fresca o no:

```python
# food/fruits.py
def is_crisp(fruit):
    if fruit:
        fruit = fruit.lower()
    if fruit in ["apple", "watermelon", "cherries"]:
        return True
    elif fruit in ["orange", "mango", "strawberry"]:
        return False
    else:
        raise ValueError(f"{fruit} not in known list of fruits.")
    return False
```

Para probar esta función, podemos utilizar sentencias assert para relacionar las entradas con las salidas esperadas.
La sentencia que sigue a la palabra assert debe devolver True. También es una buena idea agregar afirmaciones sobre las excepciones.

```python
# tests/food/test_fruits.py
def test_is_crisp():
    assert is_crisp(fruit="apple")
    assert is_crisp(fruit="Apple")
    assert not is_crisp(fruit="orange")
    with pytest.raises(ValueError):
        is_crisp(fruit=None)
        is_crisp(fruit="pear")
```

<details>
<summary>Ejemplo</summary>

```python
# tests/code/test_evaluate.py
import numpy as np
import pytest

from coe_template import evaluate


def test_get_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    classes = ["a", "b"]
    performance = evaluate.get_metrics(y_true=y_true, y_pred=y_pred, classes=classes, df=None)
    assert performance["overall"]["precision"] == 2/4
    assert performance["overall"]["recall"] == 2/4
    assert performance["class"]["a"]["precision"] == 1/2
    assert performance["class"]["a"]["recall"] == 1/2
    assert performance["class"]["b"]["precision"] == 1/2
    assert performance["class"]["b"]["recall"] == 1/2
```

</details>

## Ejecución

Podemos ejecutar nuestras pruebas anteriores utilizando varios niveles diferentes de granularidad:

```bash
python -m pytest                                           # todas las pruebas
python -m pytest tests/food                                # pruebas bajo un directorio
python -m pytest tests/food/test_fruits.py                 # pruebas para un solo archivo
python -m pytest tests/food/test_fruits.py::test_is_crisp  # pruebas para una sola función
```

Si alguna de nuestras afirmaciones en esta prueba fallara, veríamos las afirmaciones fallidas, junto con la salida esperada y real de nuestra función.

## Clases

También podemos probar las clases y sus respectivas funciones creando clases de prueba.
Dentro de nuestra clase de prueba, podemos definir opcionalmente funciones que se ejecutarán automáticamente cuando configuremos o eliminemos una instancia de la clase o utilicemos un método de la misma.

* **setup_class**: configura el estado de cualquier instancia de la clase.
* **teardown_class**: elimina el estado creado en setup_class.
* **setup_method**: se llama antes de cada método para configurar cualquier estado.
* **teardown_method**: se llama después de cada método para desmontar cualquier estado.

Podemos ejecutar todas las pruebas de nuestra clase especificando el nombre de la misma:

```bash
python -m pytest tests/food/test_fruits.py::TestFruit
```

<details>
<summary>Ejemplo de prueba de una clase</summary>

```python
# tests/code/test_data.py
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from coe_template import data


class TestLabelEncoder:
    @classmethod
    def setup_class(cls):
        """Llamada antes de cada inicialización de clase."""
        pass

    @classmethod
    def teardown_class(cls):
        """Llamada después de cada inicialización de clase."""
        pass

    def setup_method(self):
        """Llamada antes de cada método."""
        self.label_encoder = data.LabelEncoder()

    def teardown_method(self):
        """Llamada después de cada método."""
        del self.label_encoder

    def test_empty_init(self):
        label_encoder = data.LabelEncoder()
        assert label_encoder.index_to_class == {}
        assert len(label_encoder.classes) == 0

    def test_dict_init(self):
        class_to_index = {"apple": 0, "banana": 1}
        label_encoder = data.LabelEncoder(class_to_index=class_to_index)
        assert label_encoder.index_to_class == {0: "apple", 1: "banana"}
        assert len(label_encoder.classes) == 2

    def test_len(self):
        assert len(self.label_encoder) == 0

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as dp:
            fp = Path(dp, "label_encoder.json")
            self.label_encoder.save(fp=fp)
            label_encoder = data.LabelEncoder.load(fp=fp)
            assert len(label_encoder.classes) == 0

    def test_str(self):
        assert str(data.LabelEncoder()) == "<LabelEncoder(num_classes=0)>"

    def test_fit(self):
        label_encoder = data.LabelEncoder()
        label_encoder.fit(["apple", "apple", "banana"])
        assert "apple" in label_encoder.class_to_index
        assert "banana" in label_encoder.class_to_index
        assert len(label_encoder.classes) == 2

    def test_encode_decode(self):
        class_to_index = {"apple": 0, "banana": 1}
        y_encoded = [0, 0, 1]
        y_decoded = ["apple", "apple", "banana"]
        label_encoder = data.LabelEncoder(class_to_index=class_to_index)
        label_encoder.fit(["apple", "apple", "banana"])
        assert np.array_equal(label_encoder.encode(y_decoded), np.array(y_encoded))
        assert label_encoder.decode(y_encoded) == y_decoded
```

</details>

## Parametrización

Para eliminar la redundancia en las pruebas, pytest tiene el decorador `@pytest.mark.parametrize` que nos permite representar nuestras entradas y salidas como parámetros.

```python
@pytest.mark.parametrize(
    "fruit, crisp",
    [
        ("apple", True),
        ("Apple", True),
        ("orange", False),
    ],
)
def test_is_crisp_parametrize(fruit, crisp):
    assert is_crisp(fruit=fruit) == crisp
```

Del mismo modo, también podríamos pasar una excepción como resultado esperado:

```python
@pytest.mark.parametrize(
    "fruit, exception",
    [
        ("pear", ValueError),
    ],
)
def test_is_crisp_exceptions(fruit, exception):
    with pytest.raises(exception):
        is_crisp(fruit=fruit)
```

<details>
<summary>Ejemplo</summary>

```python
# tests/code/test_data.py
@pytest.mark.parametrize(
    "text, lower, stem, stopwords, cleaned_text",
    [
        ("Hello worlds", False, False, [], "Hello worlds"),
        ("Hello worlds", True, False, [], "hello worlds"),
        ("Hello worlds", False, True, [], "Hello world"),
        ("Hello worlds", True, True, [], "hello world"),
        ("Hello worlds", True, True, ["world"], "hello world"),
        ("Hello worlds", True, True, ["worlds"], "hello"),
    ],
)
def test_clean_text(text, lower, stem, stopwords, cleaned_text):
    assert (
        data.clean_text(
            text=text,
            lower=lower,
            stem=stem,
            stopwords=stopwords,
        )
        == cleaned_text
    )
```

</details>

## Fixtures

Asi como la parametrización nos permite reducir la redundancia dentro de las funciones de prueba,
los fixtures nos permite reducir la redundancia entre diferentes funciones de prueba, ya que se ejecuta antes de la función de prueba.

```python
@pytest.fixture
def my_fruit():
    fruit = Fruit(name="apple")
    return fruit

def test_fruit(my_fruit):
    assert my_fruit.name == "apple"
```

También podemos aplicar fixtures a las clases donde la función fixture será invocada cuando se llame a cualquier método de la clase.

```python
@pytest.mark.usefixtures("my_fruit")
class TestFruit:
    ...
```

<details>
<summary>Ejemplo</summary>

```python
# tests/code/test_data.py
@pytest.fixture(scope="module")
def df():
    data = [
        {"title": "a0", "description": "b0", "tag": "c0"},
        {"title": "a1", "description": "b1", "tag": "c1"},
        {"title": "a2", "description": "b2", "tag": "c1"},
        {"title": "a3", "description": "b3", "tag": "c2"},
        {"title": "a4", "description": "b4", "tag": "c2"},
        {"title": "a5", "description": "b5", "tag": "c2"},
    ]
    df = pd.DataFrame(data * 10)
    return df


@pytest.mark.parametrize(
    "labels, unique_labels",
    [
        ([], ["other"]),  # ningún conjunto de etiquetas aprobadas
        (["c3"], ["other"]),  # sin superposición de etiquetas aprobadas/reales
        (["c0"], ["c0", "other"]),  # superposición parcial
        (["c0", "c1", "c2"], ["c0", "c1", "c2"]),  # superposición completa
    ],
)
def test_replace_oos_labels(df, labels, unique_labels):
    replaced_df = data.replace_oos_labels(
        df=df.copy(), labels=labels, label_col="tag", oos_label="other"
    )
    assert set(replaced_df.tag.unique()) == set(unique_labels)
```

</details>

Los fixture pueden tener diferentes alcances dependiendo de cómo queramos utilizarlos.

* `function`: el fixture se destruye después de cada prueba. [default]
* `class`: el fixture se destruye después de la última prueba de la clase.
* `module`: el fixture se destruye después de la última prueba del módulo (script).
* `package`: el fixture se destruye después de la última prueba en el paquete.
* `session`: el fixture se destruye después de la última prueba de la sesión.

Los fixtures de mayor nivel de alcance se ejecutan primero.
Normalmente, cuando tenemos muchos fixtures en un archivo de prueba concreto, podemos organizarlos todos en un script `fixtures.py` e invocarlos según sea necesario.

## Marcadores

Podemos crear una granularidad de ejecución personalizada utilizando marcadores. Hay varios marcadores incorporados: ya hemos utilizado parametrize.
El marcador skipif nos permite saltar la ejecución de una prueba si se cumple una condición.
Por ejemplo, supongamos que sólo queremos probar el entrenamiento de nuestro modelo si hay una GPU disponible:

```python
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Las pruebas de entrenamiento completas requieren una GPU."
)
def test_training():
    pass
```

También podemos crear nuestros propios marcadores personalizados:

```python
@pytest.mark.fruits
def test_fruit(my_fruit):
    assert my_fruit.name == "apple"
```

Podemos ejecutarlos utilizando el flag `-m` que requiere una expresión marcadora:

```bash
pytest -m "fruits"      #  ejecuta todas las pruebas marcadas con `fruits`
pytest -m "not fruits"  #  ejecuta todas las pruebas excepto las marcadas con `fruits`
```

## Coverage

A medida que desarrollamos pruebas para los componentes de nuestra aplicación, es importante saber qué tan bien estamos cubriendo nuestra base de código y saber si hemos omitido algo.
Podemos utilizar la librería Coverage para rastrear y visualizar qué parte de nuestro código base cubren nuestras pruebas.
Con pytest, es aún más fácil utilizar este paquete gracias al plugin pytest-cov.

```bash
pip install pytest-cov==2.10.1
```

Y agregaremos esto a nuestro script `setup.py`:

```python
# setup.py
test_packages = ["pytest==7.1.2", "pytest-cov==2.10.1"]
```

```bash
python -m pytest --cov coe_template --cov-report html
```

Una vez que nuestras pruebas se han completado, podemos ver el informe generado (por defecto es `htmlcov/index.html`) y hacer clic en archivos individuales
para ver qué partes no fueron cubiertas por ninguna prueba. Esto es especialmente útil cuando nos olvidamos de probar ciertas condiciones, excepciones, etc.

## Exclusiones

A veces no tiene sentido escribir pruebas que cubran todas las líneas de nuestra aplicación, pero aún así queremos tener en cuenta estas líneas para poder mantener una cobertura del 100%.
Tenemos dos niveles de alcance al aplicar exclusiones:

1. Excluir líneas añadiendo este comentario `# pragma: no cover, <MESSAGE>`

    ```python
    if trial:  # pragma: no cover, optuna pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    ```

2. Excluyendo archivos al especificarlos en el archivo de configuración `pyproject.toml`:

```ini
# Pytest coverage
[tool.coverage.run]
omit = ["app/gunicorn.py"]
```

Es importante añadir una justificación a estas exclusiones mediante comentarios para que nuestro equipo pueda seguir nuestro razonamiento.
