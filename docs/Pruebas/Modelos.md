# Testing de Modelos

El último aspecto de las pruebas de los sistemas de ML consiste en probar nuestros modelos durante el entrenamiento, la evaluación, la inferencia y el despliegue.

## Entrenamiento

Queremos escribir pruebas de forma iterativa mientras desarrollamos nuestros pipelines de entrenamiento para poder detectar errores rápidamente.
Esto es especialmente importante porque, a diferencia del software tradicional, los sistemas de ML pueden ejecutarse hasta el final sin lanzar ninguna excepción/error,
pero pueden producir sistemas incorrectos. Además, queremos detectar los errores rápidamente para ahorrar tiempo y computo.

* Comprobar las formas y los valores de los resultados del modelo

    ```python
    assert model(inputs).shape == torch.Size([len(inputs), num_classes])
    ```

* Comprobar la disminución de la pérdida después de una tanda de entrenamiento

    ```python
    assert epoch_loss < prev_epoch_loss
    ```

* Sobreajuste en un batch

    ```python
    accuracy = train(model, inputs=batches[0])
    assert accuracy == pytest.approx(0.95, abs=0.05) # 0.95 ± 0.05
    ```

* Entrenar hasta la finalización (pruebas de parada anticipada, guardado, etc.)

    ```python
    train(model)
    assert learning_rate >= min_learning_rate
    assert artifacts
    ```

* En diferentes dispositivos

    ```python
    assert train(model, device=torch.device("cpu"))
    assert train(model, device=torch.device("cuda"))
    ```

Puede marcar las pruebas de cálculo intensivo con un marcador pytest y sólo ejecutarlas cuando se produzca un cambio en el sistema que afecte al modelo.

```python
@pytest.mark.training
def test_train_model():
    ...
```

## Pruebas de comportamiento

Las pruebas de comportamiento (behavioral testing) son el proceso de probar los datos de entrada y los resultados esperados mientras se trata el modelo como una caja negra.
Un artículo de referencia sobre este tema es [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://arxiv.org/abs/2005.04118),
que desglosa las pruebas de comportamiento en tres tipos de pruebas:

* **invarianza**: Los cambios no deben afectar a los resultados.

    ```python
    # INVariance a través de inyección de verbos (los cambios no deberían afectar los resultados).
    tokens = ["revolutionized", "disrupted"]
    texts = [f"Transformers applied to NLP have {token} the ML field." for token in tokens]
    predict.predict(texts=texts, artifacts=artifacts)
    ```

* **direccional**: El cambio debe afectar a las salidas.

    ```python
    # Expectativas DIReccionales (cambios con resultados conocidos).
    tokens = ["text classification", "image classification"]
    texts = [f"ML applied to {token}." for token in tokens]
    predict.predict(texts=texts, artifacts=artifacts)
    ```

* **funcionalidad mínima**: Combinación simple de entradas y salidas previstas.

    ```python
    # Pruebas de Funcionalidad Mínima (pares simples de entrada/salida).
    tokens = ["natural language processing", "mlops"]
    texts = [f"{token} is the next big wave in machine learning." for token in tokens]
    predict.predict(texts=texts, artifacts=artifacts)
    ```

Y podemos convertir estas pruebas en pruebas sistemáticas parametrizadas:

```bash
mkdir tests/model
touch tests/model/test_behavioral.py
```

<details>
<summary>Ver tests/model/test_behavioral.py</summary>

```python
# tests/model/test_behavioral.py
from pathlib import Path

import pytest

from config import config
from coe_template import main, predict


@pytest.fixture(scope="module")
def artifacts():
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = main.load_artifacts(run_id=run_id)
    return artifacts


@pytest.mark.parametrize(
    "text, tag",
    [
        (
            "Transformers applied to NLP have revolutionized machine learning.",
            "natural-language-processing",
        ),
        (
            "Transformers applied to NLP have disrupted machine learning.",
            "natural-language-processing",
        ),
    ],
)
def test_inv(text, tag, artifacts):
    """INVariance a través de inyección de verbos (los cambios no deberían afectar los resultados)."""
    predicted_tag = predict.predict(texts=[text], artifacts=artifacts)[0]["predicted_tag"]
    assert tag == predicted_tag


@pytest.mark.parametrize(
    "text, tag",
    [
        (
            "ML applied to text classification.",
            "natural-language-processing",
        ),
        (
            "ML applied to image classification.",
            "computer-vision",
        ),
        (
            "CNNs for text classification.",
            "natural-language-processing",
        )
    ],
)
def test_dir(text, tag, artifacts):
    """Expectativas DIReccionales (cambios con resultados conocidos)."""
    predicted_tag = predict.predict(texts=[text], artifacts=artifacts)[0]["predicted_tag"]
    assert tag == predicted_tag


@pytest.mark.parametrize(
    "text, tag",
    [
        (
            "Natural language processing is the next big wave in machine learning.",
            "natural-language-processing",
        ),
        (
            "MLOps is the next big wave in machine learning.",
            "mlops",
        ),
        (
            "This should not produce any relevant topics.",
            "other",
        ),
    ],
)
def test_mft(text, tag, artifacts):
    """Pruebas de Funcionalidad Mínima (pares simples de entrada/salida)."""
    predicted_tag = predict.predict(texts=[text], artifacts=artifacts)[0]["predicted_tag"]
    assert tag == predicted_tag
```

</details>

## Inferencia

Cuando nuestro modelo se despliegue, los usuarios lo utilizarán para la inferencia (directa / indirectamente), por lo que es muy importante que probemos todos sus aspectos.

**Carga de artefactos**
Esta es la primera vez que no cargamos nuestros componentes desde la memoria, por lo que queremos asegurarnos de que los artefactos necesarios
(model weights, encoders, config, etc.) son capaces de ser cargados.

```python
artifacts = main.load_artifacts(run_id=run_id)
assert isinstance(artifacts["label_encoder"], data.LabelEncoder)
...
```

**Predicción**
Una vez que tenemos nuestros artefactos cargados, estamos listos para probar el pipeline de predicción. Deberíamos probar las muestras con una sola entrada, así como con un lote de entradas.

```python
# probar nuestra llamada API directamente
data = {
    "texts": [
        {"text": "Transfer learning with transformers for text classification."},
        {"text": "Generative adversarial networks in both PyTorch and TensorFlow."},
    ]
}
response = client.post("/predict", json=data)
assert response.status_code == HTTPStatus.OK
assert response.request.method == "POST"
assert len(response.json()["data"]["predictions"]) == len(data["texts"])
...
```
