# Documentación de código

Documentación de código para su equipo y su yo futuro.

## Introducción

Podemos organizar aún más nuestro código al documentarlo para que sea más fácil para otros (y para nosotros mismos en el futuro) navegarlo y extenderlo fácilmente.
La documentación puede significar muchas cosas diferentes para los desarrolladores, así que definamos los componentes más comunes:

* comentarios: breves descripciones de por qué existe un fragmento de código.
* tipificación: especificación de los tipos de datos de entrada y salida de una función, proporcionando información relacionada con lo que una función consume y produce.
* docstrings: descripciones significativas para funciones y clases que describen la utilidad general, argumentos, retornos, etc.
* docs: página web renderizada que resume todas las funciones, clases, flujos de trabajo, ejemplos, etc.

En la sección de CI/CD aprenderemos cómo crear automáticamente y mantener nuestros documentos actualizados cada vez que hagamos cambios en nuestra base de código.

## Tipificación

Es importante ser lo más explícito posible con nuestro código.
Además de la elección de nombres explícitos para variables, funciones, etc., otra forma en que podemos ser explícitos es definiendo los tipos para las entradas y salidas de nuestra función.
En lugar de:

```python
def some_function(a, b):
    return c
```

Podemos hacerlo más legible:

```python
from typing import List

def some_function(a: List, b: int = 0) -> np.ndarray:
    return c
```

A partir de Python 3.9+, los tipos comunes están integrados, por lo que ya no es necesario importarlos escribiendo import List, Set, Dict, Tuple, Sequence.

## Docstrings

Podemos hacer que nuestro código sea aún más explícito agregando docstrings para describir la utilidad general, los argumentos, las devoluciones, las excepciones y más. Por ejemplo:

```python
from typing import List

def some_function(a: List, b: int = 0) -> np.ndarray:
    """Descripción de la función.

    ```python
    c = some_function(a=[], b=0)
    print (c)
    ```
    <pre>
    [[1 2]
     [3 4]]
    </pre>

    Args:
        a (List): descripción de `a`.
        b (int, optional): descripción de `b`. Defaults to 0.

    Raises:
        ValueError: Input list is not one-dimensional.

    Returns:
        np.ndarray: Descripción de `c`.

    """
    return c
```

Tómese este tiempo para actualizar todas las funciones y clases en el proyecto con docstrings.
Tenga en cuenta que debe importar explícitamente algunas bibliotecas a ciertos scripts porque el tipo lo requiere.
Idealmente, agregaríamos docstrings a nuestras funciones y clases a medida que las desarrollamos, en lugar de hacerlo todo de una vez al final.
Si usa Visual Studio Code, asegúrese de usar la extensión [Python Docstrings Generator](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)
para que pueda generar una plantilla docstring con `ctrl+shift+2`.
Esta extensión autocompletará partes de docstring usando la información de escritura e incluso excepción en su código.

## Docs

Mediante paquetes de código abierto podemos recopilar toda la documentación dentro de nuestro código y mostrarlo automáticamente como documentación.

1. Instale los paquetes necesarios:

    ```bash
    python -m pip install mkdocs==1.3.0 mkdocstrings==0.18.1
    ```

    En lugar de agregar directamente estos requisitos a nuestro archivo `requirements.txt`, podemos a aislarlo de nuestras bibliotecas principales requeridas.

    ```bash
    touch requirements-docs.txt
    ```

    ```bash
    # Agregar a requirements-docs.txt
    mkdocs==1.3.0
    mkdocstrings==0.18.1
    ```

2. Inicialice mkdocs:

    ```bash
    python -m mkdocs new .
    ```

3. Comenzaremos por sobrescribir el archivo `index.md` predeterminado en nuestro directorio `docs` con información específica de nuestro proyecto:

    ```markdown
    index.md
    # CoE MLOps Template

    ## Proyecto Template

    - [Workflows](coe_template/main.md): Flujos de trabajo principales.
    - [coe_template](coe_template/data.md): documentación de la funcionalidad.

    ## Documentación paso a paso

    Aprenda a combinar machine learning con la ingeniería de software para crear aplicaciones de nivel de producción.

    - Codigo: [DJuanes/coe-mlops-playbook](https://github.com/DJuanes/coe-mlops-playbook)
    ```

4. Luego, crearemos archivos de documentación para cada script en nuestro directorio coe_template:

    ```bash
    mkdir docs/coe_template
    cd docs/coe_template
    touch main.md utils.md data.md train.md evaluate.md predict.md
    cd ../../
    ```

5. A continuación, agregaremos `coe_template.<SCRIPT_NAME>` a cada archivo en `docs/coe_template`.
   Esto llenará el archivo con información sobre las funciones y clases (usando sus docstrings) de `coe_template/<SCRIPT_NAME>.py` gracias al complemento `mkdocstrings`.

    ```markdown
    # docs/coe_template/data.md
    ::: coe_template.data
    ```

6. Finalmente, agregaremos algunas configuraciones a nuestro archivo mkdocs.yml:

    ```yaml
    # mkdocs.yml
    site_name: CoE MLOps Playbook
    site_url: https://coe-mlops.com/
    repo_url: https://github.com/DJuanes/coe-mlops-playbook/
    nav:
    - Home: index.md
    - workflows:
        - main: coe_template/main.md
    - coe_template:
        - data: coe_template/data.md
        - evaluate: coe_template/evaluate.md
        - predict: coe_template/predict.md
        - train: coe_template/train.md
        - utils: coe_template/utils.md
    theme: readthedocs
    plugins:
    - mkdocstrings
    watch:
    - .  # recargar documentos para cualquier cambio de archivo
    ```

7. Disponibilice la documentación localmente:

    ```bash
    python -m mkdocs serve
    ```

## Publicación

Podemos publicar fácilmente nuestra documentación de forma gratuita utilizando páginas de GitHub para repositorios públicos, así como documentación privada para repositorios privados.
E incluso podemos alojarlo en un dominio personalizado.
