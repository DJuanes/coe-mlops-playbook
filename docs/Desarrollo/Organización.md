# Organización del código de Machine Learning

Organizar el código al pasar de notebooks a scripts de Python.

## Introducción

Tener un código organizado es tener un código legible, reproducible y robusto. Tu equipo y, también tu yo futuro, te agradecerán por hacer el esfuerzo inicial para organizar el trabajo.
Veremos cómo migrar y organizar el código de nuestro notebook a scripts de Python.

## Editor

Hay varias opciones para los editores de código, como VSCode, Atom, Sublime, PyCharm, Vim, etc.
y todos ofrecen funciones únicas al tiempo que proporcionan las operaciones básicas para la edición y ejecución de código.
Usaremos VSCode para editar y ejecutar nuestro código gracias a su simplicidad, soporte multilingüe, complementos y creciente adopción en la industria.

1. En Visual Studio Code abra la paleta de comandos (CTRL + SHIFT + P) y escriba "Preferences: Open Settings (UI)" y presione Enter.
2. Ajuste cualquier configuración relevante que desee (espaciado, tamaño de fuente, etc.)
3. Instale [extensiones de VSCode](https://marketplace.visualstudio.com/) (utilice el ícono de bloques de lego en el panel izquierdo del editor)

<details>
<summary>Extensiones VSCode recomendadas</summary>

Recomendamos instalar estas extensiones:

```bash
code --install-extension alefragnani.project-manager
code --install-extension bierner.markdown-preview-github-styles
code --install-extension christian-kohler.path-intellisense
code --install-extension euskadi31.json-pretty-printer
code --install-extension mechatroner.rainbow-csv
code --install-extension mikestead.dotenv
code --install-extension ms-azuretools.vscode-docker
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension njpwerner.autodocstring
code --install-extension PKief.material-icon-theme
code --install-extension redhat.vscode-yaml
code --install-extension shardulm94.trailing-spaces
code --install-extension streetsidesoftware.code-spell-checker-spanish
code --install-extension DavidAnson.vscode-markdownlint
code --install-extension donjayamanne.python-environment-manager
code --install-extension donjayamanne.python-extension-pack
code --install-extension equinusocio.vsc-material-theme-icons
code --install-extension etmoffat.pip-packages
code --install-extension KevinRose.vsc-python-indent
code --install-extension magicstack.MagicPython
code --install-extension mdickin.markdown-shortcuts
code --install-extension ms-azure-devops.azure-pipelines
code --install-extension ms-azuretools.vscode-azureappservice
code --install-extension ms-azuretools.vscode-azurefunctions
code --install-extension ms-azuretools.vscode-azureresourcegroups
code --install-extension ms-azuretools.vscode-azurestaticwebapps
code --install-extension ms-azuretools.vscode-azurestorage
code --install-extension ms-azuretools.vscode-azurevirtualmachines
code --install-extension ms-azuretools.vscode-bicep
code --install-extension ms-azuretools.vscode-cosmosdb
code --install-extension ms-python.isort
code --install-extension ms-toolsai.jupyter
code --install-extension ms-toolsai.jupyter-keymap
code --install-extension ms-toolsai.jupyter-renderers
code --install-extension ms-toolsai.vscode-ai
code --install-extension ms-toolsai.vscode-ai-remote
code --install-extension ms-toolsai.vscode-jupyter-cell-tags
code --install-extension ms-toolsai.vscode-jupyter-slideshow
code --install-extension ms-vscode-remote.remote-containers
code --install-extension ms-vscode-remote.remote-wsl
code --install-extension ms-vscode.azure-account
code --install-extension ms-vscode.azurecli
code --install-extension ms-vscode.powershell
code --install-extension ms-vscode.vscode-node-azure-pack
code --install-extension msazurermtools.azurerm-vscode-tools
code --install-extension VisualStudioExptTeam.vscodeintellicode
code --install-extension vscode-icons-team.vscode-icons
code --install-extension yzhang.markdown-all-in-one
```

Si agrega sus propias extensiones y desea compartirlas con otros, simplemente ejecute este comando para generar la lista de comandos:

```bash
code --list-extensions | xargs -L 1 echo code --install-extension
```

</details>

Luego de configurar VSCode, podemos comenzar creando nuestro directorio de proyectos, que usaremos para organizar todos nuestros scripts.
Hay muchas maneras de iniciar un proyecto, pero esta es nuestra ruta recomendada:

1. Utilice la terminal para crear un directorio (`mkdir <NOMBRE_PROYECTO>`).
2. Cambie al directorio del proyecto que acaba de crear (`cd <PROJECT_NAME>`).
3. Inicie VSCode desde este directorio escribiendo el `code .`
4. Abra una terminal dentro de VSCode (`View > Terminal`) para continuar creando scripts (`touch <NOMBRE_DE_ARCHIVO>`) o subdirectorios adicionales (`mkdir <SUBDIR>`) según sea necesario.

## Setup

### README

Comenzaremos nuestra organización con un archivo README.md, que proporcionará información sobre los archivos en nuestro directorio, instrucciones para ejecutar operaciones, etc.
Mantendremos constantemente actualizado este archivo para que podamos catalogar información para el futuro.

```bash
touch README.md
```

Comencemos agregando las instrucciones que usamos para crear un virtual environment:

```bash
# Dentro de README.md
conda create --prefix venv python=3.9
conda config --set env_prompt '({name})'
conda activate ./venv
pip install --upgrade pip setuptools wheel
pip install -e .
```

### Configuraciones

A continuación, crearemos un directorio de configuración llamado `config` donde podemos almacenar los componentes que serán necesarios para nuestra aplicación.
Dentro de este directorio, crearemos un `config.py` y un `args.json`.

```bash
mkdir config
touch config/config.py config/args.json
```

Dentro de `config.py`, agregaremos el código para definir las ubicaciones clave del directorio (agregaremos más configuraciones a medida que sean necesarias):

```bash
# config.py
from pathlib import Path

# Directorios
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
```

y dentro de `args.json`, agregaremos los parámetros que son relevantes para el procesamiento de datos y el entrenamiento del modelo.

```bash
{
    "shuffle": true,
    "subset": null,
    "min_freq": 75,
    "lower": true,
    "stem": false,
    "analyzer": "char",
    "ngram_max_range": 7,
    "alpha": 1e-4,
    "learning_rate": 1e-1,
    "power_t": 0.1,
    "num_epochs": 100
}
```

### Operaciones

Comenzaremos creando nuestro directorio de paquetes (`coe_template`) dentro de nuestro directorio de proyectos (`mlops`).
Dentro de este directorio de paquetes, crearemos un archivo `main.py` que definirá las operaciones principales que queremos tener disponibles.

```bash
mkdir coe_template
touch coe_template/main.py
```

Definiremos estas operaciones centrales dentro de `main.py` a medida que movemos el código de los notebooks a los scripts apropiados a continuación:

* `elt_data`: extrae, carga y transforma datos.
* `optimize`: ajustar los hiperparámetros para optimizar para el objetivo.
* `train_model`: entrena un modelo utilizando los mejores parámetros del estudio de optimización.
* `load_artifacts`: carga artefactos entrenados de una ejecución determinada.
* `predict_tag`: predecir una etiqueta para una entrada determinada.

### Utilidades

Es común tener procesos ad-hoc dentro de los notebooks porque mantiene el estado siempre que el notebook se esté ejecutando.
Por ejemplo, podemos establecer seeds en nuestros notebooks así:

```python
# Setear seeds
np.random.seed(seed)
random.seed(seed)
```

Pero en nuestros scripts, debemos envolver esta funcionalidad como una función limpia y reutilizable con los parámetros apropiados:

```python
def set_seeds(seed=42):
    """Setear seeds para la reproducibilidad."""
    np.random.seed(seed)
    random.seed(seed)
```

Podemos almacenar todas estas funciones dentro de un archivo `Utils.py` en el directorio de paquetes `coe_template`.

```bash
touch coe_template/utils.py
```

<details>
<summary>View utils.py</summary>

```python
import json
import numpy as np
import random

def load_dict(filepath):
    """Cargar un diccionario desde la ruta de archivo de un JSON."""
    with open(filepath, "r") as fp:
        d = json.load(fp)
    return d

def save_dict(d, filepath, cls=None, sortkeys=False):
    """Guardar un diccionario en una ubicación específica."""
    with open(filepath, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)

def set_seeds(seed=42):
    """Setear seeds para la reproducibilidad."""
    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
```

</details>

## Proyecto

Cuando se trata de migrar nuestro código de notebooks a scripts, es mejor organizarse en función de la utilidad.
Por ejemplo, podemos crear scripts para las diversas etapas del desarrollo de ML, como procesamiento de datos, entrenamiento, evaluación, predicción, etc.
Crearemos los diferentes archivos de Python para envolver nuestros datos y funcionalidad ML:

```bash
cd coe_template
touch data.py train.py evaluate.py predict.py
```

Es posible que tengamos scripts adicionales en otros proyectos, ya que son necesarios. Por ejemplo, normalmente tendríamos un script modelos.py.
Organizar nuestra base de código de esta manera también nos hace más fácil comprender (o modificar) la base de código.
Podríamos haber colocado todo el código en un script main.py, pero a medida que nuestro proyecto crezca, será difícil navegar por un archivo monolítico.
Por otro lado, podríamos haber asumido una postura más granular descomponiendo datos.py en split.py, preprocess.py, etc.
Esto podría tener más sentido si tenemos múltiples formas de dividir, preprocesamiento, etc. Pero en general, es suficiente estar en este nivel más alto de organización.

## Principios

A través del proceso de migración utilizaremos varios principios básicos de ingeniería de software repetidamente:

### Envolver la funcionalidad en funciones

¿Cómo decidimos cuándo deben envolver líneas de código específicas como una función separada?
Las funciones deben ser atómicas porque cada una tiene una sola responsabilidad para que podamos probarlas fácilmente.
Si no, necesitaremos dividirlos en unidades más granulares. Por ejemplo, podríamos reemplazar las etiquetas en nuestros proyectos con estas líneas:

```python
oos_tags = [item for item in df.tag.unique() if item not in tags_dict.keys()]
df.tag = df.tag.apply(lambda x: "other" if x in oos_tags else x)
```

Refactorizado quedaría así:

```python
def replace_oos_tags(df, tags_dict):
    """Reemplazar etiquetas fuera de alcance (oos)."""
    oos_tags = [item for item in df.tag.unique() if item not in tags_dict.keys()]
    df.tag = df.tag.apply(lambda x: "other" if x in oos_tags else x)
    return df
```

### Componer funciones generalizadas

```python
def replace_oos_tags(df, tags_dict):
    """Reemplazar etiquetas fuera de alcance (oos)."""
    oos_tags = [item for item in df.tag.unique() if item not in tags_dict.keys()]
    df.tag = df.tag.apply(lambda x: "other" if x in oos_tags else x)
    return df
```

Versus:

```python
def replace_oos_labels(df, labels, label_col, oos_label="other"):
    """Reemplazar etiquetas fuera de alcance (oos)."""
    oos_tags = [item for item in df[label_col].unique() if item not in labels]
    df[label_col] = df[label_col].apply(lambda x: oos_label if x in oos_tags else x)
    return df
```

De esta manera, cuando cambian los nombres de las columnas o queremos reemplazar con diferentes etiquetas, es muy fácil ajustar nuestro código.
Esto también incluye el uso de nombres generalizados en las funciones como la etiqueta en lugar del nombre de la columna de etiqueta específica.
También permite a otros reutilizar esta funcionalidad para sus casos de uso.

## Data

### Load

<details>
<summary>Cargar y guardar datos</summary>

Primero, nombraremos y crearemos el directorio para guardar nuestros activos de datos

```python
# config/config.py
from pathlib import Path

# Directorios
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")

# Crear carpeta
DATA_DIR.mkdir(parents=True, exist_ok=True)
```

A continuación, agregaremos la ubicación de nuestros activos de datos sin procesar a nuestro `config.py`.
Es importante que almacenemos esta información en nuestro archivo de configuración central para que podamos descubrirlo y actualizarla fácilmente.

```python
# config/config.py
...
# Activos
PROJECTS_URL = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/projects.csv"
TAGS_URL = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/tags.csv"
```

Definiremos esta operación en `main.py`:

```python
# coe_template/main.py
import pandas as pd
from pathlib import Path
import warnings

from config import config
from coe_template import utils

warnings.filterwarnings("ignore")

def elt_data():
    """Extraer, cargar y transformar nuestros activos de datos."""
    # Extract + Load
    projects = pd.read_csv(config.PROJECTS_URL)
    tags = pd.read_csv(config.TAGS_URL)
    projects.to_csv(Path(config.DATA_DIR, "projects.csv"), index=False)
    tags.to_csv(Path(config.DATA_DIR, "tags.csv"), index=False)

    # Transform
    df = pd.merge(projects, tags, on="id")
    df = df[df.tag.notnull()]  # eliminamos filas sin tags
    df.to_csv(Path(config.DATA_DIR, "labeled_projects.csv"), index=False)

    print("✅ Datos guardados!")
```

Debemos asegurarnos de tener los paquetes necesarios cargados en nuestro entorno.
Cargamos los paquetes requeridos y los agregemos a nuestro archivo `requirements.txt`:

```bash
pip install numpy==1.19.5 pandas==1.3.5 pretty-errors==1.2.19
```

```bash
# Agregar a requirements.txt
numpy==1.19.5
pandas==1.3.5
pretty-errors==1.2.19
```

Ejecutaremos la operación utilizando el intérprete de Python a través del terminal (escriba `python` en la terminal y luego los comandos a continuación).

```python
from coe_template import main
main.elt_data()
```

</details>

### Preprocesamiento

<details>
<summary>Preprocesamiento de features</summary>

A continuación, definiremos las funciones para preprocesar nuestros features de entrada.
Usaremos estas funciones cuando estemos preparando los datos antes de entrenar nuestro modelo.
No guardaremos los datos preprocesados en un archivo porque diferentes experimentos pueden preprocesarlos de manera diferente.

```python
# coe_template/data.py
def preprocess(df, lower, stem, min_freq):
    """Preprocesar los datos."""
    df["text"] = df.title + " " + df.description  # feature engineering
    df.text = df.text.apply(clean_text, lower=lower, stem=stem)  # limpiar texto
    df = replace_oos_labels(
        df=df, labels=config.ACCEPTED_TAGS, label_col="tag", oos_label="other"
    )  # reemplazar etiquetas OOS
    df = replace_minority_labels(
        df=df, label_col="tag", min_freq=min_freq, new_label="other"
    )  # reemplazar las etiquetas por debajo de la frecuencia mínima

    return df
```

Esta función usa la función `clean_text()`:

```python
# coe_template/data.py
from nltk.stem import PorterStemmer
import re

from config import config

def clean_text(text, lower=True, stem=False, stopwords=config.STOPWORDS):
    """Limpiar texto sin procesar."""
    # Lower
    if lower:
        text = text.lower()

    # Remover stopwords
    if len(stopwords):
        pattern = re.compile(r'\b(' + r"|".join(stopwords) + r")\b\s*")
        text = pattern.sub('', text)

    # Espaciado y filtros
    text = re.sub(
        r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text
    )  # añadir espacio entre los objetos a filtrar
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # eliminar caracteres no alfanuméricos
    text = re.sub(" +", " ", text)  # eliminar múltiples espacios
    text = text.strip()  # eliminar espacios en blanco en los extremos

    # Quitar links
    text = re.sub(r"http\S+", "", text)

    # Stemming
    if stem:
        stemmer = PorterStemmer()
        text = " ".join([stemmer.stem(word, to_lowercase=lower) for word in text.split(" ")])

    return text
```

Instale los paquetes requeridos y agréguelos a `requirements.txt`:

```bash
pip install nltk==3.7
```

```bash
# Agregar a requirements.txt
nltk==3.7
```

Tenga en cuenta que estamos usando un conjunto explícito de stopwords en lugar de la lista predeterminada de NLTK.
Esto se debe a que queremos tener una visibilidad completa de exactamente qué palabras estamos filtrando.
La lista general puede tener algunos términos valiosos que deseamos conservar y viceversa.
También agregamos la lista de tags permitidos.

```python
# config/config.py
ACCEPTED_TAGS = ["natural-language-processing", "computer-vision", "mlops", "graph-learning"]
STOPWORDS = [
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "you're",
    "you've",
    "you'll",
    "you'd",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "she's",
    "her",
    "hers",
    "herself",
    "it",
    "it's",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "that'll",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "don't",
    "should",
    "should've",
    "now",
    "d",
    "ll",
    "m",
    "o",
    "re",
    "ve",
    "y",
    "ain",
    "aren",
    "aren't",
    "couldn",
    "couldn't",
    "didn",
    "didn't",
    "doesn",
    "doesn't",
    "hadn",
    "hadn't",
    "hasn",
    "hasn't",
    "haven",
    "haven't",
    "isn",
    "isn't",
    "ma",
    "mightn",
    "mightn't",
    "mustn",
    "mustn't",
    "needn",
    "needn't",
    "shan",
    "shan't",
    "shouldn",
    "shouldn't",
    "wasn",
    "wasn't",
    "weren",
    "weren't",
    "won",
    "won't",
    "wouldn",
    "wouldn't",
]
```

A continuación, debemos definir las dos funciones a las que llamamos desde `data.py`:

```python
# coe_template/data.py
from collections import Counter

def replace_oos_labels(df, labels, label_col, oos_label="other"):
    """Reemplazar las etiquetas fuera de alcance (oos)."""
    oos_tags = [item for item in df[label_col].unique() if item not in labels]
    df[label_col] = df[label_col].apply(lambda x: oos_label if x in oos_tags else x)
    return df


def replace_minority_labels(df, label_col, min_freq, new_label="other"):
    """Reemplazar las etiquetas minoritarias con otra etiqueta."""
    labels = Counter(df[label_col].values)
    labels_above_freq = Counter(label for label in labels.elements() if (labels[label] >= min_freq))
    df[label_col] = df[label_col].apply(lambda label: label if label in labels_above_freq else None)
    df[label_col] = df[label_col].fillna(new_label)
    return df
```

</details>

### Etiquetado

<details>
<summary>Codificar etiquetas</summary>

Ahora definamos el codificador para nuestras etiquetas:

```python
# coe_template/data.py
import json
import numpy as np

class LabelEncoder():
    """Codificar las etiquetas en índices únicos."""
    def __init__(self, class_to_index={}):
        self.class_to_index = class_to_index or {}  # mutable defaults ;)
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())

    def __len__(self):
        return len(self.class_to_index)

    def __str__(self):
        return f"<LabelEncoder(num_classes={len(self)})>"

    def fit(self, y):
        classes = np.unique(y)
        for i, class_ in enumerate(classes):
            self.class_to_index[class_] = i
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())
        return self

    def encode(self, y):
        encoded = np.zeros((len(y)), dtype=int)
        for i, item in enumerate(y):
            encoded[i] = self.class_to_index[item]
        return encoded

    def decode(self, y):
        classes = []
        for i, item in enumerate(y):
            classes.append(self.index_to_class[item])
        return classes

    def save(self, fp):
        with open(fp, "w") as fp:
            contents = {"class_to_index": self.class_to_index}
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp):
        with open(fp, "r") as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)
```

</details>

### Split

<details>
<summary>Split dataset</summary>

Y finalmente, concluiremos nuestras operaciones de datos con nuestra función de división:

```python
from sklearn.model_selection import train_test_split

def get_data_splits(X, y, train_size=0.7):
    """Generar divisiones de datos equilibradas."""
    X_train, X_, y_train, y_ = train_test_split(
        X, y, train_size=train_size, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_, y_, train_size=0.5, stratify=y_)
    return X_train, X_val, X_test, y_train, y_val, y_test
```

Instale los paquetes requeridos y agréguelos a `requirements.txt`:

```bash
pip install scikit-learn==0.24.2
```

```bash
# Agregar a requirements.txt
scikit-learn==0.24.2
```

</details>

## Modelado

### Entrenamiento

<details>
<summary>Entrenar con argumentos predeterminados</summary>

Comenzaremos definiendo la operación en nuestro `main.py`:

```python
# coe_template/main.py
import json
from argparse import Namespace
from coe_template import data, train, utils


def train_model(args_fp):
    """Entrenar un modelo con argumentos dados."""
    # Cargar datos etiquetados
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))

    # Entrenar
    args = Namespace(**utils.load_dict(filepath=args_fp))
    artifacts = train.train(df=df, args=args)
    performance = artifacts["performance"]
    print(json.dumps(performance, indent=2))
```

Agregaremos más a nuestra operación `train_model()` cuando tengamos en cuenta el seguimiento de experimentos pero, por ahora, es bastante simple.

```python
# coe_template/train.py
from imblearn.over_sampling import RandomOverSampler
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

from coe_template import data, predict, utils, evaluate


def train(args, df, trial=None):
    """Entrenar modelo en datos."""

    # Setup
    utils.set_seeds()
    if args.shuffle: df = df.sample(frac=1).reset_index(drop=True)
    df = df[: args.subset]  # Ninguno = todas las muestras
    df = data.preprocess(df, lower=args.lower, stem=args.stem, min_freq=args.min_freq)
    label_encoder = data.LabelEncoder().fit(df.tag)
    X_train, X_val, X_test, y_train, y_val, y_test = \
        data.get_data_splits(X=df.text.to_numpy(), y=label_encoder.encode(df.tag))
    test_df = pd.DataFrame({"text": X_test, "tag": label_encoder.decode(y_test)})

    # Tf-idf
    vectorizer = TfidfVectorizer(analyzer=args.analyzer, ngram_range=(2,args.ngram_max_range))  # char n-grams
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)
    X_test = vectorizer.transform(X_test)

    # Oversample
    oversample = RandomOverSampler(sampling_strategy="all")
    X_over, y_over = oversample.fit_resample(X_train, y_train)

    # Model
    model = SGDClassifier(
        loss="log", penalty="l2", alpha=args.alpha, max_iter=1,
        learning_rate="constant", eta0=args.learning_rate, power_t=args.power_t,
        warm_start=True)

    # Training
    for epoch in range(args.num_epochs):
        model.fit(X_over, y_over)
        train_loss = log_loss(y_train, model.predict_proba(X_train))
        val_loss = log_loss(y_val, model.predict_proba(X_val))
        if not epoch%10:
            print(
                f"Epoch: {epoch:02d} | "
                f"train_loss: {train_loss:.5f}, "
                f"val_loss: {val_loss:.5f}"
            )

    # Threshold
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)
    args.threshold = np.quantile(
        [y_prob[i][j] for i, j in enumerate(y_pred)], q=0.25)  # Q1

    # Evaluacion
    other_index = label_encoder.class_to_index["other"]
    y_prob = model.predict_proba(X_test)
    y_pred = predict.custom_predict(y_prob=y_prob, threshold=args.threshold, index=other_index)
    performance = evaluate.get_metrics(
        y_true=y_test, y_pred=y_pred, classes=label_encoder.classes, df=test_df
    )

    return {
        "args": args,
        "label_encoder": label_encoder,
        "vectorizer": vectorizer,
        "model": model,
        "performance": performance,
    }
```

Esta función `train()` llama a dos funciones externas (`predict.custom_predict()` de `predict.py` y `Evaluation.get_metrics()` de `Evaluation.py`):

```python
# coe_template/predict.py
import numpy as np

def custom_predict(y_prob, threshold, index):
    """Función de predicción personalizada que por defecto
    es un índice si no se cumplen las condiciones."""
    y_pred = [np.argmax(p) if max(p) > threshold else index for p in y_prob]
    return np.array(y_pred)
```

```python
# coe_template/evaluate.py
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from snorkel.slicing import PandasSFApplier
from snorkel.slicing import slicing_function

@slicing_function()
def nlp_cnn(x):
    """Proyectos de NLP que utilizan convolución."""
    nlp_projects = "natural-language-processing" in x.tag
    convolution_projects = "CNN" in x.text or "convolution" in x.text
    return (nlp_projects and convolution_projects)

@slicing_function()
def short_text(x):
    """Proyectos con títulos y descripciones cortos."""
    return len(x.text.split()) < 8  # menos de 8 palabras

def get_slice_metrics(y_true, y_pred, slices):
    """Generar métricas para segmentos de datos."""
    metrics = {}
    for slice_name in slices.dtype.names:
        mask = slices[slice_name].astype(bool)
        if sum(mask):
            slice_metrics = precision_recall_fscore_support(
                y_true[mask], y_pred[mask], average="micro"
            )
            metrics[slice_name] = {}
            metrics[slice_name]["precision"] = slice_metrics[0]
            metrics[slice_name]["recall"] = slice_metrics[1]
            metrics[slice_name]["f1"] = slice_metrics[2]
            metrics[slice_name]["num_samples"] = len(y_true[mask])
    return metrics

def get_metrics(y_true, y_pred, classes, df=None):
    """Métricas de rendimiento utilizando verdades y predicciones."""
    # Performance
    metrics = {"overall": {}, "class": {}}

    # Métricas generales
    overall_metrics = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    metrics["overall"]["precision"] = overall_metrics[0]
    metrics["overall"]["recall"] = overall_metrics[1]
    metrics["overall"]["f1"] = overall_metrics[2]
    metrics["overall"]["num_samples"] = np.float64(len(y_true))

    # Métricas por clase
    class_metrics = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i, _class in enumerate(classes):
        metrics["class"][_class] = {
            "precision": class_metrics[0][i],
            "recall": class_metrics[1][i],
            "f1": class_metrics[2][i],
            "num_samples": np.float64(class_metrics[3][i]),
        }

    # Métricas de secciones
    if df is not None:
        slices = PandasSFApplier([nlp_cnn, short_text]).apply(df)
        metrics["slices"] = get_slice_metrics(
            y_true=y_true, y_pred=y_pred, slices=slices)

    return metrics
```

Instale los paquetes requeridos y agréguelos a `requirements.txt`:

```bash
pip install imbalanced-learn==0.8.1 snorkel==0.9.8
```

```bash
# Agregar a requirements.txt
imbalanced-learn==0.8.1
snorkel==0.9.8
```

Comandos para entrenar un modelo:

```python
from pathlib import Path
from config import config
from coe_template import main
args_fp = Path(config.CONFIG_DIR, "args.json")
main.train_model(args_fp)
```

Puede ser que aparezca un mensaje como este:

```bash
TypeError: Descriptors cannot not be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
```

Para solucionar este error hay que hacer un downgrade de protobuf:

```bash
pip install protobuf==3.20.*
```

```bash
# Agregar a requirements.txt
protobuf==3.20.*
```

Luego vuelva a ejecutar los comandos para entrenar el modelo.

</details>

### Optimización

<details>
<summary>Optimizar argumentos</summary>

Ahora que podemos entrenar un modelo, estamos listos para entrenar muchos modelos para optimizar nuestros hiperparámetros:

```python
# coe_template/main.py
import mlflow
from numpyencoder import NumpyEncoder
import optuna
from optuna.integration.mlflow import MLflowCallback


def optimize(args_fp, study_name, num_trials):
    """Optimizar hiperparámetros."""
    # Cargar datos etiquetados
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))

    # Optimizar
    args = Namespace(**utils.load_dict(filepath=args_fp))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(study_name="optimization", direction="maximize", pruner=pruner)
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(), metric_name="f1")
    study.optimize(
        lambda trial: train.objective(args, df, trial),
        n_trials=num_trials,
        callbacks=[mlflow_callback])

    # Mejor prueba
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(["user_attrs_f1"], ascending=False)
    utils.save_dict({**args.__dict__, **study.best_trial.params}, args_fp, cls=NumpyEncoder)
    print(f"\nMejor valor (f1): {study.best_trial.value}")
    print(f"Mejores hiperparámetros: {json.dumps(study.best_trial.params, indent=2)}")
```

Definiremos la función `objective()` dentro de `train.py`:

```python
# coe_template/train.py
def objective(args, df, trial):
    """Función objetivo para pruebas de optimización."""
    # Parámetros a tunear
    args.analyzer = trial.suggest_categorical("analyzer", ["word", "char", "char_wb"])
    args.ngram_max_range = trial.suggest_int("ngram_max_range", 3, 10)
    args.learning_rate = trial.suggest_loguniform("learning_rate", 1e-2, 1e0)
    args.power_t = trial.suggest_uniform("power_t", 0.1, 0.5)

    # Entrenar & evaluatar
    artifacts = train(args=args, df=df, trial=trial)

    # Establecer atributos adicionales
    overall_performance = artifacts["performance"]["overall"]
    print(json.dumps(overall_performance, indent=2))
    trial.set_user_attr("precision", overall_performance["precision"])
    trial.set_user_attr("recall", overall_performance["recall"])
    trial.set_user_attr("f1", overall_performance["f1"])

    return overall_performance["f1"]
```

Recuerde que en nuestro notebook modificamos la función `train()` para incluir información sobre las pruebas durante la optimización para pruning:

```python
# coe_template/train.py
import optuna

def train():
    ...
    # Entrenamiento
    for epoch in range(args.num_epochs):
        ...
        # Pruning (para optimización en la siguiente sección)
        if trial:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
```

Dado que estamos usando `MLflowCallback` aquí con Optuna, podemos permitir que todos nuestros experimentos se almacenen en el directorio `mlruns` predeterminado
que creará MLflow o podemos configurar esa ubicación:

```python
# config/config.py
import mlflow

# Directorios
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")
STORES_DIR = Path(BASE_DIR, "stores")

# Stores
MODEL_REGISTRY = Path(STORES_DIR, "model")

# Crear carpetas
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)

# Registro del modelo MLFlow
mlflow.set_tracking_uri("file:\\" + str(MODEL_REGISTRY.absolute()))
```

Instale los paquetes requeridos y agréguelos a `requirements.txt`:

```bash
pip install mlflow==1.23.1 optuna==2.10.0 numpyencoder==0.3.0
```

```bash
# Agregar a requirements.txt
mlflow==1.23.1
numpyencoder==0.3.0
optuna==2.10.0
```

Comandos para optimizar hiperparámetros:

```python
from pathlib import Path
from config import config
from coe_template import main
args_fp = Path(config.CONFIG_DIR, "args.json")
main.optimize(args_fp, study_name="optimization", num_trials=20)
```

</details>

### Seguimiento de experimentos

<details>
<summary>Seguimiento de experimentos</summary>

Ahora que tenemos nuestros hiperparámetros optimizados, podemos entrenar un modelo y almacenar sus artefactos a través del seguimiento de experimentos.
Comenzaremos modificando la operación `train()` en nuestro script `main.py`:

```python
# coe_template/main.py
import joblib
import tempfile

def train_model(args_fp, experiment_name, run_name):
    """Entrenar un modelo con argumentos dados."""
    # Cargar datos etiquetados
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))

    # Entrenar
    args = Namespace(**utils.load_dict(filepath=args_fp))
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")
        artifacts = train.train(df=df, args=args)
        performance = artifacts["performance"]
        print(json.dumps(performance, indent=2))

        # Logeo de métricas y parámetros
        performance = artifacts["performance"]
        mlflow.log_metrics({"precision": performance["overall"]["precision"]})
        mlflow.log_metrics({"recall": performance["overall"]["recall"]})
        mlflow.log_metrics({"f1": performance["overall"]["f1"]})
        mlflow.log_params(vars(artifacts["args"]))

        # Logeo de artifacts
        with tempfile.TemporaryDirectory() as dp:
            utils.save_dict(vars(artifacts["args"]), Path(dp, "args.json"), cls=NumpyEncoder)
            artifacts["label_encoder"].save(Path(dp, "label_encoder.json"))
            joblib.dump(artifacts["vectorizer"], Path(dp, "vectorizer.pkl"))
            joblib.dump(artifacts["model"], Path(dp, "model.pkl"))
            utils.save_dict(performance, Path(dp, "performance.json"))
            mlflow.log_artifacts(dp)

    # Guardar en config
    open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
    utils.save_dict(performance, Path(config.CONFIG_DIR, "performance.json"))
```

También vamos a actualizar la función `train()` dentro de `train.py` para que se capturen las métricas intermedias:

```python
# coe_template/train.py
import mlflow

def train():
    ...
    # Entrenamiento
    for epoch in range(args.num_epochs):
        ...
        # Log
        if not trial:
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
```

Comandos para entrenar un modelo con seguimiento de experimentos:

```python
from pathlib import Path
from config import config
from coe_template import main
args_fp = Path(config.CONFIG_DIR, "args.json")
main.train_model(args_fp, experiment_name="baselines", run_name="sgd")
```

Nuestro directorio de configuración ahora debería tener un archivo performance.json y un archivo run_id.txt.
Los estamos guardando para poder acceder rápidamente a estos metadatos del último entrenamiento exitoso.

</details>

### Predicción

<details>
<summary>Predecir textos</summary>

Finalmente estamos listos para usar nuestro modelo entrenado para la inferencia. Agregaremos la operación para predecir una etiqueta a `main.py`:

```python
# coe_template/main.py
from coe_template import data, predict, train, utils

def predict_tag(text, run_id=None):
    """Predecir etiqueta para texto."""
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = load_artifacts(run_id=run_id)
    prediction = predict.predict(texts=[text], artifacts=artifacts)
    print(json.dumps(prediction, indent=2))
    return prediction
```

Esto implica crear la función `load_artifacts()` dentro de nuestro script `main.py`:

```python
# coe_template/main.py
def load_artifacts(run_id):
    """Cargar artefactos para un run_id determinado."""
    # Localizar el directorio específicos de los artefactos
    experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
    artifacts_dir = Path(config.MODEL_REGISTRY, experiment_id, run_id, "artifacts")

    # Cargar objetos desde la ejecución
    args = Namespace(**utils.load_dict(filepath=Path(artifacts_dir, "args.json")))
    vectorizer = joblib.load(Path(artifacts_dir, "vectorizer.pkl"))
    label_encoder = data.LabelEncoder.load(fp=Path(artifacts_dir, "label_encoder.json"))
    model = joblib.load(Path(artifacts_dir, "model.pkl"))
    performance = utils.load_dict(filepath=Path(artifacts_dir, "performance.json"))

    return {
        "args": args,
        "label_encoder": label_encoder,
        "vectorizer": vectorizer,
        "model": model,
        "performance": performance
    }
```

Y definir la función `predict()` dentro de `predict.py`:

```python
def predict(texts, artifacts):
    """Predecir etiquetas para textos dados."""
    x = artifacts["vectorizer"].transform(texts)
    y_pred = custom_predict(
        y_prob=artifacts["model"].predict_proba(x),
        threshold=artifacts["args"].threshold,
        index=artifacts["label_encoder"].class_to_index["other"])
    tags = artifacts["label_encoder"].decode(y_pred)
    predictions = [
        {
            "input_text": texts[i],
            "predicted_tags": tags[i],
        }
        for i in range(len(tags))
    ]
    return predictions
```

Comandos para predecir la etiqueta de texto:

```python
from pathlib import Path
from config import config
from coe_template import main
text = "Transfer learning with transformers for text classification."
run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
main.predict_tag(text=text, run_id=run_id)
```

En la sección de documentación veremos como formatear nuestras funciones y clases correctamente.

</details>
