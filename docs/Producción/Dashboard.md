# Dashboard

Creación de un dashboard interactivo para inspeccionar visualmente nuestra aplicación utilizando Streamlit.

## Introducción

Al desarrollar una aplicación, hay muchas decisiones técnicas y resultados que son parte integral de nuestro sistema.
Debemos poder comunicar esto de forma eficaz a otros desarrolladores y a las partes interesadas del negocio.
Necesitamos crear un dashboard al que se pueda acceder sin ningún requisito técnico previo y que comunique eficazmente los resultados clave.
Sería aún más útil si nuestro dashboard fuera interactivo, de modo que proporcionara utilidad incluso a los desarrolladores técnicos.

## Streamlit

Dado que muchos desarrolladores que trabajan en el aprendizaje automático utilizan Python,
el panorama de las herramientas para crear dashboards para ofrecer información orientada a los datos ha evolucionado para salvar este vacío.
Estas herramientas permiten ahora a los desarrolladores de ML crear dashboards interactivos y visualizaciones en Python,
al tiempo que ofrecen una personalización completa a través de HTML, JS y CSS.
Utilizaremos [Streamlit](https://streamlit.io/) para crear nuestros dashboards debido a su intuitiva API,
sus capacidades para compartir y su creciente adopción por parte de la comunidad.

## Configuración

Con Streamlit, podemos crear rápidamente una aplicación vacía y, a medida que vayamos desarrollando, la interfaz de usuario se irá actualizando también.

```bash
# Setup
python -m pip install streamlit==1.10.0
mkdir streamlit
touch streamlit/app.py
streamlit run streamlit/app.py
```

Asegúrese de añadir este paquete y la versión a nuestro archivo `requirements.txt`.

## Referencia de la API

Antes de crear un dashboard para nuestra aplicación, debemos conocer los diferentes componentes de Streamlit.
Tómate unos minutos y revisa la [referencia de la API](https://docs.streamlit.io/library/api-reference).
Vamos a explorar los diferentes componentes en detalle, ya que se aplican a la creación de diferentes interacciones para nuestro dashboard específico.

## Secciones

Comenzaremos por delinear las secciones que queremos tener en nuestro tablero editando nuestro script `streamlit/app.py`:

```python
import pandas as pd
from pathlib import Path
import streamlit as st

from config import config
from coe_template import main, utils
```

```python
# Título
st.title("CoE Template MLOps")

# Secciones
st.header("🔢 Datos")
st.header("📈 Performance")
st.header("🔮 Inferencia")
```

### Datos

Vamos a mantener la sencillez de nuestro dashboard, así que sólo mostraremos los proyectos etiquetados.

```python
st.header("🔢 Datos")
projects_fp = Path(config.DATA_DIR, "labeled_projects.csv")
df = pd.read_csv(projects_fp)
st.text(f"Proyectos (count: {len(df)})")
st.write(df)
```

### Performance

En esta sección, mostraremos el rendimiento de nuestro último modelo entrenado.
Una vez más, vamos a mantenerlo simple, pero también podríamos superponer más información, como las mejoras o regresiones de las implementaciones anteriores, accediendo al almacén de modelos.

```python
st.header("📈 Performance")
performance_fp = Path(config.CONFIG_DIR, "performance.json")
performance = utils.load_dict(filepath=performance_fp)
st.text("Overall:")
st.write(performance["overall"])
tag = st.selectbox("Elige una etiqueta: ", list(performance["class"].keys()))
st.write(performance["class"][tag])
tag = st.selectbox("Elige un slice: ", list(performance["slices"].keys()))
st.write(performance["slices"][tag])
```

### Inferencia

Con la sección de inferencia, queremos ser capaces de predecir rápidamente con el último modelo entrenado.

```python
st.header("🔮 Inferencia")
text = st.text_input("Ingrese texto:", "Transfer learning with transformers for text classification.")
run_id = st.text_input("Ingrese run ID:", open(Path(config.CONFIG_DIR, "run_id.txt")).read())
prediction = main.predict_tag(text=text, run_id=run_id)
st.write(prediction)
```

## Caching

A veces podemos tener vistas que implican operaciones computacionalmente pesadas, como la carga de datos o artefactos del modelo.
Es una buena práctica almacenar en caché estas operaciones envolviéndolas como una función separada con el decorador @st.cache.
Esto hace que Streamlit almacene en caché la función por la combinación específica de sus entradas para entregar las respectivas salidas cuando la función es invocada con las mismas entradas.

```python
@st.cache()
def load_data():
    projects_fp = Path(config.DATA_DIR, "labeled_projects.csv")
    df = pd.read_csv(projects_fp)
    return df
```

## Despliegue

Tenemos varias opciones para desplegar y gestionar nuestro dashboard de Streamlit.
Podemos utilizar la [función de compartir](https://blog.streamlit.io/introducing-streamlit-sharing/) de Streamlit que nos permite desplegar los dashboard directamente desde GitHub.
Nuestro dashboard se mantendrá actualizado a medida que vayamos introduciendo cambios en nuestro repositorio.
Otra opción es desplegarlo junto con nuestro servicio de API.
Podemos añadirlo al ENTRYPOINT del Dockerfile con los puertos apropiados expuestos.
