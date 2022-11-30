# Empaquetado de código Python

Uso de configuraciones y entornos virtuales para crear un escenario de reproducción de resultados.

## Introducción

Vamos a definir explícitamente nuestro entorno para que podamos reproducirlo localmente y cuando lo implementemos en producción.
Hay muchas herramientas de gestión de dependencias y empaquetado, como Poetry, pero usaremos Pip, ya que es ampliamente adoptado.

## Terminal

Antes de que podamos comenzar a empaquetar, necesitamos una forma de crear archivos y ejecutar comandos. Esto lo podemos hacer a través de la terminal.
Todos los comandos que ejecutamos deben ser los mismos independientemente de su sistema operativo o lenguaje de programación de interfaz de línea de comandos (CLI).
Se recomienda utilizar [iTerm2](https://iterm2.com/)  (Mac) o [ConEmu](https://conemu.github.io/) (Windows) en lugar del terminal predeterminado por sus ricas funciones.

## Proyecto

Crearemos el directorio principal del proyecto para que podamos guardar allí nuestros componentes de empaquetado.

```bash
# Crear y cambiar al directorio
mkdir mlops
cd mlops
```

## Conda

Utilizaremos Conda como gestor de paquetes.
Luego de instalar Anaconda3, si queremos que los comandos conda estén disponibles en Bash, tenemos que agregar conda.sh al archivo .bashrc.
Navegamos a la carpeta donde tenemos instalado anaconda3, y luego a etc -> profile.d.
Hacemos clic derecho dentro del explorador de archivos y luego elejir "Git Bash Here".
Luego hay que ejecutar este comando en el terminal:

```bash
echo ". ${PWD}/conda.sh" >> ~/.bashrc
```

## Virtual environment

A continuación, configuraremos un entorno virtual para poder aislar los paquetes necesarios para nuestra aplicación.
Esto también mantendrá los componentes separados de otros proyectos que pueden tener diferentes dependencias.
Una vez que creamos nuestro entorno virtual, lo activaremos e instalaremos nuestros paquetes requeridos.

```bash
conda create --prefix venv python=3.9
```

Para eliminar el largo prefijo en el indicador de shell, modifique la configuración env_prompt en su archivo .condarc:

```bash
conda config --set env_prompt '({name})'
```

Esto editará su archivo .condarc si ya tiene uno o creará un archivo .condarc si no lo tiene.
Ahora lo podemos activar:

```bash
conda activate ./venv
# Verificamos que estamos usando 3.9
python --version
pip install pip setuptools wheel
```

Sabremos que nuestro entorno virtual está activo por su nombre en la terminal.

### Requirements

Crearemos un archivo separado llamado requirements.txt donde especificaremos los paquetes (con sus versiones) que queremos instalar.

```bash
touch requirements.txt
```

No es aconsejable instalar todos los paquetes y luego hacer pip freeze > requirements.txt porque vuelca las dependencias de todos nuestros paquetes en el archivo.
Para mitigar esto, existen herramientas como [pipreqs](https://github.com/bndr/pipreqs), [pip-tools](https://github.com/jazzband/pip-tools), [pipchill](https://github.com/rbanffy/pip-chill), etc.
que solo enumeran los paquetes que no son dependencias.

```bash
# requirements.txt
<PACKAGE>==<VERSION>  # version exacta
<PACKAGE>>=<VERSION>  # por encima de version
<PACKAGE>             # sin version
```

### Setup

Ahora crearemos un archivo llamado `setup.py` para proporcionar instrucciones sobre cómo configurar nuestro entorno virtual.

```bash
touch setup.py
```

```python
# setup.py
from pathlib import Path
from setuptools import find_namespace_packages, setup
```

Comenzaremos extrayendo los paquetes requeridos de `requirements.txt`:

```python
# Cargar paquetes desde requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]
```

El corazón del archivo `setup.py` es el objeto de instalación que describe cómo configurar nuestro paquete y sus dependencias.
Nuestro paquete se llamará `coe_template` y abarcará todos los requisitos necesarios para ejecutarlo.

```python
# setup.py
setup(
    name="coe_template",
    version=0.1,
    description="Clasificación de proyectos de machine learning.",
    author="Diego Juanes",
    author_email="diego.juanes@ypf.com",
    python_requires=">=3.9",
    install_requires=[required_packages],
)
```

## Uso

Podemos instalar nuestros paquetes así:

```bash
pip install -e .            # instala solo los paquetes requeridos
```
