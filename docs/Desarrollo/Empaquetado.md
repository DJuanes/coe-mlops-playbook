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
Crear y cambiar al directorio
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

A continuación, configuraremos un entorno virtual para poder aislar los paquetes necesarios para nuestra aplicación. Esto también mantendrá los componentes separados de otros proyectos
que pueden tener diferentes dependencias. Una vez que creamos nuestro entorno virtual, lo activaremos e instalaremos nuestros paquetes requeridos.

```bash
conda create --name .mlops python=3.9
conda activate .mlops
# Verificamos que estamos usando 3.9
python --version
python -m pip install pip setuptools wheel
```

Sabremos que nuestro entorno virtual está activo por su nombre en la terminal.

## Requirements

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

## Uso

Podemos instalar nuestros paquetes así:

```bash
python -m pip install -r requirements.txt            # instala solo los paquetes requeridos
```
