# Aplicaciones CLI (Command Line Interface)

Uso de una aplicación CLI (command line interface) para organizar los procesos de la aplicación.

## Introducción

Cuando se trata de servir nuestros modelos, debemos pensar en exponer la funcionalidad de la aplicación a nosotros mismos, a los miembros del equipo y,
en última instancia, a nuestros usuarios finales. Y las interfaces para lograr esto serán diferentes.
En las secciones anteriores ejecutamos nuestras operaciones principales a través de la terminal y el intérprete de Python.
Esto se vuelve extremadamente tedioso al tener que profundizar en el código y ejecutar funciones una a la vez.
Una solución es crear una aplicación de interfaz de línea de comandos (CLI) que permita la interacción a nivel operativo.
Debe diseñarse de tal manera que podamos ver todas las operaciones posibles (y sus argumentos requeridos) y ejecutarlas desde el shell.

## Aplicación

Vamos a crear nuestra CLI usando Typer:

```bash
python -m pip install typer==0.4.1
```

```bash
# Agregar a requirements.txt
typer==0.4.1
```

Es tan simple como inicializar la aplicación y luego agregar el decorador apropiado para cada operación de función
que deseamos usar como un comando CLI en nuestro `main.py`:

```python
# coe_template/main.py
import typer

# Inicializar la aplicación CLI de Typer
app = typer.Typer()
```

```python
@app.command()
def elt_data():
    ...
```

Repetiremos lo mismo para todas las demás funciones a las que queremos acceder a través de la CLI: elt_data(), train_model(), optimize(), predict_tag().
Haremos que todos los argumentos sean opcionales para que podamos definirlos explícitamente en nuestros comandos bash.

<details>
<summary>Ver encabezados de función coe_template/main.py</summary>

```python
@app.command()
def elt_data():
    ...


@app.command()
def train_model(
    args_fp: str = "config/args.json",
    experiment_name: str = "baselines",
    run_name: str = "sgd",
) -> None:
    ...


@app.command()
def optimize(
    args_fp: str = "config/args.json",
    study_name: str = "optimization",
    num_trials: int = 20
) -> None:
    ...

@app.command()
def predict_tag(text: str = "", run_id: str = None) -> None:
    ...
```

Al final del archivo debemos inicializar la aplicación CLI:

```python
if __name__ == "__main__":
    app()  # pragma: aplicación en vivo
```

</details>

## Comandos

Para usar nuestra aplicación CLI, primero podemos ver los comandos disponibles gracias a los decoradores:

```bash
python coe_template/main.py --help
```

## Argumentos

Con Typer, los argumentos de entrada de una función se representan automáticamente como opciones de línea de comandos.
Pero también podemos pedir ayuda con este comando específico sin tener que entrar en el código:

```bash
python coe_template/main.py predict-tag --help
```

## Uso

Finalmente, podemos ejecutar el comando específico con todos los argumentos:

```bash
python coe_template/main.py predict-tag --text="Transfer learning with transformers for text classification."
```
