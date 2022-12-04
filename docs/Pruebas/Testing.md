# Pruebas de sistemas de Machine Learning: Código, datos y modelos

Aprenda a probar modelos de ML (y su código y datos) para garantizar un comportamiento coherente.

## Introducción

En esta sección veremos como probar el código, los datos y los modelos para construir un sistema de ML con el que podamos iterar de forma fiable.
Las pruebas son una forma de asegurarnos de que algo funciona según lo previsto.
Las pruebas son muy útiles para descubrir las fuentes de error tan pronto como sea posible para que podamos reducir los costos posteriores y el tiempo perdido.
Una vez que hemos diseñado nuestras pruebas, podemos ejecutarlas automáticamente cada vez que cambiamos o añadimos algo a nuestra base de código.

## Tipos de pruebas

Hay varios tipos de pruebas que se utilizan en diferentes puntos del ciclo de desarrollo, entre los que se encuentran:

* Pruebas unitarias: pruebas sobre componentes individuales que tienen una única responsabilidad.
* Pruebas de integración: pruebas sobre la funcionalidad combinada de los componentes individuales.
* Pruebas de sistema: pruebas sobre el diseño de un sistema para los resultados esperados en función de las entradas.
* Pruebas de aceptación: pruebas para verificar el cumplimiento de los requisitos, normalmente denominadas pruebas de aceptación del usuario (UAT).
* Pruebas de regresión: pruebas basadas en errores que hemos visto antes para garantizar que los nuevos cambios no los reintroduzcan.

La distinción entre las pruebas de los sistemas de ML con los sistemas tradicionales comienza cuando pasamos de probar el código a probar los datos y los modelos.
También hay muchos otros tipos de pruebas funcionales y no funcionales, como las pruebas de humo, las pruebas de rendimiento, las pruebas de seguridad, etc.,
pero podemos generalizar todas ellas bajo las pruebas del sistema.

## ¿Cómo debemos hacer las pruebas?

El framework que hay que utilizar para componer las pruebas es la metodología Arrange Act Assert.

* Arrange: establecer las diferentes entradas que se van a probar.
* Act: aplicar las entradas en el componente que queremos probar.
* Assert: confirmar que recibimos la salida esperada.

En Python, hay muchas herramientas, como unittest, pytest, etc. que nos permiten implementar fácilmente nuestras pruebas mientras se adhieren al framework de Arrange Act Assert.
Estas herramientas vienen con una potente funcionalidad incorporada, como la parametrización, los filtros, y más, para probar muchas condiciones a escala.

## ¿Qué debemos testear?

Algunos de los aspectos que deberíamos probar son:

* inputs: tipos de datos, formato, longitud, casos límite.
* outputs: tipos de datos, formatos, excepciones, salidas intermedias y finales.

## Mejores prácticas

Independientemente del framework que utilicemos, es importante vincular fuertemente las pruebas al proceso de desarrollo.

* **atomicidad**: al crear funciones y clases, debemos asegurarnos de que tienen una única responsabilidad para poder probarlas fácilmente. Si no es así, tendremos que dividirlas en componentes más granulares.
* **composición**: cuando creamos nuevos componentes, queremos componer pruebas para validar su funcionalidad. Es una buena manera de garantizar la fiabilidad y detectar errores desde el principio.
* **reutilización**: debemos mantener repositorios centrales en los que se pruebe la funcionalidad principal en el origen y se reutilice en muchos proyectos.
  Esto reduce significativamente los esfuerzos de prueba para la base de código de cada nuevo proyecto.
* **regresión**: queremos tener en cuenta los nuevos errores que encontremos con una prueba de regresión para asegurarnos de no reintroducir los mismos errores en el futuro.
* **cobertura**: queremos garantizar una cobertura lo más cercana al 100% de nuestra base de código.
* **automatización**: queremos autoejecutar las pruebas cuando hagamos cambios en nuestra base de código.

## [Código](C%C3%B3digo.md)

## [Data](Datos.md)

## [Modelos](Modelos.md)

## Makefile

Vamos a crear un target en nuestro Makefile que nos permitirá ejecutar todas nuestras pruebas con una sola llamada:

```bash
# Test
.PHONY: test
test:
    pytest -m "not training"
    cd tests && great_expectations checkpoint run projects
    cd tests && great_expectations checkpoint run tags
    cd tests && great_expectations checkpoint run labeled_projects
```

```bash
make test
```

## Testing vs. monitoreo

Ambas son partes integrales del proceso de desarrollo de ML y dependen la una de la otra para la iteración.
Las pruebas consisten en asegurar que nuestro sistema (código, datos y modelos) supera las expectativas que hemos establecido.
Por su parte, la monitorización implica que estas expectativas sigan pasando en línea en los datos de producción en vivo,
al tiempo que se garantiza que sus distribuciones de datos son comparables a la ventana de referencia.
Cuando estas condiciones dejan de cumplirse, hay que inspeccionar más de cerca (el reentrenamiento no siempre soluciona el problema de fondo).
En el caso del monitoreo, hay bastantes preocupaciones distintas que no tuvimos que considerar durante las pruebas, ya que se trata de datos que aún no hemos visto.

* Features y distribuciones de predicción (drift), tipificación, desajustes de esquema, etc.
* Determinar el rendimiento del modelo (métricas rolling y window, en general y en slices) utilizando señales indirectas (ya que las etiquetas pueden no estar disponibles fácilmente).
* En situaciones con datos de gran tamaño, necesitamos saber qué puntos de datos hay que etiquetar y sobremuestrear para el entrenamiento.
* Identificar anomalías y valores atípicos.

Cubriremos todos estos conceptos en la sección de monitorización.
