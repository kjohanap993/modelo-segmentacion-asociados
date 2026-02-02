# Modelo de segmentación de asociados mediante K-means

Este repositorio contiene el código desarrollado en Python como parte de un proyecto de grado
en analítica de datos. El objetivo del proyecto es segmentar a los asociados de una cooperativa
mediante técnicas de aprendizaje automático no supervisado.

## Metodología
- Generación de un conjunto de datos sintético con variables demográficas y transaccionales
- Cálculo de indicadores RFM (Recencia, Frecuencia y Valor Monetario)
- Estandarización de variables
- Aplicación del algoritmo K-means
- Validación del número de clústeres mediante el Método del Codo y el coeficiente de silueta

## Tecnologías utilizadas
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Reproducibilidad
El modelo puede ser reentrenado modificando las semillas aleatorias y los parámetros del algoritmo,
permitiendo evaluar la estabilidad de los clústeres obtenidos.
