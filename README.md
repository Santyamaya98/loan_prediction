# Proyecto de Predicción de Préstamos

Este proyecto se centra en la predicción de la aprobación de préstamos utilizando modelos de Machine Learning, incluyendo regresión logística y bosques aleatorios. Se realiza un análisis exhaustivo de los datos y se generan visualizaciones para comprender mejor las relaciones entre las características del préstamo y la probabilidad de aprobación.

## Estructura del Proyecto

- `analisis.py`: Script que contiene el análisis exploratorio de datos y generación de visualizaciones.
- `feature_engineering.py`: Script que se encarga de la ingeniería de características para el modelo.
- `main.py`: Script principal que carga datos, entrena los modelos y realiza predicciones.
- `models.py`: Contiene funciones para el entrenamiento y evaluación de los modelos.
- `ploter.py`: Genera gráficos para analizar el rendimiento de los modelos.
- `sample_submission.csv`: Ejemplo de archivo de envío para el conjunto de pruebas.
- `train.csv` y `test.csv`: Conjuntos de datos para el entrenamiento y pruebas.

## Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/Santyamaya98/loan_prediction.git
   cd loan_prediction
python -m venv env
source env/bin/activate  # En Windows usa `env\Scripts\activate`
pip install -r requirements.txt
python main.py
o puedes usar el shell y probar las funciones como las ploter o la generacion de los modelos
