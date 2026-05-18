# ProyectoFinalChurn

Proyecto academico de Machine Learning para prediccion de abandono de clientes usando el dataset Telco Customer Churn.

## Contenido

- Limpieza y feature engineering con PySpark.
- Entrenamiento de modelos:
  - Regresion Logistica
  - Random Forest
  - XGBoost
- Interpretabilidad con SHAP.
- Aplicacion web con Flask para prediccion y dashboard.

## Ejecucion local

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar la aplicacion:

```bash
python app/app.py
```

Abrir:

```text
http://127.0.0.1:5000/
```

## Despliegue en Vercel

Vercel reconoce aplicaciones Flask con entrada en `app/app.py`. El proyecto incluye los modelos entrenados en `modelos/` y el dataset procesado en `processed/`.

Nota: en Vercel el guardado persistente en archivos CSV no es recomendable. Para produccion se recomienda conectar una base de datos externa para almacenar clientes predichos.
