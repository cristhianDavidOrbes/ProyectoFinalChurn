from pathlib import Path
import shutil
import tempfile

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


SERVICIOS_COLUMNAS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def crear_sesion_spark(nombre_app: str = "ChurnPredictionLimpieza") -> SparkSession:
    """Crea o reutiliza una sesion local de Spark."""
    return (
        SparkSession.builder.appName(nombre_app)
        .master("local[*]")
        .getOrCreate()
    )


def cargar_dataset(
    spark: SparkSession,
    ruta_archivo: str | Path = "data/raw/dataset_original.csv",
) -> DataFrame:
    """Carga el dataset original desde data/raw usando PySpark."""
    ruta_archivo = str(Path(ruta_archivo).resolve())
    return (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(ruta_archivo)
    )


def analizar_dataset(df: DataFrame) -> dict:
    """Genera un resumen inicial del dataset con operaciones de PySpark."""
    total_filas = df.count()
    total_columnas = len(df.columns)
    filas_sin_duplicados = df.dropDuplicates().count()

    valores_nulos = {
        columna: df.filter(F.col(columna).isNull()).count()
        for columna in df.columns
    }

    analisis = {
        "filas_columnas": (total_filas, total_columnas),
        "columnas": df.columns,
        "tipos_datos": dict(df.dtypes),
        "valores_nulos": valores_nulos,
        "duplicados": total_filas - filas_sin_duplicados,
        "valores_unicos_churn": [
            fila["Churn"] for fila in df.select("Churn").distinct().collect()
        ],
        "distribucion_churn": {
            fila["Churn"]: fila["count"]
            for fila in df.groupBy("Churn").count().collect()
        },
    }
    return analisis


def limpiar_dataset(df: DataFrame) -> DataFrame:
    """Limpia datos basicos y convierte Churn a formato binario con PySpark."""
    # TotalCharges puede venir como texto o con espacios vacios; se normaliza a double.
    df_limpio = df.withColumn(
        "TotalCharges",
        F.when(F.trim(F.col("TotalCharges").cast("string")) == "", None)
        .otherwise(F.col("TotalCharges").cast("double")),
    )

    # Se usa la mediana aproximada porque es robusta ante valores extremos.
    mediana_total_charges = df_limpio.approxQuantile(
        "TotalCharges",
        [0.5],
        0.001,
    )[0]

    df_limpio = df_limpio.fillna({"TotalCharges": mediana_total_charges})

    # Se eliminan duplicados exactos si existen.
    df_limpio = df_limpio.dropDuplicates()

    # customerID se conserva para analisis, pero no debe usarse como predictor.
    df_limpio = df_limpio.withColumn(
        "Churn",
        F.when(F.col("Churn") == "Yes", F.lit(1))
        .when(F.col("Churn") == "No", F.lit(0))
        .otherwise(None)
        .cast("int"),
    )

    return df_limpio


def crear_variables_derivadas(df: DataFrame) -> DataFrame:
    """Crea variables derivadas para enriquecer el analisis posterior."""
    # Cuenta solo servicios contratados explicitamente con valor "Yes".
    expresiones_servicios = [
        F.when(F.col(columna) == "Yes", F.lit(1)).otherwise(F.lit(0))
        for columna in SERVICIOS_COLUMNAS
    ]

    num_servicios = expresiones_servicios[0]
    for expresion in expresiones_servicios[1:]:
        num_servicios = num_servicios + expresion

    df_features = df.withColumn("NumServicios", num_servicios)

    # Marca clientes nuevos con 12 meses o menos de antiguedad.
    df_features = df_features.withColumn(
        "ClienteNuevo",
        F.when(F.col("tenure") <= 12, F.lit(1)).otherwise(F.lit(0)),
    )

    # Se suma 1 a tenure para evitar division entre cero.
    df_features = df_features.withColumn(
        "CargoPromedioPorMes",
        F.col("TotalCharges") / (F.col("tenure") + F.lit(1)),
    )

    # Identifica contratos mensuales.
    df_features = df_features.withColumn(
        "ContratoMensual",
        F.when(F.col("Contract") == "Month-to-month", F.lit(1)).otherwise(F.lit(0)),
    )

    return df_features


def guardar_dataset(
    df: DataFrame,
    ruta_salida: str | Path = "processed/telco_churn_limpio.csv",
) -> None:
    """Guarda el DataFrame de Spark como un unico archivo CSV."""
    ruta_salida = Path(ruta_salida).resolve()
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=ruta_salida.parent) as temp_dir:
        temp_path = Path(temp_dir)
        df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(temp_path))

        archivo_part = next(temp_path.glob("part-*.csv"))
        if ruta_salida.exists():
            ruta_salida.unlink()
        shutil.move(str(archivo_part), str(ruta_salida))


def ejecutar_preprocesamiento() -> DataFrame:
    """Ejecuta el flujo completo de limpieza y feature engineering con PySpark."""
    spark = crear_sesion_spark()
    spark.sparkContext.setLogLevel("ERROR")

    df_original = cargar_dataset(spark)
    analisis = analizar_dataset(df_original)

    print("Analisis inicial del dataset")
    print(f"Filas y columnas: {analisis['filas_columnas']}")
    print(f"Columnas: {analisis['columnas']}")
    print("\nTipos de datos:")
    print(analisis["tipos_datos"])
    print("\nValores nulos por columna:")
    print(analisis["valores_nulos"])
    print(f"\nRegistros duplicados: {analisis['duplicados']}")
    print(f"Valores unicos de Churn: {analisis['valores_unicos_churn']}")
    print("\nDistribucion de Churn:")
    print(analisis["distribucion_churn"])

    df_limpio = limpiar_dataset(df_original)
    df_limpio = crear_variables_derivadas(df_limpio)
    guardar_dataset(df_limpio)

    print("\nVista previa del dataset limpio con variables derivadas:")
    df_limpio.select(
        "customerID",
        "Churn",
        "NumServicios",
        "ClienteNuevo",
        "CargoPromedioPorMes",
        "ContratoMensual",
    ).show(5, truncate=False)

    return df_limpio


if __name__ == "__main__":
    ejecutar_preprocesamiento()
