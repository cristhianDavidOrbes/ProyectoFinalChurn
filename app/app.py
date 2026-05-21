from datetime import datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request, send_from_directory, url_for


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "processed" / "telco_churn_limpio.csv"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"
PREDICTIONS_PATH = PREDICTIONS_DIR / "clientes_predichos.csv"
MODELOS_DIR = BASE_DIR / "modelos"
RESULTADOS_DIR = BASE_DIR / "resultados"

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

COLUMNAS_MODELO = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "NumServicios",
    "ClienteNuevo",
    "CargoPromedioPorMes",
    "ContratoMensual",
]

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"

MODELOS = {
    "regresion_logistica": {
        "nombre": "Regresion Logistica",
        "archivo": MODELOS_DIR / "modelo_regresion_logistica.pkl",
        "metricas": RESULTADOS_DIR / "metricas_regresion_logistica.csv",
    },
    "random_forest": {
        "nombre": "Random Forest",
        "archivo": MODELOS_DIR / "modelo_random_forest.pkl",
        "metricas": RESULTADOS_DIR / "metricas_random_forest.csv",
    },
    "xgboost": {
        "nombre": "XGBoost",
        "archivo": MODELOS_DIR / "modelo_xgboost.pkl",
        "metricas": RESULTADOS_DIR / "metricas_xgboost.csv",
    },
}

app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
modelos_cargados = {}


def cargar_modelos():
    """Carga los Pipelines entrenados una sola vez."""
    for clave, config in MODELOS.items():
        modelos_cargados[clave] = joblib.load(config["archivo"])


def convertir_numero(valor, tipo=float, defecto=0):
    """Convierte campos del formulario a numero con valor por defecto."""
    try:
        return tipo(valor)
    except (TypeError, ValueError):
        return defecto


def calcular_variables_derivadas(cliente):
    """Calcula las variables creadas en la etapa de feature engineering."""
    cliente["NumServicios"] = sum(1 for columna in SERVICIOS_COLUMNAS if cliente[columna] == "Yes")
    cliente["ClienteNuevo"] = 1 if cliente["tenure"] <= 12 else 0
    cliente["CargoPromedioPorMes"] = cliente["TotalCharges"] / (cliente["tenure"] + 1)
    cliente["ContratoMensual"] = 1 if cliente["Contract"] == "Month-to-month" else 0
    return cliente


def clasificar_riesgo(probabilidad):
    """Clasifica el riesgo a partir de la probabilidad de churn."""
    if probabilidad < 0.40:
        return "Bajo riesgo"
    if probabilidad < 0.70:
        return "Riesgo medio"
    return "Alto riesgo"


def ajustar_probabilidad_por_coherencia(cliente, probabilidad):
    """Evita que perfiles claramente estables queden como alto riesgo por un solo modelo."""
    pago_automatico = cliente["PaymentMethod"] in {
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    }
    contrato_largo = cliente["Contract"] in {"One year", "Two year"}
    perfil_estable = (
        contrato_largo
        and cliente["ClienteNuevo"] == 0
        and cliente["tenure"] >= 12
        and cliente["TechSupport"] == "Yes"
        and cliente["OnlineSecurity"] == "Yes"
        and pago_automatico
        and cliente["NumServicios"] >= 4
    )

    if not perfil_estable:
        return probabilidad

    limite = 0.35 if cliente["MonthlyCharges"] <= 80 else 0.55
    return min(probabilidad, limite)


def generar_recomendaciones(cliente, probabilidad):
    """Genera recomendaciones simples de retencion segun el perfil del cliente."""
    recomendaciones = []

    if probabilidad >= 0.70:
        recomendaciones.append("Contactar al cliente de forma prioritaria con una oferta personalizada de retencion.")
    elif probabilidad >= 0.40:
        recomendaciones.append("Realizar seguimiento preventivo y ofrecer beneficios segun su perfil.")

    if cliente["Contract"] == "Month-to-month":
        recomendaciones.append("Ofrecer descuento para migrar a contrato anual o de dos anos.")
    if cliente["ClienteNuevo"] == 1:
        recomendaciones.append("Aplicar campana de bienvenida, seguimiento personalizado y beneficios iniciales.")
    if cliente["MonthlyCharges"] > 80:
        recomendaciones.append("Ofrecer revision del plan actual o un plan mas economico.")
    if cliente["TechSupport"] == "No":
        recomendaciones.append("Ofrecer soporte tecnico gratuito por un periodo limitado.")
    if cliente["OnlineSecurity"] == "No":
        recomendaciones.append("Ofrecer paquete de seguridad online como beneficio de retencion.")
    if cliente["NumServicios"] <= 2:
        recomendaciones.append("Ofrecer paquete combinado de servicios para aumentar fidelizacion.")
    if cliente["PaymentMethod"] == "Electronic check":
        recomendaciones.append("Promover cambio a pago automatico con beneficio o descuento.")
    if probabilidad < 0.40 and not recomendaciones:
        recomendaciones.append("Mantener seguimiento normal y ofrecer beneficios de fidelizacion.")

    return recomendaciones


def obtener_cliente_desde_formulario(formulario):
    """Construye el registro del cliente a partir del formulario web."""
    cliente = {
        "gender": formulario.get("gender", "Female"),
        "SeniorCitizen": convertir_numero(formulario.get("SeniorCitizen"), int, 0),
        "Partner": formulario.get("Partner", "No"),
        "Dependents": formulario.get("Dependents", "No"),
        "tenure": convertir_numero(formulario.get("tenure"), int, 0),
        "PhoneService": formulario.get("PhoneService", "No"),
        "MultipleLines": formulario.get("MultipleLines", "No"),
        "InternetService": formulario.get("InternetService", "DSL"),
        "OnlineSecurity": formulario.get("OnlineSecurity", "No"),
        "OnlineBackup": formulario.get("OnlineBackup", "No"),
        "DeviceProtection": formulario.get("DeviceProtection", "No"),
        "TechSupport": formulario.get("TechSupport", "No"),
        "StreamingTV": formulario.get("StreamingTV", "No"),
        "StreamingMovies": formulario.get("StreamingMovies", "No"),
        "Contract": formulario.get("Contract", "Month-to-month"),
        "PaperlessBilling": formulario.get("PaperlessBilling", "Yes"),
        "PaymentMethod": formulario.get("PaymentMethod", "Electronic check"),
        "MonthlyCharges": convertir_numero(formulario.get("MonthlyCharges"), float, 0.0),
        "TotalCharges": convertir_numero(formulario.get("TotalCharges"), float, 0.0),
    }
    return calcular_variables_derivadas(cliente)


def crear_dataframe_modelo(cliente):
    """Crea un DataFrame con exactamente las columnas usadas en entrenamiento."""
    datos_cliente = pd.DataFrame([cliente])
    return datos_cliente[COLUMNAS_MODELO]


def guardar_cliente_predicho(cliente, modelo_usado, probabilidad, nivel_riesgo, prediccion, recomendaciones):
    """Agrega la prediccion del cliente al historico CSV."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    registro = {
        "customerID": f"WEB-{uuid4().hex[:10].upper()}",
        "modelo_usado": modelo_usado,
        "probabilidad_churn": round(probabilidad * 100, 2),
        "nivel_riesgo": nivel_riesgo,
        "prediccion_churn": "Cliente con riesgo de abandono" if prediccion == 1 else "Cliente sin riesgo de abandono",
        "recomendaciones": " | ".join(recomendaciones),
        "fecha_prediccion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    registro.update(cliente)

    fila = pd.DataFrame([registro])
    if PREDICTIONS_PATH.exists():
        fila.to_csv(PREDICTIONS_PATH, mode="a", index=False, header=False)
    else:
        fila.to_csv(PREDICTIONS_PATH, index=False)

    return registro


def cargar_metricas():
    """Lee las metricas guardadas de cada modelo."""
    metricas = []
    for clave, config in MODELOS.items():
        if not config["metricas"].exists():
            continue
        df_metricas = pd.read_csv(config["metricas"])
        fila = df_metricas.iloc[0].to_dict()
        metricas.append(
            {
                "modelo": config["nombre"],
                "accuracy": float(fila["Accuracy"]),
                "f1": float(fila["F1"]),
                "roc_auc": float(fila["ROC_AUC"]),
            }
        )
    return metricas


def cargar_predicciones_filtradas(filtros):
    """Predice todo el dataset limpio, agrega clientes nuevos y aplica filtros."""
    modelo_clave = filtros.get("modelo") or "regresion_logistica"
    modelo_nombre = MODELOS[modelo_clave]["nombre"]
    pipeline = modelos_cargados[modelo_clave]

    df_dataset = pd.read_csv(DATASET_PATH)
    datos_modelo = df_dataset[COLUMNAS_MODELO]
    probabilidades_modelo = pipeline.predict_proba(datos_modelo)[:, 1]
    probabilidades = np.array(
        [
            ajustar_probabilidad_por_coherencia(fila.to_dict(), probabilidad)
            for (_, fila), probabilidad in zip(df_dataset.iterrows(), probabilidades_modelo)
        ]
    )
    predicciones = (probabilidades >= 0.50).astype(int)

    df_predicciones = df_dataset.copy()
    df_predicciones["modelo_usado"] = modelo_nombre
    df_predicciones["probabilidad_churn"] = np.round(probabilidades * 100, 2)
    df_predicciones["nivel_riesgo"] = [
        clasificar_riesgo(probabilidad) for probabilidad in probabilidades
    ]
    df_predicciones["prediccion_churn"] = np.where(
        predicciones == 1,
        "Cliente con riesgo de abandono",
        "Cliente sin riesgo de abandono",
    )
    df_predicciones["recomendaciones"] = [
        " | ".join(generar_recomendaciones(fila.to_dict(), probabilidad))
        for (_, fila), probabilidad in zip(df_dataset.iterrows(), probabilidades)
    ]
    df_predicciones["fecha_prediccion"] = ""

    if PREDICTIONS_PATH.exists():
        df_web = pd.read_csv(PREDICTIONS_PATH)
        df_web = df_web[df_web["modelo_usado"] == modelo_nombre]
        df_predicciones = pd.concat([df_predicciones, df_web], ignore_index=True)

    conteos = {
        "total": len(df_predicciones),
        "alto": int((df_predicciones["nivel_riesgo"] == "Alto riesgo").sum()),
        "medio": int((df_predicciones["nivel_riesgo"] == "Riesgo medio").sum()),
        "bajo": int((df_predicciones["nivel_riesgo"] == "Bajo riesgo").sum()),
    }

    if filtros.get("riesgo"):
        df_predicciones = df_predicciones[df_predicciones["nivel_riesgo"] == filtros["riesgo"]]
    if filtros.get("contrato"):
        df_predicciones = df_predicciones[df_predicciones["Contract"] == filtros["contrato"]]

    if not filtros.get("riesgo"):
        df_predicciones = df_predicciones.sort_values("probabilidad_churn", ascending=False)

    return df_predicciones, conteos


@app.route("/")
def index():
    return dashboard()


@app.route("/css/<path:filename>")
def css_publico(filename):
    return send_from_directory(BASE_DIR / "public" / "css", filename)


@app.route("/resultados/<path:filename>")
def archivos_resultados(filename):
    return send_from_directory(RESULTADOS_DIR, filename)


@app.route("/nuevo")
def nuevo_cliente():
    return render_template("index.html", modelos=MODELOS)


@app.route("/predecir", methods=["POST"])
def predecir():
    modelo_clave = request.form.get("modelo", "regresion_logistica")
    if modelo_clave not in modelos_cargados:
        return redirect(url_for("nuevo_cliente"))

    cliente = obtener_cliente_desde_formulario(request.form)
    datos_cliente = crear_dataframe_modelo(cliente)
    pipeline = modelos_cargados[modelo_clave]

    probabilidad_modelo = float(pipeline.predict_proba(datos_cliente)[0][1])
    probabilidad = ajustar_probabilidad_por_coherencia(cliente, probabilidad_modelo)
    prediccion = int(probabilidad >= 0.50)
    nivel_riesgo = clasificar_riesgo(probabilidad)
    recomendaciones = generar_recomendaciones(cliente, probabilidad)
    modelo_usado = MODELOS[modelo_clave]["nombre"]

    registro = guardar_cliente_predicho(
        cliente,
        modelo_usado,
        probabilidad,
        nivel_riesgo,
        prediccion,
        recomendaciones,
    )

    return render_template(
        "resultado.html",
        registro=registro,
        cliente=cliente,
        modelo_usado=modelo_usado,
        prediccion=registro["prediccion_churn"],
        probabilidad=round(probabilidad * 100, 2),
        nivel_riesgo=nivel_riesgo,
        recomendaciones=recomendaciones,
    )


@app.route("/dashboard")
def dashboard():
    filtros = {
        "modelo": request.args.get("modelo", ""),
        "riesgo": request.args.get("riesgo", ""),
        "contrato": request.args.get("contrato", ""),
    }
    predicciones, conteos = cargar_predicciones_filtradas(filtros)

    columnas_tabla = [
        "customerID",
        "modelo_usado",
        "probabilidad_churn",
        "nivel_riesgo",
        "Contract",
        "tenure",
        "MonthlyCharges",
        "NumServicios",
        "recomendaciones",
    ]
    if predicciones.empty:
        registros = []
    else:
        registros = predicciones[columnas_tabla].to_dict(orient="records")

    return render_template(
        "dashboard.html",
        metricas=cargar_metricas(),
        registros=registros,
        conteos=conteos,
        filtros=filtros,
        modelos=MODELOS,
    )


@app.route("/metricas")
def metricas():
    return render_template(
        "dashboard.html",
        metricas=cargar_metricas(),
        registros=[],
        conteos={"total": 0, "alto": 0, "medio": 0, "bajo": 0},
        filtros={"modelo": "", "riesgo": "", "contrato": ""},
        modelos=MODELOS,
    )


@app.route("/shap")
def shap_dashboard():
    ruta_cliente = RESULTADOS_DIR / "shap_cliente_alto_riesgo.csv"
    if ruta_cliente.exists():
        tabla_cliente = pd.read_csv(ruta_cliente).to_dict(orient="records")
    else:
        tabla_cliente = []

    return render_template(
        "shap.html",
        grafica_importancia="shap_importancia_global.png",
        grafica_beeswarm="shap_beeswarm.png",
        tabla_cliente=tabla_cliente,
    )


cargar_modelos()


if __name__ == "__main__":
    app.run(debug=True)
