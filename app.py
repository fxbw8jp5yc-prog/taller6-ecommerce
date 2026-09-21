
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Clasificador Iris - KNN",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Clasificación de agrupaciones - Iris")

st.write(
    """
    Esta aplicación utiliza un modelo **K-NN** entrenado a partir
    de las agrupaciones encontradas mediante **K-Means**
    sobre el dataset Iris.

    Ingresa las características de una nueva flor para determinar
    a qué agrupación pertenece.
    """
)

# ============================================================
# CARGAR MODELO Y ESCALADOR
# ============================================================

@st.cache_resource
def cargar_modelos():
    modelo = joblib.load("knn_model.pkl")
    escalador = joblib.load("scaler.pkl")
    return modelo, escalador


modelo, escalador = cargar_modelos()


# ============================================================
# DATOS DE ENTRADA
# ============================================================

st.subheader("Características de la flor")

sepal_length = st.number_input(
    "Longitud del sépalo (cm)",
    min_value=4.0,
    max_value=8.0,
    value=5.8,
    step=0.1
)

sepal_width = st.number_input(
    "Ancho del sépalo (cm)",
    min_value=2.0,
    max_value=4.5,
    value=3.0,
    step=0.1
)

petal_length = st.number_input(
    "Longitud del pétalo (cm)",
    min_value=1.0,
    max_value=7.0,
    value=4.3,
    step=0.1
)

petal_width = st.number_input(
    "Ancho del pétalo (cm)",
    min_value=0.1,
    max_value=2.6,
    value=1.3,
    step=0.1
)


# ============================================================
# CREAR NUEVA OBSERVACIÓN
# ============================================================

nuevo_dato = pd.DataFrame(
    [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]],
    columns=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]
)


# ============================================================
# CLASIFICACIÓN
# ============================================================

if st.button("Clasificar flor", type="primary"):

    # Escalar los valores ingresados
    dato_escalado = escalador.transform(nuevo_dato)

    # Predecir cluster
    cluster = int(
        modelo.predict(dato_escalado)[0]
    )

    # Proporción de vecinos por cluster
    probabilidades = modelo.predict_proba(
        dato_escalado
    )[0]

    confianza = float(
        np.max(probabilidades) * 100
    )

    # --------------------------------------------------------
    # RESULTADO
    # --------------------------------------------------------

    st.success(
        f"🌸 La observación pertenece al Cluster {cluster}"
    )

    st.metric(
        "Acuerdo de los vecinos",
        f"{confianza:.1f}%"
    )

    st.subheader("Resultado de la clasificación")

    st.write(
        f"""
        De acuerdo con sus características, el modelo K-NN
        clasificó la nueva observación dentro del
        **Cluster {cluster}**.

        Estos grupos fueron encontrados previamente mediante
        el algoritmo K-Means.
        """
    )

    # --------------------------------------------------------
    # GRÁFICA
    # --------------------------------------------------------

    resultados = pd.DataFrame({
        "Cluster": [
            f"Cluster {int(c)}"
            for c in modelo.classes_
        ],
        "Proporción": probabilidades
    })

    st.subheader("Distribución de los vecinos")

    st.bar_chart(
        resultados.set_index("Cluster")
    )

    # --------------------------------------------------------
    # DATOS INGRESADOS
    # --------------------------------------------------------

    with st.expander("Ver datos ingresados"):

        st.dataframe(
            nuevo_dato,
            use_container_width=True
        )


# ============================================================
# PIE DE PÁGINA
# ============================================================

st.divider()

st.caption(
    "Taller 6 - Herramientas de Inteligencia Artificial | "
    "Universidad de los Andes"
)
