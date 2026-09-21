
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Segmentación Logística E-Commerce",
    page_icon="📦",
    layout="wide"
)

st.title(
    "📦 Segmentación Logística de E-Commerce"
)

st.write(
    """
    Esta herramienta clasifica nuevos envíos dentro de
    segmentos logísticos identificados previamente mediante
    **K-Means**.

    Posteriormente, un modelo **K-NN** permite asignar un nuevo
    envío al segmento correspondiente sin tener que volver a
    ejecutar todo el proceso de clustering.
    """
)


# ============================================================
# CARGAR ARCHIVOS
# ============================================================

@st.cache_resource
def cargar_modelos():

    modelo = joblib.load(
        "knn_ecommerce.pkl"
    )

    scaler = joblib.load(
        "scaler_ecommerce.pkl"
    )

    return modelo, scaler


modelo_knn, scaler = cargar_modelos()


with open(
    "cluster_metadata.json",
    "r",
    encoding="utf-8"
) as file:

    metadata = json.load(file)


ranges = metadata[
    "feature_ranges"
]


# ============================================================
# FORMULARIO
# ============================================================

st.subheader(
    "Características del nuevo envío"
)


col1, col2 = st.columns(2)


with col1:

    customer_care_calls = st.number_input(
        "Llamadas a servicio al cliente",
        min_value=float(
            ranges[
                "Customer_care_calls"
            ]["min"]
        ),
        max_value=float(
            ranges[
                "Customer_care_calls"
            ]["max"]
        ),
        value=float(
            ranges[
                "Customer_care_calls"
            ]["median"]
        ),
        step=1.0
    )


    customer_rating = st.number_input(
        "Calificación del cliente",
        min_value=float(
            ranges[
                "Customer_rating"
            ]["min"]
        ),
        max_value=float(
            ranges[
                "Customer_rating"
            ]["max"]
        ),
        value=float(
            ranges[
                "Customer_rating"
            ]["median"]
        ),
        step=1.0
    )


    cost_product = st.number_input(
        "Costo del producto",
        min_value=float(
            ranges[
                "Cost_of_the_Product"
            ]["min"]
        ),
        max_value=float(
            ranges[
                "Cost_of_the_Product"
            ]["max"]
        ),
        value=float(
            ranges[
                "Cost_of_the_Product"
            ]["median"]
        ),
        step=1.0
    )


with col2:

    prior_purchases = st.number_input(
        "Compras anteriores",
        min_value=float(
            ranges[
                "Prior_purchases"
            ]["min"]
        ),
        max_value=float(
            ranges[
                "Prior_purchases"
            ]["max"]
        ),
        value=float(
            ranges[
                "Prior_purchases"
            ]["median"]
        ),
        step=1.0
    )


    discount = st.number_input(
        "Descuento ofrecido",
        min_value=float(
            ranges[
                "Discount_offered"
            ]["min"]
        ),
        max_value=float(
            ranges[
                "Discount_offered"
            ]["max"]
        ),
        value=float(
            ranges[
                "Discount_offered"
            ]["median"]
        ),
        step=1.0
    )


    weight = st.number_input(
        "Peso del envío (gramos)",
        min_value=float(
            ranges[
                "Weight_in_gms"
            ]["min"]
        ),
        max_value=float(
            ranges[
                "Weight_in_gms"
            ]["max"]
        ),
        value=float(
            ranges[
                "Weight_in_gms"
            ]["median"]
        ),
        step=100.0
    )


# ============================================================
# NUEVA OBSERVACIÓN
# ============================================================

nuevo_envio = pd.DataFrame(
    [[
        customer_care_calls,
        customer_rating,
        cost_product,
        prior_purchases,
        discount,
        weight
    ]],

    columns=[
        "Customer_care_calls",
        "Customer_rating",
        "Cost_of_the_Product",
        "Prior_purchases",
        "Discount_offered",
        "Weight_in_gms"
    ]
)


# ============================================================
# PREDICCIÓN
# ============================================================

if st.button(
    "Clasificar envío",
    type="primary",
    use_container_width=True
):

    envio_scaled = (
        scaler.transform(
            nuevo_envio
        )
    )

    cluster = int(
        modelo_knn.predict(
            envio_scaled
        )[0]
    )

    probabilities = (
        modelo_knn.predict_proba(
            envio_scaled
        )[0]
    )

    confidence = float(
        np.max(
            probabilities
        ) * 100
    )


    # ========================================================
    # RESULTADO PRINCIPAL
    # ========================================================

    st.success(
        f"📦 Segmento logístico asignado: "
        f"Cluster {cluster}"
    )


    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Cluster",
            cluster
        )

    with c2:

        st.metric(
            "Acuerdo de los vecinos",
            f"{confidence:.1f}%"
        )


    # ========================================================
    # INTERPRETACIÓN
    # ========================================================

    st.subheader(
        "Interpretación operativa"
    )

    cluster_data = (
        metadata[
            "cluster_info"
        ].get(
            str(cluster),
            {}
        )
    )


    description = (
        cluster_data.get(
            "description",
            "Segmento logístico identificado mediante K-Means."
        )
    )

    recommendation = (
        cluster_data.get(
            "recommendation",
            "Realizar seguimiento al comportamiento operativo del segmento."
        )
    )


    st.write(
        description
    )

    st.info(
        "💡 Recomendación: "
        + recommendation
    )


    # ========================================================
    # DISTRIBUCIÓN DE VECINOS
    # ========================================================

    st.subheader(
        "Distribución de vecinos cercanos"
    )

    results = pd.DataFrame({

        "Segmento": [
            f"Cluster {int(c)}"

            for c
            in modelo_knn.classes_
        ],

        "Proporción":
            probabilities
    })


    st.bar_chart(
        results.set_index(
            "Segmento"
        )
    )


    # ========================================================
    # DATOS INGRESADOS
    # ========================================================

    with st.expander(
        "Ver características ingresadas"
    ):

        st.dataframe(
            nuevo_envio,
            use_container_width=True
        )


# ============================================================
# PIE
# ============================================================

st.divider()

st.caption(
    "Taller 6 | K-Means + K-NN | "
    "Segmentación logística de E-Commerce"
)
