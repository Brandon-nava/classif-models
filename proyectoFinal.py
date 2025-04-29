import streamlit as st
import joblib
import numpy as np
import os
from PIL import Image

# Configuración de la página
logo = Image.open("logoUAT.png")
st.set_page_config(page_title='FIC', page_icon=logo)

# Menú lateral
st.sidebar.image(logo, use_column_width=True)
seccion = st.sidebar.selectbox(
    "Menú",
    ["Inicio", "Predicción", "Acerca de"]
)

# === SECCIÓN: Inicio ===
if seccion == "Inicio":
    st.title("Bienvenido al Clasificador de Ataques Cardíacos")
    st.write("""
    Esta aplicación permite predecir si un paciente ha sufrido un ataque cardíaco 
    basado en parámetros clínicos utilizando distintos modelos de clasificación.
    
    Navega a la sección 'Predicción' para ingresar los datos.
    """)

# === SECCIÓN: Predicción ===
elif seccion == "Predicción":
    st.title("Clasificación de Ataques Cardíacos")
    st.write("Ingresa los valores clínicos para predecir el resultado usando distintos modelos de clasificación.")

    modelos_disponibles = {
        'Regresión Logística': 'logModel_entrenado.pkl',
        'Árbol de Decisión': 'treeModel_entrenado.pkl',
        'KNN': 'knnModel_entrenado.pkl',
        'Naive Bayes': 'naiveModel_entrenado.pkl',
        'SVM': 'svmModel_entrenado.pkl',
        'MLP': 'mlpModel_entrenado.pkl'
    }

    ruta_scaler = 'escalador.joblib'

    age = st.number_input("Edad (Age)", min_value=1, max_value=100, step=1)
    ckmb = st.number_input("CK-MB", min_value=0.000, step=0.001)
    troponin = st.number_input("Troponin", min_value=0.000, step=0.001)

    modelo_seleccionado = st.selectbox("Selecciona un modelo de predicción", list(modelos_disponibles.keys()))

    if st.button("Predecir"):
        try:
            modelo = joblib.load(modelos_disponibles[modelo_seleccionado])
            scaler = joblib.load(ruta_scaler)

            entrada = np.array([[age, ckmb, troponin]])
            entrada_escalada = scaler.transform(entrada)

            prediccion = modelo.predict(entrada_escalada)[0]
            resultado = "Positivo" if prediccion == 1 else "Negativo"

            st.success(f"Resultado de la predicción: **{resultado}**")

        except Exception as e:
            st.error(f"Error: {e}")

# === SECCIÓN: Acerca de ===
elif seccion == "Acerca de":
    st.title("Acerca de")
    st.write("""
    Autor: Brandon  
    Proyecto académico de clasificación usando modelos de Machine Learning entrenados previamente.  
    Universidad Autónoma de Tamaulipas (UAT)  
    """)
