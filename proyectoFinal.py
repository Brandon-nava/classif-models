import streamlit as st
import joblib
import numpy as np
import os
from PIL import Image

logo = Image.open("logoUAT.png")
st.set_page_config(page_title = 'FIC', page_icon = logo)

# === Configuración inicial ===
st.title("Clasificación de Ataques Cardíacos")
st.write("Ingresa los valores clínicos para predecir el resultado usando distintos modelos de clasificación.")

# === Diccionario de modelos disponibles ===
modelos_disponibles = {
    'Regresión Logística': 'logModel_entrenado.pkl',
    'Árbol de Decisión': 'treeModel_entrenado.pkl',
    'KNN': 'knnModel_entrenado.pkl',
    'Naive Bayes': 'naiveModel_entrenado.pkl',
    'SVM' : 'svmModel_entrenado.pkl',
    'MLP' : 'mlpModel_entrenado.pkl'
}

# Ruta del escalador
ruta_scaler = 'escalador.joblib'

# === Inputs del usuario ===
age = st.number_input("Edad (Age)", min_value=1, max_value= 100, step=1)
ckmb = st.number_input("CK-MB", min_value=0.000, step=0.001)
troponin = st.number_input("Troponin", min_value=0.000, step=0.001)

modelo_seleccionado = st.selectbox("Selecciona un modelo de predicción", list(modelos_disponibles.keys()))

# === Botón para ejecutar predicción ===
if st.button("Predecir"):
    try:
        # Cargar modelo y escalador
        modelo = joblib.load(modelos_disponibles[modelo_seleccionado])
        scaler = joblib.load(ruta_scaler)

        # Escalar los datos ingresados
        entrada = np.array([[age, ckmb, troponin]])
        entrada_escalada = scaler.transform(entrada)

        # Realizar predicción
        prediccion = modelo.predict(entrada_escalada)[0]
        resultado = "Positivo" if prediccion == 1 else "Negativo"

        # Mostrar resultado
        st.success(f"Resultado de la predicción: **{resultado}**")

    except Exception as e:
        st.error(f"Error: {e}")
