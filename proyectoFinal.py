import streamlit as st
import joblib
import numpy as np
import os
from PIL import Image

# Configuración de la página
logo = Image.open("logoUAT.png")
st.set_page_config(page_title='FIC', page_icon=logo)

# Menú lateral
st.sidebar.image(logo, use_container_width=True)
seccion = st.sidebar.selectbox(
    "Menú",
    ["Inicio", "Clasificación", "Acerca de"]
)

# === SECCIÓN: Inicio ===
if seccion == "Inicio":
    st.title("Bienvenido al Clasificador de Ataques Cardíacos")
    st.write("""
    Esta aplicación permite clasificar si un paciente es propenso un ataque cardíaco 
    basado en parámetros clínicos utilizando distintos modelos de clasificación.
    Sin embargo, los resultados obtenidos no sustituyen la evaluación de un médico 
    especialista y no deben considerarse un diagnóstico definitivo.
    
    Navega a la sección 'Clasificación' para ingresar los datos.
    """)

# === SECCIÓN: Clasificación ===
elif seccion == "Clasificación":
    st.title("Clasificación de Ataques Cardíacos")
    st.write("Ingresa los valores clínicos para determinar si se es propenso a ataques cardiacos usando distintos modelos de clasificación.")

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
    ckmb = st.number_input("CK-MB", min_value=0.00, step=0.01)
    troponin = st.number_input("Troponin", min_value=0.00, step=0.01)

    modelo_seleccionado = st.selectbox("Selecciona un modelo de clasificación", list(modelos_disponibles.keys()))

    if st.button("Clasificar"):
        try:
            modelo = joblib.load(modelos_disponibles[modelo_seleccionado])
            scaler = joblib.load(ruta_scaler)

            entrada = np.array([[age, ckmb, troponin]])
            entrada_escalada = scaler.transform(entrada)

            prediccion = modelo.predict(entrada_escalada)[0]
            resultado = "Positivo" if prediccion == 1 else "Negativo"

            st.success(f"Resultado de la clasificación: **{resultado}**")

        except Exception as e:
            st.error(f"Error: {e}")

# === SECCIÓN: Acerca de ===
elif seccion == "Acerca de":
    st.title("Acerca de")

    st.subheader("Dataset")
    st.write("""
    Este conjunto de datos fue recopilado en el hospital Zheen en Erbil, Irak, entre enero
    de 2019 y mayo de 2019. Los atributos de este conjunto de datos son: edad, sexo,
    frecuencia cardíaca, presión arterial sistólica, presión arterial diastólica, azúcar en
    sangre, ck-mb y troponina con salida negativa o positiva. De acuerdo con la información
    proporcionada, el conjunto de datos médicos clasifica el infarto de miocardio o
    ninguno. La columna de género de los datos se normaliza: el masculino se establece
    en 1 y el femenino en 0. La columna de glucosa se establece en 1 si está > 120; de
    lo contrario, 0. En cuanto a la salida, positivo se establece en 1 y negativo en 0.
             
    Disponible en: https://data.mendeley.com/datasets/wmhctcrt5v/1


    """)

    st.subheader("Curvas ROC de los modelos entrenados")
    st.write("""
    Las curvas ROC ayudan a evaluar qué tan bueno es un modelo de clasificación para 
    distinguir entre dos categorías, en este caso, los que han sufrido ataques cardiacos y los que no. \n
    En la gráfica ROC, se comparan los aciertos (cuando el modelo predice bien) con los errores (cuando confunde los casos). 
    Un modelo perfecto tendría una curva que se acerca al vértice superior izquierdo, mientras que un modelo aleatorio se vería como una línea diagonal.\n

    El AUC-ROC (Área bajo la curva ROC) es un número que indica qué tan confiable es el modelo:\n
    
    \u2022 Si el AUC está cerca de 1 → El modelo funciona muy bien.\n
    \u2022 Si el AUC está cerca de 0.5 → El modelo no está ayudando mucho.\n
             
    A continuación se muestran las curvas ROC para cada uno de los modelos utilizados en este proyecto: \n
    

    """)


    st.image("CurvasROC.png", use_container_width=True)


# === Pie de página fijo y centrado ===
st.markdown("""
    <style>
    /* Espacio inferior al contenido para que no lo tape el footer */
    .stApp {
        padding-bottom: 80px;
    }

    /* Footer fijo en la parte inferior */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #6c757d;
        text-align: center;
        font-size: 0.75em;
        color: #f1f3f6;
        padding: 10px;
        z-index: 9999;
        box-shadow: 0 -1px 5px rgba(0,0,0,0.1);
    }
    </style>

    <div class="footer">
        Autor: Brandon Nahú Nava Martínez &nbsp;|&nbsp;
        Proyecto académico para la materia de <i>Aprendizaje Automático</i> &nbsp;|&nbsp;
        Facultad de Ingeniería y Ciencias - Universidad Autónoma de Tamaulipas
    </div>
""", unsafe_allow_html=True)

