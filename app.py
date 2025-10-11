import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def load_model_1cal():
    from joblib import load
    return load('./data/prediccion_1cal.pkl')
    
@st.cache_data
def load_model_2cal():
    from joblib import load
    return load('./data/prediccion_2cal.pkl')

st.title("Predicción de Calificaciones")
tab1, tab2, tab3, tab4 = st.tabs(["🧙‍♂️ 1 Examen", "🧙‍♂️ 2 Exámenes", "💻 Cálculo por fórmula", "❓ Información"])

with tab1:
    st.subheader("1 Examen")
    st.write("Este modelo predice si un estudiante aprobará o no basado en solo la calificación del **primer examen parcial y las primeras 2 prácticas** asumiendo que se entrega el proyecto completo.")
    st.write("*Nota: La probablidad de aprobar ya incluye realizar el repechaje y pasarlo.*")
    examen1 = st.number_input("Examen 1 (0-100):", min_value=0, max_value=100, value=30, key='1_examen1')
    practicas1 = st.number_input("Práctica 1 (0-100):", min_value=0, max_value=100, value=95, key='1_practicas', help="Calificación de la primer práctica")
    practicas2 = st.number_input("Práctica 2 (0-100):", min_value=0, max_value=100, value=95, key='1_practicas2', help="Calificación de la segunda práctica")
    proyecto1 = 84 #promedio de proyectos entregados
    if st.button("🧙‍♂️ Predecir", key="1_button"):
        rfclas1 = load_model_1cal()

        prediccion = pd.DataFrame({'Examen1':[examen1], 'Practica1':[practicas1], 'Practica2':[practicas2], 'Proyecto':[proyecto1]})
        #res = rfclas1.predict(prediccion)
        probs = rfclas1.predict_proba(prediccion)
        st.progress(probs[0][1])
        res1_1, res1_2 = st.columns(2)
        with res1_1:
            st.error(f"### {probs[0][0]*100:.2f}% No aprobará")
        with res1_2:
            st.success(f"### {probs[0][1]*100:.2f}% Aprobará")

with tab2:
    st.subheader("2 Exámenes")
    st.write("Este modelo predice si un estudiante aprobará o no basado en las calificaciones de los **dos exámenes parciales, la suma total de las 4 prácticas** y asumiendo que se entrega el proyecto completo.")
    st.write("*Nota: La probablidad de aprobar ya incluye realizar el repechaje y pasarlo.*")
    examen2_1 = st.number_input("Examen 1 (0-100):", min_value=0, max_value=100, value=30, key="2_examen1")
    examen2_2 = st.number_input("Examen 2 (0-100):", min_value=0, max_value=100, value=30, key="2_examen2")
    practicas2 = st.number_input("Prácticas (0-400):", min_value=0, max_value=400, value=95, key="2_practicas", help="Promedio de todas las prácticas del curso")
    proyecto2 = 84 #promedio de proyectos entregados
    if st.button("🧙‍♂️ Predecir", key="2_button"):
        rfclas2 = load_model_2cal()

        prediccion = pd.DataFrame({'Examen1':[examen2_1], 'Examen2':[examen2_2], 'Practicas':[practicas2/4], 'Proyecto':[proyecto2]})
        #res = rfclas2.predict(prediccion)
        probs = rfclas2.predict_proba(prediccion)
        st.progress(probs[0][1])
        res2_1, res2_2 = st.columns(2)
        with res2_1:
            st.error(f"### {probs[0][0]*100:.2f}% No aprobará")
        with res2_2:
            st.success(f"### {probs[0][1]*100:.2f}% Aprobará")
with tab3:
    st.subheader("Cálculo por fórmula")
    st.write("Calcula la calificación final basada en una fórmula ponderada.")
    examen3_1 = st.number_input("Examen 1 (0-100):", min_value=0, max_value=100, value=30, key="3_examen1", help="15% de la calificación final")
    examen3_2 = st.number_input("Examen 2 (0-100):", min_value=0, max_value=100, value=30, key="3_examen2", help="20% de la calificación final")
    examen3_3 = st.number_input("Examen 3 (0-100):", min_value=0, max_value=100, value=30, key="3_examen3", help="20% de la calificación final")
    practicas3 = st.number_input("Prácticas (0-400):", min_value=0, max_value=400, value=95, key="3_practicas", help="20% de la calificación final")
    proyecto_avances3 = st.number_input("Avances de proyecto (0-100):", min_value=0, max_value=100, value=100, key="3_proyecto_avances", help="10% de la calificación final")
    proyecto3 = st.number_input("Proyecto (0-100):", min_value=0, max_value=100, value=85, key="3_proyecto", help="15% de la calificación final")
    if st.button("💻 Calcular calificación", key="3_button"):
        calificacion_final = (examen3_1 * 0.15) + (examen3_2 * 0.20) + (examen3_3 * 0.20) + (practicas3/4 * 0.20) + (proyecto_avances3 * 0.10) + (proyecto3 * 0.15)
        if calificacion_final >= 60:
            st.success(f"### ✔ Aprobará con una calificación final de **{calificacion_final:.2f}**")
        elif calificacion_final >= 55:
            st.info(f"### ⚠ Puede recuperar repechaje, actualmente **{calificacion_final:.2f}**")
        else:
            st.error(f"### ✘ No aprobará con una calificación final de **{calificacion_final:.2f}**")
    
        min_examen3 = (60 - (examen3_1 * 0.15) - (examen3_2 * 0.20) - (practicas3/4 * 0.20) - (proyecto_avances3 * 0.10) - (proyecto3 * 0.15)) / 0.20
        min_examen3_rep = (55 - (examen3_1 * 0.15) - (examen3_2 * 0.20) - (practicas3/4 * 0.20) - (proyecto_avances3 * 0.10) - (proyecto3 * 0.15)) / 0.20
        mincol1, mincol2 = st.columns(2)
        with mincol1:
            st.success(f"⚠ Calificación **mínima** necesaria en **Examen 3** para aprobar: **{min_examen3:.2f}**")
        with mincol2:
            st.warning(f"⚠ Calificación **mínima** necesaria en **Examen 3** para repechaje: **{min_examen3_rep:.2f}**")
with tab4:
    st.subheader("❓ Información")
    st.write("""
    Esta aplicación utiliza modelos de aprendizaje automático para predecir si un estudiante aprobará o no un curso basado en sus calificaciones en exámenes, prácticas y proyectos. 
    - El **Modelo de 1 Examen** utiliza solo la calificación del primer examen parcial. Tiene una precisión del **78%**.
    - El **Modelo de 2 Exámenes** utiliza las calificaciones de los dos exámenes parciales. Tiene una precisión del **87%**.
    - La pestaña de **Cálculo por fórmula** permite calcular la calificación final basada en una fórmula ponderada. Se puede calcular la calificación necesaria en el examen 3 para aprobar.
    
    Los modelos fueron entrenados utilizando el algoritmo de Bosques Aleatorios (Random Forest) utilizando la información de alumnos de semestres pasados.
    
    **Nota:** Estas predicciones son aproximaciones y no garantizan resultados definitivos en evaluaciones académicas reales.
    """)