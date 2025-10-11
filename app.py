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
tab1, tab2, tab3, tab4 = st.tabs(["Modelo 1 Calificación", "Modelo 2 Calificaciones", "Cálculo por fórmula", "❓ Información"])

with tab1:
    st.subheader("1 Calificación")
    st.write("Este modelo predice si un estudiante aprobará o no basado en solo la calificación del primer examen parcial.")
    st.write("Tiene una precisión del 76% en datos de prueba.")
    examen1 = st.number_input("Examen 1 (0-100):", min_value=0, max_value=100, value=30, key='1_examen1')
    practicas1 = st.number_input("Prácticas (0-100):", min_value=0, max_value=100, value=95, key='1_practicas', help="Promedio de todas las prácticas del curso")
    proyecto1 = st.number_input("Proyecto (0-100):", min_value=0, max_value=100, value=85, key='1_proyecto', help="Incluyendo avances y entrega final")
    if st.button("🧙‍♂️ Predecir", key="1_button"):
        rfclas1 = load_model_1cal()

        prediccion = pd.DataFrame({'Examen1':[examen1], 'Practicas':[practicas1], 'Proyecto':[proyecto1]})
        res = rfclas1.predict(prediccion)
        if res[0] == 0:
            st.error("✘ No aprobará")
        else:
            st.success("✔ Aprobará")

with tab2:
    st.subheader("2 Calificaciones")
    st.write("Este modelo predice si un estudiante aprobará o no basado en las calificaciones de los dos exámenes parciales.")
    st.write("Tiene una precisión del 95% en datos de prueba.")
    examen2_1 = st.number_input("Examen 1 (0-100):", min_value=0, max_value=100, value=30, key="2_examen1")
    examen2_2 = st.number_input("Examen 2 (0-100):", min_value=0, max_value=100, value=30, key="2_examen2")
    practicas2 = st.number_input("Prácticas (0-100):", min_value=0, max_value=100, value=95, key="2_practicas", help="Promedio de todas las prácticas del curso")
    proyecto2 = st.number_input("Proyecto (0-100):", min_value=0, max_value=100, value=85, key="2_proyecto", help="Incluyendo avances y entrega final")
    if st.button("🧙‍♂️ Predecir", key="2_button"):
        rfclas2 = load_model_2cal()

        prediccion = pd.DataFrame({'Examen1':[examen2_1], 'Examen2':[examen2_2], 'Practicas':[practicas2], 'Proyecto':[proyecto2]})
        res = rfclas2.predict(prediccion)
        if res[0] == 0:
            st.error("✘ No aprobará")
        else:
            st.success("✔ Aprobará")
with tab3:
    st.subheader("Cálculo por fórmula")
    st.write("Calcula la calificación final basada en una fórmula ponderada.")
    examen3_1 = st.number_input("Examen 1 (0-100):", min_value=0, max_value=100, value=30, key="3_examen1", help="15% de la calificación final")
    examen3_2 = st.number_input("Examen 2 (0-100):", min_value=0, max_value=100, value=30, key="3_examen2", help="20% de la calificación final")
    examen3_3 = st.number_input("Examen 3 (0-100):", min_value=0, max_value=100, value=30, key="3_examen3", help="20% de la calificación final")
    practicas3 = st.number_input("Prácticas (0-100):", min_value=0, max_value=100, value=95, key="3_practicas", help="20% de la calificación final")
    proyecto_avances3 = st.number_input("Avances de proyecto (0-100):", min_value=0, max_value=100, value=100, key="3_proyecto_avances", help="10% de la calificación final")
    proyecto3 = st.number_input("Proyecto (0-100):", min_value=0, max_value=100, value=85, key="3_proyecto", help="15% de la calificación final")
    if st.button("💻 Calcular calificación", key="3_button"):
        calificacion_final = (examen3_1 * 0.15) + (examen3_2 * 0.20) + (examen3_3 * 0.20) + (practicas3 * 0.20) + (proyecto_avances3 * 0.10) + (proyecto3 * 0.15)
        if calificacion_final >= 60:
            st.success(f"✔ Aprobará con una calificación final de **{calificacion_final:.2f}**")
        elif calificacion_final >= 55:
            st.info(f"⚠ Puede recuperar repechaje, actualmente **{calificacion_final:.2f}**")
        else:
            st.error(f"✘ No aprobará con una calificación final de **{calificacion_final:.2f}**")
    
        min_examen3 = (60 - (examen3_1 * 0.15) - (examen3_2 * 0.20) - (practicas3 * 0.20) - (proyecto_avances3 * 0.10) - (proyecto3 * 0.15)) / 0.20
        min_examen3_rep = (55 - (examen3_1 * 0.15) - (examen3_2 * 0.20) - (practicas3 * 0.20) - (proyecto_avances3 * 0.10) - (proyecto3 * 0.15)) / 0.20
        mincol1, mincol2 = st.columns(2)
        with mincol1:
            st.success(f"Calificación **mínima** necesaria en **Examen** 3 para aprobar: **{min_examen3:.2f}**")
        with mincol2:
            st.warning(f"Calificación **mínima** necesaria en **Examen** 3 para repechaje: **{min_examen3_rep:.2f}**")
with tab4:
    st.subheader("❓ Información")
    st.write("""
    Esta aplicación utiliza modelos de aprendizaje automático para predecir si un estudiante aprobará o no un curso basado en sus calificaciones en exámenes, prácticas y proyectos. 
    - El **Modelo de 1 Calificación** utiliza solo la calificación del primer examen parcial. Tiene una precisión del **76%**.
    - El **Modelo de 2 Calificaciones** utiliza las calificaciones de los dos exámenes parciales. Tiene una precisión del **95%**.
    - La pestaña de **Cálculo por fórmula** permite calcular la calificación final basada en una fórmula ponderada. En este caso no hay predicción, solo cálculo.
    
    Los modelos fueron entrenados utilizando el algoritmo de Bosques Aleatorios (Random Forest) utilizando la información de alumnos de semestres pasados.
    
    **Nota:** Estas predicciones son aproximaciones y no garantizan resultados definitivos en evaluaciones académicas reales.
    """)