import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

UMBRAL_1CAL = 0.35
UMBRAL_2CAL = 0.4

@st.cache_data
def load_model_1cal():
    from joblib import load
    return load('./data/prediccion_1cal.pkl')
    
@st.cache_data
def load_model_2cal():
    from joblib import load
    return load('./data/prediccion_2cal.pkl')

st.title("Predicción de Calificaciones")
tab4, tab1, tab2, tab3 = st.tabs(["❓ Información", "🧙‍♂️ 1 Examen", "🧙‍♂️ 2 Exámenes", "💻 Cálculo por fórmula"])

with tab1:
    st.subheader("1 Examen")
    st.write("Este modelo predice si un estudiante aprobará o no basado en solo la calificación del **primer examen parcial y la primera práctica** asumiendo que se entrega el proyecto con 84 de calificación.")
    st.write("*Nota: La probablidad de aprobar ya incluye realizar el repechaje y pasarlo.*")
    examen1 = st.number_input("Examen 1 (0-100):", min_value=0, max_value=100, value=30, key='1_examen1')
    practicas1 = st.number_input("Práctica 1 (0-100):", min_value=0, max_value=100, value=95, key='1_practicas', help="Calificación de la primer práctica")
    proyecto1 = 84 #promedio de proyectos entregados

    if st.button("🧙‍♂️ Predecir", key="1_button"):
        rfclas1 = load_model_1cal()

        prediccion = pd.DataFrame({'Examen1':[examen1], 'Practica1':[practicas1], 'Proyecto':[proyecto1]})
        
        # probs = rfclas1.predict_proba(prediccion)
        # st.progress(probs[0][1])
        # res1_1, res1_2 = st.columns(2)
        # with res1_1:
        #     st.error(f"### {probs[0][0]*100:.2f}% No aprobará")
        # with res1_2:
        #     st.success(f"### {probs[0][1]*100:.2f}% Aprobará")
        p_si = float(rfclas1.predict_proba(prediccion)[0][1])
        umbral = UMBRAL_1CAL
        aprueba = p_si >= umbral

        res1_1, res1_2 = st.columns([1,2])
        with res1_1:
            st.metric("Score de aprobación", f"{p_si*100:.1f}%", help=f"Regla de decisión: aprueba si el score ≥ {umbral:.2f}") # prioriza detectar aprobados
        with res1_2:
            if aprueba:
                st.success(f"### ✅ Aprobará")
            else:
                st.error(f"### ⚠️ No aprobará")

with tab2:
    st.subheader("2 Exámenes")
    st.write("Este modelo predice si un estudiante aprobará o no basado en las calificaciones de los **primeros dos exámenes parciales y las primeras 2 prácticas** y asumiendo que se entrega el proyecto con 84 de calificación.")
    st.write("*Nota: La probablidad de aprobar ya incluye realizar el repechaje y pasarlo.*")
    examen2_1 = st.number_input("Examen 1 (0-100):", min_value=0, max_value=100, value=30, key="2_examen1")
    examen2_2 = st.number_input("Examen 2 (0-100):", min_value=0, max_value=100, value=30, key="2_examen2")
    practicas1 = st.number_input("Práctica 1 (0-100):", min_value=0, max_value=100, value=95, key='2_practica1', help="Calificación de la primer práctica")
    practicas2 = st.number_input("Práctica 2 (0-100):", min_value=0, max_value=100, value=95, key="2_practica2", help="Calificación de la segunda práctica")
    proyecto2 = 84 #promedio de proyectos entregados

    if st.button("🧙‍♂️ Predecir", key="2_button"):
        rfclas2 = load_model_2cal()

        prediccion = pd.DataFrame({'Examen1':[examen2_1], 'Examen2':[examen2_2], 'Practica1':[practicas1], 'Practica2':[practicas2], 'Proyecto':[proyecto2]})
        #res = rfclas2.predict(prediccion)
        # probs = rfclas2.predict_proba(prediccion)
        # st.progress(probs[0][1])
        # res2_1, res2_2 = st.columns(2)
        # with res2_1:
        #     st.error(f"### {probs[0][0]*100:.2f}% No aprobará")
        # with res2_2:
        #     st.success(f"### {probs[0][1]*100:.2f}% Aprobará")

        p_si = float(rfclas2.predict_proba(prediccion)[0][1])
        umbral = UMBRAL_2CAL
        aprueba = p_si >= umbral

        res1_1, res1_2 = st.columns([1,2])
        with res1_1:
            st.metric("Score de aprobación", f"{p_si*100:.1f}%", help=f"Regla de decisión: aprueba si el score ≥ {umbral:.2f}") # prioriza detectar aprobados
        with res1_2:
            if aprueba:
                st.success(f"### ✅ Aprobará")
            else:
                st.error(f"### ⚠️ No aprobará")
with tab3:
    st.subheader("Cálculo por fórmula")
    st.write("Calcula la calificación final basada en una fórmula ponderada.")
    examen3_1 = st.number_input("Examen 1 (0-100):", min_value=0, max_value=100, value=30, key="3_examen1", help="15% de la calificación final")
    examen3_2 = st.number_input("Examen 2 (0-100):", min_value=0, max_value=100, value=30, key="3_examen2", help="20% de la calificación final")
    examen3_3 = st.number_input("Examen 3 (0-100):", min_value=0, max_value=100, value=30, key="3_examen3", help="20% de la calificación final")
    practicas3 = st.number_input("Total de prácticas (0-400):", min_value=0, max_value=400, value=380, key="3_practicas", help="20% de la calificación final")
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
            if min_examen3 > 0:
                st.success(f"Calificación **mínima** necesaria en **Examen 3** para aprobar: **{min_examen3:.2f}**")
        with mincol2:
            if min_examen3_rep > 0:
                st.warning(f"Calificación **mínima** necesaria en **Examen 3** para repechaje: **{min_examen3_rep:.2f}**")
with tab4:
    st.subheader("❓ Información")
    st.write("""
    Esta aplicación utiliza modelos de aprendizaje automático para predecir si un estudiante aprobará o no un curso basado en sus calificaciones de exámenes, prácticas y proyecto final. 
    
    Los modelos fueron entrenados utilizando el algoritmo de _Bosques Aleatorios (Random Forest)_ utilizando la información de alumnos de semestres pasados.
    """)
    st.warning("**❕ Nota importante:** Estas predicciones son aproximaciones y no garantizan resultados definitivos en evaluaciones académicas reales.")
    