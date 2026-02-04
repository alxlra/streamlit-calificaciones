# Predicción de calificaciones

Esta aplicación utiliza modelos de aprendizaje automático para predecir si un estudiante aprobará o no un curso basado en sus calificaciones en exámenes, prácticas y proyectos. 
- El **Modelo de 1 Calificación** utiliza solo la calificación del primer examen parcial. Tiene una precisión del **76%**.
- El **Modelo de 2 Calificaciones** utiliza las calificaciones de los dos exámenes parciales. Tiene una precisión del **95%**.
- La pestaña de **Cálculo por fórmula** permite calcular la calificación final basada en una fórmula ponderada. En este caso no hay predicción, solo cálculo.


Los modelos fueron entrenados utilizando el algoritmo de Bosques Aleatorios (Random Forest) utilizando la información de alumnos de semestres pasados.
    
    Nota: Estas predicciones son aproximaciones y no garantizan resultados definitivos en evaluaciones académicas reales.

---
### Crear entorno:
`call .venv\Scripts\activate.bat`

### Ejecutar Streamlit
`".venv\Scripts\python.exe" -m streamlit run app.py --server.headless=false --server.port=8502`

_headless=false_ para que abra el navegador automáticamente
