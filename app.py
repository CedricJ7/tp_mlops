# app.py

import streamlit as st
import pandas as pd
import requests
import streamlit.components.v1 as components
from joblib import load
import os

st.set_page_config(page_title="TP IA — Student CGPA", layout="wide")

api_url = "http://localhost:8000/predict"


# Load dataset for UI helpers
try:
    response = requests.get("http://localhost:8000/data")
    if response.status_code == 200:
        res_json = response.json()
        if isinstance(res_json, list):
            data = pd.DataFrame(res_json)
        else:
            data = None
    else:
        data = None
except:
    data = None

# Fallback sur le CSV local si l'API ne renvoie pas une liste valide à enlever
# if not isinstance(data, pd.DataFrame):
#     DATA_PATH = os.path.join("data", "Student_data.csv")
#     if os.path.exists(DATA_PATH):
#         data = pd.read_csv(DATA_PATH)
#     else:
#         data = None

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["Rapport", "Prédictions"]) 

if page == "Rapport":
    st.title("Rapport d'analyse exploratoire")
    report_path = "rapport_analyse_exploratoire.html" # à remplacer par une génération du rapport au lancement de l'app 
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            rapport_html = f.read()
        components.html(rapport_html, height=900, scrolling=True)
    else:
        st.error(f"Fichier de rapport introuvable: {report_path}")

elif page == "Prédictions":
    st.title("Faire une prédiction de CGPA")

    st.markdown("Remplissez les informations de l'étudiant puis cliquez sur **Prédire**.")

    # Build input form
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender_options = data['Gender'].unique().tolist() if isinstance(data, pd.DataFrame) and 'Gender' in data.columns else ['Male','Female']
            gender = st.selectbox("Gender", options=gender_options)
            age = st.number_input("Age", min_value=15, max_value=100, value=20)
            major_options = sorted(data['Major'].unique().tolist()) if isinstance(data, pd.DataFrame) and 'Major' in data.columns else ['Computer Science','Economics', 'Business', 'Engineering', 'Mathematics', 'Psychology']
            major = st.selectbox("Major", options=major_options)

        with col2:
            attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=80.0, format="%.1f")
            study_hours = st.number_input("Study hours per day", min_value=0.0, max_value=24.0, value=4.0, format="%.1f")
            previous_gpa = st.number_input("Previous GPA", min_value=0.0, max_value=4.0, value=3.0, format="%.2f")

        with col3:
            sleep_hours = st.number_input("Sleep hours", min_value=0.0, max_value=24.0, value=7.0, format="%.1f")
            social_hours = st.number_input("Social hours per week", min_value=0, max_value=168, value=5)

        submit = st.form_submit_button("Prédire")

    if submit:
        # Prépare le dictionnaire des variables attendues par l'API
        payload = {
            'Gender': gender,
            'Age': int(age),
            'Major': major,
            'Attendance_Pct': float(attendance),
            'Study_Hours_Per_Day': float(study_hours),
            'Previous_GPA': float(previous_gpa),
            'Sleep_Hours': float(sleep_hours),
            'Social_Hours_Week': int(social_hours)
        }
        try:
            response = requests.post(api_url, json=payload)
            if response.status_code == 200:
                pred_value = response.json().get("predicted_Final_CGPA")
                st.success(f"Prédiction CGPA par l'API: {pred_value:.2f}")
            else:
                st.error(f"Erreur de l'API ({response.status_code}): {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Impossible de joindre l'API sur {api_url}. Assurez-vous qu'elle est lancée (uvicorn api:app --reload).\n\nDétails: {e}")

    # Optionally show sample of the dataset
    if isinstance(data, pd.DataFrame) and st.checkbox("Afficher un extrait des données" ):
        st.dataframe(data.head())