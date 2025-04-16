import streamlit as st
import pandas as pd
import joblib

# Load models and scaler
model_diag = joblib.load("diagnosis_model.pkl")
model_med = joblib.load("medication_model.pkl")

st.title("🧬 Personalized Healthcare & Medicine Recommendation")

st.markdown("Enter patient details below:")

# Input fields
age = st.number_input("Age", min_value=0, max_value=100, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
symptom1 = st.selectbox("Symptom 1", ["Headache", "Fever", "Cough"])
symptom2 = st.selectbox("Symptom 2", ["Fatigue", "Nausea", "Shortness of Breath"])

# Sample encoding (you should match this with training encoding)
gender = 1 if gender == "Male" else 0
symptom1 = {"Headache": 0, "Fever": 1, "Cough": 2}[symptom1]
symptom2 = {"Fatigue": 0, "Nausea": 1, "Shortness of Breath": 2}[symptom2]

# Create feature array
features = pd.DataFrame([[age, gender, symptom1, symptom2]],
                        columns=["Age", "Gender", "Symptom1", "Symptom2"])

st.write("Feature shape:", features.shape)
st.write("Expected shape:", model_diag.n_features_in_)

if st.button("Predict"):
    pred_diagnosis = model_diag.predict(features)[0]
    pred_medicine = model_med.predict(features)[0]

    st.success(f"🧾 Predicted Diagnosis: **{pred_diagnosis}**")
    st.success(f"💊 Recommended Medicine: **{pred_medicine}**")
