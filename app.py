import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load pretrained models and scaler
model_diag = joblib.load("diagnosis_model.pkl")
model_med = joblib.load("medication_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
le_med = joblib.load("le_med.pkl")  # medication label encoder

st.title("🧬 Personalized Healthcare & Medicine Recommendation")

# User input form
st.subheader("Enter Patient Details")

user_data = {}
for col in label_encoders:
    options = label_encoders[col].classes_
    val = st.selectbox(f"{col}", options)
    user_data[col] = label_encoders[col].transform([val])[0]

# For numerical columns
for col in ["Age", "Weight_kg", "Height_cm", "BMI", "Recovery_Time_Days"]:
    user_data[col] = st.number_input(f"{col}", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([user_data])

# Scale input
scaled_input = scaler.transform(input_df)

# Predict
if st.button("Predict"):
    diagnosis = model_diag.predict(scaled_input)[0]
    med_encoded = model_med.predict(scaled_input)[0]
    medication = le_med.inverse_transform([med_encoded])[0]

    st.success(f"🧾 Diagnosis: **{diagnosis}**")
    st.success(f"💊 Recommended Medicine: **{medication}**")
