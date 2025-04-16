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

# Collect user input for categorical columns
for col in label_encoders:
    options = label_encoders[col].classes_
    val = st.selectbox(f"{col}", options)
    user_data[col] = label_encoders[col].transform([val])[0]

# Collect user input for numerical columns
for col in ["Age", "Weight_kg", "Height_cm", "BMI", "Recovery_Time_Days"]:
    user_data[col] = st.number_input(f"{col}", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([user_data])

# Ensure the input matches the expected feature names used during training
model_features = scaler.feature_names_in_

# Align features with model features (in case they are not in the same order)
input_df = input_df[model_features]

# Scale the input data
scaled_input = scaler.transform(input_df)

# Predict
if st.button("Predict"):
    try:
        # Predict diagnosis
        diagnosis = model_diag.predict(scaled_input)[0]

        # Predict medication
        med_encoded = model_med.predict(scaled_input)[0]
        medication = le_med.inverse_transform([med_encoded])[0]

        # Display results
        st.success(f"🧾 Diagnosis: **{diagnosis}**")
        st.success(f"💊 Recommended Medicine: **{medication}**")

    except Exception as e:
        st.error(f"An error occurred: {e}")
