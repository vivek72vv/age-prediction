import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model pipeline
model = joblib.load("model.pkl")

# Streamlit UI
st.set_page_config(page_title="Age Group Predictor", layout="centered")
st.title("🧠 Age Group Prediction from Nutrition Survey")

st.write("Fill in the details below to predict whether the person is an **Adult** or **Senior**.")

# Define input fields (should match the model's training features)
lbxglu = st.number_input("Glucose (LBXGLU)", min_value=0.0, step=0.1)
lbxin = st.number_input("Insulin (LBXIN)", min_value=0.0, step=0.1)
bmxbmi = st.number_input("BMI (BMXBMI)", min_value=0.0, step=0.1)
ridageyr = st.number_input("Age (RIDAGEYR)", min_value=0)
riagendr = st.selectbox("Gender (RIAGENDR)", ["Male", "Female"])
dmdeduc2 = st.selectbox("Education Level (DMDEDUC2)", [1, 2, 3, 4, 5, 7, 9])  # You can customize labels
dmdmartl = st.selectbox("Marital Status (DMDMARTL)", [1, 2, 3, 4, 5, 6, 77, 99])  # Customize if needed

if st.button("Predict Age Group"):
    # Create input DataFrame
    input_data = pd.DataFrame({
        'lbxglu': [lbxglu],
        'lbxin': [lbxin],
        'bmxbmi': [bmxbmi],
        'ridageyr': [ridageyr],
        'riagendr': [riagendr],
        'dmdeduc2': [dmdeduc2],
        'dmdmartl': [dmdmartl],
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Map back to label
    label = "Adult" if prediction == 0 else "Senior"
    st.success(f"🧾 Predicted Age Group: **{label}**")
