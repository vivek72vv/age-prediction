
import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("../models/model.pkl", "rb"))

# Streamlit UI
st.set_page_config(page_title="Age Group Prediction", layout="centered")
st.title("🧠 Age Group Prediction (Adult or Senior)")
st.write("Enter your health and nutrition metrics to predict your age group.")

# Sidebar for user input
st.sidebar.header("Input Parameters")

def user_input_features():
    GLU = st.sidebar.slider("GLU - Glucose Level", 0.0, 300.0, 120.0)
    INS = st.sidebar.slider("INS - Insulin Level", 0.0, 300.0, 90.0)
    BMI = st.sidebar.slider("BMI - Body Mass Index", 10.0, 50.0, 24.0)
    PAQ605 = st.sidebar.slider("PAQ605 - Physical Activity Frequency", 1, 4, 2)
    PAD680 = st.sidebar.slider("PAD680 - Daily Activity Level", 1, 3, 1)
    DRQSPREP = st.sidebar.slider("DRQSPREP - Food Preparation Frequency", 1, 6, 3)
    features = np.array([[GLU, INS, BMI, PAQ605, PAD680, DRQSPREP]])
    return features

input_data = user_input_features()

# Prediction
prediction = model.predict(input_data)

# Output
st.subheader("Prediction")
label = "Senior (1)" if prediction[0] == 1 else "Adult (0)"
st.success(f"Predicted Age Group: **{label}**")
