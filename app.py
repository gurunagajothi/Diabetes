import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Prediction App")
st.write("Predict whether a patient is **Diabetic or Not** using Logistic Regression.")

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    with open("diabetes_logistic_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# -----------------------------
# User input
# -----------------------------
st.subheader("Enter Patient Details")

age = st.number_input("Age", min_value=1, max_value=120, value=25)
mass = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=70.0, value=25.0)
insu = st.number_input("Insulin Level", min_value=0.0, max_value=900.0, value=80.0)
plas = st.number_input("Plasma Glucose Level", min_value=0.0, max_value=300.0, value=120.0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    input_data = np.array([[age, mass, insu, plas]])

    prediction = model.predict(input_data)

    if prediction[0] == "tested_positive":
        st.error("⚠️ Prediction: **Diabetic**")
    else:
        st.success("✅ Prediction: **Not Diabetic**")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Model: Logistic Regression | Built with Streamlit")
