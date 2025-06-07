import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model and scaler
with open('diabetes_disease_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the feature names (must match what was used during training)
feature_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

# Set up Streamlit UI
st.set_page_config(page_title="Diabetes Prediction App")
st.title("🩺 Diabetes Prediction App")

st.write("Enter the patient information to predict the risk of Diabetes disease:")


age = st.number_input("Age", min_value=1, max_value=120)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=140)
bmi = st.number_input("BMI", min_value=0.0, max_value=80.0)
insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0)
pregnancies = st.slider("Pregnancies", 0, 20)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5)

# Predict button
if st.button("Predict"):
    # Create input data as a DataFrame to match scaler input
    input_df = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                              insulin, bmi, diabetes_pedigree, age]],
                            columns=feature_names)
    
    # Scale input
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Display result
    if prediction == 1:
        st.error(f"⚠️ High risk of diabetes ({probability:.2%} probability)")
    else:
        st.success(f"✅ Low risk of diabetes ({probability:.2%} probability)")