import streamlit as st
import joblib
import numpy as np

st.title("Credit Card Fraud Detection System")

# Load model
model = joblib.load("model.pkl")

amount = st.number_input("Transaction Amount", min_value=0.0)
time = st.number_input("Transaction Time", min_value=0.0)

if st.button("Check Fraud"):
    input_data = np.array([[time, amount]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠ Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")