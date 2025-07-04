import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import pickle
from utils.predict import predict_fraud
from utils.shap_plot import plot_global_shap, plot_local_shap

# Load model
with open("model/xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection")

st.write("Upload a single-row CSV or enter values manually for prediction and SHAP explanation.")


# Upload single-row CSV or Excel
uploaded_file = st.file_uploader("Upload a single-row CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        input_df = pd.read_csv(uploaded_file, encoding="utf-8", engine="python")
    elif uploaded_file.name.endswith(".xlsx"):
        input_df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format.")
        st.stop()
        
    st.markdown("---")


    # Convert to numeric
    input_df = input_df.apply(pd.to_numeric, errors='coerce')

    # Ensure only one row
    if input_df.shape[0] != 1:
        st.error("Please upload a file with only one row.")
        st.stop()

    if input_df.isnull().any().any():
        st.error("Some values could not be converted to numbers. Please check the uploaded file.")
        st.stop()

    # ✅ Prediction
    prediction, proba = predict_fraud(model, input_df)

    st.subheader("🔍 Prediction Result")
    st.write(f"**Prediction:** {'Fraud' if prediction == 1 else 'Not Fraud'}")
    st.write(f"**Probability of Fraud:** {proba:.2%}")

    # ✅ SHAP plots
    st.subheader("📊 Global Feature Importance (SHAP)")
    plot_global_shap(model, input_df)

    st.subheader("📌 Local Explanation for This Transaction")
    plot_local_shap(model, input_df)

else:
    st.info("No file uploaded yet. Please upload a file.")
