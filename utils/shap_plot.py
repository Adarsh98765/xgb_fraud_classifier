import shap
import matplotlib.pyplot as plt
import streamlit as st

explainer = None

def plot_global_shap(model, input_df):
    global explainer
    if explainer is None:
        explainer = shap.Explainer(model)

    shap_values = explainer(input_df)
    fig = plt.figure()
    shap.plots.bar(shap_values, show=False)
    st.pyplot(fig)

def plot_local_shap(model, input_df):
    global explainer
    if explainer is None:
        explainer = shap.Explainer(model)

    shap_values = explainer(input_df)
    fig = plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
