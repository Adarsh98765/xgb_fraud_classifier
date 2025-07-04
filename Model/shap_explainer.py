import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shap
import joblib
import pandas as pd
from utils.preprocess import load_and_clean_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Load trained model
model = joblib.load("model/xgb_model.pkl")

# Load and preprocess data
X, y = load_and_clean_data("data/raw/creditcard_2023.csv", target_col="Class")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Create SHAP explainer
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Create output folder
os.makedirs("model/shap_outputs", exist_ok=True)

# Plot 1: Global Feature Importance
plt.figure()
shap.plots.bar(shap_values, show=False)
plt.title("Global Feature Importance (SHAP)")
plt.savefig("model/shap_outputs/global_shap_bar.png", bbox_inches="tight")
print("✅ Saved: model/shap_outputs/global_shap_bar.png")

# Plot 2: Explanation for a single prediction
sample_idx = 0
shap.plots.waterfall(shap_values[sample_idx], show=False)
plt.savefig("model/shap_outputs/sample_explanation.png", bbox_inches="tight")
print("✅ Saved: model/shap_outputs/sample_explanation.png")
