import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import joblib
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from utils.preprocess import load_and_clean_data

# Load and preprocess the data
X, y = load_and_clean_data("data/raw/creditcard_2023.csv", target_col="Class")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train XGBoost model
model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    scale_pos_weight=(y == 0).sum() / (y == 1).sum()  # handles class imbalance
)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("✅ Classification Report:")
print(classification_report(y_test, y_pred))
print("🔍 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/xgb_model.pkl")
print("✅ Model saved as model/xgb_model.pkl")
