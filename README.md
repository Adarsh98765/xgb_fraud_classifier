#  XGBoost Fraud Classifier

This project is a machine learning-based credit card fraud detection system that leverages the power of **XGBoost**, **SHAP explainability**, and an **interactive Streamlit app** to make fraud predictions interpretable and accessible.

##  Project Highlights

-  Built using **XGBoost**, a powerful gradient boosting model.
-  **SHAP (SHapley Additive exPlanations)** provides both global and local interpretability.
-  Upload any single transaction (CSV or Excel) to get predictions and explanation plots.
-  Fully interactive **Streamlit** UI for quick exploration and usage.

---

## Project Structure

```
.
├── Model/
│   ├── train_model.py          # Trains XGBoost model and saves pickle
│   ├── xgb_model.pkl           # Trained XGBoost model
│   ├── shap_explainer.py       # Generates SHAP explainer object
│   └── shap_outputs/           # Stores SHAP plots
│
├── utils/
│   ├── preprocess.py           # Data loading and preprocessing functions
│   ├── predict.py              # Prediction logic
│   └── shap_plot.py            # SHAP visualization functions
│
├── streamlit_app/
│   └── app.py                  # Streamlit frontend application
│
├── requirements.txt            # Project dependencies
└── .gitignore                  # Files and folders ignored by Git
```

---

##  Model Training

1. Download the dataset separately (due to size constraints).
2. Place it under `Data/raw/`.
3. Run:

```bash
python Model/train_model.py
This will generate the model (xgb_model.pkl) and SHAP explainer.

 Run the Streamlit App
bash
Copy
Edit
cd streamlit_app
streamlit run app.py
Then, open the link in your browser and upload a single-row CSV or Excel file with feature values to get predictions and insights.

 Tech Stack
Python

XGBoost

SHAP

Pandas, NumPy

Streamlit

Matplotlib

 Notes
Trained model and SHAP explainer are saved as .pkl files.

The dataset (creditcard_2023.csv) is not included in the repo due to GitHub’s file size limit.

Make sure uploaded files contain exactly one row for prediction.

 License
This project is open-source and available under the MIT License.

 Future Improvements
Add Docker support for easier deployment.

Expand to accept batch predictions.

Improve EDA and feature selection via Notebooks.

 Contributions
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.