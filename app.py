import pickle

import gradio as gr
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Load models
log_reg = joblib.load("models/logistic_regression_model.pkl")
xgb = pickle.load(open("models/xgboost_model.pkl", "rb"))
ffnn = load_model("models/ffnn_model.keras")
scaler = joblib.load("models/standard_scaler.pkl")

import json

with open("data/feature_names.json", "r") as f:
    feature_names = json.load(f)


def predict_default(*inputs):
    inputs_array = np.array(inputs).reshape(1, -1)
    scaled = scaler.transform(inputs_array)
    logit = log_reg.predict_proba(scaled)[0][1]
    xgb_pred = xgb.predict_proba(inputs_array)[0][1]
    ffnn_pred = ffnn.predict(scaled)[0][0]
    return {
        "Logistic Regression": float(logit),
        "XGBoost": float(xgb_pred),
        "FFNN": float(ffnn_pred),
    }


input_components = [gr.Number(label=name) for name in feature_names]
output_components = gr.Label(num_top_classes=3)

demo = gr.Interface(
    fn=predict_default,
    inputs=input_components,
    outputs=output_components,
    title="Loan Default Risk Predictor",
    description="Enter borrower info and see the default risk prediction from 3 models.",
)

demo.launch()
