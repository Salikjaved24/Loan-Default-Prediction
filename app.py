import pickle

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load models and scaler
log_reg = joblib.load("models/logistic_regression_model.pkl")
xgb = pickle.load(open("models/xgboost_model.pkl", "rb"))
ffnn = load_model("models/ffnn_model.keras")
scaler = joblib.load("models/standard_scaler.pkl")
pt = joblib.load("models/boxcox_transformer.pkl")

# Master feature definition: order matters!
features = {
    "Age": {
        "type": "numeric",
        "default": 56.0,
        "explanation": "The age of the borrower in years.",
    },
    "Income": {
        "type": "numeric",
        "default": 85994.0,
        "explanation": "The annual income of the borrower in USD.",
    },
    "LoanAmount": {
        "type": "numeric",
        "default": 50587.0,
        "explanation": "The amount of money being borrowed in USD.",
    },
    "CreditScore": {
        "type": "numeric",
        "default": 520.0,
        "explanation": "Credit score indicating borrower creditworthiness.",
    },
    "MonthsEmployed": {
        "type": "numeric",
        "default": 80.0,
        "explanation": "Months the borrower has been employed at current job.",
    },
    "NumCreditLines": {
        "type": "numeric",
        "default": 4.0,
        "explanation": "Number of active credit lines the borrower has.",
    },
    "InterestRate": {
        "type": "numeric",
        "default": 15.23,
        "explanation": "Interest rate for the loan as a percentage.",
    },
    "LoanTerm": {
        "type": "numeric",
        "default": 36.0,
        "explanation": "Duration of the loan in months.",
    },
    "DTIRatio": {
        "type": "derived",
        "explanation": "Debt-to-Income ratio (total debt / annual income).",
    },
    "Education": {
        "type": "categorical",
        "default": 0.0,
        "categories": ["Bachelor's", "High School", "Master's", "PhD"],
        "explanation": "Highest education level attained by the borrower.",
    },
    "EmploymentType": {
        "type": "categorical",
        "default": 0.0,
        "categories": ["Full-time", "Part-time", "Self-employed", "Unemployed"],
        "explanation": "Borrower's employment status.",
    },
    "MaritalStatus": {
        "type": "categorical",
        "default": 0.0,
        "categories": ["Divorced", "Married", "Single"],
        "explanation": "Borrower's marital status.",
    },
    "HasMortgage": {
        "type": "categorical",
        "default": 1.0,
        "categories": ["No", "Yes"],
        "explanation": "Whether the borrower currently has a mortgage.",
    },
    "HasDependents": {
        "type": "categorical",
        "default": 1.0,
        "categories": ["No", "Yes"],
        "explanation": "Whether the borrower has dependents.",
    },
    "LoanPurpose": {
        "type": "categorical",
        "default": 4.0,
        "categories": ["Auto", "Business", "Education", "Home", "Other"],
        "explanation": "The reason for taking out the loan.",
    },
    "HasCoSigner": {
        "type": "categorical",
        "default": 1.0,
        "categories": ["No", "Yes"],
        "explanation": "Whether there is a co-signer on the loan.",
    },
    "AffRatio": {
        "type": "derived",
        "explanation": "LoanAmount divided by Income, a financial ratio.",
    },
    "TotalInterest": {
        "type": "derived",
        "explanation": "Total interest paid: InterestRate * LoanTerm.",
    },
    "Debt": {"type": "numeric", "default": 37837.36, "explanation": "Total debt."},
    "AvgBorrowed": {
        "type": "derived",
        "explanation": "Average borrowed amount per credit line.",
    },
}

# Gradio input components (with refs)
input_components = []
component_refs = {}

for name, meta in features.items():
    if meta["type"] == "categorical":
        dropdown = gr.Dropdown(
            label=name,
            choices=meta["categories"],
            value=meta["categories"][int(meta["default"])],
            info=meta["explanation"],
        )
        input_components.append(dropdown)
        component_refs[name] = dropdown
    elif meta["type"] == "numeric":
        number = gr.Number(label=name, value=meta["default"], info=meta["explanation"])
        input_components.append(number)
        component_refs[name] = number

# Derived components
input_components += [
    gr.Number(
        label="AffRatio",
        interactive=False,
        info=features["AffRatio"]["explanation"],
        value=lambda loan, income: round(loan / income, 5) if income else 0,
        inputs=[component_refs["LoanAmount"], component_refs["Income"]],
    ),
    gr.Number(
        label="TotalInterest",
        interactive=False,
        info=features["TotalInterest"]["explanation"],
        value=lambda rate, term: round(rate * term, 5),
        inputs=[component_refs["InterestRate"], component_refs["LoanTerm"]],
    ),
    gr.Number(
        label="DTIRatio",
        interactive=False,
        info=features["DTIRatio"]["explanation"],
        value=lambda debt, income: round(debt / income, 5) if income else 0,
        inputs=[component_refs["Debt"], component_refs["Income"]],
    ),
    gr.Number(
        label="AvgBorrowed",
        interactive=False,
        info=features["AvgBorrowed"]["explanation"],
        value=lambda loan, lines: round(loan / lines, 5) if lines else 0,
        inputs=[component_refs["LoanAmount"], component_refs["NumCreditLines"]],
    ),
]


# Inference logic
def predict_default(*inputs):
    input_map = {}
    input_index = 0

    for name, meta in features.items():
        if meta["type"] == "derived":
            continue

        val = inputs[input_index]
        if meta["type"] == "categorical":
            val = meta["categories"].index(val)
        input_map[name] = val
        input_index += 1

    # Derived features and Box-Cox transform (same as before)
    input_map["AffRatio"] = (
        round(input_map["LoanAmount"] / input_map["Income"], 5)
        if input_map["Income"]
        else 0
    )
    input_map["TotalInterest"] = round(
        input_map["InterestRate"] * input_map["LoanTerm"], 5
    )
    input_map["DTIRatio"] = (
        round(input_map["Debt"] / input_map["Income"], 5) if input_map["Income"] else 0
    )
    input_map["AvgBorrowed"] = (
        round(input_map["LoanAmount"] / input_map["NumCreditLines"], 5)
        if input_map["NumCreditLines"]
        else 0
    )

    derived_cols = ["AffRatio", "AvgBorrowed", "TotalInterest", "Debt"]
    derived_values_df = pd.DataFrame(
        [
            [
                input_map["AffRatio"],
                input_map["AvgBorrowed"],
                input_map["TotalInterest"],
                input_map["Debt"],
            ]
        ],
        columns=derived_cols,
    )

    transformed_derived = pt.transform(derived_values_df).flatten()

    (
        input_map["AffRatio"],
        input_map["AvgBorrowed"],
        input_map["TotalInterest"],
        input_map["Debt"],
    ) = transformed_derived

    input_row = [input_map[name] for name in features.keys()]
    input_df = pd.DataFrame([input_row], columns=list(features.keys()))
    scaled = scaler.transform(input_df)

    # Get probabilities
    probs = {
        "Logistic Regression": float(log_reg.predict_proba(scaled)[0][1]),
        "XGBoost": float(xgb.predict_proba(input_df.values)[0][1]),
        "FFNN": float(ffnn.predict(scaled, verbose=0)[0][0]),
    }

    # Binary labels using 0.5 threshold
    labels = {
        model: "Default" if p >= 0.5 else "No Default" for model, p in probs.items()
    }

    # Create markdown summary for labels
    label_md = "\n".join(
        [f"## {model}: *{label}*\n" for model, label in labels.items()]
    )

    # Explanatory text for the user
    explanation_md = (
        "### Prediction Explanation\n"
        "Each model predicts the probability that the borrower will default on their loan.\n"
        "- Probabilities closer to 1 indicate higher risk of default.\n"
        "- A threshold of 0.5 is used to classify 'Default' vs 'No Default'.\n"
        "- Consider the results from all models to get a comprehensive view.\n"
        "\n"
        "Please use this information as guidance and not a final decision."
    )

    # For bar chart: format data as dict with labels and values
    bar_data = pd.DataFrame(
        {
            "Model": list(probs.keys()),
            "Default Probability": list(probs.values()),
        }
    )

    return bar_data, label_md, explanation_md


output_bar = gr.BarPlot(
    x="Model", y="Default Probability", label="Model Default Probabilities", height=250
)
output_labels = gr.Markdown()
output_explanation = gr.Markdown()

demo = gr.Interface(
    fn=predict_default,
    inputs=input_components,
    outputs=[output_bar, output_labels, output_explanation],
    title="Loan Default Risk Predictor",
    description="Enter borrower info and see the default risk prediction from 3 models.",
    flagging_mode="never",
)

demo.launch()
