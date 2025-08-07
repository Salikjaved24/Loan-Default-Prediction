# Loan Default Risk Predictor

This project provides a web-based tool for predicting the probability of loan default using three machine learning models: Logistic Regression, XGBoost, and a Feedforward Neural Network (FFNN). The goal is to assist in evaluating borrower risk based on financial and demographic information.

<iframe
    src="https://sidmaji-loan-default-predictor.hf.space"
    frameborder="0"
    width="600"
    height="450"
></iframe>

<p align="center">
  <a href="https://sidmaji-loan-default-predictor.hf.space" target="_blank">
    Try the Loan Default Risk Predictor on HuggingFace Spaces 🤗
  </a>
</p>

## Features

-   **Interactive Gradio Interface**: Easily input borrower data through dropdowns and number fields.
-   **Multi-Model Predictions**: View default probabilities from Logistic Regression, XGBoost, and FFNN models.
-   **Derived Financial Metrics**: Automatically computes ratios like Affordability Ratio, Total Interest, Debt-to-Income Ratio, and Average Borrowed per Credit Line.
-   **Data Preprocessing**: Includes standardization and Box-Cox transformation for skewed features.
-   **Model Interpretability**: Outputs both probability scores and binary classification (Default / No Default) for each model.

## Technologies Used

-   Python (Pandas, NumPy, Scikit-learn, XGBoost, TensorFlow/Keras)
-   Gradio for the web interface
-   Joblib / Pickle for model and transformer serialization

## Getting Started Locally

### Prerequisites

-   Python 3.9+

### Running the App

1. Clone the repository:
    ```bash
    git clone <repository-url>
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Navigate to the project directory:
    ```bash
    cd loan_default_prediction
    ```
4. Run the app:
    ```bash
    python app.py
    ```
