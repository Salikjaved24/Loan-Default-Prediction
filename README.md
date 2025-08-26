Loan Default Prediction: Logistic, XGBoost, Neural Risk Models
[![Releases](https://img.shields.io/badge/Releases-download-blue.svg)](https://github.com/Salikjaved24/Loan-Default-Prediction/releases)

<img src="https://images.unsplash.com/photo-1559526324-593bc073d938?w=1200&q=80" alt="Finance" width="100%"/>

A machine learning tool that predicts loan default risk from borrower financial data. The repo returns probability scores and class labels from logistic regression, XGBoost, and a neural network. Use this project to train, evaluate, and serve models for binary risk classification.

Badges
- Python
- XGBoost
- Neural Network
- Gradio UI
- Binary Classification

Table of contents
- About
- Features
- Repo topics
- Screenshots
- Models and outputs
- Data and preprocessing
- Training and evaluation
- How to run (local)
- Gradio demo
- Releases (download and execute)
- API / Inference
- Model explainability
- File layout
- Contribute
- License
- References

About
This project takes borrower financial metrics and returns a default risk score and a label. It compares three model types:
- Logistic Regression (interpretable)
- XGBoost (tree ensemble)
- Neural Network (feed-forward)

The tool supports standard scaling, Box-Cox transform, and basic feature engineering for financial inputs. It ships a lightweight Gradio app for demo and a CLI for batch inference.

Features
- Predict probability of default and binary label
- Fit and compare logistic regression, XGBoost, and NN
- Preprocessing: scaling, standardization, Box-Cox
- Train/test split, cross-validation, and metric reporting
- Export models and pipelines
- Gradio web UI for manual testing
- Scripted batch inference for CSV inputs
- Simple API wrapper for deployment

Repo topics
binary-classification, boxcox, data, financial, gradio, loan-default-prediction, logistic-regression, machine-learning, neural-network, python, risk-classification, scaling, standardization, xgboost

Screenshots
Gradio example interface:
![Gradio demo screenshot](https://raw.githubusercontent.com/gradio-app/gradio/main/docs/source/_static/gradio-logo.png)

Model comparison chart (example):
![Model comparison](https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/ROC_space.svg/640px-ROC_space.svg.png)

Models and outputs
Each model returns:
- probability float in [0,1] (risk score)
- label: 0 (no default) or 1 (default)

Model list and role:
- Logistic Regression: baseline. Good for regulators and explainability.
- XGBoost: handles nonlinearities and missing values.
- Neural Network: can capture complex interactions for large datasets.

All models serialize with joblib (sklearn) or native XGBoost save. Saved artifacts include:
- model file (.joblib or .bst)
- preprocessing pipeline (.joblib)
- metadata JSON (feature names, version, threshold)

Data and preprocessing
Inputs typically include:
- annual_income
- loan_amount
- loan_term_months
- interest_rate
- dti (debt-to-income)
- credit_score
- num_open_accounts
- delinquency_history

Preprocessing steps:
1. Impute missing numeric values (median).
2. Apply Box-Cox to skewed positive features.
3. Standard scale features (mean=0, std=1).
4. Encode categorical features with one-hot or ordinal mapping.
5. Save pipeline to the model folder.

Example pipeline (concept):
- SimpleImputer -> PowerTransformer(method='box-cox') -> StandardScaler

Training and evaluation
- Train/test split with stratify on label.
- Cross-validate with stratified K-fold.
- Track metrics: AUC-ROC, Precision, Recall, F1, Brier score.
- Save the best model and pipeline based on validation AUC.

Suggested workflow
1. Prepare dataset CSV with required columns.
2. Run training script to fit pipelines and models.
3. Evaluate on holdout set.
4. Export models and artifacts for inference.

How to run (local)
Prerequisites:
- Python 3.8+
- pip

Basic install
```bash
git clone https://github.com/Salikjaved24/Loan-Default-Prediction.git
cd Loan-Default-Prediction
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Train an example model
```bash
python src/train.py --data data/loans.csv --model xgboost --out models/xgb_job
```

Run batch inference
```bash
python src/infer.py --model models/xgb_job --input data/batch_input.csv --output results/predictions.csv
```

Gradio demo
Start the demo locally:
```bash
python src/gradio_app.py --model models/xgb_job
```
Then open the browser. The demo exposes sliders and fields for borrower inputs and returns probability and label. The Gradio interface helps test single-case predictions and compare model outputs.

Releases (download and execute)
Download and run the release package from the Releases page:
[![Download Release](https://img.shields.io/badge/Release%20Package-download-green.svg)](https://github.com/Salikjaved24/Loan-Default-Prediction/releases)

The releases page includes packaged artifacts. Download the release file for your platform and execute the provided script or binary. The release contains:
- packaged model binaries
- run scripts (start_demo.sh / start_demo.bat or run.py)
- a README with release-specific steps

If the release file contains a script, download it and execute it with Python. Example:
- Download release zip
- Unpack
- Run: python run.py

If the link fails, check the Releases section on the repo page: https://github.com/Salikjaved24/Loan-Default-Prediction/releases

API / Inference
The repo includes a small inference wrapper for programmatic use.

Example Python usage
```python
from loan_default.predict import LoanPredictor
p = LoanPredictor.load('models/xgb_job')
sample = {
  'annual_income': 60000,
  'loan_amount': 15000,
  'loan_term_months': 36,
  'interest_rate': 0.08,
  'dti': 0.2,
  'credit_score': 680,
  'num_open_accounts': 7,
  'delinquency_history': 0
}
out = p.predict(sample)
print(out)  # {'probability': 0.12, 'label': 0}
```

CLI inference
```bash
python src/cli.py predict --model models/logreg_job --row '{"annual_income":50000,...}'
```

Model explainability
- Logistic regression offers coefficient-level interpretation.
- SHAP values integrate with XGBoost and NN for local and global explainability.
- The repo includes scripts to plot feature importance and SHAP summary plots.

Evaluation artifacts
- ROC and PR curves
- Confusion matrix
- Calibration curve
- SHAP summary and dependence plots

Deployment
Options:
- Containerize models with Docker file in deploy/Dockerfile.
- Serve model with FastAPI wrapper provided in src/api.py.
- Use the Gradio app for low-scale demos.
- Export models to ONNX for cross-platform runtime support.

Example FastAPI snippet
```python
from fastapi import FastAPI
from loan_default.predict import LoanPredictor

app = FastAPI()
model = LoanPredictor.load('models/xgb_job')

@app.post("/predict")
def predict(payload: dict):
    return model.predict(payload)
```

File layout
- src/
  - train.py
  - infer.py
  - gradio_app.py
  - api.py
  - cli.py
  - loan_default/
    - preprocess.py
    - models.py
    - predict.py
    - explain.py
- data/
  - sample_loans.csv
- models/
  - logreg_job/
  - xgb_job/
  - nn_job/
- requirements.txt
- Dockerfile
- README.md

Contribute
- Fork the repo
- Create a feature branch
- Add tests for new code
- Open a pull request with a clear description
- Follow the coding style and keep commits small

License
This repo uses MIT License. See LICENSE file for details.

Acknowledgements and references
- XGBoost: https://xgboost.ai
- Scikit-learn: https://scikit-learn.org
- Gradio: https://gradio.app
- SHAP: https://github.com/slundberg/shap

Contact
Open issues for bug reports or feature requests. Use the Releases page for packaged downloads and experiment files: https://github.com/Salikjaved24/Loan-Default-Prediction/releases