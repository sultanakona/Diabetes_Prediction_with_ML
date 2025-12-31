This folder contains Jupyter notebooks.
# 🩺 Diabetes Prediction — Machine Learning Project

[![Status](https://img.shields.io/badge/status-production-brightgreen.svg)](https://github.com/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Dataset](https://img.shields.io/badge/dataset-Pima%20Indians%20Diabetes-orange.svg)](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

A clean, well-documented repository for predicting diabetes using classical and modern ML models. This project demonstrates data exploration, preprocessing, model training, evaluation, and a small API for inference — built for learning, experimentation, and quick deployment.

---

## ✨ Highlights

- Clear end-to-end pipeline: Data → Features → Models → Evaluation → API
- Multiple models: Logistic Regression, Random Forest, XGBoost, and a small Neural Network
- Robust preprocessing: missing-value handling, scaling, and feature analysis
- Reproducible experiments and example inference code
- Ready-to-run scripts + optional Docker support

---

## 🧭 Project Structure

- data/ — raw and processed data (gitignored)
- notebooks/ — EDA and experimentation notebooks
- src/
  - data.py — loaders and preprocessors
  - models.py — model definitions & training helpers
  - evaluate.py — metrics and visualization helpers
  - api.py — simple Flask API for predictions
- scripts/
  - train.py — train & save models
  - predict.py — CLI inference
- requirements.txt
- README.md

---

## 🔬 Dataset

We use the popular "Pima Indians Diabetes Database" (UCI / Kaggle). It contains medical diagnostic measurements to predict whether a patient has diabetes.

Key features:
- Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- Binary target: Outcome (0 = no diabetes, 1 = diabetes)

Dataset link: https://www.kaggle.com/uciml/pima-indians-diabetes-database

---

## 🚀 Quick Start

Clone, create venv, install, and run a quick prediction:

```bash
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Train a baseline model:

```bash
python scripts/train.py --model=logistic --data=data/pima_diabetes.csv --out=models/logistic.joblib
```

Run a quick prediction:

```bash
python scripts/predict.py \
  --model=models/logistic.joblib \
  --input '{"Pregnancies":2,"Glucose":120,"BloodPressure":70,"SkinThickness":20,"Insulin":79,"BMI":25.5,"DiabetesPedigreeFunction":0.5,"Age":30}'
```

Expected output (JSON):

```json
{
  "prediction": 0,
  "probability": 0.12,
  "model": "logistic"
}
```

---

## 📈 Model & Results (example)

After training a few standard models, you might see results like:

- Logistic Regression: Accuracy 0.77, AUC 0.82
- Random Forest: Accuracy 0.78, AUC 0.84
- XGBoost: Accuracy 0.79, AUC 0.85
- Neural Network (small): Accuracy 0.77, AUC 0.83

(Exact numbers depend on preprocessing, feature engineering, train/test splits and random seeds.)

---

## 🧰 How it works — Overview

1. Data loading and validation
2. Exploratory Data Analysis (in notebooks)
3. Preprocessing:
   - Replace biologically impossible zeros with NaN for selected features
   - Impute medians (or more advanced methods)
   - Scale numeric features
4. Train multiple models with cross-validation
5. Choose best model by AUC / recall (medical applications often require high recall)
6. Save trained model and scaler for inference
7. Serve with a lightweight Flask API or export to Docker

---

## 🧾 Example: Training script (usage)

```bash
python scripts/train.py \
  --data data/pima_diabetes.csv \
  --model xgboost \
  --output models/xgboost.joblib \
  --seed 42
```

Common options:
- --model: logistic | random_forest | xgboost | mlp
- --cv: number of cross-validation folds
- --scaler: standard | robust | minmax

---

## 🧪 Evaluation & Metrics

For medical predictions, consider:
- Confusion matrix
- Precision, Recall (Sensitivity), Specificity
- ROC AUC
- Precision-Recall curve
- Calibration plots

Tip: In screening tasks prefer improving recall at a reasonable precision level.

---

## 🛠️ API Example (Flask)

Start the API:

```bash
python src/api.py --model models/xgboost.joblib --port 5000
```

POST request example (curl):

```bash
curl -X POST http://localhost:5000/predict \
 -H "Content-Type: application/json" \
 -d '{"Pregnancies":2,"Glucose":120,"BloodPressure":70,"SkinThickness":20,"Insulin":79,"BMI":25.5,"DiabetesPedigreeFunction":0.5,"Age":30}'
```

Response:

```json
{
  "prediction": 0,
  "probability": 0.12
}
```

---

## 🐳 Optional: Docker

Build:

```bash
docker build -t diabetes-ml:latest .
```

Run:

```bash
docker run -p 5000:5000 diabetes-ml:latest
```

---

## ♻️ Reproducibility

- Use fixed random seeds (e.g., --seed 42)
- Record package versions (requirements.txt)
- Save model artifacts (model + scaler + metadata)
- Keep notebooks for exploratory analysis

---

## 💡 Ideas for Improvements

- Add feature selection / SHAP explanations for interpretability
- Use richer imputation (KNN, MICE) or autoencoders
- Hyperparameter tuning (Optuna / GridSearchCV)
- Calibrate probabilities for clinical decision thresholds
- Build a small frontend to visualize risk for users / clinicians

---

## 🤝 Contributing

Contributions are welcome! Suggested ways to help:
- Improve preprocessing & imputation
- Add new models or pipelines
- Add unit tests and CI
- Improve docs and examples

Please open an issue or pull request.

---

## 📜 License

This project is MIT licensed — see the [LICENSE](./LICENSE) file for details.

---

## 🙏 Acknowledgements

- Pima Indians Diabetes Database — UCI / Kaggle
- Thanks to the open-source ML ecosystem: scikit-learn, XGBoost, pandas, Flask

---

If you'd like, I can:
- Generate a ready-to-commit README.md file for your repo,
- Create training and prediction scripts,
- Provide a sample Flask API file or Dockerfile.

Which would you like next?
