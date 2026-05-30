# 🧠 ANN Customer Churn Prediction

A deep learning project that predicts whether a bank customer will churn (leave) using an Artificial Neural Network (ANN) built with TensorFlow/Keras. Includes a live interactive **Streamlit web app** and a bonus **Salary Regression** model.

---

## 🔗 Live Demo

> Run locally with `streamlit run app.py` (see setup below)

---

## 📌 Project Overview

Customer churn is one of the most critical problems in the banking industry. This project builds an end-to-end ANN pipeline to predict churn probability based on customer demographics and account information — helping banks proactively retain at-risk customers.

---

## 🗂️ Repository Structure

```
ANN-Classifier/
│
├── experiments.ipynb              # Main EDA + ANN training notebook
├── hyperparametertuingann.ipynb   # GridSearchCV hyperparameter tuning
├── prediction.ipynb               # Single-sample prediction notebook
├── salaryregression.ipynb         # Bonus: ANN regression (salary prediction)
│
├── app.py                         # Streamlit app — Churn Classifier
├── streamlit_regression.py        # Streamlit app — Salary Regressor
│
├── model.h5                       # Saved trained classification model
├── regression_model.h5            # Saved trained regression model
├── label_encoder_gender.pkl       # Saved LabelEncoder for Gender
├── onehot_encoder_geo.pkl         # Saved OneHotEncoder for Geography
├── scaler.pkl                     # Saved StandardScaler
│
├── Churn_Modelling.csv            # Dataset
├── logs/fit/                      # TensorBoard logs (classification)
├── regressionlogs/fit/            # TensorBoard logs (regression)
├── requirements.txt               # Dependencies
└── README.md
```

---

## 📊 Dataset

**Source:** [Churn Modelling Dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)

| Feature | Description |
|---|---|
| CreditScore | Customer credit score |
| Geography | Country (France / Germany / Spain) |
| Gender | Male / Female |
| Age | Customer age |
| Tenure | Years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Has credit card (0/1) |
| IsActiveMember | Is active (0/1) |
| EstimatedSalary | Estimated annual salary |
| **Exited** | **Target — 1 = churned, 0 = stayed** |

- **Rows:** 10,000 customers
- **Class imbalance:** ~80% stayed, ~20% churned

---

## ⚙️ Pipeline

```
Raw CSV
  │
  ├── Drop irrelevant columns (RowNumber, CustomerId, Surname)
  ├── LabelEncoder       → Gender (Male=1, Female=0)
  ├── OneHotEncoder      → Geography (France, Germany, Spain)
  ├── train_test_split   → 80% train / 20% test
  ├── StandardScaler     → Normalize all features
  │
  └── ANN Model (TensorFlow/Keras)
        ├── Dense(64, relu)
        ├── Dense(32, relu)
        └── Dense(1, sigmoid)   ← Binary output
```

---

## 🔧 Hyperparameter Tuning

Used **GridSearchCV** with `KerasClassifier` to find the best architecture:

```python
param_grid = {
    'epochs':  [50, 100],
    'neurons': [16, 32, 64],
    'layers':  [1, 2]
}
```

**Best Parameters Found:**

| Parameter | Best Value |
|---|---|
| epochs | 100 |
| layers | 1 |
| neurons | 32 |
| **CV Accuracy** | **85.60%** |

---

## 📈 Model Performance

| Metric | Score |
|---|---|
| Cross-Validation Accuracy | **85.60%** |
| AUC-ROC | ~0.87 |

> **Key focus:** Recall on the Churn class — catching actual churners matters more than overall accuracy in a business context.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Preprocessing-green?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?logo=pandas)

- **Deep Learning:** TensorFlow / Keras
- **Web App:** Streamlit
- **Preprocessing:** Scikit-learn (LabelEncoder, OneHotEncoder, StandardScaler)
- **Tuning:** GridSearchCV + EarlyStopping
- **Monitoring:** TensorBoard
- **Data:** Pandas, NumPy

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/krotov22/ANN-Classifier.git
cd ANN-Classifier
```

### 2. Create and activate conda environment
```bash
conda create -n ann python=3.11 -y
conda activate ann
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

### 5. View TensorBoard logs
```bash
tensorboard --logdir logs/fit
```
Open `http://localhost:6006` in your browser.

---

## 📱 App Preview

The Streamlit app takes customer inputs and predicts churn probability in real time:

- **Geography** (France / Germany / Spain)
- **Gender**, **Age**, **Tenure**
- **Balance**, **Credit Score**, **Estimated Salary**
- **Number of Products**, **Has Credit Card**, **Is Active Member**

Output: Churn probability score + Yes/No prediction

---

## 🎁 Bonus: Salary Regression

This repo also includes an ANN regression model to predict customer **Estimated Salary**:

```bash
streamlit run streamlit_regression.py
```

Notebook: `salaryregression.ipynb`
Model: `regression_model.h5`

---

## 📬 Contact

**Satyansh Singh**
- 📧 bhikams468@gmail.com
- 💼 [LinkedIn](https://linkedin.com/in/satyansh-singh-567a22260)
- 🐙 [GitHub](https://github.com/krotov22)