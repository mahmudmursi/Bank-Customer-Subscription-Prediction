# 🏦 Bank Customer Subscription Prediction using Neural Networks

## 📌 Project Overview

This project builds an AI-powered system to predict whether a bank customer will subscribe to a term deposit campaign.

The dataset comes from a Portuguese banking institution and contains customer demographic, financial, and marketing campaign information.

The goal is to help banks identify potential customers and optimize marketing campaigns.

---

## 🎯 Business Problem

Banks spend significant resources contacting customers during marketing campaigns.

However, many customers do not subscribe to term deposits.

This project aims to:

- Predict subscription likelihood
- Reduce marketing costs
- Improve campaign targeting
- Increase conversion rates

---

## 📂 Dataset Information

Dataset: **Bank Marketing Dataset**

Target Variable:

```text
y
```

- `yes` → Customer subscribed
- `no` → Customer did not subscribe

Dataset Shape:

```text
41,188 rows × 21 columns
```

---

## ⚙️ Project Pipeline

### 1. Exploratory Data Analysis (EDA)

Performed comprehensive analysis on:

#### Numerical Features
- Distribution analysis
- Outlier detection
- Correlation analysis
- Statistical summaries

#### Categorical Features
- Class distributions
- Subscription relationships
- Campaign effectiveness

---

### 2. Feature Engineering

Created custom features:

#### `previous_contact`

Indicates whether the customer was contacted before.

```python
df["previous_contact"] = np.where(
    df["pdays"] == 999,
    0,
    1
)
```

---

#### `pdays_clean`

Replaced invalid value (`999`) with NaN.

```python
df["pdays_clean"] = df["pdays"].replace(
    999,
    np.nan
)
```

---

### 3. Data Preprocessing

Performed:

- One-Hot Encoding
- Feature Scaling using StandardScaler
- Train-Test Split
- Class imbalance analysis

Final feature size:

```text
53 Features
```

---

## 🧠 Neural Network Architecture

### Advanced Neural Network

Architecture:

```text
Input Layer (53)

↓
Linear(53 → 64)
BatchNorm1d
ReLU
Dropout(0.3)

↓
Linear(64 → 32)
BatchNorm1d
ReLU
Dropout(0.3)

↓
Linear(32 → 16)
ReLU

↓
Linear(16 → 1)
```

Loss Function:

```text
BCEWithLogitsLoss
```

Optimizer:

```text
Adam Optimizer
Learning Rate = 0.001
```

---

## 📊 Model Evaluation

### Final Model Performance

| Metric | Score |
|--------|------|
| Accuracy | 86% |
| Precision | 43% |
| Recall | 62% |
| F1 Score | 51% |
| ROC-AUC | 0.80 |
| PR-AUC | 0.47 |

---

## 📈 Key Insights

### Important Features

Top influential features:

- `nr.employed`
- `month_may`
- `cons.conf.idx`
- `euribor3m`
- `previous_contact`
- `poutcome_success`

### Business Insights

- Economic indicators strongly influence subscription behavior.
- Previous successful campaigns increase future subscription probability.
- Campaign timing significantly affects outcomes.
- Cellular contact performs better than telephone campaigns.

---

## 🎯 Threshold Optimization

Since the dataset is imbalanced:

```text
88.7% No
11.3% Yes
```

Threshold tuning was applied.

Selected threshold:

```text
0.60
```

This improved minority-class detection.

---

## 🚀 Production Pipeline

The project includes:

✅ Model saving

✅ Scaler saving

✅ Feature persistence

✅ Real customer prediction

Example output:

```text
Subscription Probability: 69.37%
Customer likely to subscribe ✅
```

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- PyTorch

---

## 📁 Project Structure

```text
Bank-Marketing-Neural-Network/

│── data/
│   └── bank.csv

│── notebooks/
│   └── Bank-Marketing-Neural-Network.ipynb

│── models/
│   ├── bank_advanced_nn.pth
│   ├── bank_scaler.pkl
│   └── feature_columns.pkl

│── README.md
│── requirements.txt
```

---

## 🔮 Future Improvements

- Hyperparameter tuning
- Cross-validation
- Explainable AI (SHAP)
- Ensemble models
- API deployment using FastAPI
- Streamlit dashboard

---

## 👨‍💻 Author

**Mahmoud Morsy**

AI Engineer | Data Science | Machine Learning
Built with Neural Networks & PyTorch for bank subscription prediction.