# ML-Powered Customer Churn Prediction Pipeline

An end-to-end machine learning project that predicts whether a customer is likely to churn based on customer demographics, account information, services, contract details, and billing information.

The project includes data preprocessing, feature engineering, model training and evaluation, automated retraining, a FastAPI REST API, MySQL prediction storage, and cloud deployment using Railway.

---

## 🚀 Live Demo

### 🌐 Frontend
https://churn-frontend-production.up.railway.app

### 🔗 FastAPI API
https://ml-churn-prediction-pipeline-production.up.railway.app

### 📖 API Documentation
https://ml-churn-prediction-pipeline-production.up.railway.app/docs

The Swagger UI allows you to test the `/predict` endpoint directly from the browser.

---

## 📌 Features

- Customer churn prediction using Machine Learning
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering and one-hot encoding
- Numerical feature scaling
- Logistic Regression classification
- Cross-validation
- Accuracy, Precision, and Recall evaluation
- Churn probability prediction
- FastAPI REST API
- MySQL database integration
- Automatic prediction storage
- Automated model retraining
- Model performance comparison before replacement
- Railway cloud deployment
- Web-based frontend for making predictions

---

## 🧠 Machine Learning Pipeline

The project follows an end-to-end ML workflow:

```text
Customer Dataset
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Feature Engineering
       ↓
Categorical Encoding
       ↓
Numerical Feature Scaling
       ↓
Train/Test Split
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Model Saving
       ↓
FastAPI Prediction API
       ↓
MySQL Prediction Storage
