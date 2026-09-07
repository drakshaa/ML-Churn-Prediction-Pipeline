import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score
)


# FIND PROJECT ROOT

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "models" / "final_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"


# LOAD DATASET

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# REMOVE CUSTOMER ID

if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)


# CONVERT TOTAL CHARGES TO NUMERIC

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df.dropna(inplace=True)


# CREATE TARGET VARIABLE

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# REMOVE TARGET FROM FEATURES

X = df.drop("Churn", axis=1)


# CATEGORICAL COLUMNS

categorical_columns = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]


# ONE-HOT ENCODE CATEGORICAL FEATURES

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)


# MAKE ALL FEATURES NUMERIC

X = X.astype(float)

print("Features:", X.shape)
print("Target:", y.shape)


# TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# SCALE NUMERICAL FEATURES

numerical_columns = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

scaler = StandardScaler()

X_train[numerical_columns] = scaler.fit_transform(
    X_train[numerical_columns]
)

X_test[numerical_columns] = scaler.transform(
    X_test[numerical_columns]
)


# TRAIN NEW MODEL

print("\nTraining new Logistic Regression model...")

new_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

new_model.fit(
    X_train,
    y_train
)


# EVALUATE NEW MODEL

y_pred = new_model.predict(X_test)

new_accuracy = accuracy_score(
    y_test,
    y_pred
)

new_precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

new_recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\nNew Model Evaluation")
print("-------------------------")
print(f"Accuracy : {new_accuracy:.4f}")
print(f"Precision: {new_precision:.4f}")
print(f"Recall   : {new_recall:.4f}")


# COMPARE WITH EXISTING MODEL

if MODEL_PATH.exists():

    print("\nLoading existing model...")

    old_model = joblib.load(MODEL_PATH)

    old_pred = old_model.predict(X_test)

    old_accuracy = accuracy_score(
        y_test,
        old_pred
    )

    print(f"Existing Model Accuracy: {old_accuracy:.4f}")

else:

    old_accuracy = 0

    print("\nNo existing model found.")


# DECIDE WHETHER TO REPLACE MODEL

if new_accuracy >= old_accuracy:

    print("\nNew model is equal or better.")
    print("Updating model...")

    joblib.dump(
        new_model,
        MODEL_PATH
    )

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    print("Model updated successfully!")

else:

    print("\nExisting model performs better.")
    print("Keeping existing model.")
    print("No model update performed.")


# FINAL MESSAGE

print("\nRetraining completed successfully!")