from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from api.database import save_prediction


# CREATE FASTAPI APP

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting whether a customer will churn",
    version="1.0"
)


# CORS SETTINGS

app.add_middleware(
    CORSMiddleware,
    # allow_origins=[
    #     "http://127.0.0.1:5500",
    #     "http://localhost:5500"
    #     "https://churn-frontend-production.up.railway.app"
    # ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# FIND PROJECT FOLDERS

# api/main.py -> project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Model path
MODEL_PATH = BASE_DIR / "models" / "final_model.pkl"

# Scaler path
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"


# LOAD MODEL

try:
    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully!")
    print("Model path:", MODEL_PATH)

except FileNotFoundError:
    print("ERROR: Model file not found!")
    print("Expected path:", MODEL_PATH)
    raise


# LOAD SCALER

try:
    scaler = joblib.load(SCALER_PATH)

    print("Scaler loaded successfully!")
    print("Scaler path:", SCALER_PATH)

except FileNotFoundError:
    print("ERROR: Scaler file not found!")
    print("Expected path:", SCALER_PATH)
    raise


# CHECK MODEL FEATURES

if hasattr(model, "n_features_in_"):
    print("\nNumber of features expected by model:")
    print(model.n_features_in_)

if hasattr(model, "feature_names_in_"):
    print("\nFeature names expected by model:")
    print(model.feature_names_in_)


# DEFINE INPUT DATA

class CustomerData(BaseModel):

    SeniorCitizen: int
    tenure: int
    MonthlyCharges: float
    TotalCharges: float

    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str


# HOME ROUTE

@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running!"
    }


# SHOW MODEL FEATURES

@app.get("/features")
def get_features():

    if hasattr(model, "feature_names_in_"):
        return {
            "number_of_features": len(model.feature_names_in_),
            "features": model.feature_names_in_.tolist()
        }

    return {
        "message": "Feature names are not available for this model"
    }


# PREDICTION ROUTE

@app.post("/predict")
def predict(customer: CustomerData):

    # STEP 1: CREATE RAW INPUT DATA

    input_data = pd.DataFrame([{

        # Numerical features
        "SeniorCitizen": customer.SeniorCitizen,
        "tenure": customer.tenure,
        "MonthlyCharges": customer.MonthlyCharges,
        "TotalCharges": customer.TotalCharges,

        # Gender
        "gender_Male":
            1 if customer.gender == "Male" else 0,

        # Partner
        "Partner_Yes":
            1 if customer.Partner == "Yes" else 0,

        # Dependents
        "Dependents_Yes":
            1 if customer.Dependents == "Yes" else 0,

        # Phone Service
        "PhoneService_Yes":
            1 if customer.PhoneService == "Yes" else 0,

        # Multiple Lines
        "MultipleLines_No phone service":
            1 if customer.MultipleLines == "No phone service" else 0,

        "MultipleLines_Yes":
            1 if customer.MultipleLines == "Yes" else 0,

        # Internet Service
        "InternetService_Fiber optic":
            1 if customer.InternetService == "Fiber optic" else 0,

        "InternetService_No":
            1 if customer.InternetService == "No" else 0,

        # Online Security
        "OnlineSecurity_No internet service":
            1 if customer.OnlineSecurity == "No internet service" else 0,

        "OnlineSecurity_Yes":
            1 if customer.OnlineSecurity == "Yes" else 0,

        # Online Backup
        "OnlineBackup_No internet service":
            1 if customer.OnlineBackup == "No internet service" else 0,

        "OnlineBackup_Yes":
            1 if customer.OnlineBackup == "Yes" else 0,

        # Device Protection
        "DeviceProtection_No internet service":
            1 if customer.DeviceProtection == "No internet service" else 0,

        "DeviceProtection_Yes":
            1 if customer.DeviceProtection == "Yes" else 0,

        # Tech Support
        "TechSupport_No internet service":
            1 if customer.TechSupport == "No internet service" else 0,

        "TechSupport_Yes":
            1 if customer.TechSupport == "Yes" else 0,

        # Streaming TV
        "StreamingTV_No internet service":
            1 if customer.StreamingTV == "No internet service" else 0,

        "StreamingTV_Yes":
            1 if customer.StreamingTV == "Yes" else 0,

        # Streaming Movies
        "StreamingMovies_No internet service":
            1 if customer.StreamingMovies == "No internet service" else 0,

        "StreamingMovies_Yes":
            1 if customer.StreamingMovies == "Yes" else 0,

        # Contract
        "Contract_One year":
            1 if customer.Contract == "One year" else 0,

        "Contract_Two year":
            1 if customer.Contract == "Two year" else 0,

        # Paperless Billing
        "PaperlessBilling_Yes":
            1 if customer.PaperlessBilling == "Yes" else 0,

        # Payment Method
        "PaymentMethod_Credit card (automatic)":
            1 if customer.PaymentMethod == "Credit card (automatic)" else 0,

        "PaymentMethod_Electronic check":
            1 if customer.PaymentMethod == "Electronic check" else 0,

        "PaymentMethod_Mailed check":
            1 if customer.PaymentMethod == "Mailed check" else 0

    }])


    # SCALE NUMERICAL FEATURES

    numerical_columns = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    input_data[numerical_columns] = scaler.transform(
        input_data[numerical_columns]
    )


    #  PUT COLUMNS IN EXACT
    # ORDER EXPECTED BY THE MODEL

    if hasattr(model, "feature_names_in_"):

        input_data = input_data[
            model.feature_names_in_
        ]


    #  MAKE PREDICTION

    prediction = model.predict(input_data)[0]


    # GET CHURN PROBABILITY

    probability = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_data)[0]

        if hasattr(model, "classes_"):

            classes = list(model.classes_)

            if 1 in classes:

                churn_index = classes.index(1)

                probability = float(
                    probabilities[churn_index]
                )

        else:
            probability = float(probabilities[1])


    #  CONVERT TO READABLE RESULT

    churn_prediction = (
        "Yes"
        if prediction == 1
        else "No"
    )


    #  SAVE PREDICTION TO MYSQL

    save_prediction(
        tenure=customer.tenure,
        monthly_charges=customer.MonthlyCharges,
        total_charges=customer.TotalCharges,
        contract=customer.Contract,
        internet_service=customer.InternetService,
        churn_prediction=int(prediction),
        churn_probability=float(probability)
    )


    # RETURN RESULT

    return {
        "prediction": int(prediction),
        "churn": churn_prediction,
        "churn_probability": probability
    }