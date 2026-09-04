from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path


# Create FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting whether a customer will churn",
    version="1.0"
)



# FIND PROJECT FOLDERS


# api/main.py -> project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Path to saved model
MODEL_PATH = BASE_DIR / "models" / "final_model.pkl"



# LOAD MODEL


try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
    print("Model path:", MODEL_PATH)

except FileNotFoundError:
    print("ERROR: Model file not found!")
    print("Expected path:", MODEL_PATH)
    raise



# DEFINE INPUT DATA


class CustomerData(BaseModel):
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float



# HOME ROUTE


@app.get("/")
def home():
    return {
        "message": "Churn Prediction API is running!"
    }



# PREDICTION ROUTE


@app.post("/predict")
def predict(customer: CustomerData):

    # Convert input into the format expected by the model
    input_data = [[
        customer.SeniorCitizen,
        customer.Partner,
        customer.Dependents,
        customer.tenure,
        customer.PhoneService,
        customer.MultipleLines,
        customer.InternetService,
        customer.OnlineSecurity,
        customer.OnlineBackup,
        customer.DeviceProtection,
        customer.TechSupport,
        customer.StreamingTV,
        customer.StreamingMovies,
        customer.Contract,
        customer.PaperlessBilling,
        customer.PaymentMethod,
        customer.MonthlyCharges,
        customer.TotalCharges
    ]]

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get probability if supported
    probability = None

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_data)[0][1])

    # Convert prediction to readable result
    churn_prediction = "Yes" if prediction == 1 else "No"

    return {
        "prediction": int(prediction),
        "churn": churn_prediction,
        "churn_probability": probability
    }