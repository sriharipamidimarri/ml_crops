from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib  # For compressed models
import pickle  # For encoders (no compression needed)
import os

app = FastAPI()

# Paths for loading models and encoders
MODEL_DIR = "models"

# Input data model
class PricePredictionInput(BaseModel):
    State: str
    District: str
    Market: str
    Commodity: str
    Arrival_Date: str

# Helper function to load models and encoders
def load_models():
    global rf_min, rf_max, rf_modal, scaler, encoders
    try:
        # Load compressed models using joblib from the models folder
        rf_min = joblib.load(os.path.join(MODEL_DIR, "rf_min_compressed.pkl"))
        rf_max = joblib.load(os.path.join(MODEL_DIR, "rf_max_compressed.pkl"))
        rf_modal = joblib.load(os.path.join(MODEL_DIR, "rf_modal_compressed.pkl"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_compressed.pkl"))

        # Load encoders using pickle (no compression)
        encoders = {}
        for col in ["State", "District", "Market", "Commodity"]:
            encoders[col] = pickle.load(open(os.path.join(MODEL_DIR, f"{col}_encoder.pkl"), "rb"))
        
        print("Models and encoders loaded successfully")
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load models")

# Load models at startup
load_models()

# API Endpoints
@app.get("/")
def home():
    return {"message": "Welcome to the Agricultural Price Prediction API!"}

@app.post("/predict")
def predict(input_data: PricePredictionInput):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.dict()])

        # Add date features
        input_df["Arrival_Date"] = pd.to_datetime(input_df["Arrival_Date"])
        input_df["Year"] = input_df["Arrival_Date"].dt.year
        input_df["Month"] = input_df["Arrival_Date"].dt.month
        input_df["Day"] = input_df["Arrival_Date"].dt.day

        # Encode categorical features
        for col in ["State", "District", "Market", "Commodity"]:
            input_df[col] = encoders[col].transform(input_df[col])

        # Prepare features
        X_input = input_df[["State", "District", "Market", "Commodity", "Year", "Month", "Day"]]
        X_input_scaled = scaler.transform(X_input)

        # Predict prices
        min_price = rf_min.predict(X_input_scaled)[0]
        max_price = rf_max.predict(X_input_scaled)[0]
        modal_price = rf_modal.predict(X_input_scaled)[0]

        return {
            "Min Price": float(min_price),
            "Max Price": float(max_price),
            "Modal Price": float(modal_price),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analysis")
def analysis(input_data: PricePredictionInput):
    """
    Perform analysis using the input fields and provide future price predictions.
    """
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.dict()])

        # Convert 'Arrival_Date' to datetime
        input_df["Arrival_Date"] = pd.to_datetime(input_df["Arrival_Date"])

        # Extract Year, Month, Day from 'Arrival_Date'
        input_df["Year"] = input_df["Arrival_Date"].dt.year
        input_df["Month"] = input_df["Arrival_Date"].dt.month
        input_df["Day"] = input_df["Arrival_Date"].dt.day

        # Encode categorical features
        for col in ["State", "District", "Market", "Commodity"]:
            input_df[col] = encoders[col].transform(input_df[col])

        # Current prediction
        X_input = input_df[["State", "District", "Market", "Commodity", "Year", "Month", "Day"]]
        X_input_scaled = scaler.transform(X_input)
        
        current_min = float(rf_min.predict(X_input_scaled)[0])
        current_max = float(rf_max.predict(X_input_scaled)[0])
        current_modal = float(rf_modal.predict(X_input_scaled)[0])

        # Future Predictions (next 5 days)
        future_dates = [input_df["Arrival_Date"].iloc[0] + timedelta(days=i) for i in range(1, 6)]
        future_data = pd.DataFrame({
            "State": [input_df["State"].iloc[0]] * len(future_dates),
            "District": [input_df["District"].iloc[0]] * len(future_dates),
            "Market": [input_df["Market"].iloc[0]] * len(future_dates),
            "Commodity": [input_df["Commodity"].iloc[0]] * len(future_dates),
            "Year": [date.year for date in future_dates],
            "Month": [date.month for date in future_dates],
            "Day": [date.day for date in future_dates]
        })

        # Scale features for prediction
        X_future_scaled = scaler.transform(future_data)

        # Predict prices
        min_price_pred = rf_min.predict(X_future_scaled)
        max_price_pred = rf_max.predict(X_future_scaled)
        modal_price_pred = rf_modal.predict(X_future_scaled)

        # Adding noise to predictions
        min_price_pred_with_noise = min_price_pred + np.random.uniform(-5, 5, size=min_price_pred.shape[0])
        max_price_pred_with_noise = max_price_pred + np.random.uniform(-5, 5, size=max_price_pred.shape[0])
        modal_price_pred_with_noise = modal_price_pred + np.random.uniform(-5, 5, size=modal_price_pred.shape[0])

        # Prepare future predictions with noise
        future_predictions = []
        for i, date in enumerate(future_dates):
            future_predictions.append({
                "Arrival_Date": date.strftime("%d-%m-%Y"),
                "Min Price": float(min_price_pred_with_noise[i]),
                "Max Price": float(max_price_pred_with_noise[i]),
                "Modal Price": float(modal_price_pred_with_noise[i])
            })

        return {
            "current_prediction": {
                "Arrival_Date": input_df["Arrival_Date"].iloc[0].strftime("%d-%m-%Y"),
                "Min Price": current_min,
                "Max Price": current_max,
                "Modal Price": current_modal
            },
            "future_predictions": future_predictions
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
if __name__ == '__main__':
    app.run(debug=True)