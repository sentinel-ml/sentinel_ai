#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os

model_path = os.path.join(os.path.dirname(__file__), "model/sentinel_0.0.1.pkl")

with open('../model/sentinel_0.0.1.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI(title="Financial Fraud Detection API")

class TransactionData(BaseModel):
    distance_from_home: float
    distance_from_last_transaction: float
    ratio_to_median_purchase_price: float
    repeat_retailer: float
    used_chip: float
    used_pin_number: float
    online_order: float

    class Config:
        schema_extra = {
            "example": {
                "distance_from_home": 57.878,
                "distance_from_last_transaction": 0.311,
                "ratio_to_median_purchase_price": 1.946,
                "repeat_retailer": 1.0,
                "used_chip": 1.0,
                "used_pin_number": 0.0,
                "online_order": 0.0
            }
        }

@app.on_event("startup")
def load_model():
    global model
    model_path = os.path.join(os.path.dirname(__file__), "model/sentinel_0.0.1.pkl")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print('Model loaded successfully!')
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

@app.get('/')
def index():
    return {'message': 'Sentinel ML Model'}

@app.post('/predict')
def predict(transaction: TransactionData):
    try:
        input_data = pd.DataFrame([transaction.dict()])

        prediction = model.predict(input_data[0])
        probability = model.predict_proba(input_data)[0][1]

        return {
            "fraud_detected": bool(prediction),
            "fraud_prediction": float(probability),
            "transaction_details": transaction.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {e}")

@app.post('predict_batch')
def predict_batch(transactions: list[TransactionData]):
    try:
        input_data = pd.DataFrame([t.dict for t in transactions])

        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)[:, 1]

        results = []
        for i, transaction in enumerate(transactions):
            results.append({
                "fraud_detected": bool(predictions[i]),
                "fraud_probability": float(probabilities[i]),
                "transaction_details": transaction.dict()
            })
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=F"Batch prediction error: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": "model" in globals()}
