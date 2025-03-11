from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os

with open('/model/sentinel_0.0.1.pkl', 'rb') as f:
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

@app.get('/', methods=['GET'])
def index():
    return {'message': 'Sentinel ML Model'}

@app.post('/predict', methods=['POST'])
def predict(data):
    
