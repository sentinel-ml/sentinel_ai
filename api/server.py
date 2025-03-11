from fastapi import FastApi, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os

with open('/model/sentinel_0.0.1.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastApi(title="Financial Fraud Detection API")

class TransactionData(BaseModel):
    distance_from_home: float
    distance_from_last_transaction: float
    ratio_to_median_purchase_price: float
    repeat_retailer: float
    used_chip: float
    used_pin_number: float
    online_order: float

@app.get('/', methods=['GET'])
def index():
    return {'message': 'Sentinel ML Model'}

@app.post('/predict', methods=['POST'])
def predict(data):
    
