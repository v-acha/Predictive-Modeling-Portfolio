# inference.py

import lightgbm as lgb
import pandas as pd
import numpy as np
import os
import json
import joblib

# Load model at startup
def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    return model

# Deserialize input data
def input_fn(request_body, request_content_type):
    if request_content_type == "text/csv":
        df = pd.read_csv(io.StringIO(request_body), header=None)
        return df
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

# Make prediction
def predict_fn(input_data, model):
    probabilities = model.predict(input_data)
    return probabilities

# Serialize output
def output_fn(prediction, response_content_type):
    if response_content_type == "application/json":
        result = {"probabilities": prediction.tolist()}
        return json.dumps(result)
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")