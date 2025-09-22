# scripts/inference.py
import joblib
import os
import numpy as np

def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))

def input_fn(request_body, content_type):
    if content_type == "text/csv":
        return np.array([float(x) for x in request_body.split(",")]).reshape(1, -1)
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, accept):
    return str(prediction[0])
