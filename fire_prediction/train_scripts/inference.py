import os
import joblib
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

# Function to load the model (required by SageMaker)
def model_fn(model_dir):
    # Adjust model path as needed
    model_path = os.path.join(model_dir, "tabnet_model.pkl")
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    model.device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    return model

# Function to process the input data (required by SageMaker)
def input_fn(request_body, content_type="application/json"):
    if content_type == "application/json":
        import json
        data = json.loads(request_body)
        input_data = torch.tensor(data["inputs"], dtype=torch.float32)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

# Function to make predictions (required by SageMaker)
def predict_fn(input_data, model):
    with torch.no_grad():
        predictions = model.predict(input_data.numpy())
    return predictions.tolist()

# Function to format the output (required by SageMaker)
def output_fn(prediction, accept="application/json"):
    import json
    return json.dumps({"predictions": prediction})
