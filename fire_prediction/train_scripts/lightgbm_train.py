# lightgbm_train.py

import pandas as pd
import boto3
import io
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sagemaker.model import Model
from sagemaker import image_uris, script_uris
import sagemaker

# ---------------------- Configuration ---------------------- #
S3_BUCKET = "testbucket1"
S3_PREFIX = "data_van_test/"
MODEL_DATA_URI = "s3://testbucket1/model_van_final_2/built-in-algo-lightgbm-classification-m-2025-03-31-23-24-42-542/output/model.tar.gz"
ENDPOINT_NAME = "my-lightgbm-endpoint-fireguard-5"
CONTENT_TYPE = "text/csv"
BATCH_SIZE = 1500
MODEL_ID = "lightgbm-classification-model"
MODEL_VERSION = "*"
INSTANCE_TYPE = "ml.m5.large"

# ---------------------- Load Test Data ---------------------- #
def load_test_data():
    s3_client = boto3.client("s3")
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX)
    csv_files = [file["Key"] for file in response.get("Contents", []) if file["Key"].endswith(".csv")]
    obj = s3_client.get_object(Bucket=S3_BUCKET, Key=csv_files[0])
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    return df

# ---------------------- Deploy Model ---------------------- #
def deploy_model():
    session = sagemaker.Session()
    role = sagemaker.get_execution_role()

    image_uri = image_uris.retrieve(
        region=session.boto_region_name,
        framework=None,
        image_scope="inference",
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        instance_type=INSTANCE_TYPE,
    )

    source_uri = script_uris.retrieve(
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        script_scope="inference"
    )

    model = Model(
        image_uri=image_uri,
        model_data=MODEL_DATA_URI,
        source_dir=source_uri,
        entry_point="inference.py",
        role=role,
        sagemaker_session=session,
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME
    )
    return predictor

# ---------------------- Query Endpoint ---------------------- #
def query_endpoint(encoded_data):
    client = boto3.client("runtime.sagemaker")
    response = client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType=CONTENT_TYPE,
        Body=encoded_data
    )
    return response

def parse_response(response):
    predictions = json.loads(response["Body"].read())
    return np.array(predictions["probabilities"])

# ---------------------- Run Inference ---------------------- #
def run_inference(df):
    ground_truth = df.iloc[:, 0]
    features = df.iloc[:, 1:]

    predictions = []
    for i in np.arange(0, len(df), step=BATCH_SIZE):
        batch = features.iloc[i:i + BATCH_SIZE]
        encoded = batch.to_csv(header=False, index=False).encode("utf-8")
        response = query_endpoint(encoded)
        batch_preds = parse_response(response)
        predictions.append(batch_preds)

    predict_prob = np.concatenate(predictions, axis=0)
    return ground_truth, predict_prob

# ---------------------- Main ---------------------- #
if __name__ == "__main__":
    df_test = load_test_data()
    deploy_model()
    ground_truth, predict_prob = run_inference(df_test)

    # Optional: convert probabilities to binary predictions
    predict_label = (predict_prob >= 0.5).astype(int)

    # Evaluate
    print("Accuracy:", accuracy_score(ground_truth, predict_label))
    print("F1 Score:", f1_score(ground_truth, predict_label))
    print("Confusion Matrix:\n", confusion_matrix(ground_truth, predict_label))