# Inside train_balanced.py
import os
import sys
import subprocess

# Install necessary packages if not already installed
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("pytorch-tabnet")
install("s3fs")
install("joblib")
install("tqdm")

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from pytorch_tabnet.tab_model import TabNetClassifier
import s3fs
import joblib
from tqdm import tqdm

# Load data in chunks to save memory
def load_csv_in_chunks(path, chunksize=1_000_000):
    chunks = []
    for chunk in tqdm(pd.read_csv(path, chunksize=chunksize, storage_options={"anon": False}), desc=f"Loading {path}"):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)

# Prepare data for TabNet
def prepare_tabnet_data(df):
    # Convert float64 to float32 for efficiency
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")

    # Convert to NumPy arrays
    X = df.drop(columns=["fire_occurrence"]).values
    y = df["fire_occurrence"].values
    return X, y

# Custom weighted loss function
def make_weighted_loss(class_weights_tensor):
    def loss_fn(y_pred, y_true):
        return F.cross_entropy(y_pred, y_true, weight=class_weights_tensor)
    return loss_fn

# Training function with weighted loss
def train_tabnet_weights(X_train, y_train, X_val, y_val, epochs=10, batch_size=8192, lr=0.03, model_dir="/opt/ml/model"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # Compute class weights
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    loss_fn = make_weighted_loss(class_weights_tensor)

    # Initialize TabNet model
    tabnet = TabNetClassifier(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=lr),
        mask_type="entmax",
        device_name=device
    )

    # Fit the model with the weighted loss function
    tabnet.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["balanced_accuracy"],
        max_epochs=epochs,
        batch_size=batch_size,
        patience=3,
        loss_fn=loss_fn
    )

    # Remove loss function reference (makes model picklable)
    tabnet.loss_fn = None
    
    # Save the model to the model_dir (SageMaker convention)
    model_path = os.path.join(model_dir, "tabnet_model.pkl")
    joblib.dump(tabnet, model_path)
    print(f"Model saved to {model_path}")

def main(args):
    # Load data from S3
    train_df = load_csv_in_chunks(args.train_path)
    val_df = load_csv_in_chunks(args.val_path)

    # Prepare data for TabNet
    X_train, y_train = prepare_tabnet_data(train_df)
    X_val, y_val = prepare_tabnet_data(val_df)

    # Train the balanced model
    train_tabnet_weights(X_train, y_train, X_val, y_val, epochs=args.epochs, batch_size=args.batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Training data paths
    parser.add_argument("--train_path", type=str, default="s3://fireguarddata/data/preprocessed_data/train.csv")
    parser.add_argument("--val_path", type=str, default="s3://fireguarddata/data/preprocessed_data/val.csv")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16384)

    # Model output directory
    parser.add_argument("--model_dir", type=str, default="/opt/ml/model")

    args = parser.parse_args()
    main(args)
