# Inside train_tabnet.py
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
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import s3fs
import joblib
import io
from tqdm import tqdm

def load_csv_in_chunks(path, chunksize=1_000_000):
    chunks = []
    for chunk in tqdm(pd.read_csv(path, chunksize=chunksize, storage_options={"anon": False}), desc=f"Loading {path}"):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)

def prepare_tabnet_data(df):
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")
    X = df.drop(columns=["fire_occurrence"]).values
    y = df["fire_occurrence"].values
    return X, y

def train_tabnet(X_train, y_train, X_val, y_val, epochs=15, batch_size=4096, lr=0.03, model_dir="/opt/ml/model"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # Initialize TabNet model
    tabnet = TabNetClassifier(
        optimizer_fn=torch.optim.AdamW,  # Use AdamW instead of Adam
        optimizer_params=dict(lr=0.02, weight_decay=0.0001),  # Add weight decay
        scheduler_fn=torch.optim.lr_scheduler.StepLR,  # Learning rate scheduler
        scheduler_params={"step_size": 5, "gamma": 0.7},  # Reduce LR every 5 epochs
        mask_type="entmax",
        device_name=device
)

    # Fit the model
    tabnet.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["balanced_accuracy"],
        max_epochs=epochs,
        batch_size=batch_size,
        patience=1,

    )

    # Save the model to the model_dir (SageMaker convention)
    model_path = os.path.join(model_dir,# Inside train.py
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
#install("pandas")
#install("numpy")


import argparse
import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import s3fs
import joblib
import os
import io
from tqdm import tqdm

def load_csv_in_chunks(path, chunksize=1_000_000):
    chunks = []
    for chunk in tqdm(pd.read_csv(path, chunksize=chunksize, storage_options={"anon": False}), desc=f"Loading {path}"):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)

def prepare_tabnet_data(df):
    # Convert float64 to float32 for efficiency
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")

    # Convert to NumPy arrays
    X = df.drop(columns=["fire_occurrence"]).values
    y = df["fire_occurrence"].values
    return X, y

def train_tabnet(X_train, y_train, X_val, y_val, epochs=5, batch_size=8192, lr=0.03, model_dir="/opt/ml/model"):
    # Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # Initialize TabNet model
    tabnet = TabNetClassifier(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=lr),
        mask_type="entmax",
        device_name=device
    )

    # Fit the model
    tabnet.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["balanced_accuracy"],
        max_epochs=epochs,
        batch_size=batch_size,
        patience=3
    )

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

    # Train the model
    train_tabnet(X_train, y_train, X_val, y_val, epochs=args.epochs, batch_size=args.batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Training data paths
    parser.add_argument("--train_path", type=str, default="s3://fireguarddata/data/preprocessed_data/train.csv")
    parser.add_argument("--val_path", type=str, default="s3://fireguarddata/data/preprocessed_data/val.csv")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8192)

    # Model output directory
    parser.add_argument("--model_dir", type=str, default="/opt/ml/model")

    args = parser.parse_args()
    main(args)
 "tabnet_model_new.pkl")
    joblib.dump(tabnet, model_path)
    print(f"Model saved to {model_path}")

def main(args):
    train_df = load_csv_in_chunks(args.train_path)
    val_df = load_csv_in_chunks(args.val_path)

    X_train, y_train = prepare_tabnet_data(train_df)
    X_val, y_val = prepare_tabnet_data(val_df)

    train_tabnet(X_train, y_train, X_val, y_val, epochs=args.epochs, batch_size=args.batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default="s3://fireguarddata/data/preprocessed_data/train.csv")
    parser.add_argument("--val_path", type=str, default="s3://fireguarddata/data/preprocessed_data/val.csv")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--model_dir", type=str, default="/opt/ml/model")

    args = parser.parse_args()
    main(args)
