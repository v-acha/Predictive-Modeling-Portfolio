
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
import torch.nn.functional as F
import s3fs
import joblib
import io
from tqdm import tqdm

# === Data Loading and Preparation === #
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
# === Focal Loss Function === #
def focal_loss_fn(gamma=2.0, alpha=None):
    def loss_fn(y_pred, y_true):
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.float()
        else:
            y_pred = torch.tensor(y_pred, dtype=torch.float32)

        y_true = y_true.long()
        logp = F.log_softmax(y_pred, dim=1)
        p = torch.exp(logp)

        # One-hot encoding of targets
        y_true_onehot = F.one_hot(y_true, num_classes=y_pred.shape[1]).float()

        if alpha is not None:
            alpha_tensor = torch.tensor(alpha).to(y_pred.device)
            logp = logp * alpha_tensor

        loss = - (1 - p) ** gamma * logp
        loss = (loss * y_true_onehot).sum(dim=1)
        return loss.mean()
    
    return loss_fn

# === Training Function with Focal Loss === #
def train_tabnet_focal(X_train, y_train, X_val, y_val, epochs=5, batch_size=8192, lr=0.02, model_dir="/opt/ml/model"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # Compute class weights for focal loss
    classes = np.unique(y_train)
    class_weights = torch.tensor(
        np.bincount(y_train) / len(y_train), dtype=torch.float32
    )
    alpha = (1.0 - class_weights).tolist()  # Inverse frequency as alpha

    # Focal loss with gamma=2.0
    loss_fn = focal_loss_fn(gamma=2.0, alpha=alpha)

    # Initialize TabNet model
    tabnet = TabNetClassifier(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=lr),
        mask_type="entmax",
        device_name=device
    )

    # Fit
    tabnet.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["balanced_accuracy"],
        max_epochs=epochs,
        batch_size=batch_size,
        patience=2,
        loss_fn=loss_fn
    )

    # Remove loss function to make model picklable
    tabnet.loss_fn = None

    # Save to /opt/ml/model (SageMaker convention)
    model_path = os.path.join(model_dir, "tabnet_focal_model.pkl")
    joblib.dump(tabnet, model_path)
    print(f"Model saved to {model_path}")

# === Main Script === #
def main(args):
    train_df = load_csv_in_chunks(args.train_path)
    val_df = load_csv_in_chunks(args.val_path)

    X_train, y_train = prepare_tabnet_data(train_df)
    X_val, y_val = prepare_tabnet_data(val_df)

    train_tabnet_focal(X_train, y_train, X_val, y_val, epochs=args.epochs, batch_size=args.batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="s3://fireguarddata/data/preprocessed_data/train.csv")
    parser.add_argument("--val_path", type=str, default="s3://fireguarddata/data/preprocessed_data/val.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--model_dir", type=str, default="/opt/ml/model")
    args = parser.parse_args()
    main(args)
