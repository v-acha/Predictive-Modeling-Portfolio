import pandas as pd
import numpy as np
import boto3
import s3fs
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import logging

# ========== SETUP ==========
bucket = "fireguarddata"
prefix = "data/csv_files/historic_merged_firms_weather_and_era_data/california_only_data/"
s3 = boto3.client("s3")
fs = s3fs.S3FileSystem(anon=False)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# ========== FILE LOADING ==========
def load_csv_in_chunks(path, chunksize=1_000_000):
    """Load CSV files from S3 in chunks."""
    chunks = []
    for chunk in tqdm(pd.read_csv(path, chunksize=chunksize, storage_options={"anon": False}), desc=f"Loading {path}"):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)

# ========== PREPROCESSING FUNCTION ==========
def preprocess_data(df):
    """Perform feature engineering and data cleanup for inference."""
    # Drop unnecessary columns
    drop_cols = ["number", "surface", "depthBelowLandLayer", "step", "valid_time"]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors="ignore")

    # Drop all unnamed columns
    df.drop(columns=[col for col in df.columns if col.startswith("Unnamed")], inplace=True, errors="ignore")

    # Convert float64 to float32
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Target encoding (if available)
    if "brightness" in df.columns and "daynight" in df.columns:
        df["fire_occurrence"] = np.where(
            df["daynight"] == "D", (df["brightness"] > 325).astype(int), (df["brightness"] > 320).astype(int)
        )
        df.drop(columns=["brightness"], inplace=True)

    # Date feature extraction
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df.drop(columns=["date"], inplace=True)

    # Wind speed and direction calculation
    if "u10" in df.columns and "v10" in df.columns:
        df["wind_speed"] = np.sqrt(df["u10"]**2 + df["v10"]**2)
        df["wind_direction"] = np.degrees(np.arctan2(df["v10"], df["u10"]))

    # Fuel features
    df["fuel_load"] = df.get("lai_hv", 0) + df.get("lai_lv", 0)
    df["fuel_moisture"] = df.get("swvl1", 0)
    df["fuel_availability"] = df["fuel_load"] * (1 - df["fuel_moisture"])

    # Day/Night encoding
    df["daynight"] = df["daynight"].map({"D": 1, "N": 0})

    return df

# ========== SCALING FUNCTION ==========
def scale_features(df):
    """Scale features for inference."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, columns=df.columns)

# ========== SAVE FUNCTION ==========
def save_to_s3(df, s3_path):
    """Save DataFrame to S3 as CSV."""
    with fs.open(s3_path, "w") as f:
        df.to_csv(f, index=False)
    print(f"Saved: {s3_path}")

# ========== MAIN ==========
response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith(".csv")]

processed_chunks = []
for file in files:
    s3_path = f"s3://{bucket}/{file}"
    print(f"Processing: {s3_path}")
    try:
        df = load_csv_in_chunks(s3_path)
        processed_df = preprocess_data(df)
        processed_chunks.append(processed_df)
    except Exception as e:
        print(f"Failed to process {file}: {e}")

# Combine all processed chunks
df_full = pd.concat(processed_chunks, ignore_index=True)
print(f"Combined data shape: {df_full.shape}")

# Scale the entire dataset for inference
df_full = scale_features(df_full)

# Save preprocessed data to S3 for inference
preprocessed_path = "s3://fireguarddata/data/preprocessed_data/inference.csv"
save_to_s3(df_full, preprocessed_path)
print("Data for inference saved successfully to S3.")
