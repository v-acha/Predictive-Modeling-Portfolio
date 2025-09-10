import os
import pandas as pd
import glob
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import geopandas as gpd
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.ensemble import IsolationForest

OUTPUT_DIR = "work_dir/wildfire_processed_data/EDA"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(folder_path):
    file_pattern = os.path.join(folder_path, "merged_era5_firms_dataset_*")
    file_list = glob.glob(file_pattern)
    print(file_pattern)
    print(file_list)
    if not file_list:
        print("No files found. Check the folder path.")
        return None
    
    df_list = [pd.read_csv(file, delimiter=',') for file in file_list]
    df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(file_list)} files. Total rows: {df.shape[0]}, Columns: {df.shape[1]}")
    return df

def save_plot(fig, filename):
    fig.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')

def data_overview(df):
    print("Data Overview:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nDuplicate Rows:", df.duplicated().sum())
    
    plt.figure(figsize=(10, 5))
    msno.matrix(df.sample(1000))
    save_plot(plt.gcf(), "missing_values_matrix.png")

def statistical_summary(df):
    print("Statistical Summary:")
    print(df.describe())
    
    df.hist(figsize=(12, 10), bins=30)
    plt.tight_layout()
    save_plot(plt.gcf(), "histograms.png")

def temporal_analysis(df):
    if 'valid_time' in df.columns and pd.to_datetime(df['valid_time'], errors='coerce').notna().all():
        df['valid_time'] = pd.to_datetime(df['valid_time'])
        df.set_index('valid_time', inplace=True)

        # Convert relevant columns to numeric
        numeric_columns = ['skt', 'brightness','u10','v10','sp','lai_hv','lai_lv','tvl','swvl1']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows where all numeric values are NaN
        df = df.dropna(subset=numeric_columns)

        if df.empty:
            print("No valid numerical data available for temporal analysis. Skipping.")
            return
        
        for col in numeric_columns:
            plt.figure(figsize=(12, 6))
            df[col].resample('D').mean().plot()
            plt.title(f"Daily Trend of {col.capitalize()}")
            save_plot(plt.gcf(), f"temporal_trend_{col}.png")

        # Perform autocorrelation based on the date
        plt.figure(figsize=(10, 6))
        plot_acf(df['skt'].resample('D').mean().dropna(), lags=30)
        plt.title("Autocorrelation of Skin Temperature Based on Date")
        save_plot(plt.gcf(), "autocorrelation_skt.png")
    else:
        print("Invalid or missing 'valid_time' column. Skipping temporal analysis.")

def spatial_analysis(df, ne_folder, sample_fraction=1):
    """
    Optimized function to plot fire hotspots on a US map using pre-downloaded shapefile.
    - Only plots non-zero brightness values.
    - Uses vectorized operations to speed up processing.
    - Allows optional subsampling to handle large datasets efficiently.
    
    Args:
        df (DataFrame): Wildfire data containing latitude, longitude, and brightness.
        ne_folder (str): Path to the Natural Earth 50m shapefile directory.
        sample_fraction (float): Fraction of data to sample (default: 10%).
    """
    # Ensure the shapefile path exists
    ne_admin_path = os.path.join(ne_folder, "ne_50m_admin_0_countries.shp")
    if not os.path.exists(ne_admin_path):
        print(f"Error: Could not find {ne_admin_path}. Check the path and file existence.")
        return

    # **Filter required columns & Drop NaNs at once**
    df = df[['latitude', 'longitude', 'brightness']].dropna()

    if df.empty:
        print("No valid data for spatial analysis. Skipping.")
        return

    # **Convert brightness to numeric & remove zeros**
    df['brightness'] = pd.to_numeric(df['brightness'], errors='coerce')
    df = df[df['brightness'] > 0]  # Only plot non-zero brightness values

    if df.empty:
        print("No nonzero brightness values. Skipping spatial analysis.")
        return

    # **Subsample dataset for speed improvement (default: 10%)**
    if sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=42)

    print(f"Processing {len(df):,} rows after filtering nonzero brightness and sampling.")

    # **Fast brightness range computation**
    vmin, vmax = df['brightness'].agg(['min', lambda x: x.quantile(0.99)])
    
    # **Load US map from the provided Natural Earth folder path**
    us_map = gpd.read_file(ne_admin_path)
    us_map = us_map[us_map.NAME == "United States of America"]

    # **Vectorized conversion to GeoDataFrame**
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")

    # **Plot map & hotspots**
    fig, ax = plt.subplots(figsize=(10, 6))
    us_map.plot(ax=ax, color='lightgray')
    scatter = df.plot(ax=ax, column='brightness', cmap='hot', alpha=0.5, vmin=vmin, vmax=vmax, legend=True)
    
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Fire Hotspots on US Map (Non-Zero Brightness Only)")

    # **Highlight the max brightness location**
    max_idx = df['brightness'].idxmax()
    max_lat, max_lon = df.loc[max_idx, ['latitude', 'longitude']]
    ax.scatter(max_lon, max_lat, color='blue', edgecolors='black', s=100, label='Max Brightness')
    
    plt.legend()
    save_plot(plt.gcf(), "fire_hotspots_nonzero_brightness.png")

def detect_anomalies(df):
    if {'brightness', 'confidence', 'skt', 'valid_time'}.issubset(df.columns):
        df[['brightness', 'confidence', 'skt']] = df[['brightness', 'confidence', 'skt']].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=['brightness', 'confidence', 'skt', 'valid_time'])
        df['valid_time'] = pd.to_datetime(df['valid_time'])
        
        # Using Isolation Forest for anomaly detection
        model = IsolationForest(contamination=0.05, random_state=42)
        df['anomaly_score'] = model.fit_predict(df[['brightness', 'confidence', 'skt']])
        
        # Visualizing anomalies over time
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x=df['valid_time'], y=df['brightness'], hue=df['anomaly_score'], palette={1: 'blue', -1: 'red'}, alpha=0.6)
        plt.title("Anomaly Detection in Brightness Over Time")
        plt.xlabel("Date")
        plt.ylabel("Brightness")
        plt.xticks(rotation=45)
        save_plot(plt.gcf(), "brightness_anomalies_by_date.png")
        plt.show()
    else:
        print("Missing required columns for anomaly detection. Skipping.")

def perform_eda(data_folder, ne_folder):
    """Perform Exploratory Data Analysis on the specified folder."""
    print(f"Performing EDA on folder: {data_folder}")
    df = load_data(data_folder)
    if df is not None:
        data_overview(df)
        statistical_summary(df)
        temporal_analysis(df)
        spatial_analysis(df, ne_folder)
        detect_anomalies(df)