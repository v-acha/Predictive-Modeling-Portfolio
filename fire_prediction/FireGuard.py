# -*- coding: utf-8 -*-
"""FireGuard - FireGuard.ipynb

Author: Krishna Rapaka, Richard Pang, KusamDeep Brar, Vanellsa Acha
Email: krishna.rapaka@berkeley.edu, ...
Description: Capstone project FireGuard ...
License:
"""

import os
import logging
import argparse
import geopandas as gpd
import pandas as pd
import requests
import cdsapi
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, box
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import datetime
from itertools import islice
import xarray as xr
import cdsapi
from EDA import *
import boto3


# TODO replace with your personal map key
MAP_KEY = ''


def setup_directories(base_folder):
    """Ensure necessary directories exist."""
    folders = [
        'wildfire_processed_data/csv_data',
        'wildfire_processed_data/plots',
        'wildfire_processed_data',
        'weather_data',
        'climate_data',
        'satellite_images',
        'logs'
    ]
    for folder in folders:
        os.makedirs(os.path.join(base_folder, folder), exist_ok=True)

def setup_logging(logs_folder):
    """Setup logging configuration."""
    log_file = os.path.join(logs_folder, 'wildfire_feature_collection.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_message(message, log_type="general"):
    logging.info(f"{log_type.upper()}: {message}")

def load_usa_map(ne_folder):
    """Load the USA map with state boundaries."""
    ne_shapefile = os.path.join(ne_folder, 'ne_50m_admin_1_states_provinces.shp')
    usa_map = gpd.read_file(ne_shapefile)
    return usa_map[usa_map['admin'] == 'United States of America']

def generate_valid_grid_points(usa_map, lat_min, lat_max, lon_min, lon_max, grid_size):
    """Generate valid grid points inside the USA boundary."""
    latitudes = np.arange(lat_min, lat_max + grid_size, grid_size)
    longitudes = np.arange(lon_min, lon_max + grid_size, grid_size)
    grid_points = [Point(lon, lat) for lat in latitudes for lon in longitudes]
    grid_gdf = gpd.GeoDataFrame(geometry=grid_points, crs="EPSG:4326")
    return grid_gdf[grid_gdf.geometry.within(usa_map.unary_union)].copy()

def plot_firms_data(firms_df, date_str, usa_map, valid_points, plots_folder):
    """Plot FIRMS wildfire data with valid grid points and save the plot."""
    log_message(f"Plotting FIRMS data for {date_str}", "firms")

    if firms_df is None or firms_df.empty:
        log_message("No FIRMS data available for plotting.", "firms")
        return

    fire_points = [Point(lon, lat) for lat, lon in zip(firms_df['latitude'], firms_df['longitude'])]
    fire_gdf = gpd.GeoDataFrame(geometry=fire_points, crs="EPSG:4326")
    fire_gdf = fire_gdf[fire_gdf.geometry.within(usa_map.unary_union)]

    # Create bounding boxes
    grid_size = 1.0  # Grid resolution
    rectangles = []
    for point in fire_gdf.geometry:
        lon, lat = point.x, point.y
        rect = box(lon - grid_size / 2, lat - grid_size / 2, lon + grid_size / 2, lat + grid_size / 2)
        rectangles.append(rect)

    rectangles_gdf = gpd.GeoDataFrame(geometry=rectangles, crs="EPSG:4326")

    # Plot results focusing only on the USA
    fig, ax = plt.subplots(figsize=(12, 8))
    usa_map.boundary.plot(ax=ax, linewidth=1, edgecolor="black")
    valid_points.plot(ax=ax, color="red", markersize=10, label="Valid Grid Points")
    rectangles_gdf.plot(ax=ax, edgecolor="blue", facecolor="none", linewidth=1)

    # Create legend handles manually
    fire_legend = mpatches.Patch(edgecolor="blue", facecolor="none", label="Fire Boundaries")
    grid_legend = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=10, label="Valid Grid Points")

    ax.set_xlim(-125, -67)
    ax.set_ylim(24, 49)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Wildfire Detection - {date_str}")
    plt.legend(handles=[grid_legend, fire_legend])  # Manually setting legend handles
    plt.grid(True, linestyle="--", linewidth=0.5)

    # Save plot to file
    plot_path = os.path.join(plots_folder, f'firms_plot_{date_str}.png')
    plt.savefig(plot_path)
    plt.close()
    log_message(f"Saved wildfire plot to {plot_path}", "firms")

def fetch_nasa_weather_data_batch(locations_batch, batch_size=10):
    """Fetch NASA weather data for a batch of locations/dates in smaller chunks."""
    try:
        params = {
            'parameters': 'T2M,T2MDEW,PS,WS10M,RH2M,PRECTOT,GWETTOP,GWETROOT,AIRMASS,CLOUD_AMT_DAY,TSOIL1,TSOIL2,FRSNO,RHOA',
            'community': 'SB',
            'format': 'JSON'
        }

        responses = {}
        for batch in chunked(locations_batch, batch_size):  # Reduce batch size to 10
            for lat, lon, date in batch:
                request_params = params.copy()
                request_params.update({
                    'longitude': lon,
                    'latitude': lat,
                    'start': date.replace('-', ''),
                    'end': date.replace('-', '')
                })

                response = requests.get('https://power.larc.nasa.gov/api/temporal/daily/point', params=request_params)
                response.raise_for_status()
                log_message(f"Fetching NASA weather data for {lat}, {lon}, {date}", "NASA Weather")

                response_data = response.json()
                if response_data:
                    responses[(lat, lon, date)] = response_data

        return responses if responses else {}  # Ensure dictionary is always returned
    except Exception as e:
        log_message(f"Error fetching NASA weather data: {e}", "error")
        return {}

def chunked(iterable, size):
    """Yield successive chunks of the iterable of a given size."""
    iterable = iter(iterable)
    while chunk := list(islice(iterable, size)):
        yield chunk

def merge_firms_with_weather(csv_folder, nasa_firm_fname_prefix):
    """Merge FIRMS wildfire data with NASA weather data using batch API requests."""
    log_message(f"Merging FIRMS data with NASA weather data", "merge")

    firms_csv_path = os.path.join(csv_folder, f'{nasa_firm_fname_prefix}.csv')
    if not os.path.exists(firms_csv_path):
        log_message(f"FIRMS CSV file not found: {firms_csv_path}", "error")
        return

    firms_df = pd.read_csv(firms_csv_path, usecols=['latitude', 'longitude', 'acq_date'])
    #locations = [(row['latitude'], row['longitude'], row['acq_date']) for _, row in firms_df.iterrows()][:10]
    locations = [(row['latitude'], row['longitude'], row['acq_date']) for _, row in firms_df.iterrows()]

    # Batch fetch weather data for every 10 locations
    weather_data_batch = fetch_nasa_weather_data_batch(locations, batch_size=10)
    if not weather_data_batch:
        log_message("No weather data retrieved. Check API or input parameters.", "error")
        return

    merged_data = []
    for lat, lon, date_str in locations:
        date_key = date_str.replace('-', '')
        weather_data = weather_data_batch.get((lat, lon, date_str), {})  # Default to empty dict to prevent NoneType error

        weather_values = weather_data.get('properties', {}).get('parameter', {})
        merged_data.append({
            'latitude': lat,
            'longitude': lon,
            'date': date_str,
            'weather_T2M': weather_values.get('T2M', {}).get(date_key, None),
            'weather_PS': weather_values.get('PS', {}).get(date_key, None),
            'weather_WS10M': weather_values.get('WS10M', {}).get(date_key, None),
            'weather_RH2M': weather_values.get('RH2M', {}).get(date_key, None),
            'weather_PRECTOT': weather_values.get('PRECTOTCORR', {}).get(date_key, None),
            'weather_GWETTOP': weather_values.get('GWETTOP', {}).get(date_key, None),
            'weather_GWETROOT': weather_values.get('GWETROOT', {}).get(date_key, None),
            'weather_AIRMASS': weather_values.get('AIRMASS', {}).get(date_key, None),
            'weather_CLOUD_AMT_DAY': weather_values.get('CLOUD_AMT_DAY', {}).get(date_key, None),
            'weather_TSOIL1': weather_values.get('TSOIL1', {}).get(date_key, None),
            'weather_TSOIL2': weather_values.get('TSOIL2', {}).get(date_key, None),
            'weather_FRSNO': weather_values.get('FRSNO', {}).get(date_key, None),
            'weather_RHOA': weather_values.get('RHOA', {}).get(date_key, None)
        })

    merged_df = pd.DataFrame(merged_data)
    output_csv = os.path.join(csv_folder, f'merged_firms_weather_{nasa_firm_fname_prefix}.csv')
    merged_df.to_csv(output_csv, index=False)
    log_message(f"Saved merged FIRMS and weather data to {output_csv}", "merge")
    return output_csv

def plot_and_save_grid(valid_points, usa_map, output_path, grid_size):
    """Plot and save the USA map with grid points."""
    rectangles = [
        box(point.x - grid_size / 2, point.y - grid_size / 2, point.x + grid_size / 2, point.y + grid_size / 2)
        for point in valid_points.geometry
    ]
    rectangles_gdf = gpd.GeoDataFrame(geometry=rectangles, crs="EPSG:4326")

    fig, ax = plt.subplots(figsize=(12, 8))
    usa_map.plot(ax=ax, color="lightgray", edgecolor="black")
    valid_points.plot(ax=ax, color="red", markersize=10)
    rectangles_gdf.plot(ax=ax, edgecolor="blue", facecolor="none", linewidth=1)
    ax.set_xlim(-125, -67)
    ax.set_ylim(24, 49)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Filtered USA Wildfire Prediction Grid (Only Inside USA)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.savefig(output_path)
    plt.close()
    log_message(f"Saved grid plot to {output_path}", "general")

def get_country_info():
    countries_url = 'https://firms.modaps.eosdis.nasa.gov/api/countries'
    df_countries = pd.read_csv(countries_url, sep=';')
    USA = df_countries[df_countries['name'] == 'United States']
    USA_id = USA['id'].values[0]
    USA_abreviation= USA['abreviation'].values[0]
    return USA_id, USA_abreviation

# Checks to see if the selected map key is valid
def validate_map_key():
    url = 'https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY=' + MAP_KEY
    try:
        df = pd.read_json(url,  typ='series')
        log_message(df)
        if df['transaction_limit'] > 0:
            return True
        else:
            log_message('Transaction limit has been exhuasted')
            return False
    except:
        # possible error, wrong MAP_KEY value, check for extra quotes, missing letters
        log_message("There is an issue with the query. \nTry in your browser: %s" % url)
        return False

def plot_firms_data_availability(firms_df, output_path):
    """Plot a bar chart showing the availability of FIRMS data by date."""
    try:
        firms_df['acq_date'] = pd.to_datetime(firms_df['acq_date'])
        date_counts = firms_df['acq_date'].value_counts().sort_index()

        plt.figure(figsize=(10, 5))
        plt.bar(date_counts.index.astype(str), date_counts.values, width=0.7)
        plt.xlabel("Date")
        plt.ylabel("Number of FIRMS Records")
        plt.title("FIRMS Data Availability by Date")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(output_path)
    except Exception as e:
        log_message(f"Error plotting FIRMS data availability: {e}", "error")

# TODO: Replace the root_path to match the root directory of your system
# REPLACE ROOT_PATH AND KEY
def register_cdsapi_config_file():
    url = 'url: https://cds.climate.copernicus.eu/api'
    key = 'key: '
    root_path = ''

    with open(root_path, 'w') as f:
        f.write('\n'.join([url, key]))

    with open(root_path) as f:
        print(f.read())

def get_dates_from_now_to_x_days(date, num_days):
    dates = [date + datetime.timedelta(days=i) for i in range(num_days+1)]
    years = []
    months = []
    days = []
    for date in dates:
        year, month, day = date.year, date.month, date.day
        years.append(year) if year not in years else None
        months.append(month) if month not in months else None
        days.append(day) if day not in days else None

    return years, months, days

# Ensure .cfgrib file is in root folder, otherwise you have to override the
# env variable. It's needed for the cdsapi
def process_era5_data(date_str, root_path, num_days=0):
    #register_cdsapi_config_file()
    date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    
    year, month, day = get_dates_from_now_to_x_days(date, num_days)
    print('year, month, and days included in the era5 data fetch')
    print(year)
    print(month)
    print(day)
    # era dataset uses a grid box of 1 degree of longitude in the east-west direction and 1 degree of latitude in the north-south direction
    input_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    
    # Get the current time
    current_time = datetime.datetime.now()
    
    # Calculate the difference between the current time and midnight of the given day
    time_difference = current_time - input_date
    hours_passed = 24 if 24 < int(time_difference.total_seconds() / 3600) else int(time_difference.total_seconds() / 3600)

    # Create a list of hours that have passed
    hours = [f"{hour:02}:00" for hour in range(hours_passed)]

    print(hours)
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "10m_u_component_of_wind", # horizontal speed of air moving towards the east, at a height of ten metres above the surface of the Earth (m/s)
            "10m_v_component_of_wind", # horizontal speed of air moving towards the north, at a height of ten metres above the surface of the Earth (m/s)
            "surface_pressure", #  pressure (force per unit area) of the atmosphere on the surface of land, sea and in-land water
            "total_precipitation", # accumulated liquid and frozen water, comprising rain and snow, that falls to the Earth's surface (m)
            "specific_humidity", # mass of water vapour per kilogram of moist air
            "leaf_area_index_high_vegetation", # describes land surface vegetation in high vegetation. 'High vegetation' consists of evergreen trees, deciduous trees, mixed forest/woodland, and interrupted forest.
            "leaf_area_index_low_vegetation", #  land surface vegetation in low vegetation area. 'Low vegetation' consists of crops and mixed farming, irrigated crops, short grass, tall grass, tundra, semidesert, bogs and marshes, evergreen shrubs, deciduous shrubs, and water and land mixtures.
            "type_of_low_vegetation", # 1 = Crops, Mixed farming, 2 = Grass, 7 = Tall grass, 9 = Tundra, 10 = Irrigated crops, 11 = Semidesert, 13 = Bogs and marshes, 16 = Evergreen shrubs, 17 = Deciduous shrubs, 20 = Water and land mixtures
            "type_of_high_vegetation",
            "geopotential",
            "instantaneous_moisture_flux",
            "skin_temperature", # temperature of the surface of the Earth (K)
            "lake_cover", # lake cover:  proportion of a grid box covered by inland water bodies,
            "volumetric_soil_water_layer_1", # volume of water in soil layer 1 (0 - 7cm)
            "2m_dewpoint_temperature",
            "2m_temperature",
            "minimum_2m_temperature_since_previous_post_processing",
            "mean_potential_evaporation_rate",
            "total_cloud_cover",
            "evaporation",
            "potential_evaporation",
            "surface_runoff",
            "convective_precipitation",
            "total_column_rain_water",
            "snow_density",
            "snow_depth",
            "snow_evaporation",
            "snowfall",
            "temperature_of_snow_layer",
            "soil_temperature_level_1",
            "soil_type"
        ],
        "year": year,
        "month": month,
        "day": day,
        "time": hours,
        "data_format": "grib",
        "download_format": "unarchived",
        "area": [42, -124, 32, -113] # limited to California -124.41060660766607,32.5342307609976,-114.13445790587905,42.00965914828148
    }
    try:
        target = root_path + '/era5_' + str(date_str) + '_to_'+ str(num_days) + '.grib'
        client = cdsapi.Client()
        client.retrieve(dataset, request, target)

        ds = xr.open_dataset(target, engine='cfgrib')

        combined_df = ds.to_dataframe()
        combined_df = combined_df.reset_index()

        combined_df.rename(columns={'time': 'date'}, inplace=True)

        # combined_df.to_csv(file_path_for_dataset, index=False, chunksize=10000, compression='gzip') -- we already have the grib file, we don't need to save as csv again
        return combined_df
    except Exception as e:
        log_message('Errored fetching era5 data with:' + str(e))
        return pd.DataFrame()

def filter_era5_data(era_df):
    if era_df.empty:
        return pd.DataFrame()
    era_df['date'] = era_df['date'].apply(lambda x: x.date())
    # round the long and lat of the firms and era5 dataset to neart quarter degree
    era_df['latitude'] = era_df['latitude'].apply(lambda x: round(x / 0.25) * 0.25)
    era_df['longitude'] = era_df['longitude'].apply(lambda x: round(x / 0.25) * 0.25)
    return era_df

def merge_firms_with_era5(firm_df, date_str, root_path, num_days=0):
    if firm_df is None:
        log_message('FIRMS data does not exist, unable to join with the era5 data', 'merge_era5_and_firms')
        return
    
    # get the era5 data for the same days
    era_df = process_era5_data(date_str, root_path, num_days)
    log_message('Successfully pulled ERA5 data, now attempting merge', 'merge_era5_and_firms')
    if era_df is None:
        log_message('ERA5 data is empty, unable to continue the join, abandoning', 'merge_era5_and_firms')
        return

    # convert the time to date only
    era_df['date'] = era_df['date'].apply(lambda x: x.date())
    
    
    # rename firm_df column acq_date to date so it's easier to join the two
    firm_df.rename(columns={'acq_date':'date'}, inplace=True)

    # round the long and lat of the firms and era5 dataset to neart quarter degree
    era_df['latitude'] = era_df['latitude'].apply(lambda x: round(x / 0.25) * 0.25)
    era_df['longitude'] = era_df['longitude'].apply(lambda x: round(x / 0.25) * 0.25)
    firm_df['latitude'] = firm_df['latitude'].apply(lambda x: round(x / 0.25) * 0.25)
    firm_df['longitude'] = firm_df['longitude'].apply(lambda x: round(x / 0.25) * 0.25)

    # convert date column to be of type datetime for the firms dataset
    firm_df['date'] = firm_df['date'].apply(lambda x: x.date())

    # attempt join
    merged_df = pd.merge(
        era_df,
        firm_df[['date', 'latitude', 'longitude', 'brightness', 'confidence', 'daynight']],
        on=['date', 'latitude', 'longitude'],
        how='outer')
    
    # replace NaN values with 0 / n based on column in the entire DataFrame
    merged_df['daynight'] = merged_df['daynight'].fillna('D')
    merged_df['confidence'] = merged_df['confidence'].fillna('n')
    merged_df['brightness'] = merged_df['brightness'].fillna('0')
    
    log_message(f'Final size of the merged dataframe is {merged_df.shape}, beginning writting to file.', 'merge_era5_and_firms')
    log_message(f'Final size of the FIRMS dataframe is {firm_df.shape}, beginning writting to file.', 'merge_era5_and_firms')
    log_message(f'Final size of the merged dataframe is {era_df.shape}, beginning writting to file.', 'merge_era5_and_firms')

    file_path_for_dataset = root_path + '/wildfire_processed_data/csv_data/merged_era5_firms_dataset_'+ date_str + '_to_'+ str(num_days) + '.csv'
    merged_df.to_csv(file_path_for_dataset, index=False, chunksize=10000)
    return file_path_for_dataset

# Using the fims API requires the use of api keys. These can be generated using
# https://firms.modaps.eosdis.nasa.gov/api/map_key . There is a call limit of 5000 and
# the transaction can run atmost for 10 minutes. For now, we can hard code this value,
# but for the future we may way want to consider storing this as a system variable if we
# want to automate this script and publish it so it's publicly accessible.
def process_firms_data(nasa_firms_folder, csv_folder, fname_prefix, date_str, num_days=1):
    """Process FIRMS data and save as CSV."""
    try:
        date_list = [(datetime.datetime.strptime(date_str, "%Y-%m-%d") + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(num_days)]

        if validate_map_key():
            _, usa_abr = get_country_info()
            # date needs to be of the form year-month-day!
            modis_url = f'https://firms.modaps.eosdis.nasa.gov/api/country/csv/{MAP_KEY}/MODIS_NRT/{usa_abr}/{str(num_days)}/{date_str}'
            viirs_url = f'https://firms.modaps.eosdis.nasa.gov/api/country/csv/{MAP_KEY}/VIIRS_NOAA20_NRT/{usa_abr}/{(num_days)}/{date_str}'
            modis_df = pd.read_csv(modis_url)
            viirs_df = pd.read_csv(viirs_url)
        else:
            modis_df = pd.read_csv(os.path.join(nasa_firms_folder, 'modis_2023_United_States.csv'))
            viirs_df = pd.read_csv(os.path.join(nasa_firms_folder, 'viirs-snpp_2023_United_States.csv'))

        firms_df = pd.concat([modis_df, viirs_df], ignore_index=True)
        firms_df = firms_df[firms_df['acq_date'].isin(date_list)]
        lat_min = 32
        lat_max = 42
        lon_min = -124
        lon_max = -113

        # reduce to lat and long that exists in California
        filtered_firms = firms_df[
            (firms_df['latitude'] >= lat_min) & (firms_df['latitude'] <= lat_max) &
            (firms_df['longitude'] >= lon_min) & (firms_df['longitude'] <= lon_max)
        ]

        plot_firms_data_availability(filtered_firms,os.path.join(csv_folder, f'{fname_prefix}.png'))

        output_csv = os.path.join(csv_folder, f'{fname_prefix}.csv')
        filtered_firms.to_csv(output_csv, index=False)
        log_message(f"Saved filtered FIRMS data to {output_csv}", "firms")

        return filtered_firms
    except Exception as e:
        log_message(f"Error processing FIRMS data: {e}", "error")
        return None
    
def generate_dates(start_date, end_date, step_days):
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    
    date_list = []
    
    current_date = start
    while current_date <= end:
        date_list.append(str(current_date.strftime("%Y-%m-%d")))  # Append the date to the list
        current_date += datetime.timedelta(days=step_days)  # Increment the date by 10 days
    
    return date_list

def augment_firms(firms_df):
    #replace this part with quantization
    #resolution 1°x1° - 69 miles x 69 miles
    firms_df['acq_date']=pd.to_datetime(firms_df['acq_date'])
    firms_df['latitude']=round(firms_df['latitude'],0)
    firms_df['longitude']=round(firms_df['longitude'],0)
    firms_df=firms_df.groupby(['latitude','longitude','acq_date']).agg(brightness=('brightness','mean'),confidence=('confidence','mean')).reset_index()


    #creating the complement of firms (USA no fire)
    max_date=firms_df['acq_date'].max()
    min_date=firms_df['acq_date'].min()
    days_difference = (max_date - min_date).days + 1
    #Lat-long coorditates for cities in United States are in range: Latitude from 19.50139 to 64.85694 and longitude from -161.75583 to -68.01197.
    date_list = [max_date - datetime.timedelta(days=x) for x in range(days_difference)]
    date_df=pd.DataFrame({'acq_date':date_list})
    lat_max=65
    lat_min=19
    lon_max=-68
    lon_min=-162


    #replace this with quantization - use numpy
    #cross join to get all possible lat and lon pairings
    lat_list=[x for x in range(lat_min,lat_max)]
    lon_list=[x for x in range(lon_min,lon_max)]
    lat_df=pd.DataFrame({'latitude':lat_list})
    lon_df=pd.DataFrame({'longitude':lon_list})
    pairings_df=pd.merge(lat_df,lon_df,how='cross')


    #cross join - all pairings with all dates
    all_df=pd.merge(pairings_df,date_df,how='cross')
    all_df['brightness']=0
    all_df['confidence']=0

    #merge with firms
    firms_all_df=pd.merge(all_df,firms_df,how='left', on=['acq_date','latitude','longitude'])
    firms_all_df=firms_all_df.rename(columns={'brightness_y':'brightness','confidence_y':'confidence'})
    #the brightness from the right side firms df will have null values - fill them
    firms_all_df=firms_all_df.fillna(0)
    firms_all_df=firms_all_df[['acq_date','latitude','longitude','brightness','confidence']]

    return firms_all_df

# make sure to have you aws credentials configured!
def upload_files_to_s3(firms_era_file_name, firms_weather_file_name):
    bucket_name = 'fireguarddata'
    subpath = 'data/csv_files/historic_merged_firms_weather_and_era_data/'
    era_file_name = subpath + firms_era_file_name.split('/')[-1]
    weather_file_name = subpath + firms_weather_file_name.split('/')[-1]

    s3 = boto3.client('s3')

    # Upload a file to the S3 bucket
    try:
        s3.upload_file(firms_era_file_name, bucket_name, era_file_name)
        print(f"File {firms_era_file_name} uploaded to {bucket_name}/{era_file_name}")

        s3.upload_file(firms_weather_file_name, bucket_name, weather_file_name)
        print(f"File {firms_weather_file_name} uploaded to {bucket_name}/{weather_file_name}")
    except Exception as e:
        print(f"Error occurred: {e}")
    return


def main():
    parser = argparse.ArgumentParser(description='Wildfire Feature Collection and Processing')
    parser.add_argument('--base_folder', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'work_dir'), help='Base directory for data storage')
    parser.add_argument('--ne_folder', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '50m_cultural'), help='Natural Earth shapefile folder')
    parser.add_argument('--nasa_firms_folder', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nasa_firms'), help='NASA FIRMS data folder')
    parser.add_argument('--date', type=str, required=False, help='Date to process data for (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=False, help='Last date to collect data up to (YYYY-MM-DD)')
    parser.add_argument('--grid_size', type=float, default=1.0, help='Grid resolution size (degrees)')
    parser.add_argument('--enable_grid_plotting', action='store_true', help='Enable plotting of grid points')
    parser.add_argument('--num_days', type=int, default=1, help='The number of days from the date provided that you would like data collected for')
    parser.add_argument('--eda', action='store_true', help='Perform EDA')
    parser.add_argument('--eda_folder', type=str, help='Folder to perform EDA on')
    parser.add_argument('--upload_to_s3',  action='store_true', help="Push files to S3")
    parser.add_argument('--daily_recordings', action='store_true', help="Get daily environmental data")
    args = parser.parse_args()

    # If EDA mode is enabled, perform EDA and exit
    if args.eda:
        if args.eda_folder is None:
            print("Error: --eda_folder must be specified when using --eda")
            return
        perform_eda(args.eda_folder, args.ne_folder)
        return  # Skip the rest of the processing

    # Regular processing workflow
    setup_directories(args.base_folder)
    setup_logging(os.path.join(args.base_folder, 'logs'))
    usa_map = load_usa_map(args.ne_folder)
    date = args.date
    if args.date == 'today':
        date = datetime.datetime.today().strftime('%Y-%m-%d')

    print("Selected date: " + date)
    plots_folder = os.path.join(args.base_folder, 'wildfire_processed_data/plots')
    csv_folder = os.path.join(args.base_folder, 'wildfire_processed_data/csv_data')
    nasa_firms_process_fname_prefix = f'firms_{args.date}_to_{args.num_days}_days'

    if args.daily_recordings:
        era_data = process_era5_data(date, csv_folder, args.num_days)
        processed_data = filter_era5_data(era_data)
        file_name = csv_folder + '/' + date + '-daily-era5-recording.csv'
        processed_data.to_csv(file_name)
        print('completed successfully')
        return

    valid_points = generate_valid_grid_points(usa_map, 24, 49, -125, -67, args.grid_size)

    if args.enable_grid_plotting:
        plot_and_save_grid(valid_points, usa_map, os.path.join(args.base_folder, 'wildfire_processed_data/plots/grid_plot.png'), args.grid_size)

    num_days = args.num_days  # max 10 as the FIRMS API doesn't allow for a larger date range
    end_date = args.end_date if args.end_date is not None else date
    days = generate_dates(date, end_date, num_days)  # dates from 2024 Jan to Dec

    for day in days:
        print('start date :' + day)

        firms_df = process_firms_data(args.nasa_firms_folder, csv_folder, nasa_firms_process_fname_prefix, day, num_days)

        if firms_df is not None:
            plot_firms_data(firms_df, day, usa_map, valid_points, plots_folder)
            log_message(f"Processed {len(firms_df)} FIRMS entries for {args.date}", "firms")
            # fetch and merge with era5 and weather data
            firms_era_file_name = merge_firms_with_era5(firms_df, day, args.base_folder, num_days)
            firms_weather_file_name = merge_firms_with_weather(csv_folder, nasa_firms_process_fname_prefix)

            if args.upload_to_s3:
                upload_files_to_s3(firms_era_file_name, firms_weather_file_name)

    log_message("Processing complete", "general")

if __name__ == "__main__":
    main()
