import pandas as pd
import requests
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import os

# === Load & train model ===
file_path = os.path.join(os.path.dirname(__file__), "pattern.xlsx")
data = pd.read_excel(file_path)
sample_size = min(300, len(data))
data_subset = data.sample(n=sample_size, random_state=42).reset_index(drop=True)
X = data_subset[["Temperature_Deviation_C", "Humidity_Percent", "Wind_Speed_kmh", "Cloud_Cover_Percent"]]
y = data_subset["Corrected_Intensity_Delta"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# weather utilities
def get_weather_data(lat, lon, days, timezone="Europe/Kiev"):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,rain,cloud_cover,wind_speed_10m",
        "timezone": timezone,
        "forecast_days": days
    }
    res = requests.get(url, params=params)
    res.raise_for_status()
    response = res.json()

    hourly_data = response['hourly']
    df = pd.DataFrame({
        'time': pd.to_datetime(hourly_data['time']),
        'temperature': hourly_data['temperature_2m'],
        'humidity': hourly_data['relative_humidity_2m'],
        'rain': hourly_data['rain'],
        'cloud_cover': hourly_data['cloud_cover'],
        'wind_speed': hourly_data['wind_speed_10m']
    })

    df['date'] = df['time'].dt.date
    daily_averages = df.groupby('date').mean()
    result = daily_averages[["temperature", "humidity", "wind_speed", "cloud_cover"]].to_numpy()
    return result

# bloom logic
def generate_blooming_graph(start, end):
    start_date = datetime.strptime(start, "%d/%m/%Y")
    end_date = datetime.strptime(end, "%d/%m/%Y")
    date_range = pd.date_range(start=start_date, end=end_date)
    midpoint = start_date + (end_date - start_date) / 2
    x_days = np.array([(date - start_date).days for date in date_range])
    peak = 5
    std_dev = (end_date - start_date).days / 6
    y = peak * np.exp(-((x_days - (midpoint - start_date).days) ** 2) / (2 * std_dev ** 2))
    return np.round(y).tolist()

def generate_dates_dict(start, intensities):
    start_date = datetime.strptime(start, "%d/%m/%Y")
    return {
        (start_date + timedelta(days=i)).strftime("%Y-%m-%d"): intensity
        for i, intensity in enumerate(intensities)
    }

def generate_intensity_map(deltas):
    today = datetime.today()
    return {
        (today + timedelta(days=i)).strftime("%Y-%m-%d"): delta
        for i, delta in enumerate(deltas)
    }

def merge_intensities(delta_map, bloom_map):
    updated = bloom_map.copy()
    for date, delta in delta_map.items():
        if date in updated:
            updated[date] = max(0, min(5, updated[date] + delta))
    return list(updated.values()), updated

# core function
def generate_updated_intensity_forecast(start, end, lat, lon, timezone):
    weather = get_weather_data(lat, lon, 16, timezone)
    test_df = pd.DataFrame(weather, columns=["Temperature_Deviation_C", "Humidity_Percent", "Wind_Speed_kmh", "Cloud_Cover_Percent"])
    predictions = rf_model.predict(test_df).round().astype(int).tolist()

    delta_map = generate_intensity_map(predictions)
    bloom_curve = generate_blooming_graph(start, end)
    bloom_map = generate_dates_dict(start, bloom_curve)

    updated_array, updated_dict = merge_intensities(delta_map, bloom_map)

    return {
        "updated_array": updated_array,
        "updated_dict": updated_dict
    }
