import pandas as pd
import requests
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, jsonify, request
from datetime import datetime, timedelta

app = Flask(__name__)

file_path = "pattern.xlsx"
data = pd.read_excel(file_path)
sample_size = min(300, len(data))
data_subset = data.sample(n=sample_size, random_state=42).reset_index(drop=True)
X = data_subset[["Temperature_Deviation_C", "Humidity_Percent", "Wind_Speed_kmh", "Cloud_Cover_Percent"]]
y = data_subset["Corrected_Intensity_Delta"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

def process_weather_data(response):
    hourly_data = response['hourly']
    data = pd.DataFrame({
        'time': pd.to_datetime(hourly_data['time']),
        'temperature': hourly_data['temperature_2m'],
        'humidity': hourly_data['relative_humidity_2m'],
        'rain': hourly_data['rain'],
        'cloud_cover': hourly_data['cloud_cover'],
        'wind_speed': hourly_data['wind_speed_10m']
    })
    data['date'] = data['time'].dt.date
    daily_averages = data.groupby('date').mean()
    return daily_averages[["temperature", "humidity", "wind_speed", "cloud_cover"]].to_numpy()

def generate_intensity_map(delta_intensities):
    today = datetime.today()
    return { (today + timedelta(days=i)).strftime("%Y-%m-%d"): delta for i, delta in enumerate(delta_intensities) }

def generate_blooming_graph(start, end):
    start_date = datetime.strptime(start, "%d/%m/%Y")
    end_date = datetime.strptime(end, "%d/%m/%Y")
    date_range = pd.date_range(start=start_date, end=end_date)
    midpoint = start_date + (end_date - start_date) / 2
    x_days = np.array([(date - start_date).days for date in date_range])
    y_values = 5 * np.exp(-((x_days - (midpoint - start_date).days) ** 2) / (2 * ((end_date - start_date).days / 6) ** 2))
    return np.round(y_values).tolist()

def generate_dates_dict(start, intensities):
    start_date = datetime.strptime(start, "%d/%m/%Y")
    return { (start_date + timedelta(days=i)).strftime("%Y-%m-%d"): intensity for i, intensity in enumerate(intensities) }

def merge_intensities(delta_intensity_map, blooming_intensity_map):
    updated_map = blooming_intensity_map.copy()
    for date, delta in delta_intensity_map.items():
        if date in updated_map:
            updated_map[date] = max(0, min(5, updated_map[date] + delta))
    return list(updated_map.values()), updated_map

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.json
        start, end = input_data['start_date'], input_data['end_date']
        params = {
            "latitude": input_data['latitude'],
            "longitude": input_data['longitude'],
            "hourly": "temperature_2m,relative_humidity_2m,rain,cloud_cover,wind_speed_10m",
            "timezone": input_data['timezone'],
            "forecast_days": 16
        }
        res = requests.get("https://api.open-meteo.com/v1/forecast", params=params)
        res.raise_for_status()
        weather_data = process_weather_data(res.json())
        test_data = pd.DataFrame(weather_data, columns=["Temperature_Deviation_C", "Humidity_Percent", "Wind_Speed_kmh", "Cloud_Cover_Percent"])
        delta_intensities = rf_model.predict(test_data).round().astype(int).tolist()
        delta_intensity_map = generate_intensity_map(delta_intensities)
        blooming_intensities = generate_blooming_graph(start, end)
        blooming_dates_dict = generate_dates_dict(start, blooming_intensities)
        updated_array, updated_dict = merge_intensities(delta_intensity_map, blooming_dates_dict)
        return jsonify({"updated_array": updated_array, "updated_dict": updated_dict})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
