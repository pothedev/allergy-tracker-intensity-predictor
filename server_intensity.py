import pandas as pd
import requests
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, jsonify, request
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from weather import get_weather_data



# Initialize Flask app
app = Flask(__name__)

# Load pre-trained Random Forest model and training data
file_path = "pattern.xlsx"
data = pd.read_excel(file_path)
sample_size = min(300, len(data))
data_subset = data.sample(n=sample_size, random_state=42).reset_index(drop=True)
X = data_subset[["Temperature_Deviation_C", "Humidity_Percent", "Wind_Speed_kmh", "Cloud_Cover_Percent"]]
y = data_subset["Corrected_Intensity_Delta"]

# Train the Random Forest model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Weather forecast processing
url = "https://api.open-meteo.com/v1/forecast"

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
    result = daily_averages[["temperature", "humidity", "wind_speed", "cloud_cover"]].to_numpy()
    return result

def generate_intensity_map(delta_intensities):
    today = datetime.today()
    intensity_map = {}
    for i, delta in enumerate(delta_intensities):
        date = today + timedelta(days=i)
        intensity_map[date.strftime("%Y-%m-%d")] = delta
    return intensity_map

def generate_blooming_graph(start, end):
    start_date = datetime.strptime(start, "%d/%m/%Y")
    end_date = datetime.strptime(end, "%d/%m/%Y")

    date_range = pd.date_range(start=start_date, end=end_date)
    midpoint = start_date + (end_date - start_date) / 2

    x_days = np.array([(date - start_date).days for date in date_range])

    peak_intensity = 5
    std_dev = (end_date - start_date).days / 6
    y_values = peak_intensity * np.exp(-((x_days - (midpoint - start_date).days) ** 2) / (2 * std_dev ** 2))

    intensity_values = np.round(y_values).tolist()
    return intensity_values

def generate_dates_dict(start, intensities):
    start_date = datetime.strptime(start, "%d/%m/%Y")

    dates_dict = {}
    for i, intensity in enumerate(intensities):
        date = start_date + timedelta(days=i)
        dates_dict[date.strftime("%Y-%m-%d")] = intensity

    return dates_dict

def merge_intensities(delta_intensity_map, blooming_intensity_map):
    updated_intensity_map = blooming_intensity_map.copy()

    for date, delta_intensity in delta_intensity_map.items():
        if date in updated_intensity_map:
            updated_value = updated_intensity_map[date] + delta_intensity
            updated_intensity_map[date] = max(0, min(5, updated_value))

    updated_intensities_array = list(updated_intensity_map.values())
    return updated_intensities_array, updated_intensity_map

# Flask route for predictions
@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.json
        start = input_data['start_date']
        end = input_data['end_date']
        latitude = input_data['latitude']
        longitude = input_data['longitude']
        

        data = get_weather_data(latitude, longitude, 16)
        weather_data = data[1]

        params = {
            "latitude": input_data['latitude'],
            "longitude": input_data['longitude'],
            "hourly": "temperature_2m,relative_humidity_2m,rain,cloud_cover,wind_speed_10m",
            "timezone": input_data['timezone'],
            "forecast_days": 16
        }

        res = requests.get(url, params=params)
        res.raise_for_status()
        response = res.json()
        weather_data = process_weather_data(response)

        test_data = pd.DataFrame(weather_data, columns=["Temperature_Deviation_C", "Humidity_Percent", "Wind_Speed_kmh", "Cloud_Cover_Percent"])
        test_predictions = rf_model.predict(test_data).round().astype(int)
        delta_intensities = test_predictions.tolist()

        delta_intensity_map = generate_intensity_map(delta_intensities)
        blooming_intensities = generate_blooming_graph(start, end)
        blooming_dates_dict = generate_dates_dict(start, blooming_intensities)

        updated_array, updated_dict = merge_intensities(delta_intensity_map, blooming_dates_dict)

        return jsonify({
            "updated_array": updated_array,
            "updated_dict": updated_dict
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "awake"}), 200


# Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
