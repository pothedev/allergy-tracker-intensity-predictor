import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from app.weather import get_weather_data  # Adjust path as needed if your structure differs

# === Load & Train Model ===
file_path = os.path.join(os.path.dirname(__file__), "pattern.xlsx")
data = pd.read_excel(file_path)

sample_size = min(300, len(data))
data_subset = data.sample(n=sample_size, random_state=42).reset_index(drop=True)

X = data_subset[["Temperature_Deviation_C", "Humidity_Percent", "Wind_Speed_kmh", "Cloud_Cover_Percent"]]
y = data_subset["Corrected_Intensity_Delta"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# === Helpers ===
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

# === Main Prediction Function ===
def generate_updated_intensity_forecast(start, end, lat, lon, timezone):
    weather_results = get_weather_data(lat, lon, forecast_days=16)
    weather_array = weather_results[1]

    test_df = pd.DataFrame(weather_array, columns=["Temperature_Deviation_C", "Humidity_Percent", "Wind_Speed_kmh", "Cloud_Cover_Percent"])
    predictions = rf_model.predict(test_df).round().astype(int).tolist()

    delta_map = generate_intensity_map(predictions)
    bloom_curve = generate_blooming_graph(start, end)
    bloom_map = generate_dates_dict(start, bloom_curve)

    updated_array, updated_dict = merge_intensities(delta_map, bloom_map)

    return {
        "updated_array": updated_array,
        "updated_dict": updated_dict
    }
