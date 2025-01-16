import pandas as pd
import requests
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pprint import pprint

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
    return daily_averages[['temperature', 'humidity', 'wind_speed', 'cloud_cover']].to_numpy()

url = "https://api.open-meteo.com/v1/forecast"
params = {
    #latitude and longitude are examples of passed values
    "latitude": 50.4504,
    "longitude": 30.5245,
    "hourly": "temperature_2m,relative_humidity_2m,rain,cloud_cover,wind_speed_10m",
    "timezone": "Africa/Cairo",
    "forecast_days": 16
}

try:
    res = requests.get(url, params=params)
    res.raise_for_status()
    response = res.json()
    weather_data = process_weather_data(response)
except requests.exceptions.RequestException as e:
    print(f"HTTP Request failed: {e}")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

file_path = "pattern.xlsx"
data = pd.read_excel(file_path)
sample_size = min(300, len(data))
data_subset = data.sample(n=sample_size, random_state=42).reset_index(drop=True)
X = data_subset[["Temperature_Delta", "Humidity_Percent", "Wind_Speed_kmh", "Cloud_Cover_Percent"]]
y = data_subset["Intensity_Delta"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test).round().astype(int)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = 1 - mse / np.var(y_test)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")
print(f"Accuracy: {accuracy:.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, label="Predicted vs Actual")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label="Ideal Fit")
plt.xlabel("Actual Intensity Delta")
plt.ylabel("Predicted Intensity Delta")
plt.title("Actual vs Predicted Intensity Delta")
plt.legend()
plt.show()

test_data = pd.DataFrame(weather_data, columns=["Temperature_Delta", "Humidity_Percent", "Wind_Speed_kmh", "Cloud_Cover_Percent"])
test_predictions = rf_model.predict(test_data).round().astype(int)
test_data["Predicted Intensity Delta"] = test_predictions
delta_intensities = test_data["Predicted Intensity Delta"].tolist()

def generate_intensity_map(delta_intensities):
    today = datetime.today()
    return { (today + timedelta(days=i)).strftime("%Y-%m-%d"): delta for i, delta in enumerate(delta_intensities) }

def generate_blooming_graph(start, end):
    start_date = datetime.strptime(start, "%d/%m/%Y")  # example of passed value
    end_date = datetime.strptime(end, "%d/%m/%Y")  # example of passed value
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

#start and end are examples of passed values 
start = "24/07/2025"
end = "22/10/2025"
blooming_intensities = generate_blooming_graph(start, end)
blooming_dates_dict = generate_dates_dict(start, blooming_intensities) 
delta_intensity_map = generate_intensity_map(delta_intensities)
updated_array, updated_dict = merge_intensities(delta_intensity_map, blooming_dates_dict)

print("\nUpdated Intensities Dictionary:")
pprint(updated_dict)
print("\nUpdated Intensities Array:")
print(updated_array)
