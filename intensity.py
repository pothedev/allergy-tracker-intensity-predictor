import pandas as pd
import requests
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pprint import pprint
from weather import get_weather_data

start = "01/02/2025"
end = "25/02/2025"
latitude = 50.0
longitude = 30.0
forecast_days = 16

data = get_weather_data(latitude, longitude, forecast_days)
weather_data = data[1]
print(data[1])

# Load and process training data
file_path = "pattern.xlsx"
data = pd.read_excel(file_path)
print(f"Dataset contains {len(data)} rows and {data.shape[1]} columns.")

sample_size = min(300, len(data))
data_subset = data.sample(n=sample_size, random_state=42).reset_index(drop=True)
X = data_subset[["Temperature_Deviation_C", "Humidity_Percent", "Wind_Speed_kmh", "Cloud_Cover_Percent"]]
y = data_subset["Corrected_Intensity_Delta"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test).round().astype(int)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = 1 - mse / np.var(y_test)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")
print(f"Accuracy: {accuracy:.2f}")

# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred, alpha=0.7, label="Predicted vs Actual")
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label="Ideal Fit")
# plt.xlabel("Actual Intensity Delta")
# plt.ylabel("Predicted Intensity Delta")
# plt.title("Actual vs Predicted Intensity Delta")
# plt.legend()
# plt.show()

#print(delta_weather)

# Predict for the 16-day forecast
test_data = pd.DataFrame(weather_data, columns=["Temperature_Deviation_C", "Humidity_Percent", "Wind_Speed_kmh", "Cloud_Cover_Percent"])
test_predictions = rf_model.predict(test_data).round().astype(int)
test_data["Predicted Intensity Delta"] = test_predictions

# Display predictions for 16 days
print("Predicted Intensity Delta for the 16-day forecast:")
print(test_data)

# Extract delta intensities as an array
delta_intensities = test_data["Predicted Intensity Delta"].tolist()


# Function to generate a map of dates to delta intensities
def generate_intensity_map(delta_intensities):
    today = datetime.today()  # Get today's date
    intensity_map = {}

    # Generate the mapping
    for i, delta in enumerate(delta_intensities):
        date = today + timedelta(days=i)  # Calculate the date for each delta
        intensity_map[date.strftime("%Y-%m-%d")] = delta  # Add to map with formatted date as key

    return intensity_map


# Generate the delta intensity map
delta_intensity_map = generate_intensity_map(delta_intensities)


# Function to generate blooming graph intensities
def generate_blooming_graph(start, end):
    # Parse the start and end dates
    start_date = datetime.strptime(start, "%d/%m/%Y")
    end_date = datetime.strptime(end, "%d/%m/%Y")

    # Generate a range of dates from start to end
    date_range = pd.date_range(start=start_date, end=end_date)

    # Calculate the midpoint of the season (peak)
    midpoint = start_date + (end_date - start_date) / 2

    # Create x-values as days from the start date
    x_days = np.array([(date - start_date).days for date in date_range])

    # Create y-values using a normal distribution
    peak_intensity = 5
    std_dev = (end_date - start_date).days / 6
    y_values = peak_intensity * np.exp(-((x_days - (midpoint - start_date).days) ** 2) / (2 * std_dev ** 2))

    # Return the rounded intensities for each day
    intensity_values = np.round(y_values).tolist()
    return intensity_values


# Function to generate dates dictionary from blooming graph
def generate_dates_dict(start, intensities):
    # Parse the start date
    start_date = datetime.strptime(start, "%d/%m/%Y")

    # Generate the dictionary
    dates_dict = {}
    for i, intensity in enumerate(intensities):
        date = start_date + timedelta(days=i)  # Calculate the date for each intensity
        dates_dict[date.strftime("%Y-%m-%d")] = intensity  # Add to the dictionary with formatted date as key

    return dates_dict


def merge_intensities(delta_intensity_map, blooming_intensity_map):
    # Initialize the updated dictionary
    updated_intensity_map = blooming_intensity_map.copy()

    # Update the blooming intensities with the delta intensities
    for date, delta_intensity in delta_intensity_map.items():
        if date in updated_intensity_map:
            updated_value = updated_intensity_map[date] + delta_intensity
            # Clamp the value between 0 and 5
            updated_intensity_map[date] = max(0, min(5, updated_value))

    # Convert the dictionary values to an array
    updated_intensities_array = list(updated_intensity_map.values())

    return updated_intensities_array, updated_intensity_map


# Example usage for blooming graph

blooming_intensities = generate_blooming_graph(start, end)
blooming_dates_dict = generate_dates_dict(start, blooming_intensities)

# Print the results
print("\nDate-to-Delta Intensity Map:")
pprint(delta_intensity_map)

print("\nDates to Blooming Intensities Dictionary:")
pprint(blooming_dates_dict)

updated_array, updated_dict = merge_intensities(delta_intensity_map, blooming_dates_dict)

print("\nUpdated Intensities Dictionary:")
pprint(updated_dict)

# Print the results
print("Updated Intensities Array:")
print(updated_array)