import openmeteo_requests
import requests_cache
import pandas as pd
import requests
import numpy as np
from retry_requests import retry
from datetime import datetime, timedelta
from collections import defaultdict
from pprint import pprint


def get_weather_data(latitude=50.0, longitude=30.0, forecast_days=16, start="2020-01-01", end="2024-01-01"):

 # 1. Fetch Weather Forecast Data
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    forecast_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover",
        "timezone": "Europe/Kiev",
        "forecast_days": forecast_days
    }

    try:
        forecast_res = requests.get(forecast_url, params=forecast_params)
        forecast_res.raise_for_status()
        forecast_response = forecast_res.json()

        # Process forecast data
        hourly_data = forecast_response['hourly']
        data = pd.DataFrame({
            'time': pd.to_datetime(hourly_data['time']),
            'temperature': hourly_data['temperature_2m'],
            'humidity': hourly_data['relative_humidity_2m'],
            'wind_speed': hourly_data['wind_speed_10m'],
            'cloud_cover': hourly_data['cloud_cover'],
        })

        # Extract date and calculate daily averages
        data['date'] = data['time'].dt.strftime("%Y-%m-%d")  # Ensure string format
        daily_avg = data.groupby('date').mean().round()

        # Convert temperature data to dictionary
        daily_temperature_dict = daily_avg['temperature'].to_dict()

    except requests.exceptions.RequestException as e:
        print(f"HTTP Request failed (forecast): {e}")
        return [{}]  # Return an empty structure if API request fails

    # 2. Fetch Historical Weather Data
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    historical_url = "https://archive-api.open-meteo.com/v1/archive"
    historical_params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_mean",
        "timezone": "Europe/Kiev"
    }

    try:
        responses = openmeteo.weather_api(historical_url, params=historical_params)
        response = responses[0]
        daily = response.Daily()

        # Extract historical data
        daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()

        # Generate date range
        num_days = len(daily_temperature_2m_mean)
        start_date = datetime.strptime(start, "%Y-%m-%d")
        date_list = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(num_days)]

        # Create historical data dictionary
        weather_data = {date: round(temp) for date, temp in zip(date_list, daily_temperature_2m_mean)}

    except requests.exceptions.RequestException as e:
        print(f"HTTP Request failed (historical): {e}")
        return [{}]

    # 3. Calculate Historical Averages for the Current Year
    def calculate_average_temps(weather_data):
        temp_by_date = defaultdict(list)
        for date, temp in weather_data.items():
            month_day = date[5:]  # Extract 'MM-DD' part
            temp_by_date[month_day].append(temp)

        current_year = datetime.now().year
        return {
            f"{current_year}-{month_day}": round(sum(temps) / len(temps))
            for month_day, temps in temp_by_date.items()
        }

    averaged_weather_data = calculate_average_temps(weather_data)

    # 4. Compute Temperature Deviation (Forecast - Historical)
    def calculate_temp_deviation(forecast_data, historical_avg_data):
        return {
            date: round(forecast_data[date] - historical_avg_data[date])
            for date in forecast_data if date in historical_avg_data
        }

    temperature_deviation = calculate_temp_deviation(daily_temperature_dict, averaged_weather_data)

    # 5. Construct Forecast Weather Array with Delta Temperature
    forecast_with_delta = []
    
    # Convert daily_avg.index to a list of string dates for easy lookup
    available_dates = set(daily_avg.index)

    for date, delta_temp in temperature_deviation.items():
        if date in available_dates:
            row = [
                delta_temp,  # First column: Delta Temperature (forecast - historical)
                daily_avg.loc[date, "humidity"],
                daily_avg.loc[date, "wind_speed"],
                daily_avg.loc[date, "cloud_cover"]
            ]
            forecast_with_delta.append(row)

    forecast_weather_array = np.array(forecast_with_delta)  # Convert list to NumPy array

    # 6. Return All Processed Data (Dictionary + Forecast Weather Array)
    return [
        {
            "daily_temperature_dict": daily_temperature_dict,
            "averaged_weather_data": averaged_weather_data,
            "temperature_deviation": temperature_deviation
        },
        forecast_weather_array
    ]


# Example Usage:
if __name__ == "__main__":
    weather_results = get_weather_data()

    # First element: Processed weather dictionary
    weather_dict = weather_results[0]
    print("\nDaily Temperature Forecast:")
    pprint(weather_dict["daily_temperature_dict"])

    print("\nHistorical Average Temperatures:")
    pprint(weather_dict["averaged_weather_data"])

    print("\nTemperature Deviation (Forecast - Historical):")
    pprint(weather_dict["temperature_deviation"])

    # Second element: Forecast weather array
    weather_array = weather_results[1]
    print("\nForecast Weather Array (Delta Temp, Humidity, Wind Speed, Cloud Cover):")
    print(weather_array)


start = "15/02/2025"
end = "23/02/2025"
latitude = 50.0
longitude = 30.0
forecast_days = 16

data= get_weather_data(latitude, longitude, forecast_days)
print(data[1])