from flask import Flask, jsonify, request
from app.intensity import (
    generate_updated_intensity_forecast,
    get_weather_data
)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.json
        start = input_data['start_date']
        end = input_data['end_date']
        latitude = input_data['latitude']
        longitude = input_data['longitude']
        timezone = input_data['timezone']

        result = generate_updated_intensity_forecast(start, end, latitude, longitude, timezone)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "awake"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
