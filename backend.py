from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/predict_price', methods=['POST'])
def predict_price():
    housing_category = request.json["housing_category"]
    city_area = request.json["city_area"]
    housing_state = request.json["housing_state"]
    living_area = request.json["living_area"]
    land_area = request.json["land_area"]
    latitude = request.json["latitude"]
    longitude = request.json["longitude"]

    print(land_area)
    print(housing_category)
    print(latitude)

    # Use the input values to predict() the price of the flat
    price = 10000

    return jsonify({'price': price})

if __name__ == '__main__':
    app.run()
