from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

@app.route('/api/predict_price', methods=['POST'])
def predict_price():
    num_rooms = request.json['num_rooms']
    square_meters = request.json['square_meters']
    num_bathrooms = request.json['num_bathrooms']

    # Use the input values to predict() the price of the flat
    price = 10000

    return jsonify({'price': price})

if __name__ == '__main__':
    app.run()
