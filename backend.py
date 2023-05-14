from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# load nn model
model = tf.keras.models.load_model('my_model.h5')

@app.route('/api/predict_price', methods=['POST'])
def predict_price():
    housing_category = request.json["housing_category"]
    city_area = request.json["city_area"]
    housing_state = request.json["housing_state"]
    living_area = request.json["living_area"]

    print(housing_category)

    # Use the input values to predict() the price of the flat
    price = 10000

    return jsonify({'price': price})

if __name__ == '__main__':
    app.run()
