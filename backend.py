from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from DataLoader import Loader

app = Flask(__name__)
CORS(app)

# load nn model
model = tf.keras.models.load_model('MLmodel/model/model_nn.model')
# use Loader class to get scalers
loader = Loader("data/data_kosice.csv")
loader.clean_data()
loader.prepare_data()

@app.route('/api/predict_price', methods=['POST'])
def predict_price():
    # get values from frontend
    housing_category = request.json["housing_category"]
    city_area = request.json["city_area"]
    housing_state = request.json["housing_state"]
    living_area = request.json["living_area"]

    # make dummmies for categorial data
    categorial_data = pd.DataFrame({"housing_category": [housing_category], "city_area": [city_area], "housing_state": [housing_state]})
    user_input_encoded = pd.get_dummies(categorial_data[["housing_category", "city_area", "housing_state"]])
    user_input_encoded = user_input_encoded.reindex(columns=loader.encoded.columns, fill_value=0)
    # write answer from user
    user_input_encoded[housing_category] = 1.0
    user_input_encoded[city_area] = 1.0
    user_input_encoded[housing_state] = 1.0


    # use numeric attribute scaler model from Loader class and standardize numeric data
    numeric_data = pd.DataFrame({"living_area": [living_area]})
    standardized = loader.scaler_numeric_data.transform(numeric_data)
    numeric_standardized = pd.DataFrame(standardized, columns=["living_area"])

    X_predict = pd.DataFrame(user_input_encoded, dtype=np.uint8)
    X_predict["living_area"] = numeric_standardized["living_area"]


    price_output = np.array(model.predict(X_predict)[0][0])
    # destandardize and convert back to float
    price = loader.scaler_price.inverse_transform(price_output.reshape(-1, 1))
    price = list(price[0])
    price = float(price[0])
    price = round(price, 2)

    return jsonify({'price': price})

if __name__ == '__main__':
    app.run()
