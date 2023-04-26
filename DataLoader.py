import pandas as pd
from IPython.display import display, HTML
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split


def show_info(data_to_show):
    data_to_show.info()


def show_sample(data_to_show):
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 5000)
    to_show = data_to_show.sample(50)
    display(to_show)


class Loader:

    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.data_prepared = None

        self.scaler_numeric_data = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_price = MinMaxScaler(feature_range=(-1, 1))

    def clean_data(self):
        self.data = self.data[self.data["housing"] == "Byty"]
        self.data = self.data[self.data["housing_type"] == "Predaj"]

        self.data.drop(self.data.loc[self.data["price"] == "Cenadohodou"].index, inplace=True)
        self.data.drop(self.data.loc[self.data["price"] == "InfovRK"].index, inplace=True)
        self.data.drop("price_per_meter", inplace=True, axis=1)

        self.data["living_area"] = [str(row).replace(',', '.') for row in self.data["living_area"]]
        self.data["land_area"] = [str(row).replace(',', '.') for row in self.data["land_area"]]
        self.data["price"] = [str(row).replace(',', '.') for row in self.data["price"]]

        self.data["title"] = self.data["title"].astype("string")
        self.data["link"] = self.data["link"].astype("string")
        self.data["date"] = self.data["date"].astype("string")
        self.data["price"] = self.data["price"].astype("float64")
        self.data["housing"] = self.data["housing"].astype("category")
        self.data["housing_category"] = self.data["housing_category"].astype("category")
        self.data["city_area"] = self.data["city_area"].astype("category")
        self.data["housing_type"] = self.data["housing_type"].astype("category")
        self.data["living_area"] = self.data["living_area"].astype("float64")
        self.data["land_area"] = self.data["land_area"].astype("float64")

    def prepare_data(self):
        numeric_data = pd.DataFrame(self.data[["living_area", "land_area", "lat", "long"]])

        # use numeric attribute scaler model and standardize numeric data
        model = self.scaler_numeric_data.fit(numeric_data)
        standardized = model.transform(numeric_data)
        numeric_standardized = pd.DataFrame(standardized, columns=["living_area", "land_area", "lat", "long"])

        # use price scaler model and standardize price
        # reason to use differend scaler model is that, after prediction we want to destandardize price,
        # we couldn't destandardize only price with model used for other attributes
        model_price = self.scaler_price.fit(np.asarray(self.data["price"]).reshape(-1, 1))
        price_standardized = model_price.transform(np.asarray(self.data["price"]).reshape(-1, 1))

        # make encoder for categorial attributes
        categorial_data = pd.DataFrame(self.data[["housing_category", "city_area", "housing_state"]])
        categorial_dummies = pd.get_dummies(categorial_data[["housing_category", "city_area", "housing_state"]])
        categorial_dummies = categorial_dummies.reset_index(drop=True)

        # put modified data back to one dataframe -> ML ready
        self.data_prepared = pd.DataFrame(categorial_dummies)
        self.data_prepared["living_area"] = numeric_standardized["living_area"]
        self.data_prepared["land_area"] = numeric_standardized["land_area"]
        self.data_prepared["lat"] = numeric_standardized["lat"]
        self.data_prepared["long"] = numeric_standardized["long"]

        self.data_prepared["price"] = price_standardized

    def split_data(self, test_size = 0.2, random_state = 123):
        X = pd.DataFrame(self.data_prepared)
        y = pd.DataFrame(self.data_prepared.price)
        del X["price"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
