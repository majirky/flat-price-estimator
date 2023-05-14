import pandas as pd
from IPython.display import display, HTML
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split


def show_info(data_to_show):
    """
    show info about dataset
    :param data_to_show: data
    :return: nothing
    """
    data_to_show.info()


def show_sample(data_to_show, sample=30):
    """
    shows sample of dataset with custom settings to show every column in non-interacitve shell
    :param data_to_show: data
    :return: nothing
    """
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 5000)
    to_show = data_to_show.sample(sample)
    display(to_show)


class Loader:
    """
    Loader() class handles methods and variables for data preparation and manipulation to get data ML ready
    """

    def __init__(self, data_path):
        """
        initialize Loader class
        :param data_path: path to dataset, that needs to be prepared
        """
        self.data = pd.read_csv(data_path)
        self.data_prepared = None

        self.scaler_numeric_data = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_price = MinMaxScaler(feature_range=(-1, 1))

    def clean_data(self):
        """
        cleans data. Only data for flats to sell remains. Erase data with no price tag.
        converts data to correct data type
        :return: modifies self.data. nothing to return
        """
        # only flats on sale
        self.data = self.data[self.data["housing"] == "Byty"]
        self.data = self.data[self.data["housing_type"] == "Predaj"]

        # drop rows without price
        self.data.drop(self.data.loc[self.data["price"] == "Cenadohodou"].index, inplace=True)
        self.data.drop(self.data.loc[self.data["price"] == "InfovRK"].index, inplace=True)

        # drop unimportant cols
        self.data.drop("price_per_meter", inplace=True, axis=1)
        self.data.drop("lat", inplace=True, axis=1)
        self.data.drop("long", inplace=True, axis=1)
        self.data.drop("land_area", inplace=True, axis=1)

        # replace , to . and unify value for ťahanovce
        self.data["living_area"] = [str(row).replace(',', '.') for row in self.data["living_area"]]
        self.data["price"] = [str(row).replace(',', '.') for row in self.data["price"]]
        self.data["city_area"] = [str(row).replace('Košice I - Sídlisko Ťahanovce', 'Košice I - Ťahanovce') for row in self.data["city_area"]]

        # astypes
        self.data["title"] = self.data["title"].astype("string")
        self.data["link"] = self.data["link"].astype("string")
        self.data["date"] = self.data["date"].astype("string")
        self.data["price"] = self.data["price"].astype("float64")
        self.data["housing"] = self.data["housing"].astype("category")
        self.data["housing_category"] = self.data["housing_category"].astype("category")
        self.data["city_area"] = self.data["city_area"].astype("category")
        self.data["housing_type"] = self.data["housing_type"].astype("category")
        self.data["living_area"] = self.data["living_area"].astype("float64")

        # drop rows that has very low price eg. 1 or 10. Those are probablly wrong inputs
        self.data.drop(self.data.loc[self.data["price"] < 10].index, inplace=True)

        # drop rows, that its city area atribute does not occur in dataset more than 5 times
        city_area_counts = self.data["city_area"].value_counts()
        city_area_few_values = city_area_counts.loc[city_area_counts < 5].index
        self.data = self.data[~self.data["city_area"].isin(city_area_few_values)]

    def prepare_data(self):
        """
        define scalers for numeric data and different scaler for price, so we can destandardize price after prediction.
        Uses encoder for categorial attributes. Creates ML ready dataset
        :return: updates self.data_prepared and scalers. nothing to return
        """
        numeric_data = pd.DataFrame(self.data[["living_area"]])

        # use numeric attribute scaler model and standardize numeric data
        model = self.scaler_numeric_data.fit(numeric_data)
        standardized = model.transform(numeric_data)
        numeric_standardized = pd.DataFrame(standardized, columns=["living_area"])

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
        self.data_prepared = pd.DataFrame(categorial_dummies, dtype=np.uint8)
        self.data_prepared["living_area"] = numeric_standardized["living_area"]

        self.data_prepared["price"] = price_standardized

    def split_data(self, test_size = 0.2, random_state = 123):
        """
        splits data to train and test datasets
        :param test_size: test dataset size in percent relative to whole dataset
        :param random_state: seed for reproductions
        :return: X_train, X_test, y_train, y_test
        """
        X = pd.DataFrame(self.data_prepared)
        y = pd.DataFrame(self.data_prepared.price)
        del X["price"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
