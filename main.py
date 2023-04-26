from DataLoader import Loader, show_sample, show_info
import pandas as pd

if __name__ == '__main__':
    loader = Loader("data/data_kosice.csv")

    loader.clean_data()
    # loader.show_sample()
    loader.prepare_data()
    # loader.show_sample(loader.data_prepared)
    X_train, X_test, y_train, y_test = loader.split_data()
    show_info(X_test)
