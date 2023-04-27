from DataLoader import Loader, show_sample, show_info
from Model import build_nn_model
from keras.optimizers import Adam

import pandas as pd

if __name__ == '__main__':

    loader = Loader("data/data_kosice.csv")

    loader.clean_data()
    # loader.show_sample()
    loader.prepare_data()
    # loader.show_sample(loader.data_prepared)
    X_train, X_test, y_train, y_test = loader.split_data()
    show_info(X_test)

    model = build_nn_model(input_dim=34, depth=2, dropout=True)
    model.compile(loss='mse', optimizer=Adam(), metrics=['mse'])
    model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, verbose=1)

