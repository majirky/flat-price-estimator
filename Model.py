# import keras.api._v2.keras as keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


def nn_model_layer(layer_in, neurons, initializer, dropout=False):
    layer = tf.keras.layers.Dense(units=neurons, activation="relu", kernel_initializer=initializer)(layer_in)
    if dropout:
        layer = tf.keras.layers.Dropout(0.2)(layer)
    return layer


def build_nn_model(input_dim, depth, dropout=True, initializer="normal", show_summary=True):
    inputs = tf.keras.Input(shape=(input_dim,))
    layers = inputs

    for i in range(0, depth):
        layers = nn_model_layer(layers, pow(2, 6 + i), dropout=dropout, initializer=initializer)

    for i in range(0, depth):
        layers = nn_model_layer(layers, pow(2, (6 + depth) - i), dropout=dropout, initializer=initializer)

    outputs = tf.keras.layers.Dense(units=64, activation="linear", kernel_initializer=initializer)(layers)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    if show_summary:
        model.summary()

    return model
