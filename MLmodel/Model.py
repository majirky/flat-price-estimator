# import keras.api._v2.keras as keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


def nn_model_layer(layer_in, nodes, initializer, dropout=False):
    """
    creates hidden layer of neural network model with n nodes and relu activation function
    :param layer_in: layer to attach to
    :param nodes: number of nodes in layer
    :param initializer: weights initializer
    :param dropout: if true, creates Dropout layer between dense layers with 20% node dropout chance
    :return: layer with defined settings
    """
    layer = tf.keras.layers.Dense(units=nodes, activation="relu", kernel_initializer=initializer)(layer_in)
    if dropout:
        layer = tf.keras.layers.Dropout(0.2)(layer)
    return layer


def build_nn_model(input_dim, depth, dropout=True, initializer="normal", show_summary=True):
    """
    creates neural network model using tensorflow API, consists of n layers.
    :param input_dim: input dimensions of dataset. (Kosice 34, Bratislava 39)
    :param depth: stands for n of layers
    :param dropout: if true, creates Dropout layer between dense layers with 20% node dropout chance
    :param initializer: weights initializer, normal is default
    :param show_summary: if true, shows summary of neural network model with model.summary()
    :return: neural network model
    """
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
