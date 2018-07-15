import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import *

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


nodes_1 = [50, 50, 50, 60, 60, 60, 70, 70, 70]
nodes_2 = [100, 150, 200, 100, 150, 200, 100, 150, 200]
nodes_3 = [50, 50, 50, 60, 60, 60, 70, 70, 70]

training_data_df = pd.read_csv("sales_data_training_scaled.csv")

X = training_data_df.drop('total_earnings', axis=1).values
Y = training_data_df[['total_earnings']].values

for index in range(len(nodes_1)):

    # Define the model
    model = Sequential()
    model.add(Dense(nodes_1[index], input_dim=9, activation='relu', name='layer_1'))
    model.add(Dense(nodes_2[index], activation='relu', name='layer_2'))
    model.add(Dense(nodes_3[index], activation='relu', name='layer_3'))
    model.add(Dense(1, activation='linear', name='output_layer'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    RUN_NAME = 'run ' + str(index) + ' with ' \
               + str(nodes_1[index]) + '--' \
               + str(nodes_2[index]) + '--' \
               + str(nodes_3[index]) + ' nodes'

    # Create a TensorBoard logger
    logger = keras.callbacks.TensorBoard(
        log_dir='logs/{}'.format(RUN_NAME),
        histogram_freq=5,
        write_graph=True
    )

    # Train the model
    model.fit(
        X,
        Y,
        epochs=50,
        shuffle=True,
        verbose=2,
        callbacks=[logger]
    )

    # Load the separate test data set
    test_data_df = pd.read_csv("sales_data_test_scaled.csv")

    X_test = test_data_df.drop('total_earnings', axis=1).values
    Y_test = test_data_df[['total_earnings']].values

    test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
    print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))