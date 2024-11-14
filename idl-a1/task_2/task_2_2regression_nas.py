import numpy as np 
import keras_tuner as kt
import gc
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def preprocess_data(data, labels, one_dimension=False):
    # For single value output 
    if one_dimension:
        labels_processed = labels[:, 0] + labels[:, 1] / 60.0
    else: 
        labels_processed = labels
        
    data_train, data_temp, labels_train, labels_temp = train_test_split(
        data, labels_processed, test_size=0.2, random_state=42, shuffle=True
    )

    data_val, data_test, labels_val, labels_test = train_test_split(
        data_temp, labels_temp, test_size=0.5, random_state=42  
    )

    return data_train, labels_train, data_test, labels_test, data_val, labels_val


def common_sense_loss(y_true, y_pred):
    y_pred = tf.math.mod(y_pred, 12)
    linear_diff = tf.abs(y_true - y_pred)
    circular_diff = tf.minimum(linear_diff, 12 - linear_diff)
    
    return tf.reduce_mean(tf.square(circular_diff))

def common_sense_metric(y_true, y_pred):
    y_pred = tf.math.mod(y_pred, 12)
    linear_diff = tf.abs(y_true - y_pred)
    circular_diff = tf.minimum(linear_diff, 12 - linear_diff)

    return tf.reduce_mean(circular_diff)


def build_model(hp):
    model = Sequential()

    # Number of convolutional layers (NAS component)
    num_conv_layers = hp.Int("num_conv_layers", min_value=2, max_value=5)

    for i in range(num_conv_layers):
        # Number of filters (NAS component)
        filters = hp.Choice(f"filters_{i}", values=[32, 64, 128])
        # Filter size (NAS component)
        kernel_size = hp.Choice(f"kernel_size_{i}", values=[3, 5])
        # Activation function (NAS component)
        activation = hp.Choice(f"activation_{i}", values=["relu", "sigmoid", "tanh"])

        if i == 0:
            model.add(Conv2D(filters, kernel_size, activation=activation, input_shape=(75, 75, 1)))
        else:
            model.add(Conv2D(filters, kernel_size, activation=activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Optionally add Dropout after each convolutional layer
        if hp.Boolean(f"dropout_{i}"):
            dropout_rate = hp.Float(f"dropout_rate_{i}", min_value=0.1, max_value=0.5, step=0.1)
            model.add(Dropout(dropout_rate))

    model.add(Flatten())

    # Dense layer with Dropout
    dense_units = hp.Choice("dense_units", values=[32, 64, 128])
    dense_activation = hp.Choice("dense_activation", values=["relu", "sigmoid", "tanh"])
    model.add(Dense(dense_units, activation=dense_activation))

    # Optional Dropout for the dense layer
    if hp.Boolean("dense_dropout"):
        dense_dropout_rate = hp.Float("dense_dropout_rate", min_value=0.1, max_value=0.5, step=0.1)
        model.add(Dropout(dense_dropout_rate))

    # Output layer
    model.add(Dense(1, activation="linear"))

    model.compile(
        optimizer=Adam(learning_rate=hp.Float("learning_rate", min_value=1e-5, max_value=1e-3, sampling="log")),
        loss=common_sense_loss,
        metrics=[common_sense_metric]
    )

    return model

SIZE = 75
if SIZE == 75: 
    data = np.load('../clocks_small/images.npy')
    labels = np.load('../clocks_small/labels.npy')
elif SIZE == 150:
    data = np.load('../clocks_large/images.npy')
    labels = np.load('../clocks_large/labels.npy')

# 100 times 10 trials to have checkpoints in between.
for i in range(100):
    tf.keras.backend.clear_session()
    gc.collect()

    tuner = kt.RandomSearch(
        build_model,
        objective="val_common_sense_metric",
        max_trials=10,
        executions_per_trial=2,
    )

    data_train, labels_train, data_test, labels_test, data_val, labels_val = preprocess_data(data, labels, True)

    tuner.search(data_train, labels_train, validation_data=(data_val, labels_val), epochs=10, callbacks=[
        EarlyStopping(monitor="val_common_sense_metric", patience=3, restore_best_weights=True, verbose=0)
    ])

    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.summary()

    test_mae = best_model.evaluate(data_val, labels_val, verbose=1)
    print(f"Best Model Test MAE: {test_mae[i]:.4f}")

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best Hyperparameters:", best_hps.values)

    with open(f'itdl_assignment_1/best_config_{i}.txt', 'w') as file:
        best_model.summary(print_fn=lambda x: file.write(x + '\n'))
        file.write(f"Best Model Test MAE: {test_mae[i]:.4f}\n")
        file.write(f'Best Hyperparameters: {best_hps.values}\n')
