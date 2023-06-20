# Nikita Oltyan
from tensorflow import keras
from tensorflow.keras.metrics import *

def make_model(input_shape):
    model = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='linear'),
        keras.layers.Dense(2, activation='sigmoid')
    ])

    metrics = [
        BinaryAccuracy(),
        Precision(),
        Recall()
    ]

    # Compile model
    model.compile(
        optimizer='adam',
        metrics=metrics,
        loss='binary_crossentropy'
    )

    return model

