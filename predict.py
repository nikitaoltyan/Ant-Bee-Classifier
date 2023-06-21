# Nikita Oltyan

from keras.models import load_model
import tensorflow as tf
import keras.utils as image
import numpy as np
import os

model_path = 'model/trained_model.h5'
classes = ['Ant', 'Bee']

def predict(img):
    print('Loading model...')
    model = load_model(model_path)
    print('Done')

    metrics = [
        tf.metrics.BinaryAccuracy(),
        tf.metrics.Precision(),
        tf.metrics.Recall()
    ]
    model.compile(metrics=metrics)

    x = np.expand_dims(img, axis=0)

    images = np.vstack([x])
    prediction = model.predict(images)
    predicted_class = classes[np.argmax(prediction)]
    return predicted_class