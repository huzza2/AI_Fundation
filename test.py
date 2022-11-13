import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd

import os
import random
import shutil
import pathlib

from keras_preprocessing.image import ImageDataGenerator

def load_and_predict():
    model = keras.models.load_model('dogs-vs-cats.h5')

    test_generator = ImageDataGenerator(rescale=1. / 255)

    test_iterator = test_generator.flow_from_directory(
        './input_test',
        target_size=(150, 150),
        shuffle=False,
        class_mode='binary',
        batch_size=1)

    ids = []
    for filename in test_iterator.filenames:
        ids.append(int(os.path.basename(filename).split('.')[0]))

    predict_result = model.predict(test_iterator, steps=len(test_iterator.filenames))
    predictions = []
    for index, prediction in enumerate(predict_result):
        predictions.append([ids[index], prediction[0]])
    predictions.sort()

    return predictions

predictions = load_and_predict()
df = pd.DataFrame(data=predictions, index=range(1, 12501), columns=['id', 'label'])
df = df.set_index(['id'])
df.to_csv('submission.csv')
