import random
from datetime import datetime

import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, BatchNormalization, Activation, PReLU, add, MaxPooling2D, Conv2D

from tools import parse_data

img_shape = (960, 960)
classif_classes = 6

def classificator_model(img_shape, classif_classes):
    inputs = tf.keras.Input(shape=(1, 960, 960))

    x = Conv2D(filters = 16, kernel_size = 5, strides = 2, padding = "same")(inputs)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format="channels_first")(x)
    x = Conv2D(filters = 32, kernel_size = 5, strides = 2, padding = "same")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format="channels_first")(x)
    x = Conv2D(filters = 64, kernel_size = 5, strides = 2, padding = "same")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format="channels_first")(x)
    x = Dense(128, activation="elu")(x)
    x = Dense(16, activation="elu")(x)
    x = Flatten()(x)
    predictions = Dense(1, activation='linear')(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    return model


def strip_futilities(data):
    images = []
    labels = []
    for i, item in enumerate(data):
        images.append(np.array([cv2.resize(item.image, (960, 960))]))
        labels.append(np.array([item.ellipse]))

    return images, labels


if __name__ == '__main__':
    random.seed(datetime.now())
    batch_size = 8
    trn_data_part = 0.7
    trn_data = parse_data("./data/ground_truths_develop.csv", "./data/images/", "./data/ground_truth/")
    tst_data = []
    number_of_tst_data = int(len(trn_data)*(1-trn_data_part))
    print("Len trn_data: {}; Len trn_data: {}" .format(len(trn_data)-number_of_tst_data, number_of_tst_data))

    tst_indexes = random.sample(range(len(trn_data)), number_of_tst_data)
    tst_indexes.reverse()

    for i, item in enumerate(tst_indexes):
        tst_data.append(trn_data[i])
        del trn_data[i]

    trn_images, trn_labels = strip_futilities(trn_data)
    tst_images, tst_labels = strip_futilities(tst_data)

    model = classificator_model(img_shape, classif_classes)
    for i in range(10000):
        model.fit(np.array(random.sample(trn_images, batch_size)), np.array(random.sample(trn_labels, batch_size)), 
                  epochs=1, batch_size=batch_size, verbose=1,
                  validation_data=(np.array(random.sample(tst_images, batch_size)), np.array(random.sample(tst_labels, batch_size))))
