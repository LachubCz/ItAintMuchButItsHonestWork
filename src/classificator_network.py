import random
from datetime import datetime

import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, BatchNormalization, Activation, PReLU, add, MaxPooling2D, Conv2D

from tools import parse_data
from batch import DataSet
img_shape = (960, 960)

def classificator_model():
    inputs = tf.keras.Input(shape=(1, img_shape[0], img_shape[1]))

    x = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = "same", activation="relu", data_format="channels_first")(inputs)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format="channels_first")(x)
    x = Conv2D(filters = 64, kernel_size = 3, strides = 2, padding = "same", data_format="channels_first")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format="channels_first")(x)
    x = Conv2D(filters = 128, kernel_size = 3, strides = 2, padding = "same", data_format="channels_first")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format="channels_first")(x)
    x = Conv2D(filters = 256, kernel_size = 3, strides = 2, padding = "same", data_format="channels_first")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format="channels_first")(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    predictions = Dense(1, activation='sigmoid')(x)

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
        images.append(np.array([cv2.resize(item.image, img_shape)]))
        labels.append(np.array([item.ellipse]))

    return images, labels


if __name__ == '__main__':
    random.seed(datetime.now())
    batch_size = 64
    tst_data = parse_data("./data/ground_truths_develop.csv", "./data/images/", "./data/ground_truths/")
    trn_data = DataSet(tst_data)
    tst_data = tst_data[:64]

    tst_images, tst_labels = strip_futilities(tst_data)

    model = classificator_model()
    for i in range(10000):

        trn_img_batch, trn_lbl_batch = trn_data.getBatch(batch_size)
        model.fit(np.array(trn_img_batch), np.array(trn_lbl_batch), epochs=1, batch_size=batch_size, verbose=1,
                  validation_data=(np.array(tst_images), np.array(tst_labels)))
