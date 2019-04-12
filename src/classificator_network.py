import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, BatchNormalization, Activation, PReLU, add

img_shape = (1024, 1024, 1)
classif_classes = 6

def classificator_model(img_shape, classif_classes):
    model = Sequential()

    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(classif_classes, activation='softmax'))

    net_input = Input(shape=img_shape)
    net_output = model(net_input)

    Model(inputs=net_input, outputs=net_output)

    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()