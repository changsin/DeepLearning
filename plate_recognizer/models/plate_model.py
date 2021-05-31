from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.applications.vgg16 import VGG16

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
dist = tfp.distributions

IMAGE_SIZE = 224

class PlateModel():
    def __init__(self, use_probability=True):
        self.use_probability = use_probability

    def create_model(self, train_size):
        kl_divergence_fn = lambda q, p, _: dist.kl_divergence(q, p) / tf.cast(train_size, dtype=tf.float32)

        model = Sequential()
        model.add(VGG16(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))

        if self.use_probability:
            model.add(tfp.layers.DenseFlipout(4, activation="sigmoid", kernel_divergence_fn=kl_divergence_fn))
        else:
            model.add(Dense(4, activation="sigmoid"))

        model.layers[-6].trainable = False
        model.summary()

        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        return model