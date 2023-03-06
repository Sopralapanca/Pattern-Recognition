# -*- coding: utf-8 -*-
"""ISPR-Midterm3.ipynb
MODEL SLECTION

"""

from tensorflow.keras import datasets

import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

"""PREPROCESSING"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np


def preprocessing(train_images, train_labels, test_images, test_labels):
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # considera immagini in bianco e nero
    train_labels = train_labels.flatten()
    test_labels = test_labels.flatten()

    # one hot encoding of the label
    train_labels = tf.one_hot(train_labels.astype(np.int32), depth=10)
    test_labels = tf.one_hot(test_labels.astype(np.int32), depth=10)

    # prova a rimuoverlo, numero di batch 1563
    # data augmentation
    batch_size = 32
    data_generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    train_generator = data_generator.flow(train_images, train_labels, batch_size)

    # shuffle dei dati
    # https://stackoverflow.com/questions/53731141/cifar10-randomize-train-and-test-set

    return train_generator, test_images, test_labels


train_generator, test_images, test_labels = preprocessing(train_images, train_labels, test_images, test_labels)

from functools import partial

DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, strides=1,
                        padding="SAME", use_bias=False)


class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            tf.keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                tf.keras.layers.BatchNormalization()]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)


# method to create the model
def create_model(starting_filters, res_unit_per_block, n_blocks, kernel_size, pool_size):
    # res unit per block > 3
    model = tf.keras.models.Sequential()
    model.add(DefaultConv2D(starting_filters, kernel_size=kernel_size, strides=2,
                            input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=2, padding="SAME"))
    prev_filters = starting_filters

    j = 1
    filter_list = []

    for i in range(n_blocks):
        for k in range(res_unit_per_block):
            filter_list.append(starting_filters * j)

        j *= 2

    for filters in filter_list:
        strides = 1 if filters == prev_filters else 2
        model.add(ResidualUnit(filters, strides=strides))
        prev_filters = filters
    model.add(tf.keras.layers.GlobalAvgPool2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    return model


import keras_tuner as kt


def build_model(hp):
    starting_filters = hp.Choice('filters', [16, 32, 64])
    res_unit_per_block = hp.Choice('units_per_block', [2, 3, 4])
    n_blocks = hp.Choice('n_blocks', [1, 2, 3])
    kernel_size = hp.Choice('kernel_size', [3, 7])
    pool_size = hp.Choice('pool_size', [2, 3])

    model = create_model(starting_filters, res_unit_per_block, n_blocks, kernel_size, pool_size)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=25)
tuner.search(train_generator, epochs=50, batch_size=128, validation_data=(test_images, test_labels), callbacks=[callback])
best_model = tuner.get_best_models()[0]

"""Models"""

starting_filters = best_model.get('filters')
res_unit_per_block = best_model.get('units_per_block')
n_blocks = best_model.get('n_blocks')
kernel_size = best_model.get('kernel_size')
pool_size = best_model.get('pool_size')

print(starting_filters, res_unit_per_block, n_blocks, kernel_size, pool_size)

model = create_model(starting_filters, res_unit_per_block, n_blocks, kernel_size, pool_size)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(train_generator,
                    validation_data=(test_images, test_labels),
                    batch_size=128,
                    callbacks=[callback],
                    epochs=100,
                    shuffle=True,
                    verbose=1)

model.evaluate(test_images, test_labels)

model.save("./ResNet")
