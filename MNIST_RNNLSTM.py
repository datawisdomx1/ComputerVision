#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 11:10:15 2021

@author: nitinsinghal
"""
# MNIST RNN LSTM 

from tensorflow.keras import datasets, layers, models, losses
import matplotlib.pyplot as plt

# Each MNIST image batch is a tensor of shape (batch_size, 28, 28).
# Each input sequence will be of size (28, 28) (height is treated like time).
input_dim = 28
batch_size = 32
units = 100
output_size = 10  # labels are from 0 to 9

mnist = datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
sample, sample_label = x_train[0], y_train[0]


model = models.Sequential()

model.add(layers.LSTM(units, input_shape=(None, input_dim), return_sequences=True))
model.add(layers.BatchNormalization())
model.add(layers.LSTM(units, return_sequences=True))
model.add(layers.BatchNormalization())
model.add(layers.LSTM(units))
model.add(layers.BatchNormalization())

model.add(layers.Dense(64, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(output_size))

model.summary()

model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer="Adam",
              metrics=["accuracy"])

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=1)

# Plot the train/test accuracy to see marginal improvement
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

# Evaluate the model using the test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print(test_acc)



































