#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 16:24:44 2021

@author: nitinsinghal
"""
# Using RNN for CIFAR-10 image classification

# Importing the tensorflow Keras libraries and packages
import numpy as np
from tensorflow.keras import datasets, layers, models, losses
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Plot the class label images matrix
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[y_train[i][0]])
plt.show()

# Each CIFAR10 image batch is a tensor of shape (batch_size, 32, 32).
# Each input sequence will be of size (28, 28) (height is treated like time).
input_dim = 32
batch_size = 32
units = 100
output_size = 10  # labels are from 0 to 9

model = models.Sequential()
model.add(layers.TimeDistributed(layers.Flatten(input_shape=(32,3))))
model.add(layers.LSTM(units, input_shape=(None, input_dim), return_sequences=True))
model.add(layers.BatchNormalization())
model.add(layers.LSTM(units, return_sequences=True))
model.add(layers.BatchNormalization())
model.add(layers.LSTM(units))
model.add(layers.BatchNormalization())

model.add(layers.Dense(64, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(output_size))

#model.summary()

model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer="Adam",
              metrics=["accuracy"])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=5)

# Plot the train/test accuracy to see marginal improvement
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

# Evaluate the model using the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print(test_acc)



