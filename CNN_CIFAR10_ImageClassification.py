#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 21 12:35:27 2022

@author: nitinsinghal
"""

# Using Convolutional Neural Networks (CNN) for CIFAR-10 image classification

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from keras.layers import LeakyReLU
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

# Build the  2D convolution layers using relu activation
# 3 Convolution layers have been created with BatchNormalization, MaxPooling and Dropout
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))

model.summary()

# Build the Flattened dense layers - 3 with decreasing features to finally give class labels
model.add(layers.Flatten())
model.add(layers.Dense(128, activation=LeakyReLU(alpha=0.3)))
model.add(layers.BatchNormalization())
model.add(layers.Dense(64, activation=LeakyReLU(alpha=0.3)))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# Compile the model using Adam optimizer
model.compile(optimizer='Adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Fit and test validate the model with 100 epochs and batch size of 10
history = model.fit(X_train, y_train, batch_size=10, epochs=100,
                    validation_data=(X_test, y_test))

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