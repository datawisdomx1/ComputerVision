#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 16:24:44 2021

@author: nitinsinghal
"""
# Using Convolutional Neural Networks (CNN) for CIFAR-10 image classification

######## MODEL 1 - FINAL CODE - ACcuracy = 86.2% ####################

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


########## OTHER CODE WITH LOWER ACCURACY ###############

##### MODEL - 2 - Accuracy = 85.35% #########
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

batch_size = 16
num_classes = 10
epochs = 40
#data_augmentation = True
num_predictions = 20

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

pochs = 50

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test, y_test),shuffle=True)
scores = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


##### MODEL - 3  Accuracy = 67%  . #########
##   Using Resnet50 pretrained model for transfer learning 

import tensorflow as tf 
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
import pandas as pd
pd.set_option('display.max_columns',None)#displaying long list of columns
pd.set_option('display.max_rows', None)#displaying long list of rows
pd.set_option('display.width', 1000)#width of window

from keras.datasets import cifar10

# Preprocess data function

def preprocess_data(X,Y):
  X_p = keras.applications.resnet50.preprocess_input(X)
  Y_p = keras.utils.to_categorical(Y,10)
  return X_p, Y_p


# load and split data
# The data, split between train and test sets:

#(x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)

# Preprocess data
## Next, we are going to call our function with the parameters loaded from the CIFAR10 database.

x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)

# Using weights of a trained neural network
# A pretrained model from the Keras Applications has the advantage of allow you to use weights that
# are already calibrated to make predictions. In this case, we use the weights from Imagenet 
# and the network is a ResNet50. The option include_top=False allows feature extraction by removing 
# the last dense layers. This let us control the output and input of the model.

input_t = keras.Input(shape=(32,32,3))
res_model = keras.applications.ResNet50(include_top=False,
                                    weights="imagenet",
                                    input_tensor=input_t)

# In this case, we ‘freeze’ all layers except for the last block of the ResNet50.

for layer in res_model.layers[:143]:
  layer.trainable=False

# We can check that we did it correctly with:
# False means that the layer is ‘freezed’ or is not trainable and 
# True that when we run our model, the weights are going to be adjusted.

for i, layer in enumerate(res_model.layers):
  print(i,layer.name,"-",layer.trainable)
  

 # Add Flatten and Dense layers on top of Resnet
 # Now, we need to connect our pretrained model with the new layers 
 # of our model. We can use global pooling or a flatten layer to connect 
 # the dimensions of the previous layers with the new layers. 
 
# to_res = (224, 224)
to_res = (32, 32)

model = keras.models.Sequential()
model.add(keras.layers.Lambda(lambda image: tf.image.resize(image, to_res))) 
model.add(res_model)
model.add(keras.layers.Flatten())
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(10, activation='softmax'))

# Compile model and train
# Results

model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(lr=2e-5),
                  metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1,
                        validation_data=(x_test, y_test)
                       )
model.summary()



