import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#pull data from keras
data = keras.datasets.fashion_mnist

#load data into 4 numpy arrays
(train_images, train_labels), (test_images,test_labels) = data.load_data()

#0-9 symbolizes different items of clothing, ex. 0 = tshirt
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# we divide the images by 255 to break each pixel into decimal values, and decrease the size
train_images = train_images/255.0
test_images = test_images/255.0


# Images are made up of 28x28 pixels, or 28 arrays of length 28 filled with pixels, for total length of 784 pixels
# Due to this, our input layer is made of of 784 neurons (each one taking a pixel).
# Our output layer will be 10 neurons, each representing one item of clothing from class_names.
# We will have a hidden layer with 128 neurons

model = keras.Sequential([
  # We want to flatten our data (move data inside nested arrays to one big list) for our input layer of 784 neurons
  keras.layers.Flatten(input_shape=(28,28)),
  # Our hidden layer has 128 neurons, using the Rectified Linear Unit activation function
  keras.layers.Dense(128,activation="relu"),
  # Our output layer has 10 neurons, using the softmax activation function
  kears.layers.Dense(10, activation="softmax")
])

model.compile(optimizer = "adam", loss= "sparse_categorical_crossentropy", metrics = ["accuracy"])
