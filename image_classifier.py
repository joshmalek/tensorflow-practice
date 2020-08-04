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

model = keras.Sequential([])
