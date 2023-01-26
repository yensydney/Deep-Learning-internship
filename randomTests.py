import sys

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Importing the fashion data
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

randomTwoCatTrainList = []
for i in range(2):
  for j in range(30000):
    randomTwoCatTrainList.append(i)
randomTwoCatTrainArray = np.array(randomTwoCatTrainList)
print(randomTwoCatTrainList)
















