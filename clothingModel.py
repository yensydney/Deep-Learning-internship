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

# Storing because it is not included in the dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

def display(plotNumberArg, index, listDisplay, labelListDisplay):
  plt.subplot(5,5,plotNumberArg)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(listDisplay[index], cmap=plt.cm.binary)
  plt.xlabel(class_names[labelListDisplay[index]])
  plt.ylabel(index, labelpad = 0.5,loc = 'top', rotation = 'horizontal')

def displayCertainCategories(desired_label, startValue, myList, labelList):
  # breakpoint()
  plotNumber = 1
  amount = 0
  fig, ax = plt.subplots(2, 2, figsize=(15,8.5))
  if desired_label == 'all':
    # breakpoint()
    for i in range(int(startValue), len(labelList)): # Loop through whole array
    # print(train_labels[i])
    # breakpoint()
      display(plotNumber, i, myList, labelList)
      plotNumber = plotNumber+1
      amount = amount+1
      if (amount >= 25):
        break
  else:
    for i in range(int(startValue), len(labelList)): # Loop through whole array
      # print(train_labels[i])
      if (labelList[i] == int(desired_label)): # Condition 
        # breakpoint()
        display(plotNumber, i, myList, labelList)
        plotNumber = plotNumber+1
        amount = amount+1
      if (amount >= 25):
        break
  plt.show()
  plt.savefig('test.png')

####################################################### Creating 1 category array #######################################################
# # The training data
# oneCatTrainList = []
# for i in range(len(train_images)):
#   oneCatTrainList.append(0)
# oneCatTrainArray = np.array(oneCatTrainList)

# # The testing data
# oneCatTestList = []
# for i in range(len(test_images)):
#   oneCatTestList.append(0)
# oneCatTestArray = np.array(oneCatTestList)

####################################################### Creating 2 category array (0 tops, 1 nontops) #######################################################
# # The training data
# twoCatTrainList = []
# numberOfTops = 0
# numberOfNonTops = 0
# for i in range(len(train_images)):
#   if (train_labels[i] == 0 or train_labels[i] == 2 or train_labels[i] == 3 or train_labels[i] == 4 or train_labels[i] == 6):
#     twoCatTrainList.append(0)
#     # numberOfTops = numberOfTops + 1
#   else:
#     twoCatTrainList.append(1)
#     # numberOfNonTops = numberOfNonTops + 1
# twoCatTrainArray = np.array(twoCatTrainList)

# # The testing data
# twoCatTestList = []
# for i in range(len(test_images)):
#   if (test_labels[i] == 0 or test_labels[i] == 2 or test_labels[i] == 3 or test_labels[i] == 4 or test_labels[i] == 6):
#     twoCatTestList.append(0)
#     # numberOfTops = numberOfTops + 1
#   else:
#     twoCatTestList.append(1)
#     # numberOfNonTops = numberOfNonTops + 1
# twoCatTestArray = np.array(twoCatTestList)

####################################################### Creating 3 category array (0 tops, 1 other, 2 shoes) #######################################################
# The training data
threeCatTrainList = []
for i in range(len(train_images)):
  if (train_labels[i] == 0 or train_labels[i] == 2 or train_labels[i] == 6):
    threeCatTrainList.append(0)
  if (train_labels[i] == 1 or train_labels[i] == 3 or train_labels[i] == 4 or train_labels[i] == 8):
    threeCatTrainList.append(1)
  if (train_labels[i] == 5 or train_labels[i] == 7 or train_labels[i] == 9):
    threeCatTrainList.append(2)
threeCatTrainArray = np.array(threeCatTrainList)

# The testing data
threeCatTestList = []
for i in range(len(test_images)):
  if (test_labels[i] == 0 or test_labels[i] == 2 or test_labels[i] == 6):
    threeCatTestList.append(0)
  if (test_labels[i] == 1 or test_labels[i] == 3 or test_labels[i] == 4 or test_labels[i] == 8):
    threeCatTestList.append(1)
  if (test_labels[i] == 5 or test_labels[i] == 7 or test_labels[i] == 9):
    threeCatTestList.append(2)
threeCatTestArray = np.array(threeCatTestList)

######################################################## Creating shoes list #######################################################
# shoesList = [] # This is a list
# shoesLabelList = []
# for i in range(len(train_images)):
#   if (train_labels[i] == 5 or train_labels[i] == 7 or train_labels[i] == 9):
#     shoesList.append(train_images[i])
#     shoesLabelList.append(1)
# # displayCertainCategories('all', 0, shoesList, shoesLabelList)
# shoesArray = np.array(shoesList)
# shoesLabelArray = np.array(shoesLabelList)

######################################################## Creating non-shoes list #######################################################
# nonShoesList = []
# nonShoesLabelList = []
# for i in range(len(train_images)):
#   if (train_labels[i] == 0 or train_labels[i] == 1 or train_labels[i] == 2 or train_labels[i] == 3 or train_labels[i] == 4 or train_labels[i] == 6 or train_labels[i] == 8):
#     nonShoesList.append(train_images[i])
#     nonShoesLabelList.append(0)
# # print(len(nonShoesList))
# # displayCertainCategories('all', 0, nonShoesList, nonShoesLabelList)
# nonShoesArray = np.array(nonShoesList)
# nonShoesLabelArray = np.array(nonShoesLabelList)

######################################################## Creating shoes and non shoes joined together. 0 = nonshoes, 1 = shoes #######################################
# shoesNonShoesLabelList = []
# for i in range(len(train_images)):
#   # Shoes
#   if (train_labels[i] == 5 or train_labels[i] == 7 or train_labels[i] == 9):
#     shoesNonShoesLabelList.append(1)
#   if (train_labels[i] == 0 or train_labels[i] == 1 or train_labels[i] == 2 or train_labels[i] == 3 or train_labels[i] == 4 or train_labels[i] == 6 or train_labels[i] == 8):
#     shoesNonShoesLabelList.append(0)
# # print(len(shoesNonShoesLabelList))
# shoesNonShoesLabelArray = np.array(shoesNonShoesLabelList)

######################################################## Creating tops list #######################################################
# topsList = []
# topsLabelList = []
# for i in range(len(train_images)):
#   if (train_labels[i] == 0 or train_labels[i] == 2 or train_labels[i] == 6):
#     topsList.append(train_images[i])
#     topsLabelList.append(2)
# # print(len(topsList))
# # displayCertainCategories('all', 0, topsList, topsLabelList)
# topsArray = np.array(topsList)
# topsLabelArray = np.array(topsLabelList)

######################################################## Creating others list #######################################################
# othersList = []
# othersLabelList = []
# for i in range(len(train_images)):
#   if (train_labels[i] == 1 or train_labels[i] == 3 or train_labels[i] == 4 or train_labels[i] == 8):
#     othersList.append(train_images[i])
#     othersLabelList.append(0)
# othersArray = np.array(othersList)
# othersLabelArray = np.array(othersLabelList)

######################################################## Creating shoes, tops, and others list joined together #######################################################
# shoesTopsOthersLabelList = []
# for i in range(len(train_images)):
#   if (train_labels[i] == 1 or train_labels[i] == 3 or train_labels[i] == 4 or train_labels[i] == 8):
#     shoesTopsOthersLabelList.append(0)
#   if (train_labels[i] == 5 or train_labels[i] == 7 or train_labels[i] == 9):
#     shoesTopsOthersLabelList.append(1)
#   if (train_labels[i] == 0 or train_labels[i] == 2 or train_labels[i] == 6):
#     shoesTopsOthersLabelList.append(2)
# shoesTopsOthersLabelArray = np.array(shoesTopsOthersLabelList)

######################################################## Scaling the values to be between 0 and 1 instead of 0 to 255 #######################################################
train_images = train_images / 255.0
test_images = test_images / 255.0

# plt.figure(figsize=(10,10))
# plotNumber = 1
# for i in range(int(start), int(start)+25):
#   plt.subplot(5, 5, plotNumber)
#   plt.xticks([])
#   plt.yticks([])
#   plt.grid(False)
#   plt.imshow(train_images[i], cmap=plt.cm.binary)
#   plt.xlabel(class_names[train_labels[i]])
#   plt.ylabel(i, labelpad = 0.5, loc = 'top', rotation = 'horizontal')
#   plotNumber = plotNumber+1
# plt.show()

######################################################## Index and category filter #######################################################
# print()
# start = input("What index do you want to start from?: ") # Prompting user to insert start value
# print()
# category = input("What is the name of the category you want to display?: ")
# isFound = False
# for i in range(len(class_names)):
#   if class_names[i] == category:
#     category = i
#     isFound = True
# if isFound == False and category != 'all':
#   print("Not an available category.")
#   sys.exit()
# # print(category)
# displayCertainCategories(category, start, train_images, train_labels)


######################################################## Building layers #######################################################
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), # Unstacking rows of pixels in image and lining them up
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

######################################################## Compiling the model #######################################################
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

####################################################### Desired index #######################################################
# startIndex = input("Enter your start index: ")
# endIndex = input("Enter your end index: ")
# train_images_50k = train_images[int(startIndex):int(endIndex)]
# train_labels_50k = train_labels[int(startIndex):int(endIndex)]

######################################################## Training and feeding the model #######################################################
model.fit(train_images, threeCatTrainArray, epochs=10)

test_loss, test_acc = model.evaluate(test_images, threeCatTestArray, verbose=2)
print('Test accuracy:', test_acc)

######################################################## Evaluating accuracy (old) #######################################################
# print()
# test_loss, test_acc = model.evaluate(train_images, allLabelArray, verbose=2)
# print('Test accuracy:', test_acc)

######################################################## Making predictions #######################################################
# probability_model = tf.keras.Sequential([model, 
#                                          tf.keras.layers.Softmax()])
# predictions = probability_model.predict(test_images)
# print(predictions[0])
# print("Prediction: ")
# print(np.argmax(predictions[0]))
# print("Real: ")
# print(test_labels[0])

######################################################## Plot the first X test images, their predicted labels, and the true labels. #####################################
# num_rows = 5
# num_cols = 5
# num_images = num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# plotNumber = 0
# for i in range(int(start), int(start)+num_images):
#   plt.subplot(num_rows, 2*num_cols, 2*plotNumber+1) # Rows, cols, # of plot
#   plot_image(i, predictions[i], test_labels, test_images)
#   plt.subplot(num_rows, 2*num_cols, 2*plotNumber+2)
#   plot_value_array(i, predictions[i], test_labels)
#   plotNumber = plotNumber+1
# plt.tight_layout()
# plt.show()

print()
print("All done!")