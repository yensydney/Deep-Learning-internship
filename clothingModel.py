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
# The training data
twoCatTrainList = []
for i in range(len(train_images)):
  if (train_labels[i] == 0 or train_labels[i] == 1 or train_labels[i] == 2 or train_labels[i] == 3 or train_labels[i] == 4):
    twoCatTrainList.append(0)
  else:
    twoCatTrainList.append(1)
twoCatTrainArray = np.array(twoCatTrainList)

# The testing data
twoCatTestList = []
for i in range(len(test_images)):
  if (test_labels[i] == 0 or test_labels[i] == 1 or test_labels[i] == 2 or test_labels[i] == 3 or test_labels[i] == 4):
    twoCatTestList.append(0)
  else:
    twoCatTestList.append(1)
twoCatTestArray = np.array(twoCatTestList)

####################################################### Creating 3 category array (0 tops, 1 other, 2 shoes) #######################################################
# The training data
# threeCatTrainList = []
# for i in range(len(train_images)):
#   if (train_labels[i] == 0 or train_labels[i] == 2 or train_labels[i] == 6):
#     threeCatTrainList.append(0)
#   if (train_labels[i] == 1 or train_labels[i] == 3 or train_labels[i] == 4 or train_labels[i] == 8):
#     threeCatTrainList.append(1)
#   if (train_labels[i] == 5 or train_labels[i] == 7 or train_labels[i] == 9):
#     threeCatTrainList.append(2)
# threeCatTrainArray = np.array(threeCatTrainList)

# # The testing data
# threeCatTestList = []
# for i in range(len(test_images)):
#   if (test_labels[i] == 0 or test_labels[i] == 2 or test_labels[i] == 6):
#     threeCatTestList.append(0)
#   if (test_labels[i] == 1 or test_labels[i] == 3 or test_labels[i] == 4 or test_labels[i] == 8):
#     threeCatTestList.append(1)
#   if (test_labels[i] == 5 or test_labels[i] == 7 or test_labels[i] == 9):
#     threeCatTestList.append(2)
# threeCatTestArray = np.array(threeCatTestList)

####################################################### Creating 4 category array #######################################################
# # The training data
# fourCatTrainList = []
# for i in range(len(train_images)):
#   if (train_labels[i] == 0 or train_labels[i] == 2 or train_labels[i] == 6):
#     fourCatTrainList.append(0)
#   if (train_labels[i] == 3 or train_labels[i] == 4):
#     fourCatTrainList.append(1)
#   if (train_labels[i] == 5 or train_labels[i] == 7 or train_labels[i] == 9):
#     fourCatTrainList.append(2)
#   if (train_labels[i] == 1 or train_labels[i] == 8):
#     fourCatTrainList.append(3)
# fourCatTrainArray = np.array(fourCatTrainList)

# # The testing data
# fourCatTestList = []
# for i in range(len(test_images)):
#   if (test_labels[i] == 0 or test_labels[i] == 2 or test_labels[i] == 6):
#     fourCatTestList.append(0)
#   if (test_labels[i] == 3 or test_labels[i] == 4):
#     fourCatTestList.append(1)
#   if (test_labels[i] == 5 or test_labels[i] == 7 or test_labels[i] == 9):
#     fourCatTestList.append(2)
#   if (test_labels[i] == 1 or test_labels[i] == 8):
#     fourCatTestList.append(3)
# fourCatTestArray = np.array(fourCatTestList)

####################################################### Creating 5 category array #######################################################
# # The training data
# fiveCatTrainList = []
# for i in range(len(train_images)):
#   if (train_labels[i] == 0 or train_labels[i] == 6):
#     fiveCatTrainList.append(0)
#   if (train_labels[i] == 1 or train_labels[i] == 2):
#     fiveCatTrainList.append(1)
#   if (train_labels[i] == 3 or train_labels[i] == 4):
#     fiveCatTrainList.append(2)
#   if (train_labels[i] == 5 or train_labels[i] == 7):
#     fiveCatTrainList.append(3)
#   if (train_labels[i] == 8 or train_labels[i] == 9):
#     fiveCatTrainList.append(4)
# fiveCatTrainArray = np.array(fiveCatTrainList)

# # The testing data
# fiveCatTestList = []
# for i in range(len(test_images)):
#   if (test_labels[i] == 0 or test_labels[i] == 6):
#     fiveCatTestList.append(0)
#   if (test_labels[i] == 1 or test_labels[i] == 2):
#     fiveCatTestList.append(1)
#   if (test_labels[i] == 3 or test_labels[i] == 4):
#     fiveCatTestList.append(2)
#   if (test_labels[i] == 5 or test_labels[i] == 7):
#     fiveCatTestList.append(3)
#   if (test_labels[i] == 8 or test_labels[i] == 9):
#     fiveCatTestList.append(4)
# fiveCatTestArray = np.array(fiveCatTestList)

####################################################### Creating 6 category array #######################################################
# # The training data
# sixCatTrainList = []
# for i in range(len(train_images)):
#   if (train_labels[i] == 0 or train_labels[i] == 6):
#     sixCatTrainList.append(0)
#   if (train_labels[i] == 1 or train_labels[i] == 2):
#     sixCatTrainList.append(1)
#   if (train_labels[i] == 3 or train_labels[i] == 4):
#     sixCatTrainList.append(2)
#   if (train_labels[i] == 5 or train_labels[i] == 7):
#     sixCatTrainList.append(3)
#   if (train_labels[i] == 8):
#     sixCatTrainList.append(4)
#   if (train_labels[i] == 9):
#     sixCatTrainList.append(5)
# sixCatTrainArray = np.array(sixCatTrainList)

# # The testing data
# sixCatTestList = []
# for i in range (len(test_images)):
#   if (test_labels[i] == 0 or test_labels[i] == 6):
#     sixCatTestList.append(0)
#   if (test_labels[i] == 1 or test_labels[i] == 2):
#     sixCatTestList.append(1)
#   if (test_labels[i] == 3 or test_labels[i] == 4):
#     sixCatTestList.append(2)
#   if (test_labels[i] == 5 or test_labels[i] == 7):
#     sixCatTestList.append(3)
#   if (test_labels[i] == 8):
#     sixCatTestList.append(4)
#   if (test_labels[i] == 9):
#     sixCatTestList.append(5)
# sixCatTestArray = np.array(sixCatTestList)

####################################################### Creating 7 category array #######################################################
# The training data
# sevenCatTrainList = []
# for i in range(len(train_images)):
#   current = train_labels[i]
#   if (current == 0 or current == 6):
#     sevenCatTrainList.append(0)
#   if (current == 1 or current == 2):
#     sevenCatTrainList.append(1)
#   if (current == 3 or current == 4):
#     sevenCatTrainList.append(2)
#   if (current == 5):
#     sevenCatTrainList.append(3)
#   if (current == 7):
#     sevenCatTrainList.append(4)
#   if (current == 8):
#     sevenCatTrainList.append(5)
#   if (current == 9):
#     sevenCatTrainList.append(6)
# sevenCatTrainArray = np.array(sevenCatTrainList)

# # The testing data
# sevenCatTestList = []
# for i in range(len(test_images)):
#   current = test_labels[i]
#   if (current == 0 or current == 6):
#     sevenCatTestList.append(0)
#   if (current == 1 or current == 2):
#     sevenCatTestList.append(1)
#   if (current == 3 or current == 4):
#     sevenCatTestList.append(2)
#   if (current == 5):
#     sevenCatTestList.append(3)
#   if (current == 7):
#     sevenCatTestList.append(4)
#   if (current == 8):
#     sevenCatTestList.append(5)
#   if (current == 9):
#     sevenCatTestList.append(6)
# sevenCatTestArray = np.array(sevenCatTestList)

####################################################### Creating 8 category array #######################################################
# # The training data
# eightCatTrainList = []
# for i in range(len(train_images)):
#   cur = train_labels[i]
#   if (cur == 0 or cur == 6):
#     eightCatTrainList.append(0)
#   if (cur == 1 or cur == 2):
#     eightCatTrainList.append(1)
#   if (cur == 3):
#     eightCatTrainList.append(2)
#   if (cur == 4):
#     eightCatTrainList.append(3)
#   if (cur == 5):
#     eightCatTrainList.append(4)
#   if (cur == 7):
#     eightCatTrainList.append(5)
#   if (cur == 8):
#     eightCatTrainList.append(6)
#   if (cur == 9):
#     eightCatTrainList.append(7)
# eightCatTrainArray = np.array(eightCatTrainList)

# # The testing data
# eightCatTestList = []
# for i in range(len(test_images)):
#   cur = test_labels[i]
#   if (cur == 0 or cur == 6):
#     eightCatTestList.append(0)
#   if (cur == 1 or cur == 2):
#     eightCatTestList.append(1)
#   if (cur == 3):
#     eightCatTestList.append(2)
#   if (cur == 4):
#     eightCatTestList.append(3)
#   if (cur == 5):
#     eightCatTestList.append(4)
#   if (cur == 7):
#     eightCatTestList.append(5)
#   if (cur == 8):
#     eightCatTestList.append(6)
#   if (cur == 9):
#     eightCatTestList.append(7)
# eightCatTestArray = np.array(eightCatTestList)

####################################################### Creating 9 category array #######################################################
# The training data
# nineCatTrainList = []
# for i in range (len(train_images)):
#   cur = train_labels[i]
#   if (cur == 0 or cur == 6):
#     nineCatTrainList.append(0)
#   if (cur == 1):
#     nineCatTrainList.append(1)
#   if (cur == 2):
#     nineCatTrainList.append(2)
#   if (cur == 3):
#     nineCatTrainList.append(3)
#   if (cur == 4):
#     nineCatTrainList.append(4)
#   if (cur == 5):
#     nineCatTrainList.append(5)
#   if (cur == 7):
#     nineCatTrainList.append(6)
#   if (cur == 8):
#     nineCatTrainList.append(7)
#   if (cur == 9):
#     nineCatTrainList.append(8)
# nineCatTrainArray = np.array(nineCatTrainList)

# # The testing data
# nineCatTestList = []
# for i in range (len(test_images)):
#   cur = test_labels[i]
#   if (cur == 0 or cur == 6):
#     nineCatTestList.append(0)
#   if (cur == 1):
#     nineCatTestList.append(1)
#   if (cur == 2):
#     nineCatTestList.append(2)
#   if (cur == 3):
#     nineCatTestList.append(3)
#   if (cur == 4):
#     nineCatTestList.append(4)
#   if (cur == 5):
#     nineCatTestList.append(5)
#   if (cur == 7):
#     nineCatTestList.append(6)
#   if (cur == 8):
#     nineCatTestList.append(7)
#   if (cur == 9):
#     nineCatTestList.append(8)
# nineCatTestArray = np.array(nineCatTestList)

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
model.fit(train_images, twoCatTrainArray, epochs=10)
# model.save('saved_model/twoNonsimilarCategoryModel')

test_loss, test_acc = model.evaluate(test_images, twoCatTestArray, verbose=2)
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