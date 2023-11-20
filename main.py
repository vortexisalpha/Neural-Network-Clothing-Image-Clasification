import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
###DATA STUFF###

#load dataset
data = keras.datasets.fashion_mnist

#split data into test and train data
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#make it into 0-1 data by dividing by 255 as it was initially 0-255
train_images = train_images/255.0
test = test_images/255.0

'''
#show a certain image
plt.imshow(train_images[7], cmap=plt.cm.binary)
plt.show()
'''
###NEURAL NETWORK###
#because it is 28rows and 28collumns of pixels we will have 784 input neurons (1 per pixel)

#we are going to have a hidden layer of 128 neurons (hidden layers add complexity)

#we are going to also have 10 output neurons for each catagory of clothes

#(generally when picking a hidden layer pick 1 hidden layer with 20% of the input size)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    #relu is an activation function (makes the network more complex)
    keras.layers.Dense(10,activation='softmax')
    #softmax is a function that will make all the neurons in the layer sum to 1
])

#using the sparse loss function to compile the network is basically just us telling
#the network how to back propagate
model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#use the scc loss function when you have 2 or more label classes 
#the accuracy metric keeps track of the number of times predictions equal labels
#the adam optimizer is a gradient decent method

#train the model
model.fit(train_images, train_labels, epochs=5)
#each epoch feeds the neural network the same data in a different order.
#this means that the network will be trained more accurately 

#evaluate the data on the testing data
test_loss, test_acc = model.evaluate(test_images,test_labels)

print('test accuracy: ',test_acc) #displays the accuracy of the model

###CREATING A PREDICTION###

#make predictions on the test images
prediction = model.predict(np.array(test_images))

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel('Actual: '+ class_names[test_labels[i]])
    plt.title('Prediction: '+class_names[np.argmax(prediction[i])])
    plt.show()
