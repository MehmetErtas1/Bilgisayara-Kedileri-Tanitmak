
# Software for detecting cats from other animals in the database

# Import some inportant functions
from Calling_program1 import *
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage
import scipy.misc
import h5py


# Loading the dataset
train_x, train_y, test_x, test_y, classes = load_data()

# Number of training images
m = train_x.shape[0]
num_px = train_x.shape[1]

# Reshaping the training and test images
train_x_flatten = train_x.reshape(train_x.shape[0],-1).T
test_x_flatten = test_x.reshape(test_x.shape[0],-1).T

# Normalize the dataset by 255
train_x = train_x_flatten/255
test_x = test_x_flatten/255


# Defining nodes for the neural network
layer_dims_deep = [12288, 20, 7, 5, 1]

# Calling the neural network function
parameters = L_layer_deep_NN(train_x, train_y, 0.0075, layer_dims_deep, 3000, True)

# Outputs the predictions
predictions_test = predict( parameters, test_x, test_y)
print("Accuracy of L layer network is ", str(predictions_test))

# Testing our own image
our_image="test6.jpg"

# Pre-process the image to be used in our algorithm
fname = our_image
my_label = 0       # Put the y-label as per our image i.e cat then put 1 other wise put 0
image = np.array(scipy.ndimage.imread(fname))
my_image = scipy.misc.imresize(image,size=(num_px,num_px)).reshape((1,num_px*num_px*3)).T
predicted_image_out=predict(parameters, my_image, my_label)
plt.imshow(image)
plt.show()
print("y = ", str(np.squeeze(predicted_image_out)) + ", The algorithm predicts a \"" + classes[int(predicted_image_out)].decode("utf-8") +  "\" picture.")






