import numpy as np
import pprint
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image, ImageOps
from mnist import MNIST

mndata = MNIST('trainandtest')

# Load Data from the MNIST dataset

image, label = mndata.load_training()
testset, testlabel = mndata.load_testing()
dataset = tf.data.Dataset.from_tensor_slices((image, label))
print(dataset) # doesn't print correct shape,takes too long to run
#print(dataset)
# print(len(image)) is 60,000
#pp = pprint.PrettyPrinter(width = 50,compact=False)
#pp.pprint(image[0])

# initialize parameters

parameters = {}

parameters["W1"] = tf.Variable(np.random.randn(20, len(image)), name = "W1")
parameters["b1"] = tf.Variable(np.zeros((20,1)), name = "b1")
parameters["W2"] = tf.Variable(np.random.randn(15, 20), name = "W2")
parameters["b2"] = tf.Variable(np.zeros((15, 1)), name = "b2")




