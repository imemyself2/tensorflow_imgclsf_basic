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
# print(len(image)) is 60,000
#pp = pprint.PrettyPrinter(width = 50,compact=False)
#pp.pprint(image[0])

