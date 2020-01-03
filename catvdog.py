import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from sklearn.utils import shuffle
#import pickle

DOGDATADIR = "D:\\Work\\College\\Spring 2020\\DeepLearningCoursera-2\\training_set\\training_set\\dogs"
CATDATADIR = "D:\\Work\\College\\Spring 2020\\DeepLearningCoursera-2\\training_set\\training_set\\cats"

IMG_SIZE = 224
dog_arr = []
cat_arr = []

try:
    dogpath = "D:\\Work\\College\\Spring 2020\\DeepLearningCoursera-2\\dogsSaved"
    catpath = "D:\\Work\\College\\Spring 2020\\DeepLearningCoursera-2\\catsSaved"
    if os.path.exists(dogpath) and os.path.exists(catpath):
        dogs = open(dogpath, 'rb')
        cats = open(catpath, 'rb')
        dog_data = np.load(dogs)
        cat_data = np.load(cats)
        print('Existing save files found....')
except IOError:
    
    for img in os.listdir(DOGDATADIR):
        try: 
            dog_img = cv2.imread(os.path.join(DOGDATADIR, img), cv2.IMREAD_GRAYSCALE)
            dog_img = cv2.resize(dog_img, (IMG_SIZE, IMG_SIZE))
            dog_arr.append(dog_img)
        except:
            continue

        
    for img in os.listdir(CATDATADIR):
        try:
            cat_img = cv2.imread(os.path.join(CATDATADIR, img), cv2.IMREAD_GRAYSCALE)
            cat_img = cv2.resize(cat_img,(IMG_SIZE, IMG_SIZE))
            cat_arr.append(cat_img)
        except:
            continue
    

    dog_data = np.asanyarray(dog_arr)
    cat_data = np.asanyarray(cat_arr)

    outfile = open('dogsSaved', 'wb')
    np.save(outfile, dog_data)
    outfile.close()
    outfile2 = open('catsSaved', 'wb')
    np.save(outfile2, cat_data)
    outfile2.close()

dog_label = np.ones((dog_data.shape[0], 1))
cat_label = np.zeros((cat_data.shape[0],1))

X_train_orig = np.concatenate((dog_data, cat_data), axis = 0)
Y_train_orig = np.concatenate((dog_label, cat_label), axis = 0)

X_train, Y_train = shuffle(X_train_orig, Y_train_orig, random_state = 0)

# Sanity check
plt.imshow(X_train[1], cmap = 'gray')
plt.show()
print(Y_train[1])