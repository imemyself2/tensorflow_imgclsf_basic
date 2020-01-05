
import numpy as np
from sklearn.utils import shuffle
import os
import cv2

def imginput(DOGDATADIR, CATDATADIR):
    IMG_SIZE = 112
    dog_arr = []
    cat_arr = []

    dog_data = np.empty(shape=(4000, 112, 112))
    cat_data = np.empty(shape=(4000, 112, 112))


    try:
        dogpath = "D:\\Work\\College\\Spring 2020\\DeepLearningCoursera-2\\dogsSaved"
        catpath = "D:\\Work\\College\\Spring 2020\\DeepLearningCoursera-2\\catsSaved"
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
        dog_data = np.expand_dims(dog_data, axis=4)
        cat_data = np.expand_dims(cat_data, axis=4)

        outfile = open(dogpath, 'wb')
        np.save(outfile, dog_data)
        outfile.close()
        outfile2 = open(catpath, 'wb')
        np.save(outfile2, cat_data)
        outfile2.close()
        print("save files created for future runs.")



    dog_label = np.ones((dog_data.shape[0], 1))
    cat_label = np.zeros((cat_data.shape[0],1))

    X_train_orig = np.concatenate((dog_data, cat_data), axis = 0)
    Y_train_orig = np.concatenate((dog_label, cat_label), axis = 0)

    X_train, Y_train = shuffle(X_train_orig, Y_train_orig, random_state = 0)
    X_train = X_train/225
    return X_train, Y_train
