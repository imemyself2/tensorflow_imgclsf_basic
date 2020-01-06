from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import MaxPooling2D
from img_input import imginput
from img_input import load_test_img
from keras.layers import BatchNormalization
from keras.layers import Activation, Dropout
from keras.models import load_model
import matplotlib.pyplot as plt
import keras.regularizers
import cv2
import numpy as np


# Paths to different files
model_save_dest = "D:\\Work\\College\\Spring 2020\\DeepLearningCoursera-2\\gitcode\\modelV1.h5"
model_save_weights_dest = "D:\\Work\\College\\Spring 2020\\DeepLearningCoursera-2\\gitcode\\modelV1_weights.h5"
dog_test_dir = "D:\\Work\\College\\Spring 2020\\DeepLearningCoursera-2\\test_set\\test_set\\dogs"
cat_test_dir = "D:\\Work\\College\\Spring 2020\\DeepLearningCoursera-2\\test_set\\test_set\\cats"
dog_train_dir = "D:\\Work\\College\\Spring 2020\\DeepLearningCoursera-2\\training_set\\training_set\\dogs"
cat_train_dir = "D:\\Work\\College\\Spring 2020\\DeepLearningCoursera-2\\training_set\\training_set\\cats"

# Loading/Creating a model
model = None
try:
    model = load_model("D:\\Work\\College\\Spring 2020\\DeepLearningCoursera-2\\gitcode\\modelV1.h5")

    model.load_weights("D:\\Work\\College\\Spring 2020\\DeepLearningCoursera-2\\gitcode\\modelV1_weights.h5")
    print("model and weights loaded successfully.")
except:
    X_train, Y_train = imginput(dog_train_dir, cat_train_dir, 128)

    model = Sequential()
    
    model.add(Conv2D(filters=32, kernel_size=(7, 7), input_shape = (128,128,3), activity_regularizer=keras.regularizers.l2(1e-4)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(filters=32, kernel_size=(4, 4), activity_regularizer=keras.regularizers.l2(0.0001)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(64, (3, 3), activity_regularizer=keras.regularizers.l2(0.0001)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train[0:600], Y_train[0:600], batch_size=32, epochs=20, verbose=1, shuffle=True)

    

## Evaluate test set
req1 = input("test model? (y/n)")
if req1 == 'y':
    testimages, testlabels = load_test_img(dog_test_dir, cat_test_dir, 128)
    result = model.evaluate(x=testimages, y=testlabels, batch_size=32, verbose=1)

    print("Loss: "+str(result[0]))
    print("Accuracy: "+str(result[1]))

req2 = input("Save weights and model? (y/n)")
if(req2 == 'y'):
    model.save(model_save_dest)
    model.save_weights(model_save_weights_dest)
else:
    print("Weights and model not saved.")

## Custom prediction data
pred_img = cv2.imread("D:\\Work\College\\Spring 2020\\DeepLearningCoursera-2\\testimg.jpg")
pred_img = cv2.resize(pred_img, (128, 128))
pred_img = pred_img/255
pred_arr = np.ndarray((1, 128, 128, 3))
pred_arr += pred_img

print(model.predict(pred_arr, 1))
# print(pred_arr.shape)
# print(pred_arr)