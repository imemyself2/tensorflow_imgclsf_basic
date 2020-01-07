from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D, BatchNormalization, Activation, Dropout, ZeroPadding2D
from img_input import imginput
from img_input import load_test_img
from keras.models import load_model, Model
import matplotlib.pyplot as plt
import keras.regularizers
import cv2
from keras.optimizers import SGD, Adam, RMSprop
import numpy as np
from keras.applications.vgg16 import VGG16
import tensorflow as tf




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
    X_train, Y_train = imginput(dog_train_dir, cat_train_dir, 64)

    # model = Sequential()

    # model.add(Conv2D(32, kernel_size=(3,3), kernel_initializer='he_uniform', input_shape=(64,64,3), activation='relu', activity_regularizer=keras.regularizers.l2(1e-8), padding='same'))
    # model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.4))
    # model.add(Dense(1, activation='sigmoid'))
    
    # opt = Adam(learning_rate=0.01)
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model = VGG16(include_top=False, input_shape=(64, 64, 3))
	# mark loaded layers as not trainable
    for layer in model.layers:
	    layer.trainable = False
	# add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)
	# define new model
    model = Model(inputs=model.inputs, outputs=output)
	# compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train[0:8000], Y_train[0:8000], batch_size=32, epochs=25, verbose=1, shuffle=True, validation_split=0.2)

    

## Evaluate test set
req1 = input("test model? (y/n)")
if req1 == 'y':
    for i in range(5):
        testimages, testlabels = load_test_img(dog_test_dir, cat_test_dir, 64)
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
pred_img = cv2.resize(pred_img, (64, 64))
pred_img = pred_img/255
pred_arr = np.ndarray((1, 64, 64, 3))
pred_arr += pred_img

for i in range(5):
    print(model.predict(pred_arr, 1))
# print(pred_arr.shape)
# print(pred_arr)