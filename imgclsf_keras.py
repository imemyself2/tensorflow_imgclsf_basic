from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import MaxPooling2D
from img_input import imginput
from img_input import load_test_img
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.models import load_model
import matplotlib.pyplot as plt

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
    X_train, Y_train = imginput(dog_train_dir, cat_train_dir)

    model = Sequential()
    model.add(Conv2D(3, (8, 8), padding='same', activation=None))
    model.add(BatchNormalization(axis=3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train[0:4000], Y_train[0:4000], batch_size=32, epochs=40)

    model.save(model_save_dest)
    model.save_weights(model_save_weights_dest)

## Predict a custom image below
testimages, testlabels = load_test_img(dog_test_dir, cat_test_dir)
result = model.evaluate(x=testimages, y=testlabels, batch_size=64, verbose=1)

print("Loss: "+str(result[0]))
print("Accuracy: "+str(result[1]))