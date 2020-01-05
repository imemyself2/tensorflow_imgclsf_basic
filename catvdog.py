import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from sklearn.utils import shuffle
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.disable_eager_execution()


#import pickle -- not used (instead used np.save and np.load)

####################
### LOAD DATASET ###
####################
DOGDATADIR = "D:\\Work\\College\\Spring 2020\\DeepLearningCoursera-2\\training_set\\training_set\\dogs"
CATDATADIR = "D:\\Work\\College\\Spring 2020\\DeepLearningCoursera-2\\training_set\\training_set\\cats"

IMG_SIZE = 224
dog_arr = []
cat_arr = []

dog_data = np.empty(shape=(4000, 224, 224))
cat_data = np.empty(shape=(4000, 224, 224))


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

    outfile = open('dogsSaved', 'wb')
    np.save(outfile, dog_data)
    outfile.close()
    outfile2 = open('catsSaved', 'wb')
    np.save(outfile2, cat_data)
    outfile2.close()
    print("save files created for future runs.")



dog_label = np.ones((dog_data.shape[0], 1))
cat_label = np.zeros((cat_data.shape[0],1))

X_train_orig = np.concatenate((dog_data, cat_data), axis = 0)
Y_train_orig = np.concatenate((dog_label, cat_label), axis = 0)

X_train, Y_train = shuffle(X_train_orig, Y_train_orig, random_state = 0)
X_train = X_train/225


## Create mini-batches
minibatches_X = np.array_split(X_train, 16, axis = 0)
minibatches_Y = np.array_split(Y_train, 16, axis = 0)


print(minibatches_Y[0].shape)


### Sanity check ###
'''
plt.imshow(X_train[1], cmap = 'gray')
plt.show()
print(Y_train[1])
'''

## Create placeholders
X = tf.compat.v1.placeholder(tf.float32, (500, 224, 224, 1), name = 'X')
Y = tf.compat.v1.placeholder(tf.float32, (500, 1), name = 'Y')

## Initializing parameters
initializer = tf.initializers.GlorotUniform()
W1 = tf.Variable(initializer(shape = (24, 24, 1, 4)), name = "W1")
# output = (201, 201, 4)
W2 = tf.Variable(initializer(shape = (3, 3, 4, 4)), name = "W2")

### Forw prop ###

Z1 = tf.nn.conv2d(X, W1, strides = [1, 1, 1, 1], padding = 'SAME')
A1 = tf.nn.relu(Z1)
P1 = tf.nn.max_pool(A1, ksize = [1, 8, 8, 1], strides= [1,8,8,1],  padding = 'SAME')

Z2 = tf.nn.conv2d(P1, W2, strides = [1, 1, 1, 1], padding = "SAME")
A2 = tf.nn.relu(Z2)
P2 = tf.nn.max_pool(A2, ksize = [1, 4, 4, 1], strides = [1, 4, 4, 1], padding = "SAME")

F = tf.compat.v1.layers.flatten(P2)
Z3 = tf.compat.v1.layers.dense(F, 1)

cost = tf.reduce_mean(tf.nn.softmax(Z3))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sess:
    sess.run(init)
    minibatch_cost = 0
    i=0
    for i in range(16):
        print("HERE")
    # for epoch in range(10):
    #     minibatch_cost = 0
    #     for i in range(16):
    #         _, t_cost = sess.run([optimizer, cost], feed_dict={X:minibatches_X[i], Y:minibatches_Y[i]})
    #         minibatch_cost += t_cost/16
    #     print("Cost after epoch %i: %f"%(epoch, minibatch_cost))
        l , t_cost = sess.run([optimizer, cost], feed_dict={X:minibatches_X[i]})
        
        print("Done")
        #minibatch_cost+=t_cost/16
        print("Cost :"+ t_cost)
        
    
    prediction = tf.argmax(Z3, 1)
    correct_pred = tf.equal(prediction, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))
    print(accuracy)
    train_acc = accuracy.eval({X:X_train, Y:Y_train})
    print("Train accuracy: "+train_acc)