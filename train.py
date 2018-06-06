import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.utils import np_utils, print_summary
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pandas as pd
from sklearn.model_selection import train_test_split

import keras.backend as K

def keras_model(image_x, image_y):
    num_of_classes = 2
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(64, (5, 5), activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(num_of_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "emojinator.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]

    return model, callbacks_list

def main():
    dataset = pd.read_csv("train_foo.csv")
    # Normalizing the training dataset (train.csv file)# Norma 
    dataset = dataset.sample(frac=1)
    X = dataset.iloc[:,1:]
    Y = np.array(dataset.iloc[:,0])
    X = X/255.                      # for normalizing the inputs
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=123, train_size = 0.8)
    
    #X_train = X_train.T
    #X_test = X_test.T
    print(Y_train.shape[0])
    Y_train = Y_train.reshape(Y_train.shape[0],1)
    Y_test = Y_test.reshape(-1,1)
    # Reshape
    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))
    image_x = 50
    image_y = 50

    import ipdb
    #ipdb.set_trace()
    train_y = np_utils.to_categorical(Y_train)
    test_y = np_utils.to_categorical(Y_test)
    train_y = train_y.reshape(train_y.shape[0], train_y.shape[1])
    test_y = test_y.reshape(test_y.shape[0], test_y.shape[1])
    X_train = np.array(X_train)
    X_train = X_train.reshape(X_train.shape[0], 50, 50, 1)
    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0], 50, 50, 1)
    print("X_train shape: " + str(X_train.shape))
    print("X_test shape: " + str(X_test.shape))

    model, callbacks_list = keras_model(image_x, image_y)
    model.fit(X_train, train_y, validation_data=(X_test, test_y), epochs=10, batch_size=64,
              callbacks=callbacks_list)
    scores = model.evaluate(X_test, test_y, verbose=0)
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
    print_summary(model)

    model.save('emojinator.h5')

main()