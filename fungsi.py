import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout,LeakyReLU

def make_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape = (100, 100, 3), activation = 'relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))
    model.add(Conv2D(256, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(7, activation="softmax" , name="classification"))
    
    return model
