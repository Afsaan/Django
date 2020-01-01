import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Conv2D , MaxPool2D , Flatten , Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


(X_train , y_train) , (X_test , y_test ) = cifar10.load_data()

classes = ['airplane' , 'automobile' , 'bird' , 'cat' , 'deer' , 'dog' , 'frog', ' horse' , 'ship' , 'truck']

#scalar = StandardScaler()
#X_train = scalar.fit_transform(X_train)

X_train ,  X_test = X_train / 255.0 , X_test / 255.0
y_train , y_test  = to_categorical(y_train) , to_categorical(y_test)

model = Sequential()
model.add(Conv2D(32 , (3,3) , activation = 'relu' , input_shape =(32,32,3)))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(64 , (3,3) , activation = 'relu'))
model.add(MaxPool2D((2,2)))
model.add(Flatten())
model.add(Dense(128 , activation='relu'))
model.add(Dense(64 , activation = 'tanh'))
model.add(Dense(10 , activation = 'softmax'))


model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train , y_train , epochs = 50 , batch_size=32)