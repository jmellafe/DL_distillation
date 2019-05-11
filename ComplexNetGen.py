import keras
from keras.datasets import cifar10
from keras import regularizers,optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes = 10;
batch_size = 32;

x_train = np.reshape(x_train,(50000,3072))
x_test = np.reshape(x_test,(10000,3072))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

nIn = int(32*32*3)
model = Sequential()
model.add(Dense(50,kernel_regularizer=regularizers.l2(0.005), input_shape=(3072,)))
model.add(Activation('relu'))
model.add(Dense(10,kernel_regularizer=regularizers.l2(0.005)))
model.add(Activation('softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=10,
          validation_data=(x_test, y_test),
          shuffle=True)
model.save("TestModel.h5")

