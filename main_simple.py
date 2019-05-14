from SimpleModel import define_model
from ShalowNetDatasetGen import getDataset

from scipy.special import softmax
import numpy as np
from keras.utils import np_utils
from keras.datasets import cifar10
from matplotlib import pyplot as plt

if __name__ == "__main__":

    # load data with soft results of the complex network (withou softmax activation)
    (x_transfer, y_transfer_hard, y_transfer_soft), (x_test, y_test_hard, y_test_soft) = getDataset()

    # aply temperature to the soft results

    T = 4

    y_transfer_soft = softmax(y_transfer_soft/T, axis = 1)

    num_classes = 10

    model = define_model(
        structure=[1, 1, 1],
        filters=[32, 64, 128],
        dropouts=[0.2, 0.3, 0.4],
        init_shape=x_transfer.shape[1:],
        weight_decay=1e-4,
        num_classes=num_classes,
        temperature= T
    )

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    batch_size = 64

    history = model.fit(x_transfer, y_transfer_soft, validation_split=0.1, batch_size=batch_size, epochs=20)

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
