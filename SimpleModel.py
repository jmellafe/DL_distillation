
import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler


def define_model(structure, filters, dropouts, init_shape, weight_decay, num_classes, temperature = 1):
    """
    Define the simple network, based in the params passed in the "structure" variable. Definitions:
    Small pack: conv2d -> elu -> BN
    Big Pack: at least 1 small pack + maxPool + DropOut
    :param structure: list of integers. Each number indicates one "Big Pack", and the number itself
    the number of "small packs" in the "Big Pack". Strcture = [2,2,2] is the one of the complex network
    :param filters: list of integers, number of filters in the conv2d of each Big Pack.
    filters = [32, 64, 128] for the complex network
    :param dropouts: drop-out ratio for the drop out layer at the end of each Big Pack. for the
    complex network dropouts = [0.2, 0.3, 0.4]
    :return: model
    """

    # Check that the input of the function has sense

    assert len(structure)==len(filters)==len(dropouts), "The length of the inputs don't match"
    assert min(structure) > 0, "All the Big Packs should include at least one Small Pack"
    assert min(dropouts) > 0 or max(dropouts) < 1, "Drop outs should be between 0 an 1"


    model = Sequential()

    for i, number_small_pack in enumerate(structure):

        for j in range(number_small_pack):
            if i + j == 0:
                # input shape should be specified in the first layer
                model.add(
                Conv2D(filters[i], (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                       input_shape=init_shape))
            else:
                model.add(
                Conv2D(filters[i], (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
            model.add(Activation('elu'))
            model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(dropouts[i]))


    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Lambda(lambda x: x / temperature))
    model.add(Activation('softmax'))

    # model.summary()

    return model