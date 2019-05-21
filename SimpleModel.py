"""
Definition of the networks.
"""



from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Lambda, Input
from keras import regularizers


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

    At this function is only used with the next params during the temperature and weight exploration:
            structure=[1, 1, 1],
            filters=[32, 64, 128],
            dropouts=[0.2, 0.3, 0.4],
            num_classes=10,
            weight_decay=1e-4
    """

    # Check that the input of the function has sense

    assert len(structure)==len(filters)==len(dropouts), "The length of the inputs don't match"
    assert min(structure) > 0, "All the Big Packs should include at least one Small Pack"
    assert min(dropouts) > 0 or max(dropouts) < 1, "Drop outs should be between 0 an 1"

    #Definition of the network
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


def define_model_weighted(structure, filters, dropouts, init_shape, weight_decay, num_classes, weight = 1., temperature = 1):
    """
    This model is a modified version of the previous one. The only difference is that this has two
    outputs: one with soft labels depending on the temperature (same as previous model) and an
    additional output with the hard labels (softmax with temperature 1). The soft labels are trained
    by distillation from the complex network and the hard labels with the real labels from the dataset.
    Cross-entropy loss for both outputs but with different weights, Temperature-normalized.

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

    # Definition of the network, Functional API in order to handle multiple outputs
    inp = Input(init_shape)

    for i, number_small_pack in enumerate(structure):

        for j in range(number_small_pack):
            if i + j == 0:
                model = Conv2D(filters[i], (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(inp)
            else:
                model = Conv2D(filters[i], (3, 3), padding='same',
                               kernel_regularizer=regularizers.l2(weight_decay))(model)
            model = Activation('elu')(model)
            model = BatchNormalization()(model)

        model = MaxPooling2D(pool_size=(2,2))(model)
        model = Dropout(dropouts[i])(model)


    model = Flatten()(model)
    dense = Dense(num_classes)(model)
    out_hard = Activation('softmax', name = 'out_hard')(dense)

    out_soft = Lambda(lambda x: x / temperature)(dense)
    out_soft = Activation('softmax', name = 'out_soft')(out_soft)

    model = Model(inputs = inp, outputs = [out_hard, out_soft])

    # handle multiple losses and weights

    losses = {
        "out_hard": "categorical_crossentropy",
        "out_soft": "categorical_crossentropy",
    }
    lossWeights = {"out_soft": temperature**2, "out_hard": weight}

    model.compile(loss=losses, loss_weights = lossWeights, optimizer='adam', metrics=['accuracy'])

    return model

