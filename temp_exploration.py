"""
This scripts explore the performance of the shallow network for different temperatures, saving
all the models and ploting accuracy vs temperature
"""

from SimpleModel import define_model
from ShalowNetDatasetGen import getDataset

from scipy.special import softmax
import numpy as np
from matplotlib import pyplot as plt
from time import gmtime, strftime
import json


def explore_config_temp(structure, filters, dropouts, num_classes, T, x_transfer, y_transfer_soft,
                        x_val, y_val_hard):
    """
    Training, testing and history ploting for a specific temperature. Returns final loss and
    accuracy for validation set (hard true labels)
    """
    #filename based in datetime and temperature, to handle correctly the different versions
    fname = 'models/%s-' % (str(structure)) + strftime("%Y%m%d-%H%M%S", gmtime()) + '-%d'%int(T)

    y_transfer_soft = softmax(y_transfer_soft / T, axis=1)

    model = define_model(
        structure=structure,
        filters=filters,
        dropouts=dropouts,
        init_shape=x_transfer.shape[1:],
        weight_decay=1e-4,
        num_classes=num_classes,
        temperature=T
    )

    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    batch_size = 64

    history = model.fit(x_transfer, y_transfer_soft, validation_data=(x_val, y_val_hard),
                        batch_size=batch_size, epochs=20)

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy, T = %.1f, structure = %s' % (T, str(structure)))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(fname + '_acc.png')
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss, T = %.1f, structure = %s' % (T, str(structure)))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(fname + '_loss.png')
    plt.clf()

    # Save model and history
    model.save(fname + '.h5')

    with open(fname + '.json', 'w') as f:
        json.dump(history.history, f)

    return history.history['val_loss'][-1], history.history['val_acc'][-1]


if __name__ == "__main__":

    # load data with soft results of the complex network (withou softmax activation)
    (x_transfer, y_transfer_hard, y_transfer_soft), (
    x_test, y_test_hard, y_test_soft) = getDataset()


    # get validation set from transfer set
    val_ratio = 0.1
    val_idx = int(x_transfer.shape[0] * (1 - val_ratio))

    x_val = x_transfer[val_idx:]
    y_val_soft = y_transfer_soft[val_idx:]
    y_val_hard = y_transfer_hard[val_idx:]

    x_transfer = x_transfer[:val_idx]
    y_transfer_hard = y_transfer_hard[:val_idx]
    y_transfer_soft = y_transfer_soft[:val_idx]

    # exploration for diferent temperatures
    temp2explore = [1,2,3,4,5,6,7]

    results = []

    for T in temp2explore:
        print("Training with temperature %d" % T)
        loss, acc = explore_config_temp(
            structure=[1, 1, 1],
            filters=[32, 64, 128],
            dropouts=[0.2, 0.3, 0.4],
            num_classes=10,
            T=T,
            x_transfer=x_transfer,
            y_transfer_soft=y_transfer_soft,
            x_val=x_val,
            y_val_hard=y_val_hard
        )

        results.append([T, acc])

    results = np.array(results)

    temps = results[:, 0]
    accs = results[:, 1]

    fig, ax1 = plt.subplots()


    ax1.plot(temps, accs, 'b')

    ax1.set_xlabel("Temperature")
    ax1.set_ylabel("Accuracy")

    plt.savefig("models/temp_exploration.png")