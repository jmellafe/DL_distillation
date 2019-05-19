from SimpleModel import define_model
from ShalowNetDatasetGen import getDataset

from scipy.special import softmax
import numpy as np
from sklearn.metrics import accuracy_score

from keras.utils import np_utils
from keras.datasets import cifar10
from keras.models import load_model

from matplotlib import pyplot as plt
from time import gmtime, strftime
import json

if __name__ == "__main__":

    # load data with soft results of the complex network (withou softmax activation)
    (x_transfer, y_transfer_hard, y_transfer_soft), (
        x_test, y_test_hard, y_test_soft) = getDataset()

    # get validation set from transfer set
    val_ratio = 0.1
    val_idx = int(x_transfer.shape[0] * (1 - val_ratio))

    x_val = x_transfer[val_idx:]
    y_val_soft = y_transfer_soft[val_idx:]
    y_val_real = y_transfer_hard[val_idx:]
    y_val_real = np.argmax(y_val_real, axis = 1)

    x_transfer = x_transfer[:val_idx]
    y_transfer_hard = y_transfer_hard[:val_idx]
    y_transfer_soft = y_transfer_soft[:val_idx]


    fnames = {
        "1": "[1, 1, 1]-20190517-000008.h5",
        "3": "[1, 1, 1]-20190517-005053.h5",
        "7": "[1, 1, 1]-20190517-014344.h5",
        "10": "[1, 1, 1]-20190517-023654.h5",
        "40": "[1, 1, 1]-20190517-033016.h5",
        "70": "[1, 1, 1]-20190517-042330.h5",
        "100": "[1, 1, 1]-20190517-051635.h5"
    }

    all_acc = []

    for key, weight_file in fnames.items():

        model = load_model('models/' + weight_file)

        # evaluate in validation
        y_preds = model.predict(x_val)
        y_preds = np.argmax(y_preds, axis=1)

        acc = accuracy_score(y_preds, y_val_real)

        all_acc.append([float(key), float(acc)])

    all_acc.sort()
    all_acc = np.array(all_acc)

    temps = all_acc[:, 0]
    accs = all_acc[:, 1]



    fig, ax1 = plt.subplots()

    # ax1.plot(temp2explore, all_loss, 'r')
    ax1.plot(temps, accs, 'b')

    ax1.set_xlabel("Temperature")
    ax1.set_ylabel("Accuracy")

    plt.savefig('models/results_final_ok.png')
