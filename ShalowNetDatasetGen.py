from keras.utils import np_utils
from keras.datasets import cifar10
from keras.models import Model,load_model
import numpy as np

def softmax(x, t = 1.0): #Softmax with temp, useful for the distillation training.
    e = np.exp(x / t)
    dist = e / np.sum(e)
    return dist

def correctWrongResults(yProbs,y1hot):
    #There are some results from the complex that are misclassified,
    #After applying softmax to those results, with this function all the
    #misclassifications are fixed with 1hot encoding of the correct answer.
    numclasses=10;
    Obtained1Hot=np_utils.to_categorical(np.argmax(yProbs, axis=1),numclasses)
    for i in range(np.shape(Obtained1Hot)[0]):
        if( not (Obtained1Hot[i,:] == y1hot[i,:]).all() ):
            yProbs[i,:]=y1hot[i,:];

    return yProbs

def getDataset():
    #Gets the dataset used to train: x_train and test are z-transformed
    #y_train/test Hard are the 1 hot encoding of the results
    #y_trin/test Soft are the results of the complex model WITHOUT SOFTMAX!
    #For less time computing, The datasets can be loaded from the file datasets
    (x_train, y_trainHard), (x_test, y_testHard) = cifar10.load_data()
    # x_train = x_train[:1000].astype('float32')
    # x_test = x_test[:1000].astype('float32')

    y_trainHard = y_trainHard[:1000]
    y_testHard = y_testHard[:1000]

    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))
    x_train = (x_train - mean) / (std + 1e-7) #Z transform of input
    x_test = (x_test - mean) / (std + 1e-7)

    num_classes = 10
    y_trainHard = np_utils.to_categorical(y_trainHard, num_classes) #2 One hot encoding
    y_testHard = np_utils.to_categorical(y_testHard, num_classes)

    model = load_model('TestModel.h5') #Loads trained net
    ModelCut = Model(inputs=model.input,outputs=model.layers[-2].output) #Gets the same network without last layer
    W = model.layers[-1].get_weights()  # Gets last layer weights

    y_testSoft = np.dot(ModelCut.predict(x_test), W[0]) + W[1]
    y_trainSoft = np.dot(ModelCut.predict(x_train), W[0]) + W[1]

    np.savez('Datasets.npz', name1=x_train, name2=y_trainHard, name3=y_trainSoft, name4=x_test, name5=y_testHard, name6=y_testSoft)

    #container = np.load('Datasets.npz') # 2 load from the file

    return (x_train, y_trainHard, y_trainSoft),(x_test, y_testHard, y_testSoft)


if __name__ == '__main__':
    getDataset()