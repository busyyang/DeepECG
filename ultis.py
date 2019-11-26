import os
import numpy as np
import scipy.io as sio
from keras.utils import np_utils


def load_data(path, len_limits=9000):
    bats = [f for f in os.listdir(path) if f.endswith('.mat')]
    mats = [f for f in bats if (np.shape(sio.loadmat(path + f)['val'])[1] >= len_limits)]
    check = np.shape(sio.loadmat(path + mats[0])['val'])[1]
    X = np.zeros((len(mats), check))
    for i in range(len(mats)):
        X[i, :] = sio.loadmat(path + mats[i])['val'][0, :len_limits]
    return X, mats


def to_one_hot(y):
    return np_utils.to_categorical(y)


if __name__ == '__main__':
    load_data('training2017', 9000)
