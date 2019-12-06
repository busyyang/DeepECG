from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint
from biosppy.signals import ecg
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import pandas as pd
import scipy.io as sio
from os import listdir
from os.path import isfile, join
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten, LSTM
import keras
from keras import regularizers
from matplotlib import pyplot as plt

np.random.seed(7)

number_of_classes = 4
inputs = 60  # Previus value for 9k check is 95
size = 0


def change(x):  # Для получения чисел от 0 до 3
    answer = np.zeros((np.shape(x)[0]))
    for i in range(np.shape(x)[0]):
        max_value = max(x[i, :])
        max_index = list(x[i, :]).index(max_value)
        answer[i] = max_index
    return answer.astype(np.int)


file_list = listdir('./dense_data/')
file_list = [file for file in file_list if file.endswith('.npy')]
if len(file_list) < 2:
    mypath = 'training2017/'
    onlyfiles = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f[0] == 'A')]
    bats = [f for f in onlyfiles if f[7] == 'm']
    check = 3000
    mats = [f for f in bats if (np.shape(sio.loadmat(mypath + f)['val'])[1] >= check)]
    size = len(mats)
    print('Training size is ', len(mats))
    X = np.zeros((len(mats), check))
    for i in range(len(mats)):
        X[i, :] = sio.loadmat(mypath + mats[i])['val'][0, :check]

    target_train = np.zeros((len(mats), 1))
    Train_data = pd.read_csv(mypath + 'REFERENCE.csv', sep=',', header=None, names=None)
    for i in range(len(mats)):
        if Train_data.loc[Train_data[0] == mats[i][:6], 1].values == 'N':
            target_train[i] = 0
        elif Train_data.loc[Train_data[0] == mats[i][:6], 1].values == 'A':
            target_train[i] = 1
        elif Train_data.loc[Train_data[0] == mats[i][:6], 1].values == 'O':
            target_train[i] = 2
        else:
            target_train[i] = 3

    Label_set = np.zeros((len(mats), number_of_classes))
    for i in range(np.shape(target_train)[0]):
        dummy = np.zeros((number_of_classes))
        dummy[int(target_train[i])] = 1
        Label_set[i, :] = dummy

    X_new = np.zeros((size, inputs))
    for i in range(size):
        print(f'processing {i}/{size}......')
        out = ecg.christov_segmenter(signal=X[i, :], sampling_rate=300.)
        A = np.hstack((0, out[0][:len(out[0]) - 1]))
        B = out[0]
        dummy = np.lib.pad(B - A, (0, inputs - len(B)), 'constant', constant_values=(0))
        X_new[i, :] = dummy

    print('All is OK')
    X = X_new
    X = (X - X.mean()) / (X.std())
    Label_set = Label_set[:size, :]
    np.save('./dense_data/X.npy', X)
    np.save('./dense_data/Label_set.npy', Label_set)
else:
    X = np.load('./dense_data/X.npy')
    Label_set = np.load('./dense_data/Label_set.npy')


def train_and_evaluate__model(model, X_train, Y_train, X_val, Y_val):
    checkpointer = ModelCheckpoint(filepath='Dense_models/Best_model of ' + '.h5', monitor='val_acc',
                                   verbose=1, save_best_only=True)
    # early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=50, verbose=1, mode='auto')
    hist = model.fit(X_train, Y_train, epochs=500, batch_size=256, validation_data=(X_val, Y_val), verbose=2,
                     shuffle=True, callbacks=[checkpointer])
    pd.DataFrame(hist.history).to_csv(path_or_buf='Dense_models/History ' + '.csv')
    model.save('my_model ' + str(i) + '.h5')
    predictions = model.predict(X_val)
    score = accuracy_score(change(Y_val), change(predictions))
    print(score)
    df = pd.DataFrame(change(predictions))
    df.to_csv(path_or_buf='Dense_models/Preds_' + str(format(score, '.4f')) + '__' + '.csv', index=None,
              header=None)
    model.save('Dense_models/' + str(format(score, '.4f')) + '__' + '_my_model.h5')
    pd.DataFrame(confusion_matrix(change(Y_val), change(predictions))).to_csv(
        path_or_buf='Dense_models/Result_Conf' + str(format(score, '.4f')) + '__' + '.csv', index=None,
        header=None)


def create_model():
    model = Sequential()
    model.add(Dense(1024, input_shape=(inputs,), kernel_initializer='normal', activation='relu'))
    model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, kernel_initializer='normal', activation='relu'))
    model.add(Dense(512, kernel_initializer='normal', activation='relu'))
    model.add(Dense(512, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number_of_classes, kernel_initializer='normal', activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


per = np.random.permutation(X.shape[0])
train_index = per[:int(X.shape[0] * 0.9)]
test_index = per[int(X.shape[0] * 0.9):]

print("TRAIN:", train_index, "TEST:", test_index)
X_train = X[train_index, :]
Y_train = Label_set[train_index, :]
X_val = X[test_index, :]
Y_val = Label_set[test_index, :]
model = create_model()
train_and_evaluate__model(model, X_train, Y_train, X_val, Y_val)

