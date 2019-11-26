import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
from os.path import isfile, join
import numpy as np
import keras
from sklearn.metrics import accuracy_score

from ultis import load_data, to_one_hot

number_of_classes = 4


def change(x):
    answer = np.zeros((np.shape(x)[0]))
    for i in range(np.shape(x)[0]):
        max_value = max(x[i, :])
        max_index = list(x[i, :]).index(max_value)
        answer[i] = max_index
    return answer.astype(np.int)


mypath = 'training2017/'
X, mats = load_data(mypath, 9000)

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

Label_set = to_one_hot(target_train)

train_len = 0.9
X_train = X[:int(train_len * len(mats)), :]
Y_train = Label_set[:int(train_len * len(mats)), :]
X_val = X[int(train_len * len(mats)):, :]
Y_val = Label_set[int(train_len * len(mats)):, :]

# reshape input to be [samples, time steps, features]
X_train = numpy.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_val = numpy.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))

# create and fit the LSTM network
batch_size = 64
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(1, 9000)))
# model.add(Dropout(0.25))
model.add(LSTM(256, return_sequences=True))
# model.add(Dropout(0.25))
model.add(LSTM(128, return_sequences=True))
# model.add(Dropout(0.25))
model.add(LSTM(64, return_sequences=True))
# model.add(Dropout(0.25))
model.add(LSTM(32))
model.add(Dense(number_of_classes, activation='softmax'))
model.summary()
early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=50, verbose=1, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=250, batch_size=batch_size, validation_data=(X_val, Y_val), verbose=2, shuffle=False,
          callbacks=[early_stopping])
# model.save('Keras_models/my_model_' + str(i) + '_' + str(j) + '_' + str() + '.h5')
predictions = model.predict(X_val)
score = accuracy_score(change(Y_val), change(predictions))
print(score)
# Data[i - starti, j - starti] = str(format(score, '.5f'))
# Output = pd.DataFrame(Data)
# name = str(batch_size) + '.csv'
# Output.to_csv(path_or_buf='Keras_models/' + name, index=None, header=None)
