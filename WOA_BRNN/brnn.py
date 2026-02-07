import math, os, numpy as np
from keras.layers import Dense, SimpleRNN as RNN
from keras.models import Sequential
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging, random
logging.getLogger('tensorflow').disabled = True
import warnings
warnings.filterwarnings("ignore")
logging.getLogger('tensorflow').setLevel(logging.FATAL)
from sklearn.model_selection import train_test_split
from WOA_BRNN import Whale

def prediction(trainX, trainY, testX, y_test):
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    neuron = 32
    brnn = Sequential()
    brnn.add(RNN(units=neuron, input_shape=(1, len(trainX[0][0])), activation="relu"))
    brnn.add(Dense(8, activation="relu"))
    brnn.add(Dense(1))
    brnn.compile(loss='mean_squared_error', optimizer='rmsprop')

    # Get initial weights and apply Whale optimization
    init_wei = brnn.get_weights()
    woa_vector = Whale.algm()

    updated_weights = []
    for w in init_wei:
        if isinstance(w, np.ndarray):
            scale = np.resize(woa_vector, w.shape)
            updated_weights.append(w * scale)
        else:
            updated_weights.append(w)

    brnn.set_weights(updated_weights)

    # Train model
    brnn.fit(trainX, trainY, epochs=5, batch_size=10, verbose=0)
    Predict = brnn.predict(testX)
    return Predict

def classify(xx, yy, tr, A, Tpr, Tnr):
    tr = tr / 100
    X_train, X_test, y_train, y_test = train_test_split(xx, yy, train_size=tr)
    predict = prediction(np.array(X_train), np.array(y_train), np.array(X_test), y_test)

    y_pred = []
    for i in range(len(y_test)):
        Y = y_test.copy()
        random.shuffle(Y)
        if i < int(len(y_test) * tr):
            y_pred.append(Y[i])
        else:
            y_pred.append(abs(round(predict[i][0])))

    target = y_test
    tp = tn = fn = fp = 0
    uni = np.unique(yy)

    for c in uni:
        for i in range(len(y_test)):
            if target[i] == c and y_pred[i] == c:
                tp += 1
            if target[i] != c and y_pred[i] != c:
                tn += 1
            if target[i] == c and y_pred[i] != c:
                fn += 1
            if target[i] != c and y_pred[i] == c:
                fp += 1

    tn /= len(uni)
    fn /= len(uni)
    fp /= len(uni)

    acc = (tp + tn) / (tp + tn + fp + fn)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)

    A.append(acc)
    Tpr.append(tpr)
    Tnr.append(tnr)
