import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import optimizers

def get_X_Y(stock, days_to_train_on):
    # Importing the training set || Training on MSFT
    dataset = pd.read_csv('stocks/' + stock + '.csv')
    # reverse datatype indexing so it's from oldest to newest
    dataset = dataset.reindex(index=dataset.index[::-1])
    dataset = dataset.reset_index()
    dataset = dataset.drop(columns=['index'])

    training_set = []
    for i in range(1, len(dataset)):
        temp_set = []
        temp_set.append(dataset['close'][i - 1] - dataset['open'][i - 1])
        temp_set.append(dataset['high'][i - 1])
        temp_set.append(dataset['low'][i - 1])
        temp_set.append(dataset['open'][i])
        temp_set.append(dataset['volume'][i-1])
        if dataset['close'][i] > dataset['open'][i]:
            temp_set.append(1)
        else:
            temp_set.append(0)
        training_set.append(temp_set)

    scx = ColumnTransformer([("normies", MinMaxScaler(feature_range=(-1, 1)), slice(0, 1)),
                            ("normies_price", MinMaxScaler(feature_range=(0, 1)), slice(1, 4)),
                            ("normies_vol", MinMaxScaler(feature_range=(0, 1)), slice(4, 5))])

    training_set = np.array(training_set)

    X = []
    y = []
    for i in range(days_to_train_on, len(training_set)):
        x_temp = scx.fit_transform(training_set[i - days_to_train_on:i, 0:5])
        X.append(x_temp)
        y.append(training_set[i, 5])

    X, y = np.array(X), np.array(y).astype(int)
    return X, y


def survey_says(prediction, for_realz, confidence=.8):
    scores = {'reacts_correctly': 0, 'loss': 0, 'percentage': 0, 'skip': 0}
    score = 0
    did_not_guess = 0
    for i in range(0, len(prediction)):
        if prediction[i] < confidence:
            score += 1
            did_not_guess += 1
        elif for_realz[i] == 1:
            score += 1
        else:
            scores['loss'] += 1

    scores['reacts_correctly'] = score
    scores['percentage'] = ((score / len(prediction)) * 100)
    scores['skip'] = did_not_guess
    scores['success'] = scores['reacts_correctly'] - scores['skip']
    scores['pass to fail'] = (scores['success'] - scores['loss'])
    scores['datapoints'] = len(prediction)
    return scores


def get_a_symbol():
    all_of_em = [line.rstrip('\n') for line in open("stocks/list.txt")]
    all_of_em = all_of_em[:2300]
    temp = all_of_em[random.randrange(len(all_of_em))]
    print(temp)
    return str(temp)

def build_Stanley(hu, ha, oa, op, loss, dtt):#Build Stanley
    stanley = Sequential()
    stanley.add(LSTM(units=hu, activation=ha, return_sequences=True, input_shape=(dtt, 5)))
    stanley.add(Dropout(0.2))
    stanley.add(LSTM(units=hu, activation=ha, return_sequences=False))
    stanley.add(Dropout(0.2))
#    stanley.add(LSTM(units=hu, activation=ha, return_sequences=True))
#    stanley.add(Dropout(0.2))
#    stanley.add(LSTM(units=hu, activation=ha))
#    stanley.add(Dropout(0.2))
    stanley.add(Dense(units=1, activation=oa))
    op = optimizers.adam(lr = 0.007, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    stanley.compile(optimizer=op, loss=loss, metrics=['accuracy'])
    return stanley

def save_it(model, name="stanley"):
    model_json = model.to_json()
    with open(name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(name+".h5")
    print("Saved model to disk")

def get_stanley(name):
    with open(name+'.json', 'r') as json_file:
        model = model_from_json(json_file.read())
    model.load_weights(name+'.h5')
    op = optimizers.adam(lr = 0.007, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=op, loss="binary_crossentropy", metrics=['accuracy'])
    return model
