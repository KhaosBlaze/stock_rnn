import numpy as np
import pandas as pd
import random
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

days_to_train_on = 60

def get_X_Y(stock):

    # Importing the training set || Training on MSFT
    dataset = pd.read_csv('stocks/' + stock + '.csv')
    #reverse datatype indexing so it's from oldest to newest
    dataset = dataset.reindex(index=dataset.index[::-1])
    dataset = dataset.reset_index()
    dataset = dataset.drop(columns=['index'])

    #Setup input nodes to have closing of the previous day, opening, high/low of previous day and the closing of today (for the y value)
    training_set = []
    for i in range(1, len(dataset)):
        temp_set = []
        temp_set.append(dataset['close'][i-1] - dataset['open'][i-1])
        temp_set.append(dataset['high'][i-1])
        temp_set.append(dataset['low'][i-1])
        temp_set.append(dataset['open'][i])
        if dataset['close'][i] > dataset['open'][i]:
            temp_set.append(1)
        else:
            temp_set.append(0)
        training_set.append(temp_set)
        

    sc = ColumnTransformer(
        [("normies", MinMaxScaler(feature_range=(-1, 1)), slice(0,4)),
         ("normie2", MinMaxScaler(feature_range=(0, 1)), slice(4,5))])
    training_set_scaled = sc.fit_transform(training_set)

    # Creating a data structure with days_to_train_on timesteps and 1 output
    X = []
    y = []
    for i in range(days_to_train_on, len(training_set_scaled)):
        X.append(training_set_scaled[i - days_to_train_on:i, 0:4].tolist())
        y.append(training_set_scaled[i, 4])

    X, y = np.array(X), np.array(y).astype(int)
    np.savetxt('output/'+stock+'.out', y, delimiter=',')
    return X, y

def survey_says(prediction, for_realz, confidence=.8):
    scores = {'reacts_correctly':0, 'loss':0, 'percentage':0, 'skip':0}
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
    all_of_em = all_of_em[:1410]
    temp = all_of_em[random.randrange(len(all_of_em))]
    print(temp)
    return str(temp)

#Build Stanley
stanley = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
stanley.add(LSTM(units=60, activation='elu', return_sequences=True, input_shape=(days_to_train_on, 4)))
stanley.add(Dropout(0.25))
# Adding a second LSTM layer and some Dropout regularisation
stanley.add(LSTM(units=60, activation='elu', return_sequences=True))
stanley.add(Dropout(0.3))
# Adding a third LSTM layer and some Dropout regularisation
stanley.add(LSTM(units=60, activation='elu', return_sequences=True))
stanley.add(Dropout(0.25))
# Adding a fourth LSTM layer and some Dropout regularisation
stanley.add(LSTM(units=60, activation='elu'))
stanley.add(Dropout(0.3))
# Adding the output layer
stanley.add(Dense(units=1, activation='sigmoid'))
# Compiling the RNN
stanley.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])

x1, y1 = get_X_Y(get_a_symbol())
stanley.fit(x1, y1, epochs=3, batch_size=60)

for i in range(0, 10):
    #Train the boi
    X_train, y_train = get_X_Y(get_a_symbol())
    # Fitting the RNN to the Training set
    stanley.fit(X_train, y_train, epochs=30, batch_size=32)

# Part 3 - Making the predictions and visualising the results
X_test, y_test = get_X_Y(get_a_symbol())

#PLace your bets boys    
predicted_stock_trend = stanley.predict(X_test)
    
#Remove first price because it's unloved and unwanted
real_stock_trend = y_test

results = survey_says(predicted_stock_trend, real_stock_trend)

#Pretty number
print('How many data points: ' + str(results['datapoints']))
print('Times guessed correctly: ' + str(results['reacts_correctly']))
print('Percentage of correct guesses: ' + str(results['percentage']))
print('How many times Stnaley was wrong: ' + str(results['loss']))
print('How many times Stanley guessed up trend:'+ str(results['success']))
print('How many times Stanley did not guess: ' +str(results['skip']))
print('Pass to Fail number, not a ratio, p shit metric, ngl:' + str(results['pass to fail']))

np.savetxt('test.out', predicted_stock_trend, delimiter=',')

