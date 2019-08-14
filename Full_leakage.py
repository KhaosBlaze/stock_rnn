import random
import numpy as np
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras import optimizers
from keras.models import Sequential
from keras.models import model_from_json
from subconscious import get_X_Y, survey_says, get_a_symbol, save_it

days_to_train_on = 45

def shuffle(array):
    random.shuffle(array)
    return array

all_of_em = [line.rstrip('\n') for line in open("stocks/list.txt")]
so_many_stocks = []
for i in range(0, 10):
    so_many_stocks += shuffle(all_of_em[:2300])

count = 0


def y_to_y_new(arr):
    y_new = []
    for i in arr:
        if i == 1:
            y_new.append([1, 0])
        else:
            y_new.append([0, 1])
    y_new = np.array(y_new)
    return y_new

def y_new_to_y(arr):
    y_test = []
    for i in arr:
        if i[0] > i[1]:
            y_test.append(1)
        else:
            y_test.append(0)
    y_test = np.array(y_test)
    return y_test

def build_Stanley(hu, oa, loss, dtt):
    stanley = Sequential()
    stanley.add(LSTM(units=hu, return_sequences=True, input_shape=(dtt, 5)))
    stanley.add(LeakyReLU(alpha=0.3))
    stanley.add(Dropout(0.2))
    stanley.add(LSTM(units=hu, return_sequences=False))
    stanley.add(LeakyReLU(alpha=0.2))
    stanley.add(Dropout(0.2))
    #    stanley.add(LSTM(units=hu, activation=ha))
    #    stanley.add(Dropout(0.2))
    stanley.add(Dense(units=2, activation=oa))
    #op = optimizers.adam(lr = 0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    op = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    stanley.compile(optimizer=op, loss=loss, metrics=['accuracy'])
    return stanley


stanley = build_Stanley(10, 'hard_sigmoid', 'categorical_crossentropy', days_to_train_on)

for i in all_of_em:
    X_train, y_train = get_X_Y(i, days_to_train_on)
    y_train = y_to_y_new(y_train)

    stanley.fit(X_train, y_train, epochs=20, batch_size=27)

    symbol = get_a_symbol()
    temp_test, temp_y = get_X_Y(symbol, days_to_train_on)
    looped_test = y_new_to_y(stanley.predict(temp_test))

    np.savetxt('output_leak/' + symbol + '.out', looped_test, delimiter=',')

    count += 1
    print(count)
    save_it(stanley, "leek")

# Part 3 - Making the predictions and visualising the results
X_test, y_test = get_X_Y(get_a_symbol(), days_to_train_on)

# PLace your bets boys
predicted_stock_trend = stanley.predict(X_test)

# Remove first price because it's unloved and unwanted
real_stock_trend = y_new_to_y(y_test)
predicted_stock_tredn = y_new_to_y(predicted_stock_trend)

results = survey_says(predicted_stock_trend, real_stock_trend)

# Pretty number
print('How many data points: ' + str(results['datapoints']))
print('Times guessed correctly: ' + str(results['reacts_correctly']))
print('Percentage of correct guesses: ' + str(results['percentage']))
print('How many times Stnaley was wrong: ' + str(results['loss']))
print('How many times Stanley guessed up trend:' + str(results['success']))
print('How many times Stanley did not guess: ' + str(results['skip']))
print('Pass to Fail number, not a ratio, p shit metric, ngl:' + str(results['pass to fail']))

np.savetxt('leak.out', predicted_stock_trend, delimiter=',')


