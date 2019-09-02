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

def build_Stanley(hu, oa, loss, dtt):
    stanley = Sequential()
    stanley.add(LSTM(units=hu, return_sequences=True, input_shape=(dtt, 5)))
    stanley.add(LeakyReLU(alpha=0.3))
    stanley.add(Dropout(0.2))
    stanley.add(LSTM(units=hu, return_sequences=False))
    stanley.add(LeakyReLU(alpha=0.2))
    stanley.add(Dropout(0.2))
    stanley.add(Dense(units=1, activation=oa))
    #op = optimizers.adam(lr = 0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    op = optimizers.Adadelta(lr=0.9, rho=0.95, epsilon=None, decay=0.0)
    stanley.compile(optimizer=op, loss=loss, metrics=['accuracy'])
    return stanley

def get_stanley(name="leek"):
    with open(name+'.json', 'r') as json_file:
        model = model_from_json(json_file.read())
    model.load_weights(name+'.h5')
    op = optimizers.Adadelta(lr=0.9, rho=0.95, epsilon=None, decay=0.0)
    model.compile(optimizer=op, loss="binary_crossentropy", metrics=['accuracy'])
    return model

#stanley = build_Stanley(20, 'hard_sigmoid', 'binary_crossentropy', days_to_train_on)
stanley = get_stanley()

for i in all_of_em:
    X_train, y_train = get_X_Y(i, days_to_train_on)
    print(i)
    stanley.fit(X_train, y_train, epochs=20, batch_size=27)

    symbol = get_a_symbol()
    temp_test, temp_y = get_X_Y(symbol, days_to_train_on)
    looped = stanley.predict(temp_test)
    print(symbol)
    np.savetxt('output_leak/' + symbol + '.out', looped, delimiter=',')

    count += 1
    print(count)
    save_it(stanley, "leek")

# Part 3 - Making the predictions and visualising the results
X_test, y_test = get_X_Y(get_a_symbol(), days_to_train_on)

# PLace your bets boys
predicted_stock_trend = stanley.predict(X_test)

y_test = y_test[1:]

results = survey_says(predicted_stock_trend, y_test)

# Pretty number
print('How many data points: ' + str(results['datapoints']))
print('Times guessed correctly: ' + str(results['reacts_correctly']))
print('Percentage of correct guesses: ' + str(results['percentage']))
print('How many times Stnaley was wrong: ' + str(results['loss']))
print('How many times Stanley guessed up trend:' + str(results['success']))
print('How many times Stanley did not guess: ' + str(results['skip']))
print('Pass to Fail number, not a ratio, p shit metric, ngl:' + str(results['pass to fail']))

np.savetxt('leak.out', predicted_stock_trend, delimiter=',')


