import numpy as np
import random
from subconscious import get_X_Y, survey_says, save_it, get_stanley, get_a_symbol

days_to_train_on = 45

stanley = get_stanley("stanley")

def shuffle(array):
    random.shuffle(array)
    return array

all_of_em = [line.rstrip('\n') for line in open("stocks/list.txt")]
so_many_stocks = []
for i in range(0, 10):
    so_many_stocks += shuffle(all_of_em[:2300])

count = 0

for i in all_of_em:

    X_train, y_train = get_X_Y(i, days_to_train_on)
    stanley.fit(X_train, y_train, epochs=20, batch_size=32)

    symbol = get_a_symbol()
    temp_test, temp_y = get_X_Y(symbol, days_to_train_on)
    looped_test = stanley.predict(temp_test)
    np.savetxt('output/' + symbol + '.out', looped_test, delimiter=',')
    count += 1
    print(count)
    save_it(stanley)

# Part 3 - Making the predictions and visualising the results
X_test, y_test = get_X_Y(get_a_symbol(), days_to_train_on)

# PLace your bets boys
predicted_stock_trend = stanley.predict(X_test)

# Remove first price because it's unloved and unwanted
real_stock_trend = y_test

results = survey_says(predicted_stock_trend, real_stock_trend)

# Pretty number
print('How many data points: ' + str(results['datapoints']))
print('Times guessed correctly: ' + str(results['reacts_correctly']))
print('Percentage of correct guesses: ' + str(results['percentage']))
print('How many times Stnaley was wrong: ' + str(results['loss']))
print('How many times Stanley guessed up trend:' + str(results['success']))
print('How many times Stanley did not guess: ' + str(results['skip']))
print('Pass to Fail number, not a ratio, p shit metric, ngl:' + str(results['pass to fail']))

np.savetxt('test.out', predicted_stock_trend, delimiter=',')

