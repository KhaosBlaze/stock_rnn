import numpy as np
from subconscious import get_X_Y, build_Stanley, get_stanley, survey_says, get_a_symbol, save_it
# Importing the Keras libraries and packages

days_to_train_on = 45

#stanley = build_Stanley(90, 'tanh', 'sigmoid', 'adam', 'binary_crossentropy', days_to_train_on)
stanley = get_stanley("stanley2")

for i in [line.rstrip('\n') for line in open("stocks/list.txt")]:
    #Train the boi
    X_train, y_train = get_X_Y(i, days_to_train_on)
    # Fitting the RNN to the Training set
    stanley.fit(X_train, y_train, epochs=20, batch_size=32)

    symbol = get_a_symbol()
    temp_test, temp_y = get_X_Y(symbol, days_to_train_on)
    looped_test = stanley.predict(temp_test)
    np.savetxt('output2/' + symbol + '.out', looped_test, delimiter=',')
    save_it(stanley, "stanley2")

# Part 3 - Making the predictions and visualising the results
X_test, y_test = get_X_Y(get_a_symbol(), days_to_train_on)

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

np.savetxt('test2.out', predicted_stock_trend, delimiter=',')


