import numpy as np
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras import optimizers
from keras.models import Sequential
from subconscious import get_X_Y, survey_says
days_to_train_on = 45

def build_Stanley(hu, oa, op, loss, dtt):
	stanley = Sequential()
	stanley.add(LSTM(units=hu, return_sequences=True, input_shape=(dtt, 5)))
	stanley.add(LeakyReLU(alpha=0.05))
	stanley.add(Dropout(0.2))
	stanley.add(LSTM(units=hu, return_sequences=False))
	stanley.add(LeakyReLU(alpha=0.05))
	stanley.add(Dropout(0.2))
	#    stanley.add(LSTM(units=hu, activation=ha))
	#    stanley.add(Dropout(0.2))
	stanley.add(Dense(units=2, activation=oa))
	op = optimizers.adam(lr = 0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	stanley.compile(optimizer=op, loss=loss, metrics=['accuracy'])
	return stanley

stanley = build_Stanley(7, 'softmax', 'adam', 'categorical_crossentropy', days_to_train_on)


X_train, y_train = get_X_Y('MSFT', days_to_train_on)

y_new = []
for i in y_train:
	if i == 1:
		y_new.append([1, 0])
	else:
		y_new.append([0, 1])
y_new = np.array(y_new)


stanley.fit(X_train, y_new, epochs=500, batch_size=32)

temp_test, temp_y = get_X_Y(symbol, days_to_train_on)
y = stanley.predict(temp_test)

# Part 3 - Making the predictions and visualising the results
X_test, y_test = get_X_Y('MSFT', days_to_train_on)

#PLace your bets boys    
predicted_stock_trend = stanley.predict(X_test)
    
#Remove first price because it's unloved and unwanted
real_stock_trend = []
for i in y_test:
	if i[0] == 1:
		new_stock_trend.append(1)
	else:
		new_stock_trend.append(0)
real_stock_trend = np.array(real_stock_trend)

results = survey_says(predicted_stock_trend, real_stock_trend)

#Pretty number
print('How many data points: ' + str(results['datapoints']))
print('Times guessed correctly: ' + str(results['reacts_correctly']))
print('Percentage of correct guesses: ' + str(results['percentage']))
print('How many times Stnaley was wrong: ' + str(results['loss']))
print('How many times Stanley guessed up trend:'+ str(results['success']))
print('How many times Stanley did not guess: ' +str(results['skip']))
print('Pass to Fail number, not a ratio, p shit metric, ngl:' + str(results['pass to fail']))

np.savetxt('single/temp.out', predicted_stock_trend, delimiter=',')
np.savetxt('single/y.out', y_test, delimiter=',')
