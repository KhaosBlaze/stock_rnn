import numpy as np
import pandas as pd

# Importing the training set || Training on MSFT
dataset_train = pd.read_csv('stocks/msft.csv')

#reverse datatype indexing so it's from oldest to newest
dataset_train = dataset_train.reindex(index=dataset_train.index[::-1])
dataset_train = dataset_train.reset_index()
dataset_train = dataset_train.drop(columns=['index'])

#Setup input nodes to have closing of the previous day, opening, high/low of previous day and the closing of today (for the y value)
training_set = []
for i in range(1, len(dataset_train)):
    temp_set = []
    temp_set.append(dataset_train['close'][i-1])
    temp_set.append(dataset_train['open'][i])
    temp_set.append(dataset_train['high'][i-1])
    temp_set.append(dataset_train['low'][i-1])
    temp_set.append(dataset_train['close'][i])
    training_set.append(temp_set)
    
    
# Get that Min/Max scale on
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i - 60:i, 0:4])
    y_train.append(training_set_scaled[i, 4])
X_train, y_train = np.array(X_train), np.array(y_train)

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 4)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs=100, batch_size=30)

# Part 3 - Making the predictions and visualising the results


# Getting the AAON stock price
test_data = pd.read_csv('stocks/aaon.csv')
#Reverse again
test_data = test_data.reindex(index=test_data.index[::-1])
test_data = test_data.reset_index()
test_data = test_data.drop(columns=['index'])
#Format the same way as the training data
test_set = []
for i in range(1, len(test_data)):
    temp_set = []
    temp_set.append(test_data['close'][i-1])
    temp_set.append(test_data['open'][i])
    temp_set.append(test_data['high'][i-1])
    temp_set.append(test_data['low'][i-1])
    temp_set.append(test_data['close'][i])
    test_set.append(temp_set)
#once again with the mining and maxing
sct = MinMaxScaler(feature_range=(0, 1))
test_set_scaled = sct.fit_transform(test_set)
#60 days of data
X_test = []
y_test = []
for i in range(60, len(test_set_scaled)):
    X_test.append(test_set_scaled[i - 60:i, 0:4])
    y_test.append(test_set_scaled[i, 4])
X_test, y_test = np.array(X_test), np.array(y_test)
#cuz lazy
real_stock_price = y_test

#Get that trend of the actual stock data
real_stock_trend = []
for i in range(0, len(real_stock_price)):
    blargh = False
    if real_stock_price[i] > real_stock_price[i-1]:
        blargh = True
    real_stock_trend.append(blargh)

#PLace your bets boys    
predicted_stock_price = regressor.predict(X_test)

#Fucking won't transform out but technically not important
#predicted_stock_price = sct.inverse_transform(predicted_stock_price)

#Get the trend of if the day closed lower or higher
predicted_stock_trend = []
for i in range(1, len(predicted_stock_price)):
    blargh = False
    if predicted_stock_price[i] > predicted_stock_price[i-1]:
        blargh = True
    predicted_stock_trend.append(blargh)
    
#Remove first price because it's unloved and unwanted
real_stock_trend = real_stock_trend[1:]

#Manually count the times rnn was right
score = 0
for i in range(0, len(real_stock_trend)):
    if real_stock_trend[i] == predicted_stock_trend[i]:
        score += 1 

#Should be a percentage between 0 and 100
final_score = ((score / len(predicted_stock_trend)) * 100)
#Pretty number
print(str(final_score) + "%")

#May use for visualize, all the fucking variables need updating
# =============================================================================
# # Visualising the results
# plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
# plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
# plt.title('Google Stock Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('Google Stock Price')
# plt.legend()
# plt.show()
# =============================================================================





