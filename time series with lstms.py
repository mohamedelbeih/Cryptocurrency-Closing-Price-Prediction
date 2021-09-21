# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 02:42:01 2021

@author: MIDO
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

train = pd.read_csv('train_imputed.csv')
test = pd.read_csv('test_imputed.csv')


training_set = train['close'].values.reshape(-1,1)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

x_train = []
y_train = []
for i in range(100,8617):
    x_train.append(training_set_scaled[i-100:i,0])
    y_train.append(training_set_scaled[i,0])
x_train , y_train = np.array(x_train) ,  np.array(y_train)   

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

from keras.models import Sequential
from keras.layers import LSTM , Dense , Dropout 

# a good start point for the number of hidden units
#units = round(x_train.shape[0]/((x_train.shape[1]+1)*3))


regressor = Sequential()
regressor.add(LSTM(units = 80 , return_sequences = True ,  input_shape = (x_train.shape[1],1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 80 , return_sequences = True ))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 80 , return_sequences = True ))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 80))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1))
regressor.compile(optimizer='adam' , loss='mean_squared_error')

regressor.fit(x_train, y_train, epochs = 500 , batch_size = 64)


x_test = training_set_scaled[-100:,0]

#inputs = x_test[-20:]
#inputs = np.reshape(inputs, (1,inputs.shape[0],1))
#y_test = regressor.predict(inputs)
#x_test = np.append(x_test,y_test)

for i in range(0,6222):
    inputs = x_test[-100:]
    inputs = np.reshape(inputs, (1,inputs.shape[0],1))
    y_test = regressor.predict(inputs)
    x_test = np.append(x_test,y_test)
    

y_test = x_test[100:]
close_values = sc.inverse_transform(y_test.reshape(1,-1))
sub = pd.DataFrame(test.id)
sub['close'] = close_values.reshape(-1,1)
sub.to_csv('sub.csv',index=False)
# the model couldn't converge to a good loss values , so we have a small data size for deep learning
# we may also think that the data have a high volatility rate so it has no time series characteristics