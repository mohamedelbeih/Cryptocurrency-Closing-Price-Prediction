# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 11:12:33 2021

@author: Mohamed Elbeih
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

data = pd.read_csv('train_imputed.csv')
test = pd.read_csv('test_imputed.csv')
imputers_results = pd.read_csv('imputers_results.csv',index_col=0)
np.mean(data['close'])
# firstly i will build the model based on the Mae / feature mean percent for each imputed features < 40%
features_for_the_model = [ x +'_imputed' for x in imputers_results[imputers_results['Mae / feature mean '] < 40].index ]
features_for_the_model.append('social_volume')
features_for_the_model.append('tweets_imputed')

x = data[features_for_the_model].values
y = data['close'].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state = 0)

# Feature scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

############################

import keras
from keras.models import Sequential
from keras.layers import Dense , Dropout
from keras.callbacks import EarlyStopping

regressor = Sequential()
regressor.add(Dense(units=200, kernel_initializer = 'uniform',activation='relu', input_dim=29))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 200, kernel_initializer = 'uniform', activation='relu'))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 200, kernel_initializer = 'uniform', activation='relu'))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1, kernel_initializer = 'uniform',activation='relu'))

regressor.compile(optimizer = 'adam', loss='mean_squared_error')

early_stop = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=25)

regressor.fit(x = x_train, y = y_train, validation_data=(x_test,y_test),
              callbacks=[early_stop],batch_size=512 , epochs = 1000)

######################################
test_data = test[features_for_the_model].values
test_data = sc.transform(test_data)
predictions = regressor.predict(test_data)

submission = pd.DataFrame(test.id)
submission['close'] = predictions
submission.to_csv('sub file.csv')
# the model couldn't converge to a good loss values , so we have a small data size for deep learning
####################################################


