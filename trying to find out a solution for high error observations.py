import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

train = train.fillna(0)
test = test.fillna(0)

t = train.close.describe([0.88,0.89,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98])

X = train.iloc[:,2:-1].values
y = train['close'].values
testt = test.iloc[:,2:].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0 )

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred =  regressor.predict(X_test)

mean_absolute_error(y_test, y_pred)
residuals = abs(y_pred)- abs(y_test)
evaluating = pd.DataFrame(residuals,columns=['residuals'])
evaluating['y_test'] = y_test
evaluating['y_pred'] = y_pred
evaluating['percent'] = residuals/y_test*100
evaluating[train.columns[2:-1]] = X_test
evaluating['close'] = y_test
high_error_predictions = evaluating[evaluating.residuals > 100]
high_error_predictions_corr = high_error_predictions.corr()


train1 = train[train.open < 23038] # high error predictions
train2 = train[train.open > 23038]



X = train2.iloc[:,2:-1].values
y = train2['close'].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0 )

from sklearn.svm import SVR
regressor = SVR(kernel = 'linear')

regressor.fit(X_train, y_train)
y_pred =  regressor.predict(X_test)
mean_absolute_error(y_test, y_pred)

evaluating = evaluating[evaluating.open > 23038]
mean_absolute_error(evaluating.y_test,evaluating.y_pred)

# some observations are underestimated in the model because it have very low number of observation
# to fix this problem we may divide them to certain intervals , then giving each interval a ( a class name )
# then we can try to use oversampling technique to increase these underestimated observations 