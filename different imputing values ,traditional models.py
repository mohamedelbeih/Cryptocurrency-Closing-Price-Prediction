# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:22:00 2021

@author: MIDO
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
train = pd.read_csv('final_train.csv')
test = pd.read_csv('final_test.csv')


train.info()
train.isnull().sum()
sns.barplot(train.isnull().sum().values , train.isnull().sum().index)
plt.yticks(size = 5)
train.isnull().sum()
sns.barplot(train.isnull().sum().values , train.isnull().sum().index)
plt.yticks(size = 5)
test.isnull().sum()
sns.barplot(test.isnull().sum().values , test.isnull().sum().index)
plt.yticks(size = 5)
#########################################################################
train_corr = train.corr()
test_corr = test.corr()
diff = np.round((np.array(test_corr)) / (np.array(train_corr)[:47,:47]))
diff = np.nan_to_num(diff,0)

num_of_matched_corr = [sum(diff[x] == 1) for x in range(len(diff)) ]
features = test_corr.columns
features_correlation = pd.DataFrame(num_of_matched_corr,columns=['num_of_matched_corr'],
                                    index = [features] )
features_correlation['corr with close'] = train_corr['close'][:-1].values
###############################################
train.tweet_replies = train.tweet_replies.fillna(method='backfill',axis =0) # ffill or backfill
test.tweet_replies = test.tweet_replies.fillna(method='backfill',axis =0) # ffill or backfill

train.tweet_replies = train.tweet_replies.interpolate(method='from_derivatives',axis =0)#axis = 0,1
test.tweet_replies = test.tweet_replies.interpolate(method='from_derivatives',axis =0)#axis = 0,1

train.close = train.close.fillna(0)
train.close.mean()
test.tweet_replies = test.tweet_replies.fillna(545)

train = train.fillna(0)
test = test.fillna(0)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imputer = IterativeImputer(random_state=0)
train.iloc[:,1:-1] = imputer.fit_transform(train.iloc[:,1:-1])
test.iloc[:,1:] = imputer.transform(test.iloc[:,1:])

close_imputer = IterativeImputer(random_state=0)
train.close = close_imputer.fit_transform(train.iloc[:,1:])[:,-1]
########################################################################3
train = pd.read_csv('train_imputed.csv')
test = pd.read_csv('test_imputed.csv')

imputed_features = [ x for x in train.columns if x[-7:] == 'imputed']
imputed_features.append('social_volume')

######################################################

train = train.drop(['percent_change_24h_rank','correlation_rank','social_volume_24h_rank',
                    'price_score','galaxy_score','market_cap_rank','social_score_24h_rank',
                    'social_impact_score','average_sentiment','news'],axis=1)
test = test.drop(['percent_change_24h_rank','correlation_rank','social_volume_24h_rank',
                    'price_score','galaxy_score','market_cap_rank','social_score_24h_rank',
                    'social_impact_score','average_sentiment','news'],axis=1)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0 )

# Training the Multiple Linear Regression model on the Training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
# Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'linear')
# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X_train[:24])
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train[:24])
y_pred = lin_reg_2.predict(poly_reg.fit_transform(X_test[:24]))
import xgboost as xgb
regressor = xgb.XGBRFRegressor()


X = train.iloc[:,1:-17].values
y = train['close'].values
testt = test.iloc[:,1:-16].values
# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)
y_pred =  regressor.predict(testt)



np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# Evaluating the Model Performance
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X, y = y, scoring='neg_mean_absolute_error', cv = 10)
print("Accuracy: {:.2f} ".format(accuracies.mean()))
print("Standard Deviation: {:.2f} %".format(accuracies.std()))
