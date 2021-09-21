# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 12:19:31 2021

@author: MIDO
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datawig import SimpleImputer
from sklearn.metrics import mean_absolute_error

train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

train = train[train.social_volume > 751]
test = test[test.social_volume > 751]
#train = train.dropna(subset=['close'])
#train.isnull().sum()


# so social_volume always exists when close value exist , also data related to
# reddit are commonly exist , medium & youtube are almost rare
# we will impute features with other features , starting from social_volume feature we will predict the most 
# correlated feature to it , the we will have 2 complete features , we will impute the most correlated feature with 
# with these 2 and so on till we complete all missing values , except for youtube and medium , will not included

# processing data for imputuing using datawig
train_ = train.drop('asset_id',axis=1)
train_ = train_.drop('close',axis=1)
train_ = train_.drop('id',axis=1)

test_ = test.drop('asset_id',axis=1)
test_ = test_.drop('id',axis=1)


sorted_corr = train_.corr().sort_values(by='social_volume',ascending=False)


#################################
# tweets prediction

imputer = SimpleImputer(input_columns=['social_volume'],  # columns containing information about the column we want to impute
                        output_column='tweets',  # the column we'd like to impute values for
                        output_path='tweets_imputer')  # stores model data and metrics

# Fit an imputer model on the train data
imputer.fit(train_, num_epochs=5)

# Impute missing values and return original dataframe with predictions
test_imputed = imputer.predict(test_)


# Calculate MAE for test data
mae = mean_absolute_error(test_imputed.dropna(subset=['tweets'])['tweets'], 
                          test_imputed.dropna(subset=['tweets'])['tweets_imputed'])

train_imputed = imputer.predict(train_) # we will use it to fill other imputable columns

                   
mae
np.mean(test_imputed.tweets)
mae/np.mean(test_imputed.tweets)

#############################################################################################
# let's see who is the most correlated feature to both of social_volume and tweets
def imputer( train_imputed, sorted_corr, test_imputed):
    
# we will drop social_volume and the imputed columns and social_volume
    features_to_drop = [x[:-8] for x in train_imputed.columns if x[-7:] == 'imputed' ]
    features_to_drop.append('social_volume')
    
    features = []
    sum_of_corrs = []
    for i in range(sorted_corr.shape[0]):
        feature_name = sorted_corr.index[i]
        sum_of_corr = sum([abs(sorted_corr[i:i+1][feature]) for feature in features_to_drop])
        features.append(feature_name)
        sum_of_corrs.append(sum_of_corr)
    
    # creating a dataframe to select the most highly correlated feature with social_volume & imputed features
    df = pd.DataFrame(index=features)
    df['sum_of_corrs'] = [float(x) for x in sum_of_corrs]
    df = df.drop(index=features_to_drop)
    df = df.sort_values(by='sum_of_corrs',ascending=False)
    output_column = df.index[0] 
    
    ###############################
    input_columns = [x for x in train_imputed.columns if x[-7:] == 'imputed' ]
    input_columns.append('social_volume')
    
    output_path = output_column+'_imputer'
    
    imputer = SimpleImputer(input_columns = input_columns,  # columns containing information about the column we want to impute
                            output_column = output_column,  # the column we'd like to impute values for
                            output_path = output_path)  # stores model data and metrics
    
    # Fit an imputer model on the train data
    imputer.fit(train_imputed, num_epochs=5)
    
    # Impute missing values and return original dataframe with predictions
    test_imputed = imputer.predict(test_imputed)
    
    imputed_feature = output_column+'_imputed'
    
    # Calculate MAE for test data
    mae = mean_absolute_error(test_imputed.dropna(subset=[output_column])[output_column], 
                              test_imputed.dropna(subset=[output_column])[imputed_feature])
    
    train_imputed = imputer.predict(train_imputed) # we will use it to fill other imputable columns
    
    feature_mean = np.mean(test_imputed[output_column])
    mae_to_mean_value_ratio = mae/np.mean(test_imputed[output_column])
    
    return train_imputed,test_imputed , output_column, mae, feature_mean, mae_to_mean_value_ratio 


features = []
mae =[]
means_of_features = []
mae_to_mean_value_ratios = []

for i in range(44):

    results = imputer(train_imputed, sorted_corr,test_imputed)

    train_imputed = results[0]
    test_imputed = results[1]
    features.append(results[2])
    mae.append(results[3])
    means_of_features.append(results[4])
    mae_to_mean_value_ratios.append(results[5])
                      
# creating a data frame for MAEs for each feature
imputers_results = pd.DataFrame(mae,index=features,columns=['MAE'])
imputers_results['feature mean'] = means_of_features
imputers_results['Mae / feature mean '] = np.round(np.array(mae_to_mean_value_ratios)*100,decimals=0)
imputers_results = imputers_results.sort_values(by='Mae / feature mean ')

# adding close & id values  
train_imputed['close'] =   train['close']
train_imputed.insert(0,'id',train['id'])
train_imputed.to_csv('train_imputed.csv')

test_imputed.insert(0,'id',test['id'])
test_imputed.to_csv('test_imputed.csv')

imputers_results.to_csv('imputers_results.csv')
train_imputed.low_imputed.min()

#imputed_features = [ x for x in data.columns if x[-7:] == 'imputed']
#imputed_features.append('social_volume')

# firstly i will build the model based on the Mae / feature mean percent for each imputed features < 40%
#features_for_the_model = [ x +'_imputed' for x in imputers_results[imputers_results['Mae / feature mean '] < 40].index ]
#features_for_the_model.append('social_volume')
#features_for_the_model.append('tweets_imputed')
