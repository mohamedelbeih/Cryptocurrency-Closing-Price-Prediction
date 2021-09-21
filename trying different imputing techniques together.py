# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 19:31:41 2021

@author: MIDO
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

tt = test.drop(['id'],axis=1)
ttt = tt.fillna(0)
np.array(ttt).min()

# observations in the training data that has a missing close values
# i will try to keep its missing values = 0 , and for the other observation i will try to impute with 
# resonable values ( calculated by datawig simpleimputer as in preprocessing file)
train0 = train[train.social_volume <= 751]
test0 = test[test.social_volume <= 751]
train0 = train0.fillna(0)
test0 = test0.fillna(0)
train0 = train0.drop("asset_id",axis=1)
test0 = test0.drop("asset_id",axis=1)


train_imputed = pd.read_csv('train_imputed.csv')
test_imputed = pd.read_csv('test_imputed.csv')

train_imputed.columns = ['id', 'social_volume', 'tweets', 'tweet_sentiment4',
                         'social_score', 'tweet_followers',
                         'tweet_sentiment_impact4', 'tweet_sentiment1',
                         'tweet_sentiment2', 'tweet_sentiment5',
                         'tweet_sentiment3', 'tweet_sentiment_impact5',
                         'tweet_favorites', 'reddit_posts', 'url_shares',
                         'tweet_sentiment_impact2', 'market_cap_global',
                         'high', 'market_cap', 'open', 'low',
                         'tweet_spam', 'reddit_comments',
                         'unique_url_shares', 'volume_24h_rank',
                         'tweet_sentiment_impact3', 'tweet_sentiment_impact1',
                         'tweet_replies', 'volume', 'reddit_posts_score',
                         'tweet_quotes', 'news', 'volatility',
                         'tweet_retweets', 'percent_change_24h',
                         'correlation_rank', 'medium', 'galaxy_score',
                         'percent_change_24h_rank', 'average_sentiment',
                         'price_score', 'social_impact_score', 'youtube',
                         'reddit_comments_score', 'social_score_24h_rank',
                         'social_volume_24h_rank', 'market_cap_rank', 'close']
test_imputed.columns = ['id', 'social_volume', 'tweets', 'tweet_sentiment4',
                         'social_score', 'tweet_followers',
                         'tweet_sentiment_impact4', 'tweet_sentiment1',
                         'tweet_sentiment2', 'tweet_sentiment5',
                         'tweet_sentiment3', 'tweet_sentiment_impact5',
                         'tweet_favorites', 'reddit_posts', 'url_shares',
                         'tweet_sentiment_impact2', 'market_cap_global',
                         'high', 'market_cap', 'open', 'low',
                         'tweet_spam', 'reddit_comments',
                         'unique_url_shares', 'volume_24h_rank',
                         'tweet_sentiment_impact3', 'tweet_sentiment_impact1',
                         'tweet_replies', 'volume', 'reddit_posts_score',
                         'tweet_quotes', 'news', 'volatility',
                         'tweet_retweets', 'percent_change_24h',
                         'correlation_rank', 'medium', 'galaxy_score',
                         'percent_change_24h_rank', 'average_sentiment',
                         'price_score', 'social_impact_score', 'youtube',
                         'reddit_comments_score', 'social_score_24h_rank',
                         'social_volume_24h_rank', 'market_cap_rank']

train0 = train0[train_imputed.columns]
test0 = test0[test_imputed.columns]

##########concatenating
final_train = pd.concat([train_imputed,train0],axis=0)
final_test = pd.concat([test_imputed,test0],axis=0)
final_train.to_csv('final_train.csv')
final_test.to_csv('final_test.csv')


############################################
# checking the features correlation with each other , also with the close values
# as i believe that manipulating or imputing missing values should not affect to much on the original corr
# if it affect the corr , we may lose some information due to this manipulation. 
train_corr = final_train.corr()
test_corr = final_test.corr()
diff = np.round((np.array(test_corr)) / (np.array(train_corr)[:46,:46]))
diff = np.nan_to_num(diff,0)

num_of_matched_corr = [sum(diff[x] == 1) for x in range(len(diff)) ]
features = test_corr.columns
features_correlation = pd.DataFrame(num_of_matched_corr,columns=['num_of_matched_corr'],
                                    index = [features] )
features_correlation['corr with close'] = train_corr['close'][:-1].values
###############################################
X = final_train.iloc[:,1:-1].values 
y = final_train['close'].values
testt = final_test.iloc[:,1:].values
# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)
y_pred =  regressor.predict(testt)
