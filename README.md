# Cryptocurrency-Closing-Price-Prediction

check it here @ Zindi https://zindi.africa/competitions/cryptocurrency-closing-price-prediction

my work will be uploaded in the repo when the competition ends @ 20-9-2021

This is a comprehensive dataset that captures the prices of a cryptocurrency along with the various features including social media attributes, trading attributes and time related attributes that were noted on an hourly basis during several months and that contribute directly or indirectly to the cryptocurrency volatile prices change.

A starter notebook is provided to help you make your first submission and land on the leaderboard.

Files available for download:

Train.csv - contains the target. This is the dataset that you will use to train your model.
Test.csv- resembles Train.csv but without the target-related columns. This is the dataset on which you will apply your model to.
SampleSubmission.csv - shows the submission format for this competition, with the ‘id’ column mirroring that of Test.csv and the close column containing your predictions. The order of the rows does not matter, but the names of the id must be correct.
Variable definitions

asset_id: An asset ID. We refer to all supported cryptocurrencies as assets

open: Open price for the time period

close: Close price for the time period

high: Highest price of the time period

low: Lowest price of the time period

volume: Number of tweets

market_cap: Total available supply multiplied by the current price in USD

url_shares: Every time an identified relevant URL is shared within relevant social posts that contain relevant terms

unique_url_shares: Number of unique url shares posted and collected on social media

reddit_posts: Number of latest Reddit posts for supported coins

reddit_posts_score: Reddit Karma score on individual posts

reddit_comments: Comments on Reddit that contain relevant terms

Reddit_comments_score: Reddit Karma score on comments

tweets: Number of crypto-specific tweets based on tuned search and filtering criteria

tweet_spam: Number of tweets classified as spam

tweet_followers: Number of followers on selected tweets

tweet_quotes: Number of quotes on selected tweets

tweet_retweets: Number of retweets of selected tweets

tweet_replies: Number of replies on selected tweets

tweet_favorites: Number of likes on an individual social post that contains a relevant term

tweet_sentiment1: Number of tweets which has a sentiment of “very bullish”

tweet_sentiment2: Number of tweets which has a sentiment of “bullish”

tweet_sentiment3: Number of tweets which has a sentiment of “neutral”

tweet_sentiment4: Number of tweets which has a sentiment of “bearish”

tweet_sentiment5: Number of tweets which has a sentiment of “very bearish”

tweet_sentiment_impact1: “Very bearish” sentiment impact

tweet_sentiment_impact2: “Bearish” sentiment impact

tweet_sentiment_impact3: “Neutral” sentiment impact

tweet_sentiment_impact4: “Bullish” sentiment impact

tweet_sentiment_impact5: “Very bullish” sentiment impact

social_score: Sum of followers, retweets, likes, reddit karma etc of social posts collected

average_sentiment: The average score of sentiments, an indicator of the general sentiment being spread about a coin

news: Number of news articles for supported coins

price_score: A score we derive from a moving average that gives the coin some indication of an upward or downward based solely on the market value

social_impact_score: A score of the volume/interaction/impact of social to give a sense of the size of the market or awareness of the coin

correlation_rank: The algorithm that determines the correlation of our social data to the coin price/volume

galaxy_score: An indicator of how well a coin is doing

volatility: Volatility indicator

market_cap_rank: The rank based on the total available supply multiplied by the current price in USD

percent_change_24h_rank: The rank based on the percent change in price since 24 hours ago

volume_24h_rank: The rank based on volume in the last 24 hours

social_volume_24h_rank: The rank based on the number of social posts that contain relevant terms in the last 24 hours

social_score_24h_rank: The rank based on the sum of followers, retweets, likes, reddit karma etc of social posts collected in the last 24 hours

medium: Number of Medium articles for supported coins

youtube: Number of videos with description that contains relevant terms

social_volume: Number of social posts that contain relevant terms

price_btc: Exchange rate with another coin

market_cap_global: Total available supply multiplied by the current price in USD

percent_change_24h: Percent change in price since 24 hours ago
