import pandas as pd
from scripts.train_and_evaluate import fetch_stock_data, fetch_sentiment_timeseries, engineer_features
ticker = "AAPL"
df = fetch_stock_data(ticker, years=2)
print("Stock data shape:", df.shape)
sentiment_dict = fetch_sentiment_timeseries(ticker)
print("Sentiment dict:", len(sentiment_dict))
df = engineer_features(df, sentiment_dict)
print("Engineered df shape:", df.shape)
