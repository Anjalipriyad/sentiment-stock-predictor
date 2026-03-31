"""
Stock Price Direction + Next-Day Price Prediction (Stacked RandomForest + LSTM Hybrid)
--------------------------------------------------------------------------------------

Adds minimal extension to predict next-day close using best model.
"""

import os, datetime, numpy as np, pandas as pd, yfinance as yf, feedparser,  joblib
from urllib import response
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
def lazy_imports():
    global tf, torch, AutoTokenizer, AutoModel
    import tensorflow as tf
    import torch
    from transformers import AutoTokenizer, AutoModel
    return tf, torch, AutoTokenizer, AutoModel


MODEL_DIR = "backend/app/ml_models/price_predictor"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- BERT for sentiment ----------
_tokenizer = None
_bert_model = None

_tokenizer = None
_bert_model = None
import requests
from backend.app.config import settings


def fetch_sentiment_timeseries(ticker: str):
    url = "https://newsapi.org/v2/everything"

    params = {
        "q": f"{ticker} stock",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": settings.NEWS_API_KEY
    }

    try:
        res = requests.get(url, params=params).json()
        articles = res.get("articles", [])

        if not articles:
            return {}

        # simple sentiment lexicon
        pos_words = ["gain", "rise", "up", "positive", "growth", "bull", "surge"]
        neg_words = ["fall", "down", "loss", "negative", "drop", "bear", "decline"]

        daily_scores = {}

        for a in articles:
            date = a.get("publishedAt", "")[:10]  # YYYY-MM-DD
            title = a.get("title", "").lower()

            score = sum(w in title for w in pos_words) - sum(w in title for w in neg_words)

            if date not in daily_scores:
                daily_scores[date] = []

            daily_scores[date].append(score)

        # average per day
        daily_avg = {d: sum(v)/len(v) for d, v in daily_scores.items()}

        # print("📊 Sample daily sentiment:", list(daily_avg.items())[:5])

        return daily_avg

    except Exception as e:
        print("❌ NewsAPI error:", e)
        return {}
def get_bert():
    global _tokenizer, _bert_model

    if _tokenizer is None:
        _, torch, AutoTokenizer, AutoModel = lazy_imports()

        _tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        _bert_model = AutoModel.from_pretrained("distilbert-base-uncased")
        _bert_model.eval()

    return _tokenizer, _bert_model



def get_bert_sentiment_features(texts, max_len=32):
    tokenizer, bert_model = get_bert()
    all_embeddings = []
    _, torch, _, _ = lazy_imports()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=max_len
            )
            outputs = bert_model(**inputs)
            # FIX: extract [CLS] token embedding and collect it
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            all_embeddings.append(embedding)
    # FIX: return mean embedding across all texts
    return np.mean(all_embeddings, axis=0) if all_embeddings else np.zeros(768)



# ---------- Fetch stock data ----------
def fetch_stock_data(ticker: str, years: int = 3):
    end = datetime.date.today()
    start = end - datetime.timedelta(days=years*365)
    df = yf.download(ticker, start=start, end=end, progress=False)
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df

# ---------- Feature engineering ----------
def engineer_features(df: pd.DataFrame, sentiment_dict):
    # convert date column
    df["Date"] = pd.to_datetime(df["Date"])
    df["date_str"] = df["Date"].dt.strftime("%Y-%m-%d")

# map sentiment per day
    df["Sentiment"] = df["date_str"].map(sentiment_dict)

# fill missing days
    df["Sentiment"].fillna(0, inplace=True)
    df["Return1"] = df["Close"].pct_change(1)
    df["Return3"] = df["Close"].pct_change(3)
    df["Volatility3"] = df["Return1"].rolling(3).std()
    df["Volatility7"] = df["Return1"].rolling(7).std()
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["Signal"]

    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI14"] = 100 - (100 / (1 + rs))

    df["Momentum"] = df["Close"] / df["MA20"] - 1
    df["VolSpike"] = df["Volume"] / df["Volume"].rolling(5).mean() - 1

    # emb_df = pd.DataFrame({f"Sentiment_t{i}_{j}": [emb[j]]*len(df)
    #                        for i, emb in enumerate(sentiment_list)
    #                        for j in range(len(emb))})
    # df = pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)

    # Direction + Price targets
    df["FutureReturn"] = df["Close"].shift(-3)/df["Close"] - 1
    df["Direction"] = np.where(df["FutureReturn"] > 0.002, 1,
                               np.where(df["FutureReturn"] < -0.002, 0, np.nan))
    df["NextClose"] = df["Close"].shift(-1)  # <-- added
    df.dropna(inplace=True)
    return df
# ---------- Full pipeline ----------
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

def run_pipeline_stacked(ticker):
    print(f"\n🚀 Running FAST stacked pipeline for {ticker}...\n")

    df = fetch_stock_data(ticker, years=2)  # reduced years → faster
    sentiment_dict = fetch_sentiment_timeseries(ticker)
    df = engineer_features(df, sentiment_dict)
    # print("📊 Sentiment features:", sentiment_dict)
    # df = engineer_features(df, sentiment_list)

    # feature_cols = [c for c in df.columns if c not in ["Date","FutureReturn","Direction","NextClose"]]
    feature_cols = [c for c in df.columns if c not in ["Date","date_str","FutureReturn","Direction","NextClose"]]
    X = df[feature_cols]
    y_dir = df["Direction"]
    y_price = df["NextClose"]

    split_idx = int(0.8 * len(df))

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train_dir, y_test_dir = y_dir[:split_idx], y_dir[split_idx:]
    y_train_price, y_test_price = y_price[:split_idx], y_price[split_idx:]

    # -------------------------------
    # ✅ SINGLE STACKED MODEL
    # -------------------------------
    clf = GradientBoostingClassifier(n_estimators=100)
    reg = GradientBoostingRegressor(n_estimators=100)

    clf.fit(X_train, y_train_dir)
    reg.fit(X_train, y_train_price)

    # Predictions
    dir_pred = clf.predict(X_test)
    dir_prob = clf.predict_proba(X_test)[-1][1]

    price_pred = reg.predict(X_test)
    next_price = float(price_pred[-1])

    # Metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    dir_metrics = {
        "stacked": {
            "accuracy": accuracy_score(y_test_dir, dir_pred),
            "precision": precision_score(y_test_dir, dir_pred, zero_division=0),
            "recall": recall_score(y_test_dir, dir_pred, zero_division=0),
            "f1": f1_score(y_test_dir, dir_pred, zero_division=0)
        }
    }

    price_metrics = {
        "stacked": {
            "MAE": mean_absolute_error(y_test_price, price_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test_price, price_pred)),
            "R2": r2_score(y_test_price, price_pred)
        }
    }

    predicted_direction = "up" if dir_prob > 0.5 else "down"

    return {
        "ticker": ticker,
        "best_model": "stacked",
        "predicted_price": next_price,
        "predicted_direction": predicted_direction,
        "direction_metrics": dir_metrics,
        "price_metrics": price_metrics
    }
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", type=str, required=True)
    args = p.parse_args()
    res = run_pipeline_stacked(args.ticker)
    print("\n✅ Final Results:\n", res)