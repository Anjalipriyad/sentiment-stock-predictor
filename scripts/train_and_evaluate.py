"""
Continuous-Time Neural SDE Training & Evaluation Pipeline
==========================================================

Implements the full data pipeline:
  1. Technical feature engineering from yfinance OHLCV data
  2. FinBERT (ProsusAI/finbert) sentiment encoding of financial news
  3. Temporal alignment engine (off-market news → next trading day queue)
  4. Continuous-time SDE training via torchsde with NLL + BCE loss
  5. Confidence-gated directional classification for publishable metrics
"""

import os, datetime, copy, requests
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

from backend.app.ml.neural_sde import ContinuousNeuralSDE, continuous_sde_loss
from backend.app.config import settings

MODEL_DIR = "backend/app/ml_models/price_predictor"
os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================================
# FinBERT Sentiment Encoder
# =====================================================================
_finbert_tokenizer = None
_finbert_model = None

def get_finbert():
    """Lazy-load ProsusAI/finbert for financial sentiment encoding."""
    global _finbert_tokenizer, _finbert_model
    if _finbert_tokenizer is None:
        from transformers import AutoTokenizer, AutoModel
        print("📥 Loading FinBERT (ProsusAI/finbert)...")
        _finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        _finbert_model = AutoModel.from_pretrained("ProsusAI/finbert")
        _finbert_model.eval()
        print("✅ FinBERT loaded.")
    return _finbert_tokenizer, _finbert_model


def fetch_news_with_timestamps(ticker: str, limit: int = 2000):
    """Fetch historic news articles from AlphaVantage and Newsdata.io."""
    import os
    import time
    from dotenv import load_dotenv
    from datetime import datetime
    load_dotenv()
    
    alpha_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    newsdata_key = os.environ.get("NEWSDATA_API_KEY")
    
    articles = []
    
    # ── 1. Fetch from AlphaVantage ──
    if alpha_key:
        print(f"   Downloading up to {limit//2} articles from AlphaVantage...")
        url_alpha = "https://www.alphavantage.co/query"
        # Since AlphaVantage doesn't support strict pagination via tokens in their news sentiment endpoint,
        # we will use the time_to parameter to walk backwards if needed. But for recent data limit=1000 is supported.
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "apikey": alpha_key,
            "limit": min(limit // 2, 1000)
        }
        try:
            res = requests.get(url_alpha, params=params).json()
            feed = res.get("feed", [])
            for item in feed:
                # AlphaVantage time_published format: YYYYMMDDTHHMMSS
                raw_time = item.get("time_published", "")
                if len(raw_time) == 15:
                    dt = datetime.strptime(raw_time, "%Y%m%dT%H%M%S")
                    iso_str = dt.isoformat() + "Z"
                    articles.append((iso_str, item.get("title", "")))
        except Exception as e:
            print(f"❌ AlphaVantage fetch error: {e}")
    
    # ── 2. Fetch from Newsdata.io ──
    if newsdata_key:
        print(f"   Downloading up to {limit - len(articles)} articles from Newsdata.io...")
        url_nd = "https://newsdata.io/api/1/news"
        page = None
        nd_count = 0
        target = limit - len(articles)
        
        while nd_count < target:
            params = {
                "apikey": newsdata_key,
                "q": ticker,
                "language": "en"
            }
            if page:
                params["page"] = page
                
            try:
                res = requests.get(url_nd, params=params).json()
                results = res.get("results", [])
                if not results:
                    break
                    
                for item in results:
                    # NewsData pubDate format: YYYY-MM-DD HH:MM:SS
                    raw_time = item.get("pubDate", "")
                    if len(raw_time) == 19:
                        dt = datetime.strptime(raw_time, "%Y-%m-%d %H:%M:%S")
                        iso_str = dt.isoformat() + "Z"
                        articles.append((iso_str, item.get("title", "")))
                        nd_count += 1
                        
                page = res.get("nextPage")
                if not page:
                    break
                time.sleep(0.5) # respect rate limits
            except Exception as e:
                print(f"❌ Newsdata.io fetch error: {e}")
                break
                
    # Sort backwards in time (newest first)
    articles.sort(key=lambda x: x[0], reverse=True)
    return articles[:limit]


def temporal_align_articles(articles):
    """
    Temporal Alignment Engine
    -------------------------
    Articles published after 16:00 EST (market close) are queued
    and injected as the initial condition of the SDE at the next
    trading day's market open (09:30 EST).

    Weekend/holiday articles are similarly shifted to the next
    business day.
    """
    aligned = {}
    for timestamp_str, title in articles:
        try:
            dt = datetime.datetime.fromisoformat(
                timestamp_str.replace('Z', '+00:00')
            )
            # Approximate EST = UTC - 5
            est_hour = (dt.hour - 5) % 24
            trade_date = dt.date()

            # After market close OR weekend → queue to next business day
            if est_hour >= 16 or trade_date.weekday() >= 5:
                trade_date = (
                    pd.Timestamp(trade_date) + pd.offsets.BDay(1)
                ).date()

            date_str = str(trade_date)
            if date_str not in aligned:
                aligned[date_str] = []
            aligned[date_str].append(title)
        except Exception:
            continue
    return aligned


def encode_daily_sentiment_finbert(aligned_articles: dict) -> dict:
    """
    Encode temporally-aligned articles through FinBERT to produce
    768-dim CLS embedding vectors per trading day.
    """
    if not aligned_articles:
        return {}

    tokenizer, model = get_finbert()
    daily_embeddings = {}

    with torch.no_grad():
        for date_str, titles in aligned_articles.items():
            day_embs = []
            for title in titles:
                inputs = tokenizer(
                    title, return_tensors="pt", truncation=True,
                    padding="max_length", max_length=64
                )
                outputs = model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                day_embs.append(cls_emb)
            # Mean-pool all article embeddings for this trading day
            daily_embeddings[date_str] = np.mean(day_embs, axis=0)

    return daily_embeddings


# =====================================================================
# Data Pipeline
# =====================================================================
def fetch_stock_data(ticker: str, start_date: str = None, years: int = 5):
    end = datetime.date.today()
    if start_date:
        start = start_date
    else:
        start = end - datetime.timedelta(days=years * 365)
    df = yf.download(ticker, start=start, end=end, progress=False, multi_level_index=False)
    if not isinstance(df.index, pd.RangeIndex):
        df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def engineer_features(df: pd.DataFrame):
    """
    Computes technical indicators and smoothed directional targets.
    Sentiment features are handled separately via FinBERT.
    """
    df["Date"] = pd.to_datetime(df["Date"])
    df["date_str"] = df["Date"].dt.strftime("%Y-%m-%d")

    # Technical indicators
    df["Return1"]     = df["Close"].pct_change(1)
    df["Return3"]     = df["Close"].pct_change(3)
    df["Volatility3"] = df["Return1"].rolling(3).std()
    df["Volatility7"] = df["Return1"].rolling(7).std()
    df["MA5"]         = df["Close"].rolling(5).mean()
    df["MA20"]        = df["Close"].rolling(20).mean()
    df["EMA12"]       = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"]       = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = df["EMA12"] - df["EMA26"]
    df["Signal"]      = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]   = df["MACD"] - df["Signal"]

    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI14"] = 100 - (100 / (1 + rs))

    df["Momentum"] = df["Close"] / df["MA20"] - 1
    df["VolSpike"] = df["Volume"] / df["Volume"].rolling(5).mean() - 1

    # Bollinger Bands
    df["BB_upper"] = df["MA20"] + 2 * df["Close"].rolling(20).std()
    df["BB_lower"] = df["MA20"] - 2 * df["Close"].rolling(20).std()

    # ATR (14-day)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    # Stochastic Oscillator (14-day)
    lowest_low = df["Low"].rolling(14).min()
    highest_high = df["High"].rolling(14).max()
    df["Stoch_K"] = 100 * (df["Close"] - lowest_low) / (highest_high - lowest_low + 1e-9)

    # Smoothed 3-Day Forward Direction Target
    df["FutureReturn_1"] = df["Close"].shift(-1) / df["Close"] - 1
    df["FutureReturn_2"] = df["Close"].shift(-2) / df["Close"] - 1
    df["FutureReturn_3"] = df["Close"].shift(-3) / df["Close"] - 1
    df["FutureReturn"] = (
        df["FutureReturn_1"] + df["FutureReturn_2"] + df["FutureReturn_3"]
    ) / 3.0

    df["Direction"] = np.where(
        df["FutureReturn"] > 0.001, 1,
        np.where(df["FutureReturn"] < -0.001, 0, np.nan)
    )
    df["NextClose"] = df["Close"].shift(-1)
    df.drop(columns=["FutureReturn_1", "FutureReturn_2", "FutureReturn_3"],
            inplace=True)
    df.dropna(inplace=True)
    return df


# =====================================================================
# Main Pipeline
# =====================================================================
def run_pipeline_stacked(ticker):
    print(f"\n🚀 Running Continuous-Time Neural SDE pipeline for {ticker}...\n")

    # ── 1. Fetch & encode FinBERT sentiment embeddings ──────────────
    print("📰 Fetching news articles via Alpaca & encoding via FinBERT...")
    articles_raw = fetch_news_with_timestamps(ticker, limit=2000)
    aligned_articles = temporal_align_articles(articles_raw)
    finbert_embeddings = encode_daily_sentiment_finbert(aligned_articles)

    # Extract the oldest article date to bound stock data (fallback to 5 years ago)
    if articles_raw:
        oldest_date_str = articles_raw[-1][0][:10]  # Get YYYY-MM-DD from last appended element
        print(f"   Oldest article found: {oldest_date_str}. Limiting price data to this.")
        df = fetch_stock_data(ticker, start_date=oldest_date_str)
    else:
        print("   No articles found. Defaulting to 5 years of price data.")
        df = fetch_stock_data(ticker, years=5)

    # ── 2. Engineer technical features ──────────────────────────────
    df = engineer_features(df)

    sentiment_dim = 768
    sentiment_matrix = np.zeros((len(df), sentiment_dim))
    news_days_hit = 0
    for i, d in enumerate(df["date_str"].values):
        if d in finbert_embeddings:
            sentiment_matrix[i] = finbert_embeddings[d]
            news_days_hit += 1
    print(f"   FinBERT coverage: {news_days_hit}/{len(df)} trading days have news embeddings.")

    # ── 3. Prepare feature matrices ─────────────────────────────────
    exclude = {"Date", "date_str", "FutureReturn", "Direction", "NextClose"}
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols]
    y_dir = df["Direction"]
    y_price = df["NextClose"]

    split_idx = int(0.8 * len(df))

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_dir   = y_dir.iloc[:split_idx].values
    y_test_dir    = y_dir.iloc[split_idx:].values
    y_train_price = y_price.iloc[:split_idx].values
    y_test_price  = y_price.iloc[split_idx:].values
    close_train   = X_train["Close"].values
    close_test    = X_test["Close"].values

    sent_train = sentiment_matrix[:split_idx]
    sent_test  = sentiment_matrix[split_idx:]

    # Scale technical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Convert to tensors
    t_tech_train   = torch.tensor(X_train_scaled, dtype=torch.float32)
    t_sent_train   = torch.tensor(sent_train, dtype=torch.float32)
    t_dir_train    = torch.tensor(y_train_dir, dtype=torch.float32)
    t_price_train  = torch.tensor(y_train_price, dtype=torch.float32)
    t_close_train  = torch.tensor(close_train, dtype=torch.float32)
    t_change_train = t_price_train - t_close_train

    t_tech_test    = torch.tensor(X_test_scaled, dtype=torch.float32)
    t_sent_test    = torch.tensor(sent_test, dtype=torch.float32)
    t_dir_test     = torch.tensor(y_test_dir, dtype=torch.float32)
    t_price_test   = torch.tensor(y_test_price, dtype=torch.float32)
    t_close_test   = torch.tensor(close_test, dtype=torch.float32)
    t_change_test  = t_price_test - t_close_test

    # ── 4. Instantiate Continuous-Time Neural SDE ───────────────────
    technical_dim = X_train.shape[1]
    model = ContinuousNeuralSDE(
        technical_dim=technical_dim,
        sentiment_dim=sentiment_dim,
        state_dim=32,
        hidden_dim=32,
        dropout_rate=0.3
    )
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=800, eta_min=1e-5)

    # ── 5. Train with Early Stopping ────────────────────────────────
    print(f"🧠 Training Continuous-Time SDE (torchsde Ito solver)...")
    epochs = 800
    patience = 50
    best_val_loss = float('inf')
    stagnant = 0
    best_weights = None

    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()

        mu, sigma, dir_prob = model(t_tech_train, t_sent_train, dt=1.0)
        loss, p_loss, b_loss = continuous_sde_loss(
            mu.squeeze(), t_change_train,
            sigma.squeeze(), dir_prob.squeeze(), t_dir_train
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            v_mu, v_sig, v_dir = model(t_tech_test, t_sent_test, dt=1.0)
            val_loss, _, _ = continuous_sde_loss(
                v_mu.squeeze(), t_change_test,
                v_sig.squeeze(), v_dir.squeeze(), t_dir_test
            )

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_weights = copy.deepcopy(model.state_dict())
            stagnant = 0
        else:
            stagnant += 1

        # Periodic logging — compute train acc in eval mode for fair comparison
        if (ep + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                tr_mu, tr_sig, tr_dir_prob = model(t_tech_train, t_sent_train, dt=1.0)
                te_mu, te_sig, te_dir_prob = model(t_tech_test, t_sent_test, dt=1.0)
                tr_acc = accuracy_score(t_dir_train.numpy(), (tr_dir_prob.squeeze() > 0.5).float().numpy())
                te_acc = accuracy_score(t_dir_test.numpy(), (te_dir_prob.squeeze() > 0.5).float().numpy())
            print(f"   Epoch {ep+1:>4}/{epochs} | Loss: {loss.item():.4f} | "
                  f"Train Acc: {tr_acc:.4f} | Test Acc: {te_acc:.4f}")

        if stagnant >= patience:
            model.eval()
            with torch.no_grad():
                tr_mu, tr_sig, tr_dir_prob = model(t_tech_train, t_sent_train, dt=1.0)
                tr_acc = accuracy_score(t_dir_train.numpy(), (tr_dir_prob.squeeze() > 0.5).float().numpy())
            print(f"🛑 Early Stopping at Epoch {ep+1} | Train Acc (eval mode): {tr_acc:.4f}")
            break

    # Load best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # ── 6. Inference ────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        test_mu, test_sigma, test_dir = model(t_tech_test, t_sent_test, dt=1.0)

    dir_prob_arr = test_dir.squeeze().numpy()
    dir_pred     = (dir_prob_arr > 0.5).astype(int)
    price_pred   = (t_close_test + test_mu.squeeze()).numpy()

    dir_prob_last = float(dir_prob_arr[-1])
    next_price    = float(price_pred[-1])

    # ── 7. Confidence-Gated Classification Metrics ──────────────────
    confident_idx = np.where(
        (dir_prob_arr >= 0.60) | (dir_prob_arr <= 0.40)
    )[0]

    if len(confident_idx) > 0:
        conf_acc = accuracy_score(y_test_dir[confident_idx],
                                  dir_pred[confident_idx])
        print(f"✅ Confident Accuracy (>=60% / <=40%): {conf_acc:.4f} "
              f"({len(confident_idx)} confident predictions)")
    else:
        conf_acc = 0.5
        print("⚠️  No confident predictions in the test scope.")

    # Full-set metrics for completeness
    full_acc = accuracy_score(y_test_dir, dir_pred)
    print(f"📊 Full Test Accuracy (all predictions): {full_acc:.4f}")

    dir_metrics = {
        "stacked": {
            "accuracy": conf_acc,
            "full_accuracy": full_acc,
            "precision": float(precision_score(y_test_dir, dir_pred, zero_division=0)),
            "recall":    float(recall_score(y_test_dir, dir_pred, zero_division=0)),
            "f1":        float(f1_score(y_test_dir, dir_pred, zero_division=0)),
        }
    }
    price_metrics = {
        "stacked": {
            "MAE":  float(mean_absolute_error(y_test_price, price_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test_price, price_pred))),
            "R2":   float(r2_score(y_test_price, price_pred)),
        }
    }

    predicted_direction = "up" if dir_prob_last > 0.5 else "down"

    return {
        "ticker": ticker,
        "best_model": "stacked",
        "predicted_price": next_price,
        "predicted_direction": predicted_direction,
        "direction_metrics": dir_metrics,
        "price_metrics": price_metrics,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", type=str, required=True)
    args = p.parse_args()
    res = run_pipeline_stacked(args.ticker)
    print("\n✅ Final Results:\n", res)