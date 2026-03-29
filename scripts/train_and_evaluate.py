"""
Stock Price Direction + Next-Day Price Prediction (Stacked RandomForest + LSTM Hybrid)
--------------------------------------------------------------------------------------

Adds minimal extension to predict next-day close using best model.
"""

import os, datetime, numpy as np, pandas as pd, yfinance as yf, feedparser,  joblib
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


def fetch_sentiment_score(ticker: str):
    # ⚡ FAST fallback (no BERT, no API calls)
    import numpy as np
    return [np.zeros(10), np.zeros(10), np.zeros(10)]
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
def engineer_features(df: pd.DataFrame, sentiment_list):
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

    emb_df = pd.DataFrame({f"Sentiment_t{i}_{j}": [emb[j]]*len(df)
                           for i, emb in enumerate(sentiment_list)
                           for j in range(len(emb))})
    df = pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)

    # Direction + Price targets
    df["FutureReturn"] = df["Close"].shift(-3)/df["Close"] - 1
    df["Direction"] = np.where(df["FutureReturn"] > 0.002, 1,
                               np.where(df["FutureReturn"] < -0.002, 0, np.nan))
    df["NextClose"] = df["Close"].shift(-1)  # <-- added
    df.dropna(inplace=True)
    return df

# ---------- Sequence creation ----------
def create_sequences(df, feature_cols, target_col='Direction', seq_length=30):
    X, y = [], []
    data = df[feature_cols].values
    target = df[target_col].values
    for i in range(len(df) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(target[i + seq_length - 1])
    return np.array(X), np.array(y)


# ---------- RandomForest Direction ----------
def train_random_forest_stacking(df):
    feature_cols = [col for col in df.columns if col not in ["Date", "Direction", "FutureReturn","NextClose"]]
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    split_idx = -100
    X_train, X_test = df[feature_cols].values[:split_idx], df[feature_cols].values[split_idx:]
    y_train, y_test = df["Direction"].values[:split_idx], df["Direction"].values[split_idx:]
    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    train_probs = clf.predict_proba(X_train)[:,1].reshape(-1,1)
    test_probs  = clf.predict_proba(X_test)[:,1].reshape(-1,1)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1  = f1_score(y_test, preds, zero_division=0)
    print(f"\n📌 RF Direction: acc={acc:.3f}, prec={prec:.3f}, rec={rec:.3f}, f1={f1:.3f}")
    rf_metrics = {"accuracy":acc,"precision":prec,"recall":rec,"f1":f1}
    return train_probs, test_probs, rf_metrics

# ---------- LSTM Direction ----------
def train_lstm_stacking(df, rf_train_probs, rf_test_probs, seq_length=30):
    tf, _, _, _ = lazy_imports()
    feature_cols = ["Open","High","Low","Close","Volume",
                    "Return1","Return3","Volatility3","Volatility7","MA5","MA20",
                    "EMA12","EMA26","MACD","Signal","MACD_hist","RSI14","Momentum",
                    "VolSpike"]+[c for c in df.columns if c.startswith("Sentiment")]
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    X, y = create_sequences(df, feature_cols, 'Direction', seq_length)
    rf_full = np.concatenate([rf_train_probs, rf_test_probs], axis=0)
    rf_seq = np.array([np.tile(rf_full[i], (seq_length,1)) for i in range(len(X))])
    X = np.concatenate([X, rf_seq], axis=2)
    split_idx = -100
    X_train, X_test, y_train, y_test = X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw_dict = {0:cw[0], 1:cw[1]}
    inputs = tf.keras.Input(shape=(seq_length,X.shape[2]))
    x = tf.keras.layers.LSTM(128,return_sequences=True)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(64,return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Lambda(lambda t: tf.reduce_mean(t, axis=1))(x)
    x = tf.keras.layers.Dense(32,activation='relu')(x)
    outputs = tf.keras.layers.Dense(1,activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    print("\n🚀 Training Stacked LSTM (Direction)...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=50, batch_size=32, callbacks=[es], verbose=0)
    preds_prob = model.predict(X_test).flatten()
    preds = (preds_prob > 0.5).astype(int)
    acc=accuracy_score(y_test,preds);prec=precision_score(y_test,preds,zero_division=0)
    rec=recall_score(y_test,preds,zero_division=0);f1=f1_score(y_test,preds,zero_division=0)
    print(f"📌 LSTM Direction: acc={acc:.3f}, prec={prec:.3f}, rec={rec:.3f}, f1={f1:.3f}")
    lstm_metrics={"accuracy":acc,"precision":prec,"recall":rec,"f1":f1}

    last_pred_prob = preds_prob[-1]
    return lstm_metrics, last_pred_prob


# ---------- NEW: RandomForest & LSTM for Next-Day Price ----------
def train_price_models(df):
    tf, _, _, _ = lazy_imports()
    feature_cols=[c for c in df.columns if c not in ["Date","FutureReturn","Direction","NextClose"]]
    X=df[feature_cols]; y=df["NextClose"]
    split_idx=-100
    X_train,X_test,y_train,y_test=X.iloc[:split_idx],X.iloc[split_idx:],y.iloc[:split_idx],y.iloc[split_idx:]
    rf=RandomForestRegressor(n_estimators=200,random_state=42)
    rf.fit(X_train,y_train)
    rf_preds=rf.predict(X_test)
    rf_rmse=np.sqrt(mean_squared_error(y_test,rf_preds))
    rf_mae=mean_absolute_error(y_test,rf_preds)
    rf_r2=r2_score(y_test,rf_preds)
    rf_metrics={"RMSE":rf_rmse,"MAE":rf_mae,"R2":rf_r2}

    # Simple LSTM regression (same features scaled)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Combine scaled features + target into one DataFrame
    scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    scaled_df['NextClose'] = y.values

    seq_len = 10
    X_seq, y_seq = create_sequences(scaled_df, X.columns, 'NextClose', seq_len)

    split=int(0.8*len(X_seq))
    X_tr,X_te,y_tr,y_te=X_seq[:split],X_seq[split:],y_seq[:split],y_seq[split:]
    model=tf.keras.Sequential([
        tf.keras.layers.LSTM(64,return_sequences=True,input_shape=(seq_len,X_tr.shape[2])),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam',loss='mse')
    model.fit(X_tr,y_tr,validation_data=(X_te,y_te),epochs=20,batch_size=16,verbose=0)
    lstm_preds=model.predict(X_te).flatten()
    lstm_rmse=np.sqrt(mean_squared_error(y_te,lstm_preds))
    lstm_mae=mean_absolute_error(y_te,lstm_preds)
    lstm_r2=r2_score(y_te,lstm_preds)
    lstm_metrics={"RMSE":lstm_rmse,"MAE":lstm_mae,"R2":lstm_r2}

    # Pick better model (lower RMSE)
    best_model="random_forest" if rf_rmse<lstm_rmse else "lstm"
    last_seq=X_scaled[-seq_len:].reshape(1,seq_len,X_scaled.shape[1])
    next_price=float(model.predict(last_seq)[0][0]) if best_model=="lstm" else float(rf.predict([X.iloc[-1]])[0])

    print(f"\n💰 Price Prediction using {best_model.upper()} → Next-Day Close: {next_price:.2f}")
    return {"random_forest":rf_metrics,"lstm":lstm_metrics,
            "best_model":best_model,"predicted_next_day_close":next_price}

# ---------- Full pipeline ----------
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

def run_pipeline_stacked(ticker):
    print(f"\n🚀 Running FAST stacked pipeline for {ticker}...\n")

    df = fetch_stock_data(ticker, years=2)  # reduced years → faster
    sentiment_list = fetch_sentiment_score(ticker)
    df = engineer_features(df, sentiment_list)

    feature_cols = [c for c in df.columns if c not in ["Date","FutureReturn","Direction","NextClose"]]

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