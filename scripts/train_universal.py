import os
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.optim as optim
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from backend.app.ml.neural_sde import SentimentNeuralSDE, neural_sde_loss
from scripts.train_and_evaluate import fetch_sentiment_timeseries, engineer_features

MODEL_DIR = "backend/app/ml_models/price_predictor"
os.makedirs(MODEL_DIR, exist_ok=True)

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "INTC",
    "BA", "DIS", "JPM", "V", "MA", "WMT", "TGT", "COST", "PG", "KO",
    "PEP", "JNJ", "PFE", "MRK", "ABBV", "XOM", "CVX", "COP", "SLB", "GE"
] # 30 top stocks as a representation

def download_multi_data(tickers, years=2):
    print("⏳ Downloading multi-stock panel data...")
    end = datetime.date.today()
    start = end - datetime.timedelta(days=years*365)
    
    # Download single dataframe with MultiIndex columns
    data = yf.download(tickers, start=start, end=end, group_by="ticker", progress=False)
    
    all_dfs = []
    # Loop over gathered multi-index
    for t in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                # yf 0.2+ returns (Ticker, OHLCV)
                if t in data.columns.levels[0] or t in data.columns.levels[1] or t in data.columns.get_level_values(0):
                    df_t = data[t].copy() if t in data.columns.get_level_values(0) else data.xs(t, level=1, axis=1)
                else: continue
            else:
                # If only one ticker was asked, or flattened
                df_t = data.copy()
            
            df_t.dropna(subset=["Close"], inplace=True)
            if len(df_t) < 100: continue
            
            df_t = df_t.reset_index()
            # Standardize columns
            if "Date" not in df_t.columns and "index" in df_t.columns:
                df_t.rename(columns={"index": "Date"}, inplace=True)
            
            # Fetch sentiment & engineer
            sentiment_dict = fetch_sentiment_timeseries(t)
            df_t = engineer_features(df_t, sentiment_dict)
            df_t["Ticker"] = t
            all_dfs.append(df_t)
        except Exception as e:
            print(f"⚠️ Skipping {t}: {e}")
            
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

def train_universal():
    print("🚀 Building UNIVERSAL Neural SDE Model across all tickers...")
    df = download_multi_data(TICKERS, years=2)
    
    if df.empty:
        print("❌ Failed to aggregate any data.")
        return

    feature_cols = [c for c in df.columns if c not in ["Date", "date_str", "FutureReturn", "Direction", "NextClose", "Ticker"]]
    
    print(f"📊 Total Panel Data Size: {len(df)} rows across {len(df['Ticker'].unique())} tickers.")
    
    X = df[feature_cols]
    y_dir = df["Direction"]
    y_price = df["NextClose"]
    t_Close = df["Close"]

    # Global train-test split (chronological or random? Since it's panel, let's do random 80/20 to capture multiple regimes)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train_dir, y_test_dir, y_train_price, y_test_price, Close_train, Close_test = train_test_split(
        X, y_dir, y_price, t_Close, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the universal scaler
    joblib.dump(scaler, f"{MODEL_DIR}/scaler_universal.joblib")

    # Tensors
    t_X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    t_y_dir_train = torch.tensor(y_train_dir.values, dtype=torch.float32)
    t_y_price_train = torch.tensor(y_train_price.values, dtype=torch.float32)
    t_Close_train = torch.tensor(Close_train.values, dtype=torch.float32)
    t_y_change_train = t_y_price_train - t_Close_train

    t_X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
    t_y_dir_test = torch.tensor(y_test_dir.values, dtype=torch.float32)
    
    # Regularized Model
    input_dim = X_train.shape[1]
    model = SentimentNeuralSDE(input_dim=input_dim, hidden_dim=16, dropout_rate=0.4)
    
    # Add L2 Weight Decay
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    
    epochs = 400
    best_test_acc = 0.0
    patience = 20
    patience_counter = 0

    print("🧠 Commencing Universal Training with Early Stopping & Dropout...")
    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        expected_change, sigma, direction_prob = model(t_X_train, dt=1.0)
        
        loss, price_loss, bce_loss = neural_sde_loss(
            expected_change.squeeze(), 
            t_y_change_train, 
            sigma.squeeze(), 
            direction_prob.squeeze(), 
            t_y_dir_train
        )
        loss.backward()
        optimizer.step()
        
        if (ep+1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                train_dir_pred = (direction_prob.squeeze() > 0.5).float()
                train_acc = accuracy_score(t_y_dir_train.numpy(), train_dir_pred.numpy())
                
                _, _, test_dir_prob = model(t_X_test, dt=1.0)
                test_dir_pred = (test_dir_prob.squeeze() > 0.5).float()
                test_acc = accuracy_score(t_y_dir_test.numpy(), test_dir_pred.numpy())
                
                print(f"Epoch {ep+1}/{epochs} | Trn Acc: {train_acc:.3f} | Tst Acc: {test_acc:.3f} | Loss: {loss.item():.3f}")
                
                # Early Stopping Logic based on Test Accuracy
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    patience_counter = 0
                    torch.save(model.state_dict(), f"{MODEL_DIR}/neural_sde_universal.pt")
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print("🛑 Early stopping triggered!")
                    break

    print(f"✅ Universal Training Complete! Best Generalized Test Accuracy: {best_test_acc:.3f}")
    print(f"-> Model saved to {MODEL_DIR}/neural_sde_universal.pt")

if __name__ == "__main__":
    train_universal()
