# backend/app/routers/fetch_data.py

from fastapi import APIRouter, HTTPException
import yfinance as yf
import pandas as pd
import feedparser
from backend.app.routers.sentiment import get_bert_sentiment_features
from backend.app.config import settings
import requests
import pandas as pd
router = APIRouter(
    prefix="/fetch",
    tags=["FetchData"]
)

@router.get("/{ticker}")
def fetch_stock_data(ticker: str, years=2):

    """
    Fetch historical stock data. Primary: Finnhub, Fallback: yfinance.
    """
    # 1. Try Finnhub (requires API key)
    if settings.FINNHUB_API_KEY:
        try:
            url = "https://finnhub.io/api/v1/stock/candle"
            params = {
                "symbol": ticker,
                "resolution": "D",
                "count": 500,
                "token": settings.FINNHUB_API_KEY
            }
            response = requests.get(url, params=params)
            data = response.json()
            if data.get("s") == "ok":
                df = pd.DataFrame({
                    "Date": pd.to_datetime(data["t"], unit="s").strftime("%Y-%m-%d"),
                    "Open": data["o"],
                    "High": data["h"],
                    "Low": data["l"],
                    "Close": data["c"],
                    "Volume": data["v"]
                })
                # FastAPI cannot serialize DataFrames directly, must convert to dict
                return df.to_dict(orient="records")

        except Exception as e:
            print(f"Finnhub error for {ticker}: {e}")

    # 2. Fallback to yfinance (free, reliable, no key needed)
    try:
        data = yf.download(ticker, period=f"{years}y", interval="1d")
        if data.empty:
            raise Exception("No data returned from yfinance")
        
        # Reset index to make Date a column
        df = data.reset_index()
        
        # FastAPI/JSON cannot serialize pandas Timestamp objects, must convert to string
        if "Date" in df.columns:
            df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
            
        # Ensure column names are clean (yfinance sometimes adds multi-index levels)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        return df.to_dict(orient="records")
    except Exception as e:
        print(f"yfinance error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch stock data: {str(e)}")


@router.get("/news/{ticker}")
async def fetch_news(ticker: str, count: int = 10):
    """
    Fetch latest news headlines for a ticker using Google News RSS feed.
    """
    try:
        rss_url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(rss_url)
        entries = feed.entries[:count]

        news = [{"title": entry.title, "link": entry.link, "published": entry.published} for entry in entries]

        return {
            "status": "success",
            "ticker": ticker,
            "news": news
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
