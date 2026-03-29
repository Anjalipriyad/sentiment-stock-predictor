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
    url = "https://finnhub.io/api/v1/stock/candle"

    params = {
        "symbol": ticker,
        "resolution": "D",
        "count": 500,
        "token": settings.FINNHUB_API_KEY  # ✅ from .env
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data.get("s") != "ok":
        raise Exception(f"Finnhub error: {data}")

    df = pd.DataFrame({
        "Date": pd.to_datetime(data["t"], unit="s"),
        "Open": data["o"],
        "High": data["h"],
        "Low": data["l"],
        "Close": data["c"],
        "Volume": data["v"]
    })

    return df
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
