# backend/app/routers/fetch_data.py

from fastapi import APIRouter, HTTPException
import yfinance as yf
import pandas as pd
import feedparser
from backend.app.routers.sentiment import get_bert_sentiment_features

router = APIRouter(
    prefix="/fetch",
    tags=["FetchData"]
)

@router.get("/{ticker}")
async def fetch_stock_data(ticker: str, years: int = 3):
    """
    Fetch historical stock data (OHLCV) for the given ticker.
    """
    try:
        end = pd.Timestamp.today()
        start = end - pd.Timedelta(days=years*365)
        df = yf.download(ticker, start=start, end=end, progress=False)
        df.reset_index(inplace=True)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # Convert dates to string for JSON serialization
        df['Date'] = df['Date'].astype(str)

        return {
            "status": "success",
            "ticker": ticker,
            "data": df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ticker}/news")
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
