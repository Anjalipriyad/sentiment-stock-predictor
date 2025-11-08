# backend/app/routers/sentiment.py

from fastapi import APIRouter, HTTPException
import torch
from transformers import AutoTokenizer, AutoModel
import feedparser
import numpy as np

router = APIRouter(
    prefix="/sentiment",
    tags=["Sentiment"]
)

# ---------------- BERT for sentiment ----------------
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = AutoModel.from_pretrained("distilbert-base-uncased")
bert_model.eval()

def get_bert_sentiment_features(texts, max_len=32):
    """
    Compute mean CLS token embedding for a list of texts.
    """
    all_embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               padding="max_length", max_length=max_len)
            outputs = bert_model(**inputs)
            cls_emb = outputs.last_hidden_state[:,0,:]
            all_embeddings.append(cls_emb.numpy().squeeze())
    if all_embeddings:
        return np.mean(all_embeddings, axis=0).tolist()
    return np.zeros(768).tolist()


@router.get("/{ticker}")
async def compute_sentiment(ticker: str, count: int = 10):
    """
    Fetch latest news headlines for the ticker and compute BERT embeddings as sentiment features.
    """
    try:
        rss_url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(rss_url)
        titles = [entry.title for entry in feed.entries[:count]]

        sentiment_features = get_bert_sentiment_features(titles)

        return {
            "status": "success",
            "ticker": ticker,
            "sentiment_features": sentiment_features
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
