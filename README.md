# 📈 Sentiment Stock Predictor

## Overview
**Sentiment Stock Predictor** is an end-to-end machine learning web application that:
- Collects real-time stock prices and financial news from multiple sources (Google News, Twitter, etc.).
- Performs **sentiment analysis** on textual data using transformer models like **FinBERT**.
- Combines sentiment features with historical prices to **forecast next-week stock trends**.
- Displays results on an interactive **frontend dashboard** with charts, news feeds, and sentiment graphs.

---

## 🚀 Features
- Fetches live stock and news data using APIs (e.g., Yahoo Finance, NewsAPI).
- Applies sentiment analysis using VADER and FinBERT.
- Predicts 7-day stock prices using ML (Prophet / LSTM).
- React-based frontend dashboard with price charts, sentiment trends, and news display.
- Modular FastAPI backend and containerized deployment using Docker.

---

## 🏗️ Project Structure
sentiment-stock-predictor/
│
├── backend/ # FastAPI backend (data, ML, API)
├── frontend/ # React frontend (UI)
├── data/ # Raw, processed, and prediction data
├── scripts/ # Data collection & model training scripts
├── notebooks/ # Jupyter notebooks for experiments
├── docker-compose.yml # Multi-container setup
├── .env.example # Environment variable template
├── LICENSE # Open-source license
└── README.md # Project documentation
