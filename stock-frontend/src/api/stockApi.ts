// stockApi.ts
// ------------------------
// Handles stock tickers and API calls to backend for predictions
// ------------------------

import axios from "axios";

// ---------- 1️⃣ Static ticker list ----------
export const getAllTickers = (): string[] => {
  return [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "JNJ",
    "WMT", "PG", "DIS", "MA", "HD", "BAC", "XOM", "KO", "PFE", "CSCO",
    "INTC", "VZ", "CVX", "ADBE", "NFLX", "T", "MRK", "PEP", "ABBV", "CRM",
    "NKE", "ORCL", "ABT", "ACN", "LLY", "AVGO", "COST", "QCOM", "MDT", "MCD",
    "TXN", "NEE", "UNH", "HON", "DHR", "LIN", "AMGN", "BMY", "SBUX", "TMUS"
  ];
};

// ---------- 2️⃣ Fetch prediction for a ticker ----------
export interface PredictionResult {
  ticker: string;
  best_model: string;
  predicted_price: number;
  predicted_direction: string;
  direction_metrics: {
    random_forest: { accuracy: number; precision: number; recall: number; f1: number };
    lstm_stacked: { accuracy: number; precision: number; recall: number; f1: number };
  };
  price_metrics: {
    random_forest: { RMSE: number; MAE: number; R2: number };
    lstm: { RMSE: number; MAE: number; R2: number };
  };
  created_at?: string; // ✅ Add this optional
}


const BASE_URL = "http://localhost:8000/predict"; // Update if your backend runs on a different port

export const fetchPrediction = async (
  ticker: string
): Promise<PredictionResult> => {
  try {
    const response = await axios.get(`${BASE_URL}/${ticker}`);
    return response.data as PredictionResult;
  } catch (err: any) {
    console.error("Error fetching prediction:", err);
    throw new Error(
      err.response?.data?.detail || "Failed to fetch prediction from backend"
    );
  }
};

// ---------- 3️⃣ Optionally: fetch prediction history ----------
export const fetchPredictionHistory = async (
  ticker: string
): Promise<PredictionResult[]> => {
  try {
    const response = await axios.get(`${BASE_URL}/history/${ticker}`);
    return response.data.records as PredictionResult[];
  } catch (err: any) {
    console.error("Error fetching prediction history:", err);
    throw new Error(
      err.response?.data?.detail || "Failed to fetch history from backend"
    );
  }
};
