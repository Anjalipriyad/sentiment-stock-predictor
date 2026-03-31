import axios from "axios";
import { type HistoricalPrice } from "../types";


export interface TickerInfo {
  ticker: string;
  name: string;
}

export const getAllTickers = (): TickerInfo[] => [
  { ticker: "AAPL", name: "Apple Inc." },
  { ticker: "MSFT", name: "Microsoft Corp." },
  { ticker: "GOOGL", name: "Alphabet Inc." },
  { ticker: "AMZN", name: "Amazon.com Inc." },
  { ticker: "TSLA", name: "Tesla Inc." },
  { ticker: "META", name: "Meta Platforms Inc." },
  { ticker: "NVDA", name: "NVIDIA Corp." },
  { ticker: "JPM", name: "JPMorgan Chase & Co." },
  { ticker: "V", name: "Visa Inc." },
  { ticker: "JNJ", name: "Johnson & Johnson" },
  { ticker: "WMT", name: "Walmart Inc." },
  { ticker: "PG", name: "Procter & Gamble Co." },
  { ticker: "DIS", name: "Walt Disney Co." },
  { ticker: "MA", name: "Mastercard Inc." },
  { ticker: "HD", name: "Home Depot Inc." },
  { ticker: "BAC", name: "Bank of America Corp." },
  { ticker: "XOM", name: "Exxon Mobil Corp." },
  { ticker: "KO", name: "Coca-Cola Co." },
  { ticker: "PFE", name: "Pfizer Inc." },
  { ticker: "CSCO", name: "Cisco Systems Inc." },
  { ticker: "INTC", name: "Intel Corp." },
  { ticker: "VZ", name: "Verizon Communications Inc." },
  { ticker: "CVX", name: "Chevron Corp." },
  { ticker: "ADBE", name: "Adobe Inc." },
  { ticker: "NFLX", name: "Netflix Inc." },
  { ticker: "T", name: "AT&T Inc." },
  { ticker: "MRK", name: "Merck & Co. Inc." },
  { ticker: "PEP", name: "PepsiCo Inc." },
  { ticker: "ABBV", name: "AbbVie Inc." },
  { ticker: "CRM", name: "Salesforce Inc." },
  { ticker: "NKE", name: "NIKE Inc." },
  { ticker: "ORCL", name: "Oracle Corp." },
  { ticker: "ABT", name: "Abbott Laboratories" },
  { ticker: "ACN", name: "Accenture plc" },
  { ticker: "LLY", name: "Eli Lilly & Co." },
  { ticker: "AVGO", name: "Broadcom Inc." },
  { ticker: "COST", name: "Costco Wholesale Corp." },
  { ticker: "QCOM", name: "Qualcomm Inc." },
  { ticker: "MDT", name: "Medtronic plc" },
  { ticker: "MCD", name: "McDonald's Corp." },
  { ticker: "TXN", name: "Texas Instruments Inc." },
  { ticker: "NEE", name: "NextEra Energy Inc." },
  { ticker: "UNH", name: "UnitedHealth Group Inc." },
  { ticker: "HON", name: "Honeywell International Inc." },
  { ticker: "DHR", name: "Danaher Corp." },
  { ticker: "LIN", name: "Linde plc" },
  { ticker: "AMGN", name: "Amgen Inc." },
  { ticker: "BMY", name: "Bristol-Myers Squibb Co." },
  { ticker: "SBUX", name: "Starbucks Corp." },
  { ticker: "TMUS", name: "T-Mobile US Inc." }
];


// ---------- Prediction Result Type ----------
// ---------- Prediction Result Type ----------
export interface PredictionResult {
  id?: number;
  ticker: string;
  best_model: string;
  predicted_price: number;
  predicted_direction: string;
  created_at?: string;

  // Price metrics (legacy)
  price_mae?: number;
  price_rmse?: number;
  price_r2?: number;

  // ✅ Direction + Price metrics from backend
  metrics?: {
    random_forest?: { RMSE: number; MAE: number; R2: number };
    lstm?: { RMSE: number; MAE: number; R2: number };
  };

  dir_metrics?: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1?: number;
  };
}

export interface NewsArticle {
  title: string;
  link: string;
  published: string;
}



const BASE_URL = "http://localhost:8000";

// ---------- Fetch prediction ----------
export const fetchPrediction = async (ticker: string): Promise<PredictionResult> => {
  try {
    const response = await axios.get(`${BASE_URL}/predict/${ticker}`);
    return response.data as PredictionResult;
  } catch (err: any) {
    console.error("Error fetching prediction:", err);
    throw new Error(err.response?.data?.detail || "Failed to fetch prediction");
  }
};

// ---------- Fetch prediction history ----------
export const fetchPredictionHistory = async (ticker: string): Promise<PredictionResult[]> => {
  try {
    const response = await axios.get(`${BASE_URL}/predict/history/${ticker}`);
    return response.data.records as PredictionResult[];
  } catch (err: any) {
    console.error("Error fetching prediction history:", err);
    throw new Error(err.response?.data?.detail || "Failed to fetch prediction history");
  }
};

// ---------- Fetch News ----------
export const fetchNews = async (ticker: string): Promise<NewsArticle[]> => {
  try {
    const response = await axios.get(`${BASE_URL}/fetch/news/${ticker}`);
    return response.data.news as NewsArticle[];
  } catch (err: any) {
    console.error("Error fetching news:", err);
    return [];
  }
};

// ---------- Fetch Actual OHLC History ----------
export const fetchActualHistory = async (ticker: string): Promise<HistoricalPrice[]> => {
  try {
    const response = await axios.get(`${BASE_URL}/fetch/${ticker}`);
    const data = (response.data || []) as any[];
    // Filter out the last 6 days for a cleaner view (the user mentioned "not dramatic")
    return data.slice(-6).map((item) => ({
      date: item.Date || item.date || new Date().toISOString(),
      open: item.Open ?? item.open ?? 0,
      high: item.High ?? item.high ?? 0,
      low: item.Low ?? item.low ?? 0,
      close: item.Close ?? item.close ?? 0,
      volume: item.Volume ?? item.volume ?? 0,
      predicted: false
    })) as HistoricalPrice[];
  } catch (err: any) {
    console.error("Error fetching actual history:", err);
    return [];
  }
};



