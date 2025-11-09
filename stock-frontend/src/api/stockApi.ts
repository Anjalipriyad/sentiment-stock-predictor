import axios from "axios";

// ---------- Static ticker list ----------
export const getAllTickers = (): string[] => [
  "AAPL","MSFT","GOOGL","AMZN","TSLA","META","NVDA","JPM","V","JNJ",
  "WMT","PG","DIS","MA","HD","BAC","XOM","KO","PFE","CSCO",
  "INTC","VZ","CVX","ADBE","NFLX","T","MRK","PEP","ABBV","CRM",
  "NKE","ORCL","ABT","ACN","LLY","AVGO","COST","QCOM","MDT","MCD",
  "TXN","NEE","UNH","HON","DHR","LIN","AMGN","BMY","SBUX","TMUS"
];

// ---------- Prediction Result Type ----------
export interface PredictionResult {
  id?: number;
  ticker: string;
  best_model: string;
  predicted_price: number;
  predicted_direction: string;
  created_at?: string;

  direction_accuracy?: number;
  direction_precision?: number;
  direction_recall?: number;
  direction_f1?: number;
  price_mae?: number;
  price_rmse?: number;
  price_r2?: number;

  metrics?: {
    random_forest?: { RMSE: number; MAE: number; R2: number; accuracy?: number; precision?: number; recall?: number; f1?: number };
    lstm?: { RMSE: number; MAE: number; R2: number };
  };
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
