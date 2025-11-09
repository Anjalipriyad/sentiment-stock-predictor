// src/types/index.ts

// ---------- 1️⃣ Prediction Result Type ----------
export interface PredictionResult {
  ticker: string;
  best_model: string;
  predicted_price: number;
  predicted_direction: string;
  direction_metrics: {
    random_forest: {
      accuracy: number;
      precision: number;
      recall: number;
      f1: number;
    };
    lstm_stacked: {
      accuracy: number;
      precision: number;
      recall: number;
      f1: number;
    };
  };
  price_metrics: {
    random_forest: {
      RMSE: number;
      MAE: number;
      R2: number;
    };
    lstm: {
      RMSE: number;
      MAE: number;
      R2: number;
    };
  };
}

// ---------- 2️⃣ Historical Data ----------
export interface HistoricalPrice {
  date: string;
  close: number;
  predicted?: boolean; // optional, true for predicted points
}

// ---------- 3️⃣ API Response Types ----------
export interface PredictionApiResponse {
  ticker: string;
  best_model: string;
  predicted_price: number;
  predicted_direction: string;
  metrics: {
    direction_metrics: PredictionResult["direction_metrics"];
    price_metrics: PredictionResult["price_metrics"];
  };
}

export interface PredictionHistoryResponse {
  ticker: string;
  records: PredictionResult[];
}
