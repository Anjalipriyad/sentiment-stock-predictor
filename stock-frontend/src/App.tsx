// src/App.tsx
import React, { useState } from "react";
import TickerSelect from "./components/TickerSelect";
import PredictionResult from "./components/PredictionResult";
import type { PredictionResult as PredictionResultType } from "./api/stockApi";
import { fetchPredictionHistory } from "./api/stockApi";
import type { HistoricalPrice } from "./types";
import "./index.css";

const App: React.FC = () => {
  const [prediction, setPrediction] = useState<PredictionResultType | null>(null);
  const [historicalData, setHistoricalData] = useState<HistoricalPrice[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // When user selects a ticker and prediction is fetched
  const handlePredictionFetched = async (pred: PredictionResultType) => {
    setPrediction(pred);
    setLoadingHistory(true);
    setError(null);

    try {
      // Fetch historical data + predicted price for chart
      const history: PredictionResultType[] = await fetchPredictionHistory(pred.ticker);

      // Transform backend data to HistoricalPrice[]
      const historical: HistoricalPrice[] = history.map((item) => ({
        date: item.created_at || new Date().toISOString(),
        close: item.predicted_price,
        predicted: false,
      }));

      // Add latest predicted price as predicted
      const lastDate = new Date(historical[historical.length - 1]?.date);
      lastDate.setDate(lastDate.getDate() + 1);
      historical.push({
        date: lastDate.toISOString().split("T")[0],
        close: pred.predicted_price,
        predicted: true,
      });

      setHistoricalData(historical);
    } catch (err: any) {
      setError(err.message || "Failed to fetch historical data");
    } finally {
      setLoadingHistory(false);
    }
  };

  return (
    <div className="app-container" style={{ fontFamily: "Arial, sans-serif", padding: "2rem" }}>
      <h1 style={{ textAlign: "center", marginBottom: "2rem" }}>📈 Stock Market Predictor</h1>

      {!prediction && (
        <div style={{ maxWidth: "500px", margin: "0 auto" }}>
          <TickerSelect onPredictionFetched={handlePredictionFetched} />
        </div>
      )}

      {prediction && (
        <div style={{ marginTop: "2rem" }}>
          {loadingHistory ? (
            <p>Loading historical data...</p>
          ) : error ? (
            <p style={{ color: "red" }}>{error}</p>
          ) : (
            <PredictionResult
              prediction={prediction}
              historicalData={historicalData}
            />
          )}

          <div style={{ marginTop: "2rem", textAlign: "center" }}>
            <button
              onClick={() => {
                setPrediction(null);
                setHistoricalData([]);
              }}
              style={{
                padding: "0.5rem 1rem",
                fontSize: "1rem",
                cursor: "pointer",
                borderRadius: "5px",
                border: "1px solid #8884d8",
                background: "#fff",
                color: "#8884d8",
                transition: "all 0.3s",
              }}
              onMouseOver={(e) => (e.currentTarget.style.background = "#8884d8")}
              onMouseOut={(e) => (e.currentTarget.style.background = "#fff")}
            >
              🔄 Select Another Ticker
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
