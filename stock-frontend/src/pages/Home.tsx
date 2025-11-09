// src/pages/Home.tsx
import React, { useState } from "react";
import TickerSelect from "../components/TickerSelect";
import PredictionResult from "../components/PredictionResult";
import type { PredictionResult as PredictionResultType } from "../api/stockApi";

// Dummy historical data generator
const generateHistoricalData = (lastPrice: number) => {
  const data = [];
  const today = new Date();
  for (let i = 10; i > 0; i--) {
    const date = new Date(today);
    date.setDate(today.getDate() - i);
    data.push({
      date: date.toISOString().split("T")[0],
      close: lastPrice + Math.floor(Math.random() * 10 - 5),
    });
  }
  return data;
};

const Home: React.FC = () => {
  const [prediction, setPrediction] = useState<PredictionResultType | null>(null);
  const [historicalData, setHistoricalData] = useState<{ date: string; close: number }[]>([]);

  const handlePredictionFetched = (pred: PredictionResultType) => {
    console.log("Prediction fetched:", pred);
    setPrediction(pred);
    setHistoricalData(generateHistoricalData(pred.predicted_price));
  };

  return (
    <div style={{ padding: "2rem" }}>
      {!prediction ? (
        <TickerSelect onPredictionFetched={handlePredictionFetched} />
      ) : (
        <PredictionResult prediction={prediction} historicalData={historicalData} />
      )}
    </div>
  );
};

export default Home;
