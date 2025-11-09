// src/components/PredictionResult.tsx
import React from "react";
import type { PredictionResult as PredictionResultType } from "../api/stockApi";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface Props {
  prediction: PredictionResultType;
  historicalData: { date: string; close: number }[]; // past stock prices
}

// Merge historical + predicted price
const prepareChartData = (
  historical: { date: string; close: number }[],
  predictedPrice: number
) => {
  if (!historical.length) {
    const today = new Date();
    return [{ date: today.toISOString().split("T")[0], close: predictedPrice, predicted: true }];
  }

  const lastDate = new Date(historical[historical.length - 1].date);
  const nextDate = new Date(lastDate);
  nextDate.setDate(lastDate.getDate() + 1);

  return [
    ...historical.map((item) => ({ ...item, predicted: false })),
    { date: nextDate.toISOString().split("T")[0], close: predictedPrice, predicted: true },
  ];
};



const PredictionResult: React.FC<Props> = ({ prediction, historicalData }) => {
  const chartData = prepareChartData(historicalData, prediction.predicted_price);

  return (
    <div style={{ padding: "2rem" }}>
      <h2>{prediction.ticker} Prediction</h2>
      <p>
        <strong>Best Model:</strong> {prediction.best_model.toUpperCase()}
      </p>
      <p>
        <strong>Predicted Direction:</strong>{" "}
        {prediction.predicted_direction.toUpperCase()}
      </p>
      <p>
        <strong>Predicted Next-Day Price:</strong> $
        {prediction.predicted_price.toFixed(2)}
      </p>

      <h3>Direction Metrics (RF / LSTM)</h3>
      <ul>
        <li>
          <strong>Accuracy:</strong>{" "}
          {prediction.direction_metrics.random_forest?.accuracy} /{" "}
          {prediction.direction_metrics.lstm_stacked?.accuracy}
        </li>
        <li>
          <strong>Precision:</strong>{" "}
          {prediction.direction_metrics.random_forest?.precision} /{" "}
          {prediction.direction_metrics.lstm_stacked?.precision}
        </li>
        <li>
          <strong>Recall:</strong>{" "}
          {prediction.direction_metrics.random_forest?.recall} /{" "}
          {prediction.direction_metrics.lstm_stacked?.recall}
        </li>
        <li>
          <strong>F1:</strong>{" "}
          {prediction.direction_metrics.random_forest?.f1} /{" "}
          {prediction.direction_metrics.lstm_stacked?.f1}
        </li>
      </ul>

      <h3>Price Metrics</h3>
      <ul>
        <li>
          <strong>RF:</strong> RMSE={prediction.price_metrics.random_forest?.RMSE.toFixed(2)}, MAE={prediction.price_metrics.random_forest?.MAE.toFixed(2)}, R2={prediction.price_metrics.random_forest?.R2.toFixed(2)}
        </li>
        <li>
          <strong>LSTM:</strong> RMSE={prediction.price_metrics.lstm?.RMSE.toFixed(2)}, MAE={prediction.price_metrics.lstm?.MAE.toFixed(2)}, R2={prediction.price_metrics.lstm?.R2.toFixed(2)}
        </li>
      </ul>

      <h3>Price Chart</h3>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis domain={["auto", "auto"]} />
          <Tooltip
            formatter={(value: any, name: any, props: any) => {
              return [value, props.payload.predicted ? "Predicted" : "Actual"];
            }}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="close"
            stroke="#8884d8"
            dot={false}
            strokeDasharray={chartData.some((d) => d.predicted) ? "5 5" : ""}
            name="Price"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PredictionResult;
