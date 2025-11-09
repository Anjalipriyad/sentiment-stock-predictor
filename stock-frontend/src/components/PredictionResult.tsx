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
  historicalData: { date: string; close: number }[];
}

// --------- Prepare chart data ----------
const prepareChartData = (
  historical: { date: string; close: number }[],
  predictedPrice: number
) => {
  if (!historical || historical.length === 0) {
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

// --------- Metrics Types ----------
type DirectionMetrics = {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
};

type PriceMetrics = {
  RMSE?: number;
  MAE?: number;
  R2?: number;
};

type ModelMetrics = DirectionMetrics & PriceMetrics;

// --------- Component ----------
const PredictionResult: React.FC<Props> = ({ prediction, historicalData }) => {
  if (!prediction) return null;

  // Prepare chart data
  const chartData = prepareChartData(historicalData, prediction.predicted_price ?? 0);

  // Get metrics for the best model safely
  const modelMetrics: ModelMetrics =
    prediction.metrics?.[prediction.best_model as keyof typeof prediction.metrics] || {};
  console.log("Model Metrics:", prediction);
  return (
    <div style={{ padding: "2rem" }}>
      <h2>{prediction.ticker ?? "Ticker"} Prediction</h2>
      <p><strong>Best Model:</strong> {prediction.best_model?.toUpperCase() || "N/A"}</p>
      <p><strong>Predicted Direction:</strong> {prediction.predicted_direction?.toUpperCase() || "N/A"}</p>
      <p><strong>Predicted Next-Day Price:</strong> ${prediction.predicted_price?.toFixed(2) ?? "N/A"}</p>

      <h3>Direction Metrics</h3>
      <ul>
        <li>Accuracy: {modelMetrics.accuracy?.toFixed(2) ?? "N/A"}</li>
        <li>Precision: {modelMetrics.precision?.toFixed(2) ?? "N/A"}</li>
        <li>Recall: {modelMetrics.recall?.toFixed(2) ?? "N/A"}</li>
        <li>F1: {modelMetrics.f1?.toFixed(2) ?? "N/A"}</li>
      </ul>

      <h3>Price Metrics</h3>
      <ul>
        <li>
          RMSE: {modelMetrics.RMSE?.toFixed(2) ?? "N/A"}, 
          MAE: {modelMetrics.MAE?.toFixed(2) ?? "N/A"}, 
          R2: {modelMetrics.R2?.toFixed(2) ?? "N/A"}
        </li>
      </ul>

      <h3>Price Chart</h3>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis domain={["auto","auto"]} />
          <Tooltip
            formatter={(value: any, name: any, props: any) => [
              value,
              props.payload.predicted ? "Predicted" : "Actual",
            ]}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="close"
            stroke="#8884d8"
            dot={false}
            strokeDasharray={chartData.some(d => d.predicted) ? "5 5" : ""}
            name="Price"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PredictionResult;
