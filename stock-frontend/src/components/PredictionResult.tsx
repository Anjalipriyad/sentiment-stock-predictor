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
    return [
      {
        date: today.toISOString().split("T")[0],
        close: predictedPrice,
        predicted: true,
      },
    ];
  }

  const lastDate = new Date(historical[historical.length - 1].date);
  const nextDate = new Date(lastDate);
  nextDate.setDate(lastDate.getDate() + 1);

  return [
    ...historical.map((item) => ({ ...item, predicted: false })),
    {
      date: nextDate.toISOString().split("T")[0],
      close: predictedPrice,
      predicted: true,
    },
  ];
};

// --------- Component ----------
const PredictionResult: React.FC<Props> = ({ prediction, historicalData }) => {
  if (!prediction) return null;

  const chartData = prepareChartData(
    historicalData,
    prediction.predicted_price ?? 0
  );

  return (
    <div style={{ padding: "2rem" }}>
      <h2>{prediction.ticker ?? "Ticker"} Prediction</h2>

      <p>
        <strong>Predicted Direction:</strong>{" "}
        {prediction.predicted_direction?.toUpperCase() || "N/A"}
      </p>
      <p>
        <strong>Predicted Next-Day Price:</strong>{" "}
        ${prediction.predicted_price?.toFixed(2) ?? "N/A"}
      </p>

      <h3>Price Chart</h3>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis domain={["auto", "auto"]} />
          <Tooltip
            formatter={(value: any, _name: any, props: any) => [
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
            strokeDasharray={chartData.some((d) => d.predicted) ? "5 5" : ""}
            name="Price"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PredictionResult;
