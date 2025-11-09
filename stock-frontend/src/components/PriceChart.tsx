// src/components/PriceChart.tsx
import React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";

interface PricePoint {
  date: string;
  close: number;
  predicted: boolean;
}

interface PriceChartProps {
  data: PricePoint[];
}

const PriceChart: React.FC<PriceChartProps> = ({ data }) => {
  return (
    <div style={{ width: "100%", height: 400 }}>
      <ResponsiveContainer>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis />
          <Tooltip
            formatter={(value: any, name: any, props: any) => {
              const { predicted } = props.payload;
              return [`${value}`, predicted ? "Predicted" : "Actual"];
            }}
          />
          <Line
            type="monotone"
            dataKey="close"
            stroke="#8884d8"
            dot={(point: any) =>
              point && point.payload.predicted ? (
                <circle r={4} fill="red" stroke="red" />
              ) : (
                <circle r={4} fill="blue" stroke="blue" />
              )
            }
            activeDot={{ r: 6 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PriceChart;
