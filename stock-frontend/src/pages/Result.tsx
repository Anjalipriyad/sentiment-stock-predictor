// src/pages/Result.tsx
import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import PredictionResult from "../components/PredictionResult";
import type { PredictionResult as PredictionResultType } from "../api/stockApi";

// Define the type for the router state
interface LocationState {
  prediction: PredictionResultType;
  historicalData: { date: string; close: number }[];
}

const Result: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();

  // Type assertion for router state
  const state = location.state as LocationState | undefined;

  if (!state) {
    // If no state, redirect back to Home
    navigate("/", { replace: true });
    return null;
  }

  const { prediction, historicalData } = state;

  return (
    <div style={{ padding: "2rem" }}>
      <button
        onClick={() => navigate("/")}
        style={{ marginBottom: "1rem", padding: "0.5rem 1rem" }}
      >
        ← Back
      </button>
      <PredictionResult prediction={prediction} historicalData={historicalData} />
    </div>
  );
};

export default Result;
