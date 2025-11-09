// src/components/TickerSelect.tsx
import React, { useState } from "react";
import { getAllTickers, fetchPrediction } from "../api/stockApi";
import type { PredictionResult } from "../api/stockApi";

interface TickerSelectProps {
  onPredictionFetched: (prediction: PredictionResult) => void;
}

const TickerSelect: React.FC<TickerSelectProps> = ({ onPredictionFetched }) => {
  const allTickers = getAllTickers();
  const [search, setSearch] = useState<string>("");
  const [filteredTickers, setFilteredTickers] = useState<string[]>(allTickers);
  const [selectedTicker, setSelectedTicker] = useState<string>(""); 
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value.toUpperCase();
    setSearch(value);
    setFilteredTickers(allTickers.filter((ticker) => ticker.includes(value)));
  };

  const handleSelect = async (ticker: string) => {
    setSelectedTicker(ticker);
    setLoading(true);
    setError(null);
    try {
      const prediction: PredictionResult = await fetchPrediction(ticker);
      onPredictionFetched(prediction);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="ticker-select" style={{ padding: "2rem" }}>
      <h2>Select a Stock Ticker</h2>

      <input
        type="text"
        placeholder="Search ticker..."
        value={search}
        onChange={handleSearchChange}
        style={{ padding: "0.5rem", width: "200px", marginBottom: "1rem" }}
      />

      <ul style={{ maxHeight: "300px", overflowY: "auto", padding: 0 }}>
        {filteredTickers.slice(0, 20).map((ticker) => (
          <li
            key={ticker}
            onClick={() => handleSelect(ticker)}
            style={{
              cursor: "pointer",
              padding: "5px 10px",
              background: selectedTicker === ticker ? "#e0e0e0" : "white",
              borderBottom: "1px solid #eee",
              listStyle: "none",
            }}
          >
            {ticker}
          </li>
        ))}
      </ul>

      {loading && <p>Fetching prediction...</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}
    </div>
  );
};

export default TickerSelect;
