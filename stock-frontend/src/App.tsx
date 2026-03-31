import React, { useState } from "react";
import { AnimatePresence, motion, type Variants } from "framer-motion";
import { TrendingUp, AlertCircle, RotateCcw } from "lucide-react";
import TickerSelect from "./components/TickerSelect";
import PredictionResult from "./components/PredictionResult";
import type { PredictionResult as PredictionResultType } from "./api/stockApi";
import { fetchPredictionHistory } from "./api/stockApi";
import type { HistoricalPrice } from "./types";
import "./App.css";

const pageVariants: Variants = {
  initial: { opacity: 0, y: 24 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.25, 0.1, 0.25, 1] } },
  exit:    { opacity: 0, y: -16, transition: { duration: 0.25 } },
};

const App: React.FC = () => {
  const [prediction, setPrediction] = useState<PredictionResultType | null>(null);
  const [historicalData, setHistoricalData] = useState<HistoricalPrice[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePredictionFetched = async (pred: PredictionResultType) => {
    if (!pred) return;
    setPrediction(pred);
    setLoadingHistory(true);
    setError(null);

    try {
      const history: PredictionResultType[] = await fetchPredictionHistory(pred.ticker);

      const historical: HistoricalPrice[] = (history || []).map((item) => ({
        date: item.created_at || new Date().toISOString(),
        close: item.predicted_price ?? 0,
        predicted: false,
      }));

      const lastDate = new Date(historical[historical.length - 1]?.date || new Date());
      lastDate.setDate(lastDate.getDate() + 1);
      historical.push({
        date: lastDate.toISOString().split("T")[0],
        close: pred.predicted_price ?? 0,
        predicted: true,
      });

      setHistoricalData(historical);
    } catch (err: any) {
      setError(err.message || "Failed to fetch historical data");
      setHistoricalData([
        { date: new Date().toISOString(), close: pred.predicted_price ?? 0, predicted: true },
      ]);
    } finally {
      setLoadingHistory(false);
    }
  };

  const handleReset = () => {
    setPrediction(null);
    setHistoricalData([]);
    setError(null);
  };

  return (
    <div className="app-shell">
      {/* Header */}
      <header className="app-header">
        <div className="app-logo">
          <div className="app-logo-icon">
            <TrendingUp size={18} />
          </div>
          <span className="app-logo-text">
            Stock<span>Sense</span>
          </span>
        </div>
        <span className="header-tag">AI-Powered Predictions</span>
      </header>

      {/* Main */}
      <main className="app-main">
        <AnimatePresence mode="wait">
          {!prediction ? (
            <motion.div
              key="search"
              variants={pageVariants}
              initial="initial"
              animate="animate"
              exit="exit"
            >
              {/* Hero */}
              <div className="hero">
                <div className="hero-eyebrow">
                  <TrendingUp size={12} />
                  Machine Learning · LSTM · Random Forest
                </div>
                <h1 className="hero-title">
                  Predict Tomorrow's<br />
                  <span>Stock Price</span>
                </h1>
                <p className="hero-sub">
                  Select a ticker to get an AI-powered next-day price prediction with confidence metrics.
                </p>
              </div>

              <TickerSelect onPredictionFetched={handlePredictionFetched} />
            </motion.div>
          ) : (
            <motion.div
              key="result"
              variants={pageVariants}
              initial="initial"
              animate="animate"
              exit="exit"
            >
              {/* Back row */}
              <div className="back-row">
                <button className="btn-ghost" onClick={handleReset}>
                  <RotateCcw size={14} />
                  New Prediction
                </button>
                <span className="header-tag">{prediction.ticker}</span>
              </div>

              {/* Error banner */}
              {error && (
                <div className="error-banner">
                  <AlertCircle size={16} />
                  {error}
                </div>
              )}

              <PredictionResult
                prediction={prediction}
                historicalData={historicalData}
                loadingHistory={loadingHistory}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
};

export default App;
