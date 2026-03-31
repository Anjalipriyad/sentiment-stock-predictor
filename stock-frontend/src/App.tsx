import React, { useState } from "react";
import { AnimatePresence, motion, type Variants } from "framer-motion";
import { TrendingUp, AlertCircle, RotateCcw } from "lucide-react";
import TickerSelect from "./components/TickerSelect";
import PredictionResult from "./components/PredictionResult";
import type { PredictionResult as PredictionResultType } from "./api/stockApi";
import { fetchActualHistory } from "./api/stockApi";
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
      const actualHistory = await fetchActualHistory(pred.ticker);
      
      // If we have actual history, use it. Append the prediction as the last "candle"
      if (actualHistory.length > 0) {
        const lastActual = actualHistory[actualHistory.length - 1];
        const nextDate = new Date(lastActual.date);
        nextDate.setDate(nextDate.getDate() + 1);

        const fullHistory: HistoricalPrice[] = [
          ...actualHistory,
          {
            date: nextDate.toISOString().split("T")[0],
            open: lastActual.close, // Start next candle at last close
            high: Math.max(lastActual.close, pred.predicted_price), 
            low: Math.min(lastActual.close, pred.predicted_price),
            close: pred.predicted_price,
            volume: 0,
            predicted: true
          }
        ];
        setHistoricalData(fullHistory);
      } else {
        // Fallback if no actual history
        setHistoricalData([
          { 
            date: new Date().toISOString(), 
            open: pred.predicted_price, 
            high: pred.predicted_price, 
            low: pred.predicted_price, 
            close: pred.predicted_price, 
            volume: 0,
            predicted: true 
          },
        ]);
      }
    } catch (err: any) {
      setError(err.message || "Failed to fetch historical data");
      setHistoricalData([
        { 
          date: new Date().toISOString(), 
          open: pred.predicted_price, 
          high: pred.predicted_price, 
          low: pred.predicted_price, 
          close: pred.predicted_price, 
          volume: 0,
          predicted: true 
        },
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

                <h1 className="hero-title">
                  Predict Tomorrow's<br />
                  <span>Stock Price</span>
                </h1>
                <p className="hero-sub">
                  Select a company to get an AI-powered next-day price prediction with confidence metrics.
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
