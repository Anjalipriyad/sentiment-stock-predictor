import React, { useState } from "react";
import { motion, AnimatePresence, type Variants } from "framer-motion";
import { Search, Loader2, AlertCircle, ChevronRight } from "lucide-react";
import { getAllTickers, fetchPrediction } from "../api/stockApi";
import type { PredictionResult } from "../api/stockApi";

interface TickerSelectProps {
  onPredictionFetched: (prediction: PredictionResult) => void;
}

const containerVariants: Variants = {
  hidden: {},
  show: {
    transition: { staggerChildren: 0.03 },
  },
};

const itemVariants: Variants = {
  hidden: { opacity: 0, scale: 0.9 },
  show:   { opacity: 1, scale: 1, transition: { type: "spring" as const, stiffness: 300, damping: 24 } },
};

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
    if (loading) return;
    setSelectedTicker(ticker);
    setLoading(true);
    setError(null);
    try {
      const prediction: PredictionResult = await fetchPrediction(ticker);
      if (prediction) onPredictionFetched(prediction);
    } catch (err: any) {
      setError(err?.message || "Failed to fetch prediction");
      setLoading(false);
    }
  };

  const visible = filteredTickers.slice(0, 24);

  return (
    <motion.div
      className="card"
      style={{ padding: "var(--space-6)" }}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
    >
      {/* Search Input */}
      <div style={{ position: "relative", marginBottom: "var(--space-5)" }}>
        <Search
          size={16}
          style={{
            position: "absolute",
            left: "var(--space-4)",
            top: "50%",
            transform: "translateY(-50%)",
            color: "var(--text-muted)",
            pointerEvents: "none",
          }}
        />
        <input
          type="text"
          placeholder="Search ticker symbol…"
          value={search}
          onChange={handleSearchChange}
          autoFocus
        />
      </div>

      {/* Count label */}
      <p style={{ fontSize: "var(--text-xs)", color: "var(--text-muted)", marginBottom: "var(--space-4)" }}>
        {filteredTickers.length} ticker{filteredTickers.length !== 1 ? "s" : ""} available
        {search && ` for "${search}"`}
      </p>

      {/* Ticker Grid */}
      <motion.div
        key={search}
        variants={containerVariants}
        initial="hidden"
        animate="show"
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(88px, 1fr))",
          gap: "var(--space-2)",
          maxHeight: "320px",
          overflowY: "auto",
          paddingRight: "var(--space-1)",
        }}
      >
        {visible.map((ticker) => {
          const isSelected = selectedTicker === ticker;
          const isLoading  = isSelected && loading;
          return (
            <motion.button
              key={ticker}
              variants={itemVariants}
              onClick={() => handleSelect(ticker)}
              disabled={loading}
              className="card-hover"
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "var(--space-3) var(--space-3)",
                borderRadius: "var(--radius-md)",
                border: isSelected
                  ? "1px solid rgba(99,102,241,0.6)"
                  : "1px solid var(--border)",
                background: isSelected ? "var(--accent-dim)" : "var(--bg-card)",
                color: isSelected ? "var(--accent-light)" : "var(--text-primary)",
                fontSize: "var(--text-sm)",
                fontWeight: isSelected ? 700 : 500,
                cursor: loading ? "not-allowed" : "pointer",
                opacity: loading && !isSelected ? 0.5 : 1,
                transition: "all 0.15s ease",
              }}
              whileHover={!loading ? { scale: 1.04 } : {}}
              whileTap={!loading  ? { scale: 0.97 } : {}}
            >
              {ticker}
              {isLoading ? (
                <Loader2 size={12} style={{ animation: "spin 1s linear infinite" }} />
              ) : isSelected ? (
                <ChevronRight size={12} />
              ) : null}
            </motion.button>
          );
        })}
      </motion.div>

      {/* Loading state full-card overlay message */}
      <AnimatePresence>
        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "var(--space-3)",
              marginTop: "var(--space-5)",
              color: "var(--accent-light)",
              fontSize: "var(--text-sm)",
            }}
          >
            <Loader2 size={16} style={{ animation: "spin 1s linear infinite" }} />
            Fetching prediction for <strong>{selectedTicker}</strong>…
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="error-banner"
            style={{ marginTop: "var(--space-4)" }}
          >
            <AlertCircle size={15} />
            {error}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Spinner keyframe */}
      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
      `}</style>
    </motion.div>
  );
};

export default TickerSelect;
