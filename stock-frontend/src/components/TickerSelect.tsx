import React, { useState } from "react";
import { motion, AnimatePresence, type Variants } from "framer-motion";
import { Search, Loader2, AlertCircle, ChevronRight, Building2 } from "lucide-react";
import { getAllTickers, fetchPrediction, type TickerInfo, type PredictionResult } from "../api/stockApi";

interface TickerSelectProps {
  onPredictionFetched: (prediction: PredictionResult) => void;
}

const listVariants: Variants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.05 },
  },
};

const itemVariants: Variants = {
  hidden: { opacity: 0, x: -10 },
  show:   { opacity: 1, x: 0, transition: { type: "spring", stiffness: 300, damping: 24 } },
};

const TickerSelect: React.FC<TickerSelectProps> = ({ onPredictionFetched }) => {
  const allTickers = getAllTickers();
  const [search, setSearch] = useState<string>("");
  const [filteredTickers, setFilteredTickers] = useState<TickerInfo[]>(allTickers);
  const [selectedTicker, setSelectedTicker] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value.toLowerCase();
    setSearch(value);
    setFilteredTickers(
      allTickers.filter(
        (t) => t.ticker.toLowerCase().includes(value) || t.name.toLowerCase().includes(value)
      )
    );
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

  return (
    <motion.div
      className="card"
      style={{ padding: "var(--space-4)", maxWidth: "600px", margin: "0 auto" }}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
    >
      {/* Search Input */}
      <div style={{ position: "relative", marginBottom: "var(--space-3)" }}>
        <Search
          size={18}
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
          placeholder="Search by company name or ticker…"
          value={search}
          onChange={handleSearchChange}
          style={{ paddingLeft: "3.25rem", height: "48px" }}
          autoFocus
        />
      </div>

      {/* Count label */}
      <p style={{ fontSize: "var(--text-xs)", color: "var(--text-muted)", marginBottom: "var(--space-4)", fontWeight: 500 }}>
        {filteredTickers.length} companies available
        {search && ` matching "${search}"`}
      </p>

      {/* Scrolling List */}
      <motion.div
        key={search}
        variants={listVariants}
        initial="hidden"
        animate="show"
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "var(--space-1)",
          maxHeight: "300px",
          overflowY: "auto",
          paddingRight: "var(--space-2)",
          marginRight: "calc(-1 * var(--space-2))",
        }}
      >
        {filteredTickers.map((t) => {
          const isSelected = selectedTicker === t.ticker;
          const isLoading  = isSelected && loading;
          return (
            <motion.button
              key={t.ticker}
              variants={itemVariants}
              onClick={() => handleSelect(t.ticker)}
              disabled={loading}
              className="ticker-list-item"
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "var(--space-3) var(--space-4)",
                borderRadius: "var(--radius-md)",
                border: "1px solid transparent",
                background: isSelected ? "var(--accent-dim)" : "transparent",
                cursor: loading ? "not-allowed" : "pointer",
                textAlign: "left",
                width: "100%",
                transition: "all 0.2s ease",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "var(--space-3)" }}>
                <div style={{
                  width: "32px",
                  height: "32px",
                  borderRadius: "var(--radius-sm)",
                  background: isSelected ? "rgba(99,102,241,0.2)" : "rgba(255,255,255,0.03)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  color: isSelected ? "var(--accent-light)" : "var(--text-muted)",
                  flexShrink: 0
                }}>
                  <Building2 size={16} />
                </div>
                <div style={{ display: "flex", flexDirection: "column" }}>
                  <span style={{
                    fontSize: "var(--text-sm)",
                    fontWeight: 600,
                    color: isSelected ? "var(--accent-light)" : "var(--text-primary)"
                  }}>
                    {t.name}
                  </span>
                  <span style={{ fontSize: "var(--text-xs)", color: "var(--text-muted)" }}>
                    {t.ticker}
                  </span>
                </div>
              </div>
              
              <div style={{ color: isSelected ? "var(--accent-light)" : "var(--text-muted)" }}>
                {isLoading ? (
                  <Loader2 size={16} style={{ animation: "spin 1s linear infinite" }} />
                ) : (
                  <ChevronRight size={16} />
                )}
              </div>
            </motion.button>
          );
        })}
        {filteredTickers.length === 0 && (
          <div style={{ padding: "var(--space-8)", textAlign: "center", color: "var(--text-muted)" }}>
            No companies found matching your search.
          </div>
        )}
      </motion.div>

      {/* Loading state message */}
      <AnimatePresence>
        {loading && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            style={{
              overflow: "hidden",
              display: "flex",
              alignItems: "center",
              gap: "var(--space-3)",
              marginTop: "var(--space-3)",
              color: "var(--accent-light)",
              fontSize: "var(--text-sm)",
              paddingTop: "var(--space-4)",
              borderTop: "1px solid var(--border)"
            }}
          >
            <Loader2 size={16} style={{ animation: "spin 1s linear infinite" }} />
            Fetching prediction for <strong>{allTickers.find(t => t.ticker === selectedTicker)?.name}</strong>…
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

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        .ticker-list-item:hover {
          background: rgba(255,255,255,0.03) !important;
          border-color: var(--border) !important;
        }
      `}</style>
    </motion.div>
  );
};

export default TickerSelect;
