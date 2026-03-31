import React, { useState, useEffect } from "react";
import { motion, type Variants } from "framer-motion";
import { TrendingUp, TrendingDown, Minus, ExternalLink, Newspaper, Calendar } from "lucide-react";
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Bar,
  ComposedChart,
  Cell,
  Line,
} from "recharts";
import { fetchNews, type NewsArticle, type PredictionResult as PredictionResultType } from "../api/stockApi";

interface Props {
  prediction: PredictionResultType;
  historicalData: { date: string; close: number; open: number; high: number; low: number; volume: number; predicted?: boolean }[];
  loadingHistory?: boolean;
}


const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  const p = payload[0]?.payload;
  const isUp = p.close >= p.open;

  return (
    <div style={{
      background: "rgba(15,15,42,0.95)",
      border: `1px solid rgba(255,255,255,0.1)`,
      borderRadius: "var(--radius-md)",
      padding: "var(--space-3) var(--space-4)",
      fontSize: "var(--text-sm)",
      boxShadow: "0 10px 15px -3px rgba(0,0,0,0.4)"
    }}>
      <p style={{ color: "var(--text-muted)", marginBottom: "var(--space-2)", fontSize: "var(--text-xs)" }}>{label}</p>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px 16px" }}>
        <span style={{ color: "var(--text-muted)", fontSize: "var(--text-xs)" }}>Open:</span>
        <span style={{ fontWeight: 600 }}>${p.open.toFixed(2)}</span>
        <span style={{ color: "var(--text-muted)", fontSize: "var(--text-xs)" }}>Close:</span>
        <span style={{ fontWeight: 600, color: isUp ? "#4fb58b" : "#ef4444" }}>${p.close.toFixed(2)}</span>
        <span style={{ color: "var(--text-muted)", fontSize: "var(--text-xs)" }}>High:</span>
        <span style={{ fontWeight: 600 }}>${p.high.toFixed(2)}</span>
        <span style={{ color: "var(--text-muted)", fontSize: "var(--text-xs)" }}>Low:</span>
        <span style={{ fontWeight: 600 }}>${p.low.toFixed(2)}</span>
      </div>
      {p.predicted && (
        <p style={{ fontSize: "var(--text-xs)", color: "var(--accent-light)", marginTop: "8px", borderTop: "1px solid rgba(255,255,255,0.05)", paddingTop: "8px" }}>
          Predicted Price
        </p>
      )}
    </div>
  );
};

const stagger: Variants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.08 } },
};
const fadeUp: Variants = {
  hidden: { opacity: 0, y: 20 },
  show:   { opacity: 1, y: 0, transition: { duration: 0.4 } },
};

const SkeletonBlock = ({ h, w = "100%" }: { h: number; w?: string }) => (
  <div className="skeleton" style={{ height: h, width: w, borderRadius: "var(--radius-md)" }} />
);

const PredictionResult: React.FC<Props> = ({ prediction, historicalData, loadingHistory }) => {
  const [news, setNews] = useState<NewsArticle[]>([]);
  const [loadingNews, setLoadingNews] = useState(false);

  useEffect(() => {
    if (prediction?.ticker) {
      const loadNews = async () => {
        setLoadingNews(true);
        const data = await fetchNews(prediction.ticker);
        setNews(data.slice(0, 5)); // Show top 5
        setLoadingNews(false);
      };
      loadNews();
    }
  }, [prediction?.ticker]);

  if (!prediction) return null;

  const direction  = prediction.predicted_direction?.toLowerCase() || "";
  const predPoint  = historicalData.find((d) => d.predicted);

  const formattedData = historicalData.map(item => ({
    ...item,
    ohlc: [item.open, item.close],
    range: [item.low, item.high]
  }));

  const prices = historicalData.flatMap(d => [d.low, d.high]);
  const minPriceVal = Math.min(...prices);
  const maxPriceVal = Math.max(...prices);
  
  // Ensure we don't have a flat chart (min == max)
  const range = maxPriceVal - minPriceVal;
  const padding = range === 0 ? 1 : range * 0.15;
  const minPrice = minPriceVal - padding;
  const maxPrice = maxPriceVal + padding;

  const maxVolume = Math.max(...historicalData.map(d => d.volume || 0), 1);

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" style={{ display: "flex", flexDirection: "column", gap: "var(--space-5)" }}>

      {/* ── Hero Card ──────────────────────────────────────────────── */}
      <motion.div variants={fadeUp} className="card" style={{
        padding: "var(--space-4) var(--space-6)",
        background: "linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(15,15,42,0.6) 100%)",
        border: "1px solid rgba(99,102,241,0.2)",
      }}>
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", flexWrap: "wrap", gap: "var(--space-4)" }}>
          <div>
            <p style={{ fontSize: "var(--text-sm)", color: "var(--text-muted)", marginBottom: "var(--space-2)" }}>
              Next-Day Predicted Price
            </p>
            <p style={{ fontSize: "var(--text-3xl)", fontWeight: 800, letterSpacing: "-0.03em", lineHeight: 1.1 }}>
              ${prediction.predicted_price?.toFixed(2) ?? "—"}
            </p>
          </div>

          <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "var(--space-3)" }}>
            <span className={`badge ${direction === 'up' ? "badge-up" : direction === 'down' ? "badge-down" : "badge-neutral"}`}>
              {direction === 'up'   ? <TrendingUp  size={12} /> :
               direction === 'down' ? <TrendingDown size={12} /> :
                        <Minus size={12} />}
              {direction.toUpperCase() || "NEUTRAL"}
            </span>
            <span style={{ fontSize: "var(--text-xs)", color: "var(--text-muted)" }}>
              {prediction.ticker} · AI Prediction
            </span>
          </div>
        </div>
      </motion.div>

      {/* ── Main Content Split ─────────────────────────────────────── */}
      <div className="split-layout" style={{ display: "flex", gap: "var(--space-6)", flexWrap: "wrap" }}>
        
        {/* Left: News (40%) */}
        <motion.div variants={fadeUp} className="card" style={{ flex: "0 0 320px", padding: "var(--space-4)", maxHeight: "420px", overflowY: "auto" }}>
          <p style={{ fontSize: "var(--text-xs)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-muted)", marginBottom: "var(--space-4)", display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
            <Newspaper size={14} /> Latest News: {prediction.ticker}
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
            {loadingNews ? (
              [1, 2, 3, 4, 5].map((i) => <SkeletonBlock key={i} h={80} />)
            ) : news.length === 0 ? (
              <p style={{ fontSize: "var(--text-sm)", color: "var(--text-muted)", padding: "var(--space-4)", textAlign: "center" }}>
                No recent news found for this ticker.
              </p>
            ) : (
              news.map((item, idx) => (
                <a key={idx} href={item.link} target="_blank" rel="noopener noreferrer" className="news-item-link" style={{ textDecoration: "none", color: "inherit" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "var(--space-3)" }}>
                    <span style={{ fontSize: "var(--text-sm)", fontWeight: 600, color: "var(--text-primary)", lineHeight: 1.4 }}>
                      {item.title}
                    </span>
                    <ExternalLink size={12} style={{ flexShrink: 0, color: "var(--accent-light)", marginTop: "2px" }} />
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)", fontSize: "var(--text-xs)", color: "var(--text-muted)", marginTop: "4px" }}>
                    <Calendar size={10} />
                    {item.published}
                  </div>
                </a>
              ))
            )}
          </div>
        </motion.div>

        {/* Right: Candlestick Chart (60%) */}
        <motion.div variants={fadeUp} className="card" style={{ flex: "1", minWidth: "400px", padding: "var(--space-4)" }}>
          <p style={{ fontSize: "var(--text-sm)", fontWeight: 600, color: "var(--text-secondary)", marginBottom: "var(--space-5)" }}>
            Technical Chart · Daily OHLC
          </p>

          {loadingHistory ? (
            <SkeletonBlock h={320} />
          ) : (
            <ResponsiveContainer width="100%" height={320}>
              <ComposedChart data={formattedData} margin={{ top: 10, right: 30, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis 
                  dataKey="date" 
                  tick={{ fontSize: 10, fill: "var(--text-muted)" }} 
                  tickFormatter={(d) => new Date(d).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                  axisLine={false}
                  tickLine={false}
                  dy={10}
                />
                <YAxis
                  yAxisId="price"
                  domain={[minPrice, maxPrice]}
                  tick={{ fontSize: 11, fill: "var(--text-muted)" }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(v) => `$${v.toFixed(0)}`}
                  width={50}
                />
                <YAxis
                  yAxisId="volume"
                  orientation="right"
                  domain={[0, maxVolume * 4]}
                  hide
                />
                <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.05)' }} />
                
                {/* Volume Bar */}
                <Bar 
                  yAxisId="volume" 
                  dataKey="volume" 
                  barSize={20}
                  opacity={0.2}
                >
                  {formattedData.map((entry, index) => (
                    <Cell key={`vol-${index}`} fill={entry.close >= entry.open ? "#4fb58b" : "#ef4444"} />
                  ))}
                </Bar>

                {/* Candlestick Wick */}
                <Bar yAxisId="price" dataKey="range" barSize={1} stroke="none">
                  {formattedData.map((entry, index) => (
                    <Cell key={`wick-${index}`} fill={entry.close >= entry.open ? "#4fb58b" : "#ef4444"} />
                  ))}
                </Bar>
                
                {/* Candlestick Body */}
                <Bar yAxisId="price" dataKey="ohlc">
                  {formattedData.map((entry, index) => (
                    <Cell key={`body-${index}`} fill={entry.close >= entry.open ? "#4fb58b" : "#ef4444"} />
                  ))}
                </Bar>

                {/* Trend Line (as seen in Image 2) */}
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="close"
                  stroke="rgba(255, 255, 255, 0.4)"
                  strokeWidth={1.5}
                  dot={{ r: 2, fill: "var(--accent-light)", strokeWidth: 0 }}
                  activeDot={{ r: 4, fill: "var(--accent-light)" }}
                />
                
                {predPoint && (
                  <ReferenceLine
                    x={predPoint.date}
                    stroke="#6366f1"
                    strokeDasharray="4 4"
                    label={{ value: "Predicted", position: "insideTopRight", fill: "#818cf8", fontSize: 10 }}
                  />
                )}
              </ComposedChart>
            </ResponsiveContainer>
          )}

          <div style={{ display: "flex", gap: "16px", marginTop: "16px", fontSize: "10px", color: "var(--text-muted)", justifyContent: "center" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
              <div style={{ width: "8px", height: "8px", background: "#4fb58b", borderRadius: "2px" }} /> Bullish
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
              <div style={{ width: "8px", height: "8px", background: "#ef4444", borderRadius: "2px" }} /> Bearish
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
              <div style={{ width: "8px", height: "2px", background: "#6366f1" }} /> Prediction
            </div>
          </div>
        </motion.div>

      </div>
    </motion.div>
  );
};

export default PredictionResult;
