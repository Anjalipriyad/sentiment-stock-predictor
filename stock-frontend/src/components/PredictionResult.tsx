import React from "react";
import { motion, type Variants } from "framer-motion";
import { TrendingUp, TrendingDown, Minus, BarChart2, Target, Activity } from "lucide-react";
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  AreaChart,
} from "recharts";
import type { PredictionResult as PredictionResultType } from "../api/stockApi";

interface Props {
  prediction: PredictionResultType;
  historicalData: { date: string; close: number; predicted?: boolean }[];
  loadingHistory?: boolean;
}

const prepareChartData = (
  historical: { date: string; close: number; predicted?: boolean }[],
  predictedPrice: number
) => {
  if (!historical || historical.length === 0) {
    return [{ date: new Date().toISOString().split("T")[0], close: predictedPrice, predicted: true }];
  }
  const last = historical[historical.length - 1];
  if (last?.predicted) return historical;

  const lastDate = new Date(last.date);
  const nextDate = new Date(lastDate);
  nextDate.setDate(lastDate.getDate() + 1);
  return [
    ...historical.map((item) => ({ ...item, predicted: false })),
    { date: nextDate.toISOString().split("T")[0], close: predictedPrice, predicted: true },
  ];
};

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  const isPredicted = payload[0]?.payload?.predicted;
  return (
    <div style={{
      background: "rgba(15,15,42,0.95)",
      border: `1px solid ${isPredicted ? "rgba(99,102,241,0.4)" : "rgba(255,255,255,0.1)"}`,
      borderRadius: "var(--radius-md)",
      padding: "var(--space-3) var(--space-4)",
      fontSize: "var(--text-sm)",
    }}>
      <p style={{ color: "var(--text-muted)", marginBottom: "var(--space-1)", fontSize: "var(--text-xs)" }}>{label}</p>
      <p style={{ fontWeight: 700, color: isPredicted ? "var(--accent-light)" : "var(--text-primary)" }}>
        ${Number(payload[0]?.value).toFixed(2)}
      </p>
      {isPredicted && (
        <p style={{ fontSize: "var(--text-xs)", color: "var(--accent-light)", marginTop: "2px" }}>Predicted</p>
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
  if (!prediction) return null;

  const direction  = prediction.predicted_direction?.toLowerCase() || "";
  const isUp       = direction === "up";
  const isDown     = direction === "down";
  const chartData  = prepareChartData(historicalData, prediction.predicted_price ?? 0);
  const predPoint  = chartData.find((d) => d.predicted);

  const rf   = prediction.metrics?.random_forest;
  const lstm = prediction.metrics?.lstm;
  const dir  = prediction.dir_metrics;

  const metrics = [
    { label: "Best Model",  value: prediction.best_model?.toUpperCase() || "N/A", icon: <Activity size={14} /> },
    { label: "Price MAE",   value: prediction.price_mae   != null ? `$${prediction.price_mae.toFixed(2)}`   : rf?.MAE  != null ? `$${rf.MAE.toFixed(2)}`  : "N/A", icon: <Target size={14} /> },
    { label: "Price RMSE",  value: prediction.price_rmse  != null ? `$${prediction.price_rmse.toFixed(2)}`  : rf?.RMSE != null ? `$${rf.RMSE.toFixed(2)}` : "N/A", icon: <BarChart2 size={14} /> },
    { label: "R²",          value: prediction.price_r2    != null ? prediction.price_r2.toFixed(3)          : rf?.R2   != null ? rf.R2.toFixed(3)         : "N/A", icon: <BarChart2 size={14} /> },
    { label: "Dir Accuracy",value: dir?.accuracy != null ? `${(dir.accuracy * 100).toFixed(1)}%` : "N/A",   icon: <Target size={14} /> },
    { label: "Dir F1",      value: dir?.f1       != null ? `${(dir.f1       * 100).toFixed(1)}%` : "N/A",   icon: <Activity size={14} /> },
  ];

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" style={{ display: "flex", flexDirection: "column", gap: "var(--space-5)" }}>

      {/* ── Hero Card ──────────────────────────────────────────────── */}
      <motion.div variants={fadeUp} className="card" style={{
        padding: "var(--space-8)",
        background: "linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(15,15,42,0.6) 100%)",
        border: "1px solid rgba(99,102,241,0.2)",
      }}>
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", flexWrap: "wrap", gap: "var(--space-4)" }}>
          <div>
            <p style={{ fontSize: "var(--text-sm)", color: "var(--text-muted)", marginBottom: "var(--space-2)" }}>
              Next-Day Predicted Price
            </p>
            <p style={{ fontSize: "var(--text-4xl)", fontWeight: 800, letterSpacing: "-0.03em", lineHeight: 1.1 }}>
              ${prediction.predicted_price?.toFixed(2) ?? "—"}
            </p>
          </div>

          <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "var(--space-3)" }}>
            <span className={`badge ${isUp ? "badge-up" : isDown ? "badge-down" : "badge-neutral"}`}>
              {isUp   ? <TrendingUp  size={12} /> :
               isDown ? <TrendingDown size={12} /> :
                        <Minus size={12} />}
              {direction.toUpperCase() || "NEUTRAL"}
            </span>
            <span style={{ fontSize: "var(--text-xs)", color: "var(--text-muted)" }}>
              {prediction.ticker} · AI Prediction
            </span>
          </div>
        </div>
      </motion.div>

      {/* ── Metrics Grid ───────────────────────────────────────────── */}
      <motion.div variants={fadeUp}>
        <p style={{ fontSize: "var(--text-xs)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-muted)", marginBottom: "var(--space-3)" }}>
          Model Metrics
        </p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(130px, 1fr))", gap: "var(--space-3)" }}>
          {metrics.map(({ label, value, icon }) => (
            <div key={label} className="stat-card">
              <span className="stat-label" style={{ display: "flex", alignItems: "center", gap: "var(--space-1)" }}>
                {icon} {label}
              </span>
              <span className="stat-value" style={{ fontSize: "var(--text-lg)" }}>{value}</span>
            </div>
          ))}
        </div>
      </motion.div>

      {/* ── Chart ──────────────────────────────────────────────────── */}
      <motion.div variants={fadeUp} className="card" style={{ padding: "var(--space-6)" }}>
        <p style={{ fontSize: "var(--text-sm)", fontWeight: 600, color: "var(--text-secondary)", marginBottom: "var(--space-5)" }}>
          Price History & Prediction
        </p>

        {loadingHistory ? (
          <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
            <SkeletonBlock h={240} />
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={320}>
            <AreaChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%"   stopColor="#6366f1" stopOpacity={0.25} />
                  <stop offset="100%" stopColor="#6366f1" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis
                dataKey="date"
                tick={{ fontSize: 11, fill: "var(--text-muted)" }}
                tickLine={false}
                axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
                interval="preserveStartEnd"
              />
              <YAxis
                domain={["auto", "auto"]}
                tick={{ fontSize: 11, fill: "var(--text-muted)" }}
                tickLine={false}
                axisLine={false}
                tickFormatter={(v) => `$${v}`}
                width={60}
              />
              <Tooltip content={<CustomTooltip />} />
              {predPoint && (
                <ReferenceLine
                  x={predPoint.date}
                  stroke="rgba(99,102,241,0.5)"
                  strokeDasharray="4 4"
                  label={{ value: "Predicted", position: "insideTopRight", fontSize: 11, fill: "var(--accent-light)" }}
                />
              )}
              <Area
                type="monotone"
                dataKey="close"
                stroke="#6366f1"
                strokeWidth={2}
                fill="url(#areaGradient)"
                dot={(props: any) => {
                  const { cx, cy, payload } = props;
                  if (!payload?.predicted) return <></>;
                  return (
                    <circle
                      key={`dot-${cx}-${cy}`}
                      cx={cx}
                      cy={cy}
                      r={5}
                      fill="#818cf8"
                      stroke="rgba(99,102,241,0.4)"
                      strokeWidth={3}
                    />
                  );
                }}
                activeDot={{ r: 5, fill: "#818cf8", strokeWidth: 0 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}

        {/* LSTM metrics row if available */}
        {lstm && (
          <div style={{ marginTop: "var(--space-5)", paddingTop: "var(--space-4)", borderTop: "1px solid var(--border)" }}>
            <p style={{ fontSize: "var(--text-xs)", color: "var(--text-muted)", marginBottom: "var(--space-3)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em" }}>
              LSTM Metrics
            </p>
            <div style={{ display: "flex", gap: "var(--space-4)", flexWrap: "wrap" }}>
              {[["RMSE", lstm.RMSE], ["MAE", lstm.MAE], ["R²", lstm.R2]].map(([k, v]) => (
                <div key={String(k)} style={{ fontSize: "var(--text-sm)" }}>
                  <span style={{ color: "var(--text-muted)" }}>{k}: </span>
                  <span style={{ fontWeight: 600, color: "var(--text-secondary)" }}>
                    {typeof v === "number" ? v.toFixed(3) : "N/A"}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </motion.div>
    </motion.div>
  );
};

export default PredictionResult;
