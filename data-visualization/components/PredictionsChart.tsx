"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Brush,
  Legend,
} from "recharts";
import type { SelectedModel, PredictionResponse, PredictionPoint } from "@/lib/types";

interface PredictionsChartProps {
  selectedModels: SelectedModel[];
}

interface MergedPoint {
  index: number;
  date: string;
  day: number;
  hour: number;
  y_test: number;
  [key: string]: number | string | [number, number];
}

const QUANTILE_BANDS = [
  { lower: "q_0.025", upper: "q_0.975", label: "95% CI", opacity: 0.08 },
  { lower: "q_0.1", upper: "q_0.9", label: "80% CI", opacity: 0.12 },
  { lower: "q_0.25", upper: "q_0.75", label: "50% CI", opacity: 0.18 },
];

const DURATION_OPTIONS = [
  { label: "1w", days: 7 },
  { label: "2w", days: 14 },
  { label: "1m", days: 30 },
  { label: "2m", days: 60 },
  { label: "3m", days: 90 },
  { label: "6m", days: 182 },
];

const TEST_START_DATE = new Date("2025-08-08T00:00:00Z");

function dayToDate(day: number): string {
  const d = new Date(TEST_START_DATE);
  d.setUTCDate(d.getUTCDate() + day);
  return d.toISOString().split("T")[0];
}

function dateToDay(dateStr: string): number {
  const d = new Date(dateStr + "T00:00:00Z");
  const diff = d.getTime() - TEST_START_DATE.getTime();
  return Math.round(diff / (1000 * 60 * 60 * 24));
}

export default function PredictionsChart({ selectedModels }: PredictionsChartProps) {
  const [predictions, setPredictions] = useState<Map<string, PredictionResponse>>(new Map());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [totalDays, setTotalDays] = useState(0);
  const [showQuantiles, setShowQuantiles] = useState(true);

  const [startDate, setStartDate] = useState(dayToDate(0));
  const [duration, setDuration] = useState(14);

  const startDay = Math.max(0, dateToDay(startDate));
  const endDay = Math.min(totalDays || 9999, startDay + duration);

  const minDate = dayToDate(0);
  const maxDate = totalDays > 0 ? dayToDate(totalDays) : dayToDate(180);
  const endDate = dayToDate(endDay);

  const fetchPredictions = useCallback(async () => {
    if (selectedModels.length === 0) return;
    setLoading(true);
    setError(null);

    try {
      const results = new Map<string, PredictionResponse>();
      const fetches = selectedModels.map(async (m) => {
        const key = `${m.experiment}/${m.model}`;
        const params = new URLSearchParams({
          experiment: m.experiment,
          model: m.model,
          run: m.run,
          startDay: String(startDay),
          endDay: String(endDay),
        });
        const res = await fetch(`/api/predictions?${params}`);
        if (!res.ok) throw new Error(`Failed to fetch predictions for ${key}`);
        const data: PredictionResponse = await res.json();
        results.set(key, data);
        if (data.totalDays > 0) setTotalDays(data.totalDays);
      });

      await Promise.all(fetches);
      setPredictions(results);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load predictions");
    } finally {
      setLoading(false);
    }
  }, [selectedModels, startDay, endDay]);

  useEffect(() => {
    fetchPredictions();
  }, [fetchPredictions]);

  const mergedData = useMemo(() => {
    if (predictions.size === 0) return [];

    const firstPred = predictions.values().next().value;
    if (!firstPred) return [];

    const points: MergedPoint[] = firstPred.data.map((p: PredictionPoint) => ({
      index: p.index,
      date: p.date,
      day: p.day,
      hour: p.hour,
      y_test: p.y_test,
    }));

    for (const [key, pred] of predictions) {
      const prefix = key.split("/").pop()?.replace(/_/g, " ") ?? key;
      for (let i = 0; i < pred.data.length && i < points.length; i++) {
        const d = pred.data[i];
        points[i][`${prefix}_mean`] = d.y_pred_mean ?? d["y_pred_mean"] ?? 0;
        for (const qKey of pred.keys) {
          if (qKey.startsWith("q_")) {
            points[i][`${prefix}_${qKey}`] = d[qKey] as number;
          }
        }
        for (const band of QUANTILE_BANDS) {
          const lo = Number(d[band.lower]) || 0;
          const hi = Number(d[band.upper]) || 0;
          points[i][`${prefix}_band_${band.label}`] = [lo, hi];
        }
      }
    }

    return points;
  }, [predictions]);

  const modelKeys = useMemo(() => {
    return selectedModels.map((m) => ({
      prefix: m.model.replace(/_/g, " "),
      color: m.color,
      label: m.model.replace(/_/g, " "),
    }));
  }, [selectedModels]);

  const formatXTick = (value: string) => {
    if (!value) return "";
    const parts = value.split(" ");
    if (parts.length >= 2) {
      return parts[0].substring(5);
    }
    return value;
  };

  const sliderMax = totalDays > 0 ? totalDays : 180;

  return (
    <div className="p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-zinc-900">
          Predictions vs Actual Prices
        </h2>
        <label className="flex items-center gap-2 text-xs text-zinc-500">
          <input
            type="checkbox"
            checked={showQuantiles}
            onChange={(e) => setShowQuantiles(e.target.checked)}
            className="rounded border-zinc-300 text-blue-600 h-3 w-3"
          />
          Show confidence intervals
        </label>
      </div>

      <div className="p-4 bg-zinc-50 rounded-lg border border-zinc-200 space-y-3">
        <div className="flex flex-wrap items-end gap-4">
          <div className="flex flex-col gap-1">
            <label className="text-xs font-medium text-zinc-500">Start date</label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              min={minDate}
              max={maxDate}
              className="text-xs px-2.5 py-1.5 rounded border border-zinc-200 bg-white text-zinc-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>

          <div className="flex flex-col gap-1">
            <label className="text-xs font-medium text-zinc-500">Duration</label>
            <div className="flex items-center gap-1">
              {DURATION_OPTIONS.map(({ label, days }) => (
                <button
                  key={days}
                  onClick={() => setDuration(days)}
                  className={`text-xs px-2.5 py-1.5 rounded border transition-colors font-medium ${
                    duration === days
                      ? "border-blue-300 bg-blue-50 text-blue-700"
                      : "border-zinc-200 bg-white text-zinc-600 hover:bg-zinc-100"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          <span className="text-xs text-zinc-400 ml-auto self-center">
            {startDate} to {endDate} ({duration} days)
          </span>
        </div>

        <div className="space-y-1.5">
          <label className="text-xs font-medium text-zinc-500">Navigate start date</label>
          <div className="relative h-6 flex items-center">
            <div className="absolute inset-x-0 h-1 bg-zinc-200 rounded-full" />
            <div
              className="absolute h-1 bg-blue-400 rounded-full"
              style={{
                left: `${(startDay / sliderMax) * 100}%`,
                width: `${(Math.min(duration, sliderMax - startDay) / sliderMax) * 100}%`,
              }}
            />
            <input
              type="range"
              min={0}
              max={Math.max(0, sliderMax - duration)}
              value={startDay}
              onChange={(e) => setStartDate(dayToDate(parseInt(e.target.value)))}
              className="absolute inset-x-0 w-full appearance-none bg-transparent z-10"
              style={{ background: "transparent" }}
            />
          </div>
          <div className="flex justify-between text-xs text-zinc-400">
            <span>{minDate}</span>
            <span>{maxDate}</span>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-96 border border-zinc-200 rounded-lg">
          <div className="text-sm text-zinc-500">Loading predictions...</div>
        </div>
      ) : error ? (
        <div className="flex items-center justify-center h-96 border border-red-200 rounded-lg bg-red-50">
          <div className="text-sm text-red-600">{error}</div>
        </div>
      ) : mergedData.length === 0 ? (
        <div className="flex items-center justify-center h-96 border border-zinc-200 rounded-lg">
          <div className="text-sm text-zinc-500">No prediction data available</div>
        </div>
      ) : (
        <div className="border border-zinc-200 rounded-lg p-4 bg-white">
          <ResponsiveContainer width="100%" height={500}>
            <ComposedChart data={mergedData} margin={{ top: 10, right: 20, left: 10, bottom: 30 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e4e4e7" />
              <XAxis
                dataKey="date"
                tick={{ fontSize: 9, fill: "#71717a" }}
                tickFormatter={formatXTick}
                interval="preserveStartEnd"
                stroke="#d4d4d8"
                angle={-30}
                textAnchor="end"
                height={50}
              />
              <YAxis
                tick={{ fontSize: 10, fill: "#71717a" }}
                stroke="#d4d4d8"
                label={{
                  value: "Price (EUR/MWh)",
                  angle: -90,
                  position: "insideLeft",
                  style: { fontSize: 10, fill: "#a1a1aa" },
                }}
              />
              <Tooltip
                contentStyle={{
                  fontSize: 11,
                  backgroundColor: "#ffffff",
                  border: "1px solid #e4e4e7",
                  borderRadius: 6,
                }}
                labelFormatter={(label) => String(label)}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />

              {showQuantiles &&
                modelKeys.map(({ prefix, color }) =>
                  QUANTILE_BANDS.map((band) => {
                    const bandKey = `${prefix}_band_${band.label}`;
                    if (mergedData.length > 0 && bandKey in mergedData[0]) {
                      return (
                        <Area
                          key={bandKey}
                          dataKey={bandKey}
                          stroke="none"
                          fill={color}
                          fillOpacity={band.opacity}
                          name={`${prefix} ${band.label}`}
                          legendType="none"
                          isAnimationActive={false}
                          type="monotone"
                        />
                      );
                    }
                    return null;
                  })
                )}

              <Line
                dataKey="y_test"
                stroke="#6b7280"
                strokeWidth={1.2}
                dot={false}
                name="Actual Price"
                isAnimationActive={false}
              />

              {modelKeys.map(({ prefix, color, label }) => (
                <Line
                  key={prefix}
                  dataKey={`${prefix}_mean`}
                  stroke={color}
                  strokeWidth={1.5}
                  dot={false}
                  name={label}
                  isAnimationActive={false}
                />
              ))}

              <Brush
                dataKey="date"
                height={30}
                stroke="#d4d4d8"
                fill="#fafafa"
                tickFormatter={formatXTick}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}

      {mergedData.length > 0 && (
        <div className="text-xs text-zinc-400 text-center">
          Showing {mergedData.length} hourly data points ({Math.ceil(mergedData.length / 24)} days).
          Drag the slider or pick a start date to navigate. Use the brush at the bottom of the chart to zoom within the loaded range.
        </div>
      )}
    </div>
  );
}
