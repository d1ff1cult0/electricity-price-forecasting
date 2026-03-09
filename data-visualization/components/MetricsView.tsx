"use client";

import { useEffect, useState, useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { SelectedModel, MetricsResponse, RunMetrics } from "@/lib/types";
import { METRIC_LABELS, KEY_METRICS } from "@/lib/types";

interface MetricsViewProps {
  selectedModels: SelectedModel[];
}

interface ModelMetrics {
  key: string;
  label: string;
  color: string;
  summary: { avg: RunMetrics; std: RunMetrics } | null;
  runs: RunMetrics[];
}

export default function MetricsView({ selectedModels }: MetricsViewProps) {
  const [modelMetrics, setModelMetrics] = useState<ModelMetrics[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(KEY_METRICS);
  const [showAllMetrics, setShowAllMetrics] = useState(false);

  useEffect(() => {
    if (selectedModels.length === 0) return;
    setLoading(true);

    const fetches = selectedModels.map((m) =>
      fetch(`/api/metrics?experiment=${encodeURIComponent(m.experiment)}&model=${encodeURIComponent(m.model)}`)
        .then((r) => r.json())
        .then((data: MetricsResponse) => ({
          key: `${m.experiment}/${m.model}`,
          label: m.model.replace(/_/g, " "),
          color: m.color,
          summary: data.summary ? { avg: data.summary.avg, std: data.summary.std } : null,
          runs: data.runs,
        }))
    );

    Promise.all(fetches)
      .then(setModelMetrics)
      .finally(() => setLoading(false));
  }, [selectedModels]);

  const allMetricKeys = useMemo(() => {
    if (modelMetrics.length === 0) return [];
    const keys = new Set<string>();
    for (const m of modelMetrics) {
      const source = m.summary?.avg ?? m.runs[0];
      if (source) Object.keys(source).forEach((k) => keys.add(k));
    }
    return Array.from(keys).filter((k) => k in METRIC_LABELS);
  }, [modelMetrics]);

  const displayMetrics = showAllMetrics ? allMetricKeys : selectedMetrics;

  const formatValue = (val: number | null | undefined, metric: string): string => {
    if (val === null || val === undefined) return "N/A";
    const meta = METRIC_LABELS[metric];
    if (!meta) return val.toFixed(2);
    const fmtMatch = meta.format.match(/\.(\d+)f/);
    const decimals = fmtMatch ? parseInt(fmtMatch[1]) : 2;
    return val.toFixed(decimals);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-sm text-zinc-500">Loading metrics...</div>
      </div>
    );
  }

  if (modelMetrics.length === 0) return null;

  return (
    <div className="p-6 space-y-6 max-w-full">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-zinc-900">
          Metrics Comparison
        </h2>
        <button
          onClick={() => setShowAllMetrics(!showAllMetrics)}
          className="text-xs px-3 py-1.5 rounded-md border border-zinc-200 text-zinc-600 hover:bg-zinc-50 transition-colors"
        >
          {showAllMetrics ? "Show Key Metrics" : "Show All Metrics"}
        </button>
      </div>

      {!showAllMetrics && (
        <div className="flex flex-wrap gap-1.5">
          {allMetricKeys.map((key) => (
            <button
              key={key}
              onClick={() =>
                setSelectedMetrics((prev) =>
                  prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]
                )
              }
              className={`text-xs px-2 py-1 rounded-md border transition-colors ${
                selectedMetrics.includes(key)
                  ? "border-blue-300 bg-blue-50 text-blue-700"
                  : "border-zinc-200 text-zinc-500 hover:bg-zinc-50"
              }`}
            >
              {METRIC_LABELS[key]?.label ?? key}
            </button>
          ))}
        </div>
      )}

      <div className="overflow-x-auto rounded-lg border border-zinc-200">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-zinc-50 border-b border-zinc-200">
              <th className="text-left py-2 px-3 font-medium text-zinc-600 text-xs">
                Metric
              </th>
              {modelMetrics.map((m) => (
                <th key={m.key} className="text-right py-2 px-3 text-xs font-medium">
                  <div className="flex items-center justify-end gap-1.5">
                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: m.color }} />
                    <span className="text-zinc-600 truncate max-w-[140px]">
                      {m.label}
                    </span>
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {displayMetrics.map((metric, idx) => {
              const meta = METRIC_LABELS[metric];
              const values = modelMetrics.map((m) => {
                return m.summary?.avg?.[metric] ?? m.runs[0]?.[metric] ?? null;
              });
              const validValues = values.filter((v): v is number => v !== null);
              const best = meta?.lowerBetter
                ? Math.min(...validValues)
                : Math.max(...validValues);

              return (
                <tr
                  key={metric}
                  className={`border-b border-zinc-100 ${
                    idx % 2 === 0 ? "" : "bg-zinc-50/50"
                  }`}
                >
                  <td className="py-2 px-3 font-medium text-zinc-700 text-xs">
                    {meta?.label ?? metric}
                    {meta && (
                      <span className="ml-1 text-zinc-400">
                        ({meta.lowerBetter ? "lower better" : "higher better"})
                      </span>
                    )}
                  </td>
                  {modelMetrics.map((m, mIdx) => {
                    const val = values[mIdx];
                    const std = m.summary?.std?.[metric];
                    const isBest = val !== null && val === best;

                    return (
                      <td
                        key={m.key}
                        className={`py-2 px-3 text-right text-xs tabular-nums ${
                          isBest
                            ? "font-semibold text-emerald-600"
                            : "text-zinc-600"
                        }`}
                      >
                        {formatValue(val, metric)}
                        {std !== undefined && std !== null && (
                          <span className="text-zinc-400 ml-1">
                            ± {formatValue(std, metric)}
                          </span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {displayMetrics.map((metric) => {
          const meta = METRIC_LABELS[metric];
          const barData = modelMetrics.map((m) => ({
            name: m.label.length > 20 ? m.label.substring(0, 20) + "..." : m.label,
            value: m.summary?.avg?.[metric] ?? m.runs[0]?.[metric] ?? 0,
            std: m.summary?.std?.[metric] ?? 0,
            color: m.color,
          }));

          return (
            <div
              key={metric}
              className="border border-zinc-200 rounded-lg p-4 bg-white"
            >
              <h3 className="text-xs font-medium text-zinc-600 mb-3">
                {meta?.label ?? metric}
              </h3>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={barData} layout="vertical" margin={{ left: 10, right: 20, top: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e4e4e7" />
                  <XAxis type="number" tick={{ fontSize: 10, fill: "#71717a" }} stroke="#d4d4d8" />
                  <YAxis
                    dataKey="name"
                    type="category"
                    tick={{ fontSize: 10, fill: "#71717a" }}
                    width={100}
                    stroke="#d4d4d8"
                  />
                  <Tooltip
                    contentStyle={{
                      fontSize: 11,
                      backgroundColor: "#ffffff",
                      border: "1px solid #e4e4e7",
                      borderRadius: 6,
                    }}
                    formatter={(value, _name, props) => {
                      const v = Number(value);
                      const std = (props.payload as Record<string, number>)?.std ?? 0;
                      return [`${v.toFixed(3)} ± ${std.toFixed(3)}`, meta?.label ?? metric];
                    }}
                  />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={20}>
                    {barData.map((entry, i) => (
                      <Cell key={i} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          );
        })}
      </div>

      {modelMetrics.some((m) => m.runs.length > 1) && (
        <div className="border border-zinc-200 rounded-lg p-4 bg-white">
          <h3 className="text-sm font-medium text-zinc-700 mb-3">
            Per-Run Breakdown
          </h3>
          {modelMetrics
            .filter((m) => m.runs.length > 1)
            .map((m) => (
              <div key={m.key} className="mb-4 last:mb-0">
                <div className="flex items-center gap-2 mb-2">
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: m.color }} />
                  <span className="text-xs font-medium text-zinc-600">{m.label}</span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-zinc-500">
                        <th className="text-left py-1 px-2 font-medium">Run</th>
                        {KEY_METRICS.map((k) => (
                          <th key={k} className="text-right py-1 px-2 font-medium">
                            {METRIC_LABELS[k]?.label ?? k}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {m.runs.map((run, idx) => (
                        <tr
                          key={idx}
                          className="border-t border-zinc-100 text-zinc-600"
                        >
                          <td className="py-1 px-2 font-medium">#{idx}</td>
                          {KEY_METRICS.map((k) => (
                            <td key={k} className="text-right py-1 px-2 tabular-nums">
                              {formatValue(run[k], k)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
        </div>
      )}
    </div>
  );
}
