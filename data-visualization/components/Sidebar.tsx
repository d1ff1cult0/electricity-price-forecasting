"use client";

import { useEffect, useState } from "react";
import type { Experiment, SelectedModel } from "@/lib/types";
import { MODEL_COLORS } from "@/lib/types";

interface SidebarProps {
  selectedModels: SelectedModel[];
  onSelectionChange: (models: SelectedModel[]) => void;
  activeTab: "metrics" | "predictions";
  onTabChange: (tab: "metrics" | "predictions") => void;
}

export default function Sidebar({
  selectedModels,
  onSelectionChange,
  activeTab,
  onTabChange,
}: SidebarProps) {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [expandedExp, setExpandedExp] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/experiments")
      .then((r) => r.json())
      .then((data) => {
        setExperiments(data.experiments);
        setLoading(false);
        if (data.experiments.length > 0) {
          setExpandedExp(data.experiments[0].name);
        }
      })
      .catch(() => setLoading(false));
  }, []);

  const isSelected = (experiment: string, model: string) =>
    selectedModels.some((m) => m.experiment === experiment && m.model === model);

  const toggleModel = (experiment: string, model: string) => {
    if (isSelected(experiment, model)) {
      onSelectionChange(
        selectedModels.filter((m) => !(m.experiment === experiment && m.model === model))
      );
    } else {
      const nextColor = MODEL_COLORS[selectedModels.length % MODEL_COLORS.length];
      onSelectionChange([
        ...selectedModels,
        { experiment, model, run: "average", color: nextColor },
      ]);
    }
  };

  const updateRun = (experiment: string, model: string, run: string) => {
    onSelectionChange(
      selectedModels.map((m) =>
        m.experiment === experiment && m.model === model ? { ...m, run } : m
      )
    );
  };

  const getRunOptions = (experiment: string, modelName: string): number => {
    const exp = experiments.find((e) => e.name === experiment);
    return exp?.models.find((m) => m.name === modelName)?.runs ?? 0;
  };

  const formatModelName = (name: string) =>
    name.replace(/_/g, " ").replace(/\(([^)]+)\)/g, "($1)");

  return (
    <aside className="w-72 min-h-screen border-r border-zinc-200 bg-zinc-50 flex flex-col">
      <div className="p-4 border-b border-zinc-200">
        <h1 className="text-sm font-semibold tracking-tight text-zinc-900">
          EPF Results
        </h1>
        <p className="text-xs text-zinc-500 mt-0.5">Electricity Price Forecasting</p>
      </div>

      <div className="flex border-b border-zinc-200">
        <button
          onClick={() => onTabChange("metrics")}
          className={`flex-1 px-3 py-2 text-xs font-medium transition-colors ${
            activeTab === "metrics"
              ? "text-blue-600 border-b-2 border-blue-600"
              : "text-zinc-500 hover:text-zinc-700"
          }`}
        >
          Metrics
        </button>
        <button
          onClick={() => onTabChange("predictions")}
          className={`flex-1 px-3 py-2 text-xs font-medium transition-colors ${
            activeTab === "predictions"
              ? "text-blue-600 border-b-2 border-blue-600"
              : "text-zinc-500 hover:text-zinc-700"
          }`}
        >
          Predictions
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-1">
        {loading ? (
          <div className="text-xs text-zinc-500 p-2">Loading experiments...</div>
        ) : experiments.length === 0 ? (
          <div className="text-xs text-zinc-500 p-2">No experiments found</div>
        ) : (
          experiments.map((exp) => (
            <div key={exp.name}>
              <button
                onClick={() => setExpandedExp(expandedExp === exp.name ? null : exp.name)}
                className="w-full text-left px-2 py-1.5 rounded text-xs font-medium text-zinc-700 hover:bg-zinc-200/60 transition-colors flex items-center justify-between"
              >
                <span className="truncate">{exp.displayName}</span>
                <svg
                  className={`w-3 h-3 shrink-0 transition-transform ${expandedExp === exp.name ? "rotate-90" : ""}`}
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                </svg>
              </button>

              {expandedExp === exp.name && (
                <div className="ml-2 mt-0.5 space-y-0.5">
                  {exp.models.map((model) => {
                    const selected = isSelected(exp.name, model.name);
                    const selModel = selectedModels.find(
                      (m) => m.experiment === exp.name && m.model === model.name
                    );

                    return (
                      <div key={model.name} className="space-y-0.5">
                        <label className="flex items-center gap-2 px-2 py-1 rounded text-xs cursor-pointer hover:bg-zinc-200/60 transition-colors">
                          <input
                            type="checkbox"
                            checked={selected}
                            onChange={() => toggleModel(exp.name, model.name)}
                            className="rounded border-zinc-300 text-blue-600 focus:ring-blue-500 focus:ring-offset-0 h-3 w-3"
                          />
                          <span
                            className="truncate text-zinc-600"
                            title={formatModelName(model.name)}
                          >
                            {formatModelName(model.name)}
                          </span>
                          {selected && selModel && (
                            <span
                              className="ml-auto w-2 h-2 rounded-full shrink-0"
                              style={{ backgroundColor: selModel.color }}
                            />
                          )}
                        </label>

                        {selected && (
                          <div className="ml-6 pb-1">
                            <select
                              value={selModel?.run ?? "average"}
                              onChange={(e) =>
                                updateRun(exp.name, model.name, e.target.value)
                              }
                              className="w-full text-xs bg-white border border-zinc-200 rounded px-1.5 py-0.5 text-zinc-600"
                            >
                              <option value="average">Average (all runs)</option>
                              {Array.from(
                                { length: getRunOptions(exp.name, model.name) },
                                (_, i) => (
                                  <option key={i} value={String(i)}>
                                    Run {i}
                                  </option>
                                )
                              )}
                            </select>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          ))
        )}
      </div>

      {selectedModels.length > 0 && (
        <div className="border-t border-zinc-200 p-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-zinc-600">
              {selectedModels.length} model{selectedModels.length > 1 ? "s" : ""} selected
            </span>
            <button
              onClick={() => onSelectionChange([])}
              className="text-xs text-red-500 hover:text-red-600 transition-colors"
            >
              Clear
            </button>
          </div>
          <div className="space-y-1">
            {selectedModels.map((m) => (
              <div
                key={`${m.experiment}/${m.model}`}
                className="flex items-center gap-1.5 text-xs text-zinc-500"
              >
                <span
                  className="w-2 h-2 rounded-full shrink-0"
                  style={{ backgroundColor: m.color }}
                />
                <span className="truncate">{formatModelName(m.model)}</span>
                <span className="text-zinc-400 ml-auto shrink-0">
                  {m.run === "average" ? "avg" : `#${m.run}`}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </aside>
  );
}
