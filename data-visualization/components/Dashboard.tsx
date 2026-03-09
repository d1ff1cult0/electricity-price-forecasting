"use client";

import { useState } from "react";
import Sidebar from "./Sidebar";
import MetricsView from "./MetricsView";
import PredictionsChart from "./PredictionsChart";
import type { SelectedModel } from "@/lib/types";

export default function Dashboard() {
  const [selectedModels, setSelectedModels] = useState<SelectedModel[]>([]);
  const [activeTab, setActiveTab] = useState<"metrics" | "predictions">("metrics");

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        selectedModels={selectedModels}
        onSelectionChange={setSelectedModels}
        activeTab={activeTab}
        onTabChange={setActiveTab}
      />
      <main className="flex-1 overflow-y-auto bg-white">
        {selectedModels.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center max-w-md">
              <div className="w-12 h-12 rounded-xl bg-zinc-100 flex items-center justify-center mx-auto mb-4">
                <svg className="w-6 h-6 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
                </svg>
              </div>
              <h2 className="text-lg font-semibold text-zinc-900 mb-1">
                Select models to compare
              </h2>
              <p className="text-sm text-zinc-500">
                Choose one or more models from the sidebar to view their metrics and prediction charts.
                You can compare models from different experiments.
              </p>
            </div>
          </div>
        ) : activeTab === "metrics" ? (
          <MetricsView selectedModels={selectedModels} />
        ) : (
          <PredictionsChart selectedModels={selectedModels} />
        )}
      </main>
    </div>
  );
}
