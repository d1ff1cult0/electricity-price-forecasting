export interface Experiment {
  name: string;
  displayName: string;
  models: ModelInfo[];
  hasResults: boolean;
}

export interface ModelInfo {
  name: string;
  displayName: string;
  runs: number;
  hasSummary: boolean;
}

export interface RunMetrics {
  MAE: number;
  MSE: number;
  RMSE: number;
  MAPE: number;
  R2: number;
  PICP: number;
  MPIW: number;
  PINAW: number;
  IntervalScore: number;
  CRPS: number;
  NLL: number;
  Pinball_10: number;
  Pinball_50: number;
  Pinball_90: number;
  Avg_Pinball: number;
  training_time: number;
  [key: string]: number;
}

export interface ModelSummary {
  avg: RunMetrics;
  std: RunMetrics;
  n_runs: number;
  config?: Record<string, unknown>;
  experiment_version?: string;
}

export interface MetricsResponse {
  experiment: string;
  model: string;
  summary: ModelSummary | null;
  runs: RunMetrics[];
}

export interface PredictionPoint {
  index: number;
  hour: number;
  day: number;
  date: string;
  y_test: number;
  [key: string]: number | string;
}

export interface PredictionResponse {
  experiment: string;
  model: string;
  run: string;
  totalDays: number;
  startDay: number;
  endDay: number;
  keys: string[];
  data: PredictionPoint[];
}

export interface SelectedModel {
  experiment: string;
  model: string;
  run: string;
  color: string;
}

export const METRIC_LABELS: Record<string, { label: string; lowerBetter: boolean; format: string }> = {
  MAE: { label: "MAE", lowerBetter: true, format: ".2f" },
  MSE: { label: "MSE", lowerBetter: true, format: ".1f" },
  RMSE: { label: "RMSE", lowerBetter: true, format: ".2f" },
  MAPE: { label: "MAPE (%)", lowerBetter: true, format: ".1f" },
  R2: { label: "R²", lowerBetter: false, format: ".4f" },
  PICP: { label: "PICP", lowerBetter: false, format: ".4f" },
  MPIW: { label: "MPIW", lowerBetter: true, format: ".2f" },
  PINAW: { label: "PINAW", lowerBetter: true, format: ".4f" },
  IntervalScore: { label: "Interval Score", lowerBetter: true, format: ".2f" },
  CRPS: { label: "CRPS", lowerBetter: true, format: ".2f" },
  NLL: { label: "NLL", lowerBetter: true, format: ".2f" },
  Pinball_10: { label: "Pinball 10%", lowerBetter: true, format: ".2f" },
  Pinball_50: { label: "Pinball 50%", lowerBetter: true, format: ".2f" },
  Pinball_90: { label: "Pinball 90%", lowerBetter: true, format: ".2f" },
  Avg_Pinball: { label: "Avg Pinball", lowerBetter: true, format: ".2f" },
  training_time: { label: "Training Time (s)", lowerBetter: true, format: ".1f" },
};

export const MODEL_COLORS = [
  "#2563eb", "#dc2626", "#16a34a", "#ca8a04", "#9333ea",
  "#0891b2", "#e11d48", "#65a30d", "#c2410c", "#7c3aed",
  "#0d9488", "#d97706", "#4f46e5", "#059669", "#be123c",
];

export const KEY_METRICS = ["MAE", "RMSE", "CRPS", "R2", "PICP", "IntervalScore"];
