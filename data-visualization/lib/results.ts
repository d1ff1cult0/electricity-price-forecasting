import fs from "fs";
import path from "path";
import type { Experiment, ModelInfo, ModelSummary, RunMetrics } from "./types";

const RESULTS_DIR = process.env.RESULTS_DIR || path.resolve(process.cwd(), "../results");

const TEST_START_DATE = new Date("2025-08-08T00:00:00Z");

export function getResultsDir(): string {
  return RESULTS_DIR;
}

export function generateDateLabels(nDays: number, startDay: number = 0): string[] {
  const labels: string[] = [];
  for (let d = startDay; d < startDay + nDays; d++) {
    const dayDate = new Date(TEST_START_DATE);
    dayDate.setUTCDate(dayDate.getUTCDate() + d);
    const dateStr = dayDate.toISOString().split("T")[0];
    for (let h = 0; h < 24; h++) {
      labels.push(`${dateStr} ${String(h).padStart(2, "0")}:00`);
    }
  }
  return labels;
}

function isExperimentDir(dirPath: string): boolean {
  if (!fs.existsSync(dirPath) || !fs.statSync(dirPath).isDirectory()) return false;
  const entries = fs.readdirSync(dirPath);
  for (const entry of entries) {
    const subPath = path.join(dirPath, entry);
    if (!fs.statSync(subPath).isDirectory()) continue;
    const subEntries = fs.readdirSync(subPath);
    if (subEntries.some((e) => e.startsWith("run_"))) return true;
  }
  return false;
}

function getModelInfo(modelPath: string): ModelInfo | null {
  if (!fs.existsSync(modelPath) || !fs.statSync(modelPath).isDirectory()) return null;
  const entries = fs.readdirSync(modelPath);
  const runDirs = entries.filter((e) => e.startsWith("run_") && fs.statSync(path.join(modelPath, e)).isDirectory());
  if (runDirs.length === 0) return null;

  const hasSummary = fs.existsSync(path.join(modelPath, "summary.json"));
  const dirName = path.basename(modelPath);

  return {
    name: dirName,
    displayName: dirName.replace(/_/g, " ").replace(/\(([^)]+)\)/g, "($1)"),
    runs: runDirs.length,
    hasSummary,
  };
}

export function listExperiments(): Experiment[] {
  if (!fs.existsSync(RESULTS_DIR)) return [];

  const entries = fs.readdirSync(RESULTS_DIR);
  const experiments: Experiment[] = [];

  for (const entry of entries) {
    const dirPath = path.join(RESULTS_DIR, entry);
    if (!isExperimentDir(dirPath)) continue;

    const subEntries = fs.readdirSync(dirPath);
    const models: ModelInfo[] = [];

    for (const sub of subEntries) {
      const modelInfo = getModelInfo(path.join(dirPath, sub));
      if (modelInfo) models.push(modelInfo);
    }

    if (models.length > 0) {
      experiments.push({
        name: entry,
        displayName: entry.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
        models,
        hasResults: fs.existsSync(path.join(dirPath, "results.json")),
      });
    }
  }

  return experiments;
}

export function readRunMetrics(experiment: string, model: string, runIdx: number): RunMetrics | null {
  const metricsPath = path.join(RESULTS_DIR, experiment, model, `run_${runIdx}`, "metrics.json");
  if (!fs.existsSync(metricsPath)) return null;
  const raw = fs.readFileSync(metricsPath, "utf-8");
  return JSON.parse(raw) as RunMetrics;
}

export function readAllRunMetrics(experiment: string, model: string): RunMetrics[] {
  const modelDir = path.join(RESULTS_DIR, experiment, model);
  if (!fs.existsSync(modelDir)) return [];

  const entries = fs.readdirSync(modelDir);
  const runDirs = entries
    .filter((e) => e.startsWith("run_"))
    .sort((a, b) => {
      const aIdx = parseInt(a.replace("run_", ""));
      const bIdx = parseInt(b.replace("run_", ""));
      return aIdx - bIdx;
    });

  const metrics: RunMetrics[] = [];
  for (const dir of runDirs) {
    const metricsPath = path.join(modelDir, dir, "metrics.json");
    if (fs.existsSync(metricsPath)) {
      const raw = fs.readFileSync(metricsPath, "utf-8");
      metrics.push(JSON.parse(raw) as RunMetrics);
    }
  }
  return metrics;
}

export function readModelSummary(experiment: string, model: string): ModelSummary | null {
  const summaryPath = path.join(RESULTS_DIR, experiment, model, "summary.json");
  if (!fs.existsSync(summaryPath)) return null;
  const raw = fs.readFileSync(summaryPath, "utf-8");
  return JSON.parse(raw) as ModelSummary;
}

export function getPredictionsPath(experiment: string, model: string, runIdx: number): string {
  return path.join(RESULTS_DIR, experiment, model, `run_${runIdx}`, "predictions.npz");
}

export function getRunCount(experiment: string, model: string): number {
  const modelDir = path.join(RESULTS_DIR, experiment, model);
  if (!fs.existsSync(modelDir)) return 0;
  return fs.readdirSync(modelDir).filter((e) => e.startsWith("run_")).length;
}
