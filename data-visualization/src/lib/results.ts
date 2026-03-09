import fs from 'fs';
import path from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

const RESULTS_DIR = path.resolve(process.cwd(), '../results');

export function getExperiments() {
    if (!fs.existsSync(RESULTS_DIR)) return [];
    const entries = fs.readdirSync(RESULTS_DIR, { withFileTypes: true });
    return entries
        .filter((entry) => entry.isDirectory() && entry.name !== '.DS_Store')
        .map((entry) => entry.name);
}

export function getModels(experiment: string) {
    const expPath = path.join(RESULTS_DIR, experiment);
    if (!fs.existsSync(expPath)) return [];
    const entries = fs.readdirSync(expPath, { withFileTypes: true });
    return entries
        .filter((entry) => entry.isDirectory() && entry.name !== '.DS_Store')
        .map((entry) => entry.name);
}

export function getRuns(experiment: string, model: string) {
    const modelPath = path.join(RESULTS_DIR, experiment, model);
    if (!fs.existsSync(modelPath)) return [];
    const entries = fs.readdirSync(modelPath, { withFileTypes: true });
    return entries
        .filter((entry) => entry.isDirectory() && entry.name.startsWith('run_'))
        .map((entry) => entry.name);
}

export function getModelMetrics(experiment: string, model: string) {
    const modelPath = path.join(RESULTS_DIR, experiment, model);
    const summaryPath = path.join(modelPath, 'summary.json');
    if (fs.existsSync(summaryPath)) {
        return JSON.parse(fs.readFileSync(summaryPath, 'utf8'));
    }
    return null;
}

export function getRunMetrics(experiment: string, model: string, run: string) {
    const runPath = path.join(RESULTS_DIR, experiment, model, run);
    const metricsPath = path.join(runPath, 'metrics.json');
    if (fs.existsSync(metricsPath)) {
        return JSON.parse(fs.readFileSync(metricsPath, 'utf8'));
    }
    return null;
}

export async function getPredictions(experiment: string, model: string, run: string) {
    const npzPath = path.join(RESULTS_DIR, experiment, model, run, 'predictions.npz');
    if (!fs.existsSync(npzPath)) return null;

    // We write a quick inline python script to read numpy arrays and print as JSON
    const pythonScript = `
import numpy as np
import json
import sys

def default_encoder(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.flatten().tolist()
    else:
        return str(obj)

try:
    data = np.load('${npzPath}')
    out = {k: data[k].flatten().tolist() for k in data.files}
    print(json.dumps(out, default=default_encoder))
except Exception as e:
    print(json.dumps({"error": str(e)}))
`;

    // We use maxBuffer to 50MB since time series arrays might be sizable (e.g. 1MB json)
    const { stdout } = await execAsync(`python -c "${pythonScript}"`, { maxBuffer: 1024 * 1024 * 50 });
    return JSON.parse(stdout);
}

export async function getAveragePredictions(experiment: string, model: string) {
    const runs = getRuns(experiment, model);
    if (runs.length === 0) return null;

    // Get all runs
    const allPredictions = await Promise.all(runs.map(run => getPredictions(experiment, model, run)));
    const validRuns = allPredictions.filter(p => p && !p.error);

    if (validRuns.length === 0) return null;

    // Average them out
    const averaged: Record<string, number[]> = {};
    const keys = Object.keys(validRuns[0]);

    for (const key of keys) {
        if (!Array.isArray(validRuns[0][key])) continue;
        const length = validRuns[0][key].length;
        averaged[key] = new Array(length).fill(0);

        for (const runData of validRuns) {
            for (let i = 0; i < length; i++) {
                averaged[key][i] += runData[key][i];
            }
        }

        for (let i = 0; i < length; i++) {
            averaged[key][i] /= validRuns.length;
        }
    }

    return averaged;
}
