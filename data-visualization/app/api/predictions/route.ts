import { NextRequest, NextResponse } from "next/server";
import { readNpzSlice, getNpzTotalDays, getNpzTotalRows, averageArrays } from "@/lib/npz-reader";
import { getPredictionsPath, getRunCount, generateDateLabels } from "@/lib/results";
import type { PredictionPoint, PredictionResponse } from "@/lib/types";

export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl;
  const experiment = searchParams.get("experiment");
  const model = searchParams.get("model");
  const run = searchParams.get("run") ?? "0";
  const startDay = parseInt(searchParams.get("startDay") ?? "0");
  const endDay = searchParams.get("endDay");

  if (!experiment || !model) {
    return NextResponse.json(
      { error: "experiment and model parameters required" },
      { status: 400 }
    );
  }

  try {
    const firstPath = getPredictionsPath(experiment, model, 0);
    const totalDays = getNpzTotalDays(firstPath);
    const totalRows = getNpzTotalRows(firstPath);

    const end = endDay ? parseInt(endDay) : Math.min(totalDays, startDay + 30);

    const STRIDE = 24;
    const startRow = startDay * STRIDE;
    const endRow = Math.min(end * STRIDE, totalRows);

    let arrays;
    if (run === "average") {
      const nRuns = getRunCount(experiment, model);
      const allRuns = [];
      for (let i = 0; i < nRuns; i++) {
        const p = getPredictionsPath(experiment, model, i);
        allRuns.push(readNpzSlice(p, startRow, endRow, STRIDE));
      }
      arrays = averageArrays(allRuns);
    } else {
      const runIdx = parseInt(run);
      const p = getPredictionsPath(experiment, model, runIdx);
      arrays = readNpzSlice(p, startRow, endRow, STRIDE);
    }

    const nDays = arrays.y_test?.length ?? 0;
    const dates = generateDateLabels(nDays, startDay);
    const keys = Object.keys(arrays).filter((k) => k !== "y_test");

    const data: PredictionPoint[] = [];
    for (let d = 0; d < nDays; d++) {
      for (let h = 0; h < 24; h++) {
        const idx = d * 24 + h;
        const point: PredictionPoint = {
          index: idx,
          hour: h,
          day: startDay + d,
          date: dates[idx] ?? "",
          y_test: round(arrays.y_test?.[d]?.[h] ?? 0),
        };
        for (const key of keys) {
          point[key] = round(arrays[key]?.[d]?.[h] ?? 0);
        }
        data.push(point);
      }
    }

    const response: PredictionResponse = {
      experiment,
      model,
      run,
      totalDays,
      startDay,
      endDay: end,
      keys,
      data,
    };

    return NextResponse.json(response);
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to read predictions: ${error}` },
      { status: 500 }
    );
  }
}

function round(val: number): number {
  return Math.round(val * 100) / 100;
}
