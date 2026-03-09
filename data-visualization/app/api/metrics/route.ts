import { NextRequest, NextResponse } from "next/server";
import { readAllRunMetrics, readModelSummary, listExperiments } from "@/lib/results";
import type { MetricsResponse } from "@/lib/types";

export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl;
  const experiment = searchParams.get("experiment");
  const model = searchParams.get("model");

  if (!experiment) {
    return NextResponse.json({ error: "experiment parameter required" }, { status: 400 });
  }

  try {
    if (model) {
      const runs = readAllRunMetrics(experiment, model);
      const summary = readModelSummary(experiment, model);
      const response: MetricsResponse = {
        experiment,
        model,
        summary,
        runs,
      };
      return NextResponse.json(response);
    }

    const experiments = listExperiments();
    const exp = experiments.find((e) => e.name === experiment);
    if (!exp) {
      return NextResponse.json({ error: `Experiment not found: ${experiment}` }, { status: 404 });
    }

    const allMetrics: MetricsResponse[] = [];
    for (const m of exp.models) {
      const runs = readAllRunMetrics(experiment, m.name);
      const summary = readModelSummary(experiment, m.name);
      allMetrics.push({ experiment, model: m.name, summary, runs });
    }

    return NextResponse.json({ metrics: allMetrics });
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to read metrics: ${error}` },
      { status: 500 }
    );
  }
}
