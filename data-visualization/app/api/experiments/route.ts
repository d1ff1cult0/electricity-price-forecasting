import { NextResponse } from "next/server";
import { listExperiments } from "@/lib/results";

export async function GET() {
  try {
    const experiments = listExperiments();
    return NextResponse.json({ experiments });
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to list experiments: ${error}` },
      { status: 500 }
    );
  }
}
