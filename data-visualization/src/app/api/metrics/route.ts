import { NextResponse } from 'next/server';
import { getModelMetrics, getRunMetrics } from '@/lib/results';

export async function GET(request: Request) {
    const { searchParams } = new URL(request.url);
    const experiment = searchParams.get('experiment');
    const model = searchParams.get('model');
    const run = searchParams.get('run');

    if (!experiment || !model) {
        return NextResponse.json({ error: 'Missing experiment or model param' }, { status: 400 });
    }

    try {
        const metrics = run
            ? getRunMetrics(experiment, model, run)
            : getModelMetrics(experiment, model);

        return NextResponse.json(metrics || {});
    } catch (error: any) {
        return NextResponse.json({ error: error.message }, { status: 500 });
    }
}
