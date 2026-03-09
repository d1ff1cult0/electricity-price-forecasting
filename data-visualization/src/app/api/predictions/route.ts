import { NextResponse } from 'next/server';
import { getPredictions, getAveragePredictions } from '@/lib/results';

export async function GET(request: Request) {
    const { searchParams } = new URL(request.url);
    const experiment = searchParams.get('experiment');
    const model = searchParams.get('model');
    const run = searchParams.get('run'); // Can be a specific run or 'average'

    if (!experiment || !model || !run) {
        return NextResponse.json({ error: 'Missing experiment, model, or run param' }, { status: 400 });
    }

    try {
        const predictions = run === 'average'
            ? await getAveragePredictions(experiment, model)
            : await getPredictions(experiment, model, run);

        if (!predictions) {
            return NextResponse.json({ error: 'Predictions not found' }, { status: 404 });
        }

        if (predictions.error) {
            return NextResponse.json({ error: predictions.error }, { status: 500 });
        }

        // Transform raw arrays into chart-friendly array of objects:
        // [{ index, actual, predicted, q_025, q_975 ... }]
        const length = predictions.y_test?.length || predictions.actual_prices?.length || predictions.y_pred_mean?.length || 0;
        const chartData = [];

        // Some models might name it 'actual_prices' or 'y_test' 
        const actualKey = 'y_test' in predictions ? 'y_test' : 'actual_prices';
        const meanKey = 'y_pred_mean';

        for (let i = 0; i < length; i++) {
            const pt: Record<string, number> = { index: i };
            if (predictions[actualKey]) pt.actual = predictions[actualKey][i];
            if (predictions[meanKey]) pt.predicted = predictions[meanKey][i];

            // Map all quantiles generically
            Object.keys(predictions).forEach(key => {
                if (key.startsWith('q_')) {
                    pt[key] = predictions[key][i];
                }
            });
            chartData.push(pt);
        }

        return Response.json(chartData); // NextResponse.json has a strict size limit bug with Next.js sometimes, Response.json is safer for large payloads
    } catch (error: any) {
        return NextResponse.json({ error: error.message }, { status: 500 });
    }
}
