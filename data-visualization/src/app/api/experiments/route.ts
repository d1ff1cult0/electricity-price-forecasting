import { NextResponse } from 'next/server';
import { getExperiments, getModels, getRuns } from '@/lib/results';

export async function GET() {
    try {
        const experiments = getExperiments();
        const data = experiments.map((exp) => {
            const models = getModels(exp);
            return {
                name: exp,
                models: models.map((model) => ({
                    name: model,
                    runs: getRuns(exp, model),
                })),
            };
        });
        return NextResponse.json(data);
    } catch (error: any) {
        return NextResponse.json({ error: error.message }, { status: 500 });
    }
}
